"""Per-tool / per-type policy — the selective shell in front of arbitrary tool output.

Measurement shows terse's value is strongly per-tool (0-30%): big on record/symbol-
shaped verbose output (gh APIs, runecho), ~0 on already-projected high-cardinality
data (kb) and single compact objects. So compression must be SELECTIVE, not blanket.

A policy is an ordered list of rules. The first rule whose `match.tool` glob matches
the tool name wins; if none match, `defaults` applies. Each rule names the lossless
TIERS to run (subset of minify/tabularize/dictionary; empty = passthrough) and an
optional per-field map.

Fail-closed (principle #37): an unmatched tool gets the lossless default and NEVER a
lossy op. Lossy field modes are parsed and surfaced but NOT executed in v1 — those
tiers aren't built — so this layer is 100% lossless regardless of policy. `critical`
fields are honored trivially today (lossless preserves everything); they become the
denylist once lossy tiers exist.

Format is JSON (stdlib, zero deps). YAML/TOML can be added behind the same loader.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import lossy as lossy_mod
from . import transforms

VALID_TIERS = ("minify", "tabularize", "dictionary", "embedded")

# What `defaults.tiers` means when a policy file omits it. Deliberately NOT `VALID_TIERS`:
# `embedded` is opt-in, so widening the valid set must not silently switch a new tier on for
# every policy file already on disk. Each costs a primer paragraph on that server's primer
# (#168), and #170 is the precedent for what that costs when the tier rarely fires —
# so a tier joins this tuple on measured evidence (`policy generate`), never by default.
DEFAULT_TIERS = ("minify", "tabularize", "dictionary")
LOSSY_MODES = ("truncate", "drop-to-retrieve")  # implemented; summarize is still deferred

# multiproxy's peer-qualifier separator (e.g. "gh__search" for peer "gh"'s "search"
# tool). Defined here, not in multiproxy.py, so Policy.select can recognize and strip
# it: a corpus captured through multiproxy stores each payload under its peer-qualified
# name (to keep same-named tools on different peers from colliding in the corpus), but
# a policy rule is authored against the downstream tool's own bare name — without this,
# every rule silently misses for a multiproxy-captured corpus (fnmatch("gh__search",
# "search") is False), and drop-eval/measure would score it as having nothing to test.
PREFIX_SEP = "__"

# Non-overridable floor for the never-lossy exclusion: a server whose NAME matches this
# is treated as carrying secrets and can NEVER receive a lossy transform, no matter what a
# policy file says. This is the belt-and-suspenders the security review required — for the
# one class of data (credentials) where "policy typo" and "leak" are the same failure, a
# policy-file exclusion alone is not a strong enough boundary. The install-time classifier
# and the policy's `never_lossy_servers` list are the PRIMARY, per-server mechanism (they
# also catch sensitive servers whose names don't match, e.g. a personal KB or a launcher
# alias); this pattern is the additional structural backstop that cannot be turned off.
SENSITIVE_SERVER_RE = re.compile(
    # Credential-shaped names.
    r"secret|credential|vault|passwd|password|token|api[-_ ]?key|keyring|auth"
    # Password managers, which carry nothing BUT credentials and whose names contain none
    # of the words above. `\bop\b` is deliberately word-bounded: unbounded it would match
    # the "op" inside "openapi"/"operator" and make the floor fire on unrelated servers.
    r"|1password|onepassword|bitwarden|lastpass|dashlane|keepass|\bop\b"
    # Personal-data servers. Not credentials, but the `never_lossy` floor exists to stop a
    # lossy transform silently eating content the operator cannot afford to lose, and a
    # mailbox or contact list is exactly that.
    #
    # `mail` is UNBOUNDED on the right so it also catches `mailbox`/`mailstore` — a
    # trailing `\b` looked tighter but put the boundary between `l` and `b`, where there
    # isn't one, so `mailbox` sailed past the very floor this line adds. Erring toward
    # over-match here is correct: the cost is one server needlessly lossless.
    #
    # `drive` is BOUNDED, for the opposite reason: unbounded it matched `webdriver`,
    # `chromedriver` and `selenium-driver` — browser-automation servers, whose verbose
    # output is precisely what lossy exists for. Bounded plus the real product spellings
    # keeps Google Drive covered without disabling lossy on Playwright-shaped servers.
    r"|gmail|mail|inbox|contacts|calendar|photos|\bdrive\b|gdrive|onedrive|google-?drive",
    re.I
)


@dataclass
class Rule:
    tool_glob: str
    tiers: tuple[str, ...]
    # Per-field map: {path: {"lossy": mode, "max": N} | {"critical": true}}. `critical`
    # is a field flag (never made lossy); see lossy.critical_paths.
    fields: dict[str, dict] = field(default_factory=dict)
    # `"capture": false` — never PERSIST this tool's payloads: the proxy skips both the
    # --capture-dir corpus tee and the --debug-log replay trace for it (#85). Distinct
    # from `tiers: ()`, which only stops compression/diff state: the capture tee sits
    # above the tier logic and would otherwise write a payload to disk regardless.
    #
    # This exists so "this tool's output must never hit disk" is a declarative property
    # of the policy — surviving re-wraps and reviewable in one place — instead of an
    # operator remembering never to pass --capture-dir to one particular wrapper. That
    # is what makes a credential-returning server (secret-broker's reveal_credential
    # returns a plaintext value by design) safe to wrap by construction rather than by
    # discipline. Default True = exactly the pre-#85 behavior.
    capture: bool = True
    # `"require_server_name": true` — refuse to start the proxy at all if `--server-name`
    # was not given. Closes a fail-OPEN gap: a server-scoped rule like this one (glob
    # `secret-broker.*`) is only reachable via the candidate `_match_candidates` inserts
    # when `server` is truthy — omit `--server-name` and the candidate is never
    # synthesized, the rule never matches, and the tool falls through to
    # `Policy.select`'s unmatched-tool fallback, whose `capture` is the dataclass
    # default `True`. A credential-returning tool guarded ONLY by a server-scoped
    # deny-all rule is therefore safe by a CLI flag being present, not by the policy —
    # exactly the "discipline, not construction" failure #85 (`capture`, above) exists
    # to close for the disk-write case. This closes it for the "rule never reached" case.
    #
    # Deliberately a per-rule, author-declared flag rather than something inferred from
    # the glob shape: a glob like `secret-broker.*` needs server-qualification to ever
    # match (its tools are literally named `secret.*`, never self-prefixed), but a glob
    # like `kb.*` does NOT — `kb`'s tools self-prefix (`kb.read.list_principles` matches
    # `kb.*` on the bare candidate alone). No syntactic rule tells these apart; only the
    # policy author, who knows the server's real tool names, does. `_match_candidates`'s
    # `bare.startswith(f"{server}.")` skip-check (below) already carries a comment on
    # exactly this ambiguity from #199 — same reason a heuristic here would eventually
    # be wrong in one direction or the other. Default False = every existing policy is
    # unaffected until its author opts a specific rule in.
    require_server_name: bool = False
    # `"structured": "compress"` — also run this tool's `structuredContent` through the
    # codec, replacing the typed field with a terse envelope (#128).
    #
    # Default "leave" = the pre-#128 behavior, and deliberately so despite the measurement
    # that motivated this. MCP 2025-06-18 lets a tool return `structuredContent` beside a
    # text block that mirrors it; measured against `claude` 2.1.218, the client forwards
    # the TYPED field to the model and discards the text block terse compresses — so on
    # such a tool terse currently delivers ~0% (see `scripts/probe/structured_content/`
    # and BENCHMARKS §6's scope note). Compressing it recovers ≈61%.
    #
    # It stays opt-in because the risk is ASYMMETRIC. Leaving it off means terse keeps
    # being a no-op for anyone who never reads the docs: bad, but inert. Turning it on by
    # default means that for any client which validates the typed field against the tool's
    # `outputSchema` — which the spec says clients SHOULD do — terse starts BREAKING tools
    # that worked. terse cannot detect which client it sits behind, and a default that
    # silently violates a declared schema for unknown clients is the exact failure class
    # #131 and #133 were about.
    #
    # "auto" (the default) resolves that per CONNECTED CLIENT instead of per guess. The
    # MCP `initialize` request carries `params.clientInfo` — a field the client declares —
    # and the proxy sees it, so terse compresses the typed field for clients measured not
    # to validate it (`STRUCTURED_SAFE_CLIENTS`) and leaves it alone for every other
    # client, including any that never identifies itself. That removes the premise the
    # #134 default rested on ("terse cannot detect which client it sits behind"), which
    # was simply wrong. Explicit "leave"/"compress" always override the resolution.
    #
    # `"replace"` goes one step further: compress the typed field AND drop the text block
    # that mirrors it. Read the measurement before enabling it, because it is NOT a
    # context saving for the client this was all measured against:
    #
    #   raw 2,596 chars in context -> "compress" 1,008 -> "replace" 1,008.
    #
    # No change. `claude` 2.1.218 had *already* discarded the block, so removing it takes
    # 2,596 chars off the stdio pipe and nothing at all off the model's context. Its real
    # beneficiary is a client that forwards BOTH fields — for which it roughly halves the
    # result — and no such client has been measured. Two things it does deliver today: it
    # ends the shape contradiction (under "compress" the block can be a cross-call DIFF
    # while the typed field is a full envelope, so a both-forwarding client would receive
    # two disagreeing renderings), and it makes the wire match what the model receives.
    #
    # It is NOT what "auto" resolves to, and deliberately so. Every mode up to "compress"
    # is invisible to a client that ignores `structuredContent` — the block is still there,
    # still complete. "replace" is the first mode that removes information from the wire,
    # so a client that changes its mind about which field it reads gets an empty result
    # rather than a suboptimal one. That is a loud failure, and recoverable with a one-line
    # `"structured": "leave"`, but it is not a failure worth defaulting anyone into — least
    # of all for a measured benefit of zero.
    structured: str = "auto"

    def lossy_fields(self) -> list[str]:
        return [k for k, v in self.fields.items() if v.get("lossy")]


@dataclass
class Policy:
    rules: list[Rule]
    default_tiers: tuple[str, ...] = DEFAULT_TIERS
    # Cross-call diffing is OFF by default (#170). Opt IN per-policy (`"diff": true`) or
    # with `proxy --diff`; the tier itself is unchanged and still falls back to the full
    # compressed form whenever a diff doesn't apply or win.
    #
    # It was default-ON from PR #76, on the strength of a validation program that measured
    # CORRECTNESS — pair fluency (`fluency --diff`), nested-record coverage (#72), drift
    # soak (#75: mechanical zero-drift, depth-1..5 behavioral PASS). All of that still
    # holds. What it did not measure is net token value, because the primer was not yet
    # known to be the dominant cost term.
    #
    # It is: the primer paragraph explaining the diff envelopes adds 190 cl100k tokens to
    # that server's primer (the largest single section), attached once per session to
    # each wrapped server that emits a terse form. The live all-time ledger over 13.3 days:
    # 7 diff emissions in 1,828 blocks (0.38%), saving 5,052 tokens. At 190 tokens per
    # attach that saving is gone after ~27 session-server pairs — about two a day across
    # the whole fleet over that window, which an active install passes immediately.
    #
    # This was originally computed against the pre-#211 per-turn charge (190 x 24,252
    # turns / 5,052 = 912x at one wrapped server, 2,736x at three). #211 made the primer
    # lazy and once-per-session, which shrank the cost side by the session turn count and
    # did not come close to changing the verdict. The restatement above is deliberately in
    # attaches rather than a multiplier: turning it back into one needs a session count,
    # which a stdio proxy cannot observe (see `stats.py` on the #144/#186/#188 family).
    #
    # The dominant refusal is `multiblock` (444 vs 7 emissions): most real results arrive
    # as several content blocks. If #140's partial multi-block join converts those into
    # emissions, re-measure and flip back — the tier is not the problem, its hit rate is.
    diff: bool = False
    # Join every text content block of a multi-block result into ONE record array before
    # compressing (#116). Several servers return one record per block; independently each
    # block is a single object, so `tabularize` never sees an array and the cross-call
    # diff tier — which reasons about one logical payload — is skipped entirely (it was
    # 71% of real traffic). Joining folds the records together AND makes the result
    # diff-eligible. ON by default: the per-result verify-before-emit gate proves
    # losslessness on every join, and a join is a one-shot reshape with none of the
    # accumulating drift risk that kept `diff` opt-in longer. Opt out per-policy
    # (`"join_blocks": false`) or with `proxy --no-join-blocks`. Independent of `diff`:
    # with diffing off, joining still folds records into one compressed block.
    join_blocks: bool = True
    # Bound dangling-reference drift (#8): force a self-contained full result (a
    # keyframe, like video I-frames) after this many consecutive diffs per tool, so a
    # chained diff can never drift more than K turns from an anchor the model can
    # reconstruct from scratch. 0 disables keyframing (diff whenever it wins).
    diff_keyframe_interval: int = 5
    # Servers on which lossy transforms are structurally forbidden (forced 100% lossless,
    # regardless of any field marked lossy). Populated at `install-mcp` time by the
    # sensitivity classifier — the point where terse knows the server's identity and can
    # confirm with the operator — so the decision is baked per-server, reviewable in one
    # place, and never re-derived by a runtime heuristic. See `server_never_lossy`, which
    # ORs this with the non-overridable SENSITIVE_SERVER_RE floor.
    never_lossy_servers: frozenset[str] = frozenset()

    def server_never_lossy(self, server: str | None) -> bool:
        """True when lossy transforms must be suppressed for this server. Two layers: the
        baked-at-install `never_lossy_servers` list (primary — catches sensitive servers by
        identity, including ones whose names look innocuous), OR the non-overridable
        SENSITIVE_SERVER_RE name floor (structural backstop a policy typo cannot defeat).
        A None/empty server (identity unknown) is NOT auto-excluded — the exclusion is a
        deliberate, identified decision, and lossy is opt-in per field regardless."""
        if not server:
            return False
        return server in self.never_lossy_servers or bool(SENSITIVE_SERVER_RE.search(server))

    def select(self, tool: str, server: str | None = None) -> Rule:
        """First rule whose glob matches the tool name, else the lossless default.

        Candidates are tried in order and the first rule matching any of them wins:

        1. `{server}.{bare tool}` when `server` is known (#83) — a *server-scoped* glob
           like `runecho.*` only ever matched servers that happen to self-prefix their
           own tool names (kb names its tools `kb.read.*`; runecho calls its tool plain
           `structure`), so such a rule silently missed for everyone else and fell
           through to the defaults. Synthesizing the qualified name makes a
           server-scoped rule mean the same thing for every server.
        2. `tool` as given.
        3. The bare part after PREFIX_SEP, for a multiproxy peer-qualified name, so a
           rule authored against a downstream tool's own name still matches a
           multiproxy-captured corpus entry for it.

        Candidate order is MAJOR over rule order: candidate 1 is tried against EVERY rule
        before candidate 2 is tried against any. So adding a candidate ahead of another does
        not merely widen what matches — it can hand the match to a different rule, and a
        broad rule reached one candidate earlier outranks a specific rule that would have
        matched one candidate later. That is why candidate 1 is synthesized only when the
        bare name cannot already carry a server-scoped rule; see `_match_candidates`, where
        #199 proposed relaxing that guard and the relaxation was measured to be both
        unnecessary and unsafe.
        """
        for candidate in self._match_candidates(tool, server):
            for rule in self.rules:
                if fnmatch.fnmatch(candidate, rule.tool_glob):
                    return rule
        return Rule(tool_glob="*", tiers=self.default_tiers)

    @staticmethod
    def _match_candidates(tool: str, server: str | None = None) -> list[str]:
        bare = tool.partition(PREFIX_SEP)[2] if PREFIX_SEP in tool else tool
        candidates = [tool] if bare == tool else [tool, bare]
        # Qualified form first: a server-scoped rule is the more specific intent, so it
        # should win over a bare-name rule the same way multiproxy's peer-qualified
        # candidate already outranks its bare fallback. Skipped when the tool already
        # carries the server as its own prefix (kb's `kb.read.*`), which would otherwise
        # synthesize a double-qualified `kb.kb.read.search` and miss the `kb.*` rule.
        #
        # #199 proposed extending the insertion to every PREFIX_SEP tool, on the theory that
        # a peer-qualified `gh__gh.api.items` self-prefixes `gh.` and so was skipped, while
        # candidate[0] `gh__gh.api.items` cannot be fnmatch-ed by `gh.*` — leaving the
        # server-scoped rule with nothing to match. NOT DONE, because the premise is false:
        # `select` also tries the BARE candidate `gh.api.items`, which `gh.*` matches, so the
        # rule was never missing. Verified across `gh__gh.api.items`, `gh__gh.rate_limit`,
        # `mcp__gh.search` and friends — `gh.*` matches all of them on the unmodified code.
        #
        # And the insertion is not free. Inserting at position 0 makes `gh.*` win one
        # candidate BEFORE a specific rule can match the bare name, which on terse's own
        # `policy.example.json` turns `{"tool": "*.rate_limit", "tiers": []}` — an explicit
        # passthrough — into `gh.*` with `result[].body` truncation (a 2,125-byte lossless
        # payload became 626 bytes with the body cut), and turns a
        # `{"tiers": [], "capture": false}` rule (USAGE.md's documented way to keep a
        # credential tool's output off disk) into a rule with `capture: true`. An operator's
        # exclusion silently becoming lossy — and capturable — is a far worse failure than
        # the one #199 set out to fix, which does not exist.
        if server and not bare.startswith(f"{server}."):
            candidates.insert(0, f"{server}.{bare}")
        return candidates

    def has_drop(self, server: str | None = None) -> bool:
        """True if a field drop-to-retrieve is reachable for `server`. Gates whether the
        proxy injects the synthetic terse.retrieve tool into tools/list (#10) AND whether
        its primer carries the 64-token dropped-field paragraph (#168).

        Server-aware for the same reason — and by exactly the same walk — as
        `reachable_tiers`: `select` returns the FIRST matching rule, so a rule that totally
        covers this server ends the walk, and a drop rule sitting after it can never fire
        here. This was the one primer gate still scanning every rule unconditionally while
        the other four took a server, so a wrapped server paid to explain (and advertised a
        tool for) a lossy form only some OTHER server's rule could produce.

        Same asymmetry of risk as `reachable_tiers`, resolved the same way: over-inclusion
        costs tokens, under-inclusion costs a handle nobody can retrieve. A rule whose glob
        merely LOOKS scoped elsewhere still counts, because `_match_candidates`' second
        candidate is the tool's own unqualified name.

        Two sound narrowings: glob-based termination AND `server_never_lossy(server)`, which
        structurally forbids every drop for this server regardless of what the policy rules
        say (#199).

        `_glob_covers_server`'s two documented unsound cases are both closed as of #199 and
        this docstring no longer inherits them. The metacharacter case — `kb[1]` vs `kb[1].*`,
        equal as strings but not as fnmatch patterns — is gone because cover is now proved
        structurally rather than by string equality. The `PREFIX_SEP` case — the claim that
        `gh__gh.api.items` or `mcp__kb.search` lets `select` reach a drop rule this walk
        terminated before — was never real: `select` tries the BARE candidate too, so the
        covering rule still matches and the termination was correct. Confirmed by
        differential enumeration against the pre-#199 code, and by a 200k-policy fuzz that
        found no server where `has_drop` is False while some tool selects a drop rule.

        What has NOT changed is the shape of the risk: over-inclusion costs tokens,
        under-inclusion costs a handle nobody can retrieve, so any future narrowing needs the
        same standard of proof. In particular, "the drop rule comes first" is not that proof —
        it does not come first: index 4 of 8 live, index 2 of the example, both after covering
        rules. Validating a policy edit against rule ORDER alone would conclude "safe"
        incorrectly.

        `server=None` scans every rule, which is both the previous behaviour and the right
        answer for a caller with no server in hand (`terse fluency --drop-eval` evaluating a
        policy file on its own).
        """
        if server and self.server_never_lossy(server):
            return False
        for r in self.rules:
            if any(isinstance(f, dict) and f.get("lossy") == "drop-to-retrieve"
                   for f in r.fields.values()):
                return True
            if server and self._glob_covers_server(r.tool_glob, server):
                return False
        return False

    # -- primer gates (#168) ------------------------------------------------------------
    #
    # Each wire form terse can put on the wire is gated by policy the proxy already knows
    # at `initialize` time. These predicates let the proxy explain ONLY the forms it can
    # actually produce. That is a correctness property before it is a cost one: a primer
    # that documents a table for a `tiers: ["minify"]` server describes behavior that
    # cannot occur.
    #
    # They answer "could ANY tool on this server emit this form", not "will this one" —
    # the primer is injected once at initialize, before any tool is named, so the union
    # over every reachable rule is the only sound scope. Erring toward including a section
    # is the safe direction: a surplus paragraph costs tokens, a missing one costs
    # comprehension on every payload that uses the form.

    @classmethod
    def _glob_covers_server(cls, glob: str, server: str) -> bool:
        """Does this glob match EVERY tool `server` can serve? If so, `select` never falls
        through to the default for that server, and later rules are dead.

        Cover means the glob matches ``{server}.`` followed by ANY bare tool name, since
        that is the shape of `_match_candidates`' first candidate and `select` tries it
        before any later rule. Proved structurally, not by probing a representative name:
        a single probe such as `fnmatch(f"{server}.x", glob)` is unsound in the OTHER
        direction — ``kb.?`` and ``kb.*x`` both match ``kb.x`` while matching no real tool,
        so cover would claim later rules are dead when `select` still reaches them, #199.

        So: the whole-world glob, or a literal prefix followed by ``*`` where the prefix is
        no longer than ``{server}.``. Everything else is refused. That refusal is the safe
        direction — a missed cover only means the walk continues and the caller
        over-approximates, which is `reachable_tiers`' documented contract and, for
        `has_drop`, the direction that keeps a primer paragraph rather than dropping one.

        Metacharacters are handled by the same rule rather than a special case: a server
        literally named ``kb[1]`` is NOT covered by ``kb[1].*``, because `select` fnmatches
        ``[1]`` as a character class and never matches that server's tools at all. String
        equality answered True here, which is the bug this replaces."""
        if glob == "*":
            return True
        if not glob.endswith("*"):
            return False
        prefix = glob[:-1]
        if any(ch in prefix for ch in "*?["):
            return False
        return f"{server}.".startswith(prefix)

    def reachable_tiers(self, server: str | None = None) -> set[str]:
        """Union of tiers any tool on `server` could select — an OVER-approximation.

        Every rule counts. It is tempting to skip rules whose glob looks scoped to a
        different server (`gh.*` when serving `kb`), but that is unsound: `select` is
        candidate-major over `_match_candidates`, whose second candidate is the tool's own
        UNQUALIFIED name — so `gh.*` matches any tool literally named `gh.something`
        regardless of which server serves it. The primer is built at `initialize`, before
        `tools/list`, so the tool inventory is unknown and no name-shape exclusion is
        available. An earlier revision filtered on the glob's literal prefix and produced
        an EMPTY primer for a server named `knowledge` serving `kb.read.*` tools under a
        `kb.*` rule, while `select` still returned all three tiers for them — a client
        receiving a `__terse_table__` envelope with nothing explaining it.

        The one sound narrowing is termination: a rule that totally covers the server ends
        the walk, because `select` returns the first match, so neither later rules nor
        `default_tiers` are reachable past it. Without it every server would inherit
        `default_tiers` (all three tiers in the shipped policy) and the primer could never
        shrink.

        Invariant, pinned by test: `select(tool, server).tiers <= reachable_tiers(server)`
        for every tool. Over-inclusion costs tokens; under-inclusion costs comprehension.
        """
        seen: set[str] = set()
        for r in self.rules:
            seen.update(r.tiers)
            if server and self._glob_covers_server(r.tool_glob, server):
                return seen
        seen.update(self.default_tiers)
        return seen

    def emits_table(self, server: str | None = None) -> bool:
        """True if `tabularize` is reachable, so a `__terse_table__` envelope can appear."""
        return "tabularize" in self.reachable_tiers(server)

    def emits_dict(self, server: str | None = None) -> bool:
        """True if `dictionary` is reachable, so a `__terse_dict__` legend can appear."""
        return "dictionary" in self.reachable_tiers(server)

    def emits_embedded(self, server: str | None = None) -> bool:
        """True if `embedded` is reachable, so a `__terse_json__` envelope can appear."""
        return "embedded" in self.reachable_tiers(server)

    def emits_diff(self, server: str | None = None) -> bool:
        """True if a `__terse_diff__` / `__terse_textdiff__` envelope can appear.

        Diffing is per-policy (`diff`), but it is also downstream of compression: a tool
        whose rule is `tiers: ()` keeps no diff base at all (see the interceptor's
        `if not ...tiers` guards). So the flag alone is not sufficient — some tier must
        also be reachable."""
        return self.diff and bool(self.reachable_tiers(server))


def default_policy() -> Policy:
    """Lossless-everywhere default: full Tier-0/0.5 on every tool, no lossy."""
    return Policy(rules=[], default_tiers=DEFAULT_TIERS)


def _coerce_tiers(raw: Any, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{where}: 'tiers' must be a list, got {type(raw).__name__}")
    bad = [t for t in raw if t not in VALID_TIERS]
    if bad:
        raise ValueError(f"{where}: unknown tier(s) {bad}; valid: {list(VALID_TIERS)}")
    return tuple(raw)


def _coerce_strict_bool(raw: Any, where: str, key: str) -> bool:
    """Strictly a real bool — a mistyped `"capture": "false"` must NOT become True (#85),
    and the same holds for `require_server_name`.

    Deliberately stricter than the `bool(...)` coercion the non-security knobs use:
    both fields are safety switches, and every wrong-typed value in Python is TRUTHY
    (`bool("false") is True`), so a lax coercion would silently turn the guard back ON —
    the one direction a typo must never fail in. Fail loudly instead, at load, before a
    single payload is proxied."""
    if not isinstance(raw, bool):
        raise ValueError(f"{where}: {key!r} must be true or false, got "
                         f"{type(raw).__name__} {raw!r}")
    return raw


VALID_STRUCTURED = ("auto", "leave", "compress", "replace")

# The modes under which `structuredContent` is run through the codec. `replace` does that
# AND drops the text mirror; ordering the literals by invasiveness (leave < compress <
# replace) keeps "is the typed field being rewritten" a single membership test.
STRUCTURED_REWRITING = frozenset({"compress", "replace"})

# Clients measured NOT to validate `structuredContent` against a tool's `outputSchema`,
# and therefore safe to hand a terse envelope in that field under `"structured": "auto"`.
# Matched on `initialize`'s `clientInfo.name`, exactly as declared.
#
# `claude-code`: measured 2026-07-23 against 2.1.218 with the `badtype` and `enveloped`
# probes in `scripts/probe/structured_content/` — neither a wrong-typed payload nor a
# terse table envelope in place of the declared record array drew any complaint; both were
# forwarded to the model verbatim.
#
# Deliberately keyed on NAME, not name+version. A version ceiling would block every future
# release on the strength of one measurement; and if a listed client ever starts
# validating, the failure is LOUD — tool calls error visibly — not silent, and
# `"structured": "leave"` is a one-line override. The client version is logged under
# --debug so such a regression is diagnosable from a log rather than a bisect.
STRUCTURED_SAFE_CLIENTS = frozenset({"claude-code"})


def structured_mode_for_client(mode: str, client_name: str | None) -> str:
    """Resolve a rule's `structured` setting against the connected client's declared name.

    Fail-closed: an unknown client, or none at all (no handshake seen, a client that omits
    `clientInfo`, or a library caller driving `Interceptor` directly), resolves to "leave"
    — the conservative behavior, never the rewriting one."""
    if mode != "auto":
        return mode
    return "compress" if client_name in STRUCTURED_SAFE_CLIENTS else "leave"


def _coerce_structured(raw: Any, where: str) -> str:
    """Strict, for the same reason `capture` is: this decides whether terse rewrites a
    field that carries a declared `outputSchema`. A typo silently reverting to "leave"
    would be a quiet no-op; a typo silently enabling "compress" would quietly rewrite a
    typed field for a client that may validate it. Neither is acceptable, so accept only
    the exact literals and fail at load."""
    if raw not in VALID_STRUCTURED:
        raise ValueError(f"{where}: 'structured' must be one of "
                         f"{list(VALID_STRUCTURED)}, got {raw!r}")
    return raw


# The keys load_policy understands, per level. Anything else (except an "_"-prefixed
# comment/annotation key, the convention policy_gen's `_comment`/`_suggested_fields*`
# already use) is rejected loudly: this file governs what gets rewritten on the wire,
# so a typo'd key ("polices", "diff_keyframe_intervall") silently reverting to default
# behavior is a trap, not a convenience.
_TOP_KEYS = frozenset({"version", "defaults", "policies", "diff", "diff_keyframe_interval",
                       "join_blocks", "never_lossy_servers"})
_DEFAULTS_KEYS = frozenset({"tiers"})
_RULE_KEYS = frozenset({"match", "tiers", "fields", "capture", "require_server_name",
                        "structured"})
_MATCH_KEYS = frozenset({"tool"})


def _reject_unknown_keys(obj: Any, allowed: frozenset[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: must be an object, got {type(obj).__name__}")
    unknown = sorted(k for k in obj if k not in allowed and not k.startswith("_"))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)} "
                         "(prefix a key with '_' for a comment/annotation)")


def load_policy(path: str | Path) -> Policy:
    """Parse + validate a JSON policy file. Raises ValueError on a malformed policy —
    including any unknown key (see `_reject_unknown_keys`): fail loudly on a typo
    rather than silently running with default behavior."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    _reject_unknown_keys(doc, _TOP_KEYS, str(path))
    if doc.get("version") != 1:
        raise ValueError(f"unsupported policy version: {doc.get('version')!r} (expected 1)")
    defaults = doc.get("defaults", {})
    _reject_unknown_keys(defaults, _DEFAULTS_KEYS, "defaults")
    default_tiers = _coerce_tiers(defaults.get("tiers", list(DEFAULT_TIERS)), "defaults")
    rules: list[Rule] = []
    for i, r in enumerate(doc.get("policies", [])):
        _reject_unknown_keys(r, _RULE_KEYS, f"policies[{i}]")
        match = r.get("match", {})
        _reject_unknown_keys(match, _MATCH_KEYS, f"policies[{i}].match")
        glob = match.get("tool", "*")
        rules.append(Rule(tool_glob=glob, tiers=_coerce_tiers(r.get("tiers", []), f"policies[{i}]"),
                          fields=r.get("fields", {}),
                          capture=_coerce_strict_bool(r.get("capture", True),
                                                      f"policies[{i}]", "capture"),
                          require_server_name=_coerce_strict_bool(
                              r.get("require_server_name", False),
                              f"policies[{i}]", "require_server_name"),
                          structured=_coerce_structured(r.get("structured", "auto"),
                                                        f"policies[{i}]")))
    return Policy(rules=rules, default_tiers=default_tiers, diff=bool(doc.get("diff", False)),
                  join_blocks=bool(doc.get("join_blocks", True)),
                  diff_keyframe_interval=int(doc.get("diff_keyframe_interval", 5)),
                  never_lossy_servers=frozenset(doc.get("never_lossy_servers", ())))


@dataclass
class Applied:
    text: str
    tool: str
    tiers: tuple[str, ...]
    skipped: bool
    warnings: list[str]
    # `handle -> (tool, path)` for every drop COMMITTED by this call — empty unless a
    # drop-to-retrieve rule actually fired and passed its recoverability gate. The proxy
    # keeps it so a later `terse.retrieve` can be billed back to the rule that caused the
    # drop (#251); nothing else reads it, and it holds no payload content. Defaulted so
    # every existing `Applied(...)` construction and test fake stays valid.
    drop_origins: dict[str, tuple[str, str]] = field(default_factory=dict)


def _apply_text_drops(raw: str, tool: str, rule: Rule, warnings: list[str],
                      drop_sink: Any, never_lossy: bool,
                      drop_origins: dict[str, tuple[str, str]] | None = None) -> str:
    """Tier-1 lossy (drop-to-retrieve) over a NON-JSON payload, addressed by span.

    Same staged-then-committed contract as the JSON drop path: handles reach the session
    store only after the gate proves the emitted text restores to the original byte for
    byte, so a gate failure leaves no orphan handles behind. Returns `raw` unchanged
    whenever anything at all is off — this is the fail-closed leg of a lossy transform."""
    if never_lossy or not lossy_mod._text_drop_specs(rule):
        return raw
    if drop_sink is None:
        warnings.append("lossy: drop-to-retrieve needs the proxy store; left lossless")
        return raw
    staging: dict[str, Any] = {}
    # Staged alongside the values and published only on the same commit, so a gate failure
    # leaves no provenance behind either — orphan attribution would bill a retrieve to a
    # rule whose drop never reached the store.
    staged_origins: dict[str, tuple[str, str]] = {}
    try:
        cand = lossy_mod.apply_text_drops(raw, rule, tool, staging.__setitem__, staged_origins)
    except Exception as exc:  # noqa: BLE001 — fail closed to the lossless text
        warnings.append(f"lossy step skipped: {exc} (kept lossless)")
        return raw
    if cand == raw:
        return raw
    if not lossy_mod.text_droppable_loss(raw, cand, staging.__getitem__):
        warnings.append("lossy step skipped: text droppable-loss gate failed (kept lossless)")
        return raw
    for handle, value in staging.items():
        drop_sink(handle, value)  # commit only once recoverability is proven
    if drop_origins is not None:
        drop_origins.update(staged_origins)
    warnings.append("lossy: dropped text span(s) to retrieve handle(s) — "
                    "output is NOT lossless")
    return cand


def _lossy_warnings(rule: Rule) -> list[str]:
    """Warn about field lossy requests that won't be executed as asked: deferred modes,
    unknown modes, and the truncate-vs-critical contradiction."""
    critical = lossy_mod.critical_paths(rule)
    out: list[str] = []
    for path, spec in rule.fields.items():
        mode = spec.get("lossy") if isinstance(spec, dict) else None
        if not mode:
            continue
        if mode == "summarize":
            out.append(f"field '{path}': lossy mode '{mode}' not implemented yet (left lossless)")
        elif mode not in LOSSY_MODES:
            out.append(f"field '{path}': unknown lossy mode '{mode}' (ignored)")
        elif path in critical:
            out.append(f"field '{path}': marked '{mode}' AND critical — kept lossless")
    for path in lossy_mod.unknown_text_selectors(rule):
        out.append(f"field '{path}': unknown text selector; known: "
                   f"{sorted(lossy_mod.TEXT_SELECTORS)} (ignored)")
    for path, mode in lossy_mod.unsupported_text_modes(rule):
        out.append(f"field '{path}': lossy mode '{mode}' is not span-addressable; a text "
                   "selector supports only 'drop-to-retrieve' (ignored)")
    return out


def _lossy_stage(obj: Any, rule: Rule, *, tool: str, never_lossy: bool,
                 drop_sink: Any, warnings: list[str],
                 drop_origins: dict[str, tuple[str, str]] | None = None) -> Any:
    """Tier-1 lossy transforms (truncate, then drop-to-retrieve) over one already-parsed
    JSON object, fail-closed: any unresolved path or failed acceptable/droppable-loss gate
    keeps the fully-lossless object. Returns the (possibly reduced) object; appends any
    warnings in place. Extracted so `apply` (one payload) and `apply_joined` (one per
    block, before the join) run the EXACT same lossy logic — the per-block guarantee that
    a field path like `$.results[*].body` resolves against a single payload's shape (#116),
    never against a joined array that would silently change what it selects."""
    data = obj
    # Truncate BEFORE drop, both fail-closed.
    if not never_lossy and lossy_mod._truncate_specs(rule):
        try:
            cand = lossy_mod.apply_lossy(obj, rule)
            if cand is not obj and cand != obj and lossy_mod.acceptable_loss(obj, cand, rule):
                data = cand
                warnings.append("lossy: truncated marked field(s) — output is NOT lossless")
            elif cand != obj:
                warnings.append("lossy step skipped: acceptable-loss gate failed (kept lossless)")
        except lossy_mod.PathError as exc:
            warnings.append(f"lossy step skipped: {exc} (kept lossless)")
        for bad in lossy_mod.unsupported_truncate_paths(obj, rule):
            warnings.append(f"lossy: field {bad!r} is marked truncate but is not a "
                            "string/list; truncate left it unchanged")

    # Drop-to-retrieve (#10): staged writes are committed to the real store only after the
    # gate proves recoverability, so a gate failure leaves no orphan handles.
    if not never_lossy and lossy_mod._drop_specs(rule):
        if drop_sink is None:
            warnings.append("lossy: drop-to-retrieve needs the proxy store; left lossless")
        else:
            staging: dict[str, Any] = {}
            # Same staged-then-committed contract as the values (see `_apply_text_drops`).
            staged_origins: dict[str, tuple[str, str]] = {}
            try:
                cand = lossy_mod.apply_drops(data, rule, tool, staging.__setitem__,
                                             staged_origins)
                if cand != data and lossy_mod.droppable_loss(data, cand, rule, staging.__getitem__):
                    for handle, value in staging.items():
                        drop_sink(handle, value)  # commit only once recoverability is proven
                    if drop_origins is not None:
                        drop_origins.update(staged_origins)
                    data = cand
                    warnings.append("lossy: dropped marked field(s) to retrieve handle(s) — "
                                    "output is NOT lossless")
                elif cand != data:
                    warnings.append("lossy step skipped: droppable-loss gate failed (kept lossless)")
            except lossy_mod.PathError as exc:
                warnings.append(f"lossy step skipped: {exc} (kept lossless)")
    return data


def _lossless_stage(data: Any, rule: Rule, warnings: list[str]) -> tuple[str, tuple[str, ...]]:
    """Always-on Tier-0/0.5/0.6 codec (minify/tabularize/dictionary/embedded) over `data`,
    with the
    verify-before-emit self-check: re-parse what we're about to emit and confirm it
    reconstructs `data`; on any mismatch (or decode error) fall back to the plain minified
    form, which is lossless by construction, and record why. This is the codec's
    counterpart to the lossy tiers' acceptable_loss/droppable_loss gates — fail closed to
    lossless. Returns `(text, tiers)` where `tiers` is `()` when the fallback fired."""
    text = transforms.compress_with(
        data,
        tabularize="tabularize" in rule.tiers,
        dictionary="dictionary" in rule.tiers,
        embedded="embedded" in rule.tiers,
    )
    try:
        emit_ok = transforms.values_equal(transforms.decompress(text), data)
    except Exception:  # noqa: BLE001 — any decode failure is a failed self-check
        emit_ok = False
    if not emit_ok:
        text = transforms.minify(data)
        warnings.append("codec self-check failed (tabularize/dictionary/embedded did not "
                        "round-trip); emitted minified-lossless form instead")
        return text, ()
    return text, rule.tiers


def apply(raw: str, tool: str, policy: Policy,
          drop_sink: Any = None, server: str | None = None,
          force_lossless: bool = False) -> Applied:
    """Compress one raw payload per policy. Lossless by default; a field marked
    `truncate` (and not `critical`) is reduced, gated by the acceptable-loss invariant.
    Non-JSON passes through.

    `drop_sink` — the per-session drop-to-retrieve store, `handle -> value` — exists only
    in the running proxy, per issue #10. When it is None a drop-marked field can't be made
    recoverable, so it is left lossless with a warning instead of silently vanishing.

    `server` — the downstream server's name, when known, so a server-scoped rule
    (`runecho.*`) matches a server whose tools aren't self-prefixed (#83). See
    `Policy.select`.

    Returns the (possibly unchanged) text plus what was applied — so a caller/proxy
    can log why a payload was or wasn't compressed, and whether anything was dropped.
    """
    rule = policy.select(tool, server)
    warnings = _lossy_warnings(rule)

    # Server-level lossy exclusion: on a never-lossy server (credential/personal store),
    # lossy transforms are structurally forbidden — forced 100% lossless — even if a rule
    # marks a field lossy. Enforced here on the VERIFIED server identity (#83), not on a
    # tool-name match, so a mislabeled/renamed rule cannot leak a credential payload through
    # a truncate/drop. Warn (not silently) when this actually suppresses a lossy request.
    # `force_lossless` is the caller-side twin of the never-lossy SERVER floor: the proxy
    # sets it per-RESULT for an `isError` payload, which no policy could express because
    # it is a property of the response, not of the server or tool.
    never_lossy = policy.server_never_lossy(server) or force_lossless
    if never_lossy and rule.lossy_fields():
        warnings.append(f"lossy fields suppressed: server '{server}' is never-lossy "
                        "(credential/personal store) — kept fully lossless")

    if not rule.tiers:
        # `tiers: []` is an explicit hands-off passthrough, so it suppresses the text-drop
        # path too. Say so: a rule carrying BOTH is a contradiction that would otherwise
        # look like a working drop config that silently never fires.
        if lossy_mod._text_drop_specs(rule):
            warnings.append("text drop selector(s) ignored: rule has 'tiers': [] "
                            "(explicit passthrough) — remove one or the other")
        return Applied(text=raw, tool=tool, tiers=(), skipped=True, warnings=warnings)
    if "minify" not in rule.tiers:
        warnings.append("'minify' implied by serialization; added")

    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError, RecursionError):
        # RecursionError: nesting so deep even the C parser blows the stack — the
        # depth guard below can't run on what never parsed.
        #
        # The lossless codec has nothing to fold in prose, so a long-text payload used to
        # end here at 0% saved. Tier-1 lossy CAN reach it, addressed by span instead of by
        # field (`$text.code_blocks`) — same opt-in, same fail-closed contract, same
        # handle/retrieve protocol. Still `skipped=True`: no JSON tier ran, and the proxy
        # relies on that flag to reset its JSON diff state for a non-JSON result.
        text_origins: dict[str, tuple[str, str]] = {}
        text = _apply_text_drops(raw, tool, rule, warnings, drop_sink, never_lossy,
                                 text_origins)
        return Applied(text=text, tool=tool, tiers=(), skipped=True,
                       warnings=warnings + ["payload is not JSON; passed through"],
                       drop_origins=text_origins)

    # Depth guard (#79): the transforms recurse without a depth argument, so a payload
    # nested past the codec-wide cap is screened out here — same passthrough contract
    # as the marker guard. Checked iteratively, before any recursive walk (including
    # has_terse_marker below).
    if transforms.exceeds_depth(obj):
        return Applied(text=raw, tool=tool, tiers=(), skipped=True,
                       warnings=warnings + [f"payload nests deeper than {transforms.MAX_DEPTH} "
                                            "levels; passed through uncompressed"])

    # Marker-collision guard: a payload that already carries a reserved terse marker key
    # can't be compressed without the consumer mis-reading its own data as a terse
    # envelope (the codec has no escape convention). Pass it through untouched (#6).
    if transforms.has_terse_marker(obj):
        return Applied(text=raw, tool=tool, tiers=(), skipped=True,
                       warnings=warnings + ["payload contains a reserved terse marker key; "
                                            "passed through uncompressed to stay lossless"])

    # Tier-1 lossy (truncate then drop) is fail-closed; the always-on Tier-0/0.5 codec
    # then runs with its own verify-before-emit gate. Both stages are extracted so the
    # multi-block join path (`apply_joined`) reuses the exact same logic (#116).
    drop_origins: dict[str, tuple[str, str]] = {}
    data = _lossy_stage(obj, rule, tool=tool, never_lossy=never_lossy,
                        drop_sink=drop_sink, warnings=warnings, drop_origins=drop_origins)
    text, tiers = _lossless_stage(data, rule, warnings)
    return Applied(text=text, tool=tool, tiers=tiers, skipped=False, warnings=warnings,
                   drop_origins=drop_origins)


# Reasons `apply_joined` returns when it declines to join, recorded in the ledger (the
# proxy prefixes them `multiblock_`). Enumerated so the post-#116 measurement can read
# exactly WHY a multi-block result did or didn't collapse.
JOIN_REFUSED_OFF = "off"                    # join_blocks disabled by policy/flag
JOIN_REFUSED_PASSTHROUGH = "passthrough"    # rule is `tiers: []` (explicit hands-off)
JOIN_REFUSED_NON_JSON = "non_json"          # some block isn't JSON
JOIN_REFUSED_HETEROGENEOUS = "heterogeneous"  # some block isn't a dict — not a record seq
JOIN_REFUSED_MARKER = "marker"              # a reserved terse marker key is present
JOIN_REFUSED_DEPTH = "depth"                # joined array nests past the codec cap


def apply_joined(raws: list[str], tool: str, policy: Policy,
                 drop_sink: Any = None, server: str | None = None,
                 force_lossless: bool = False) -> tuple[Applied | None, list | None, str]:
    """Compress a MULTI-block result as ONE record array (#116).

    Parse every block, run the lossy stage per-block (so a field path like
    `$.results[*].body` keeps its single-payload meaning — the whole reason lossy runs
    before the join, not after), join the results into an array, then run the always-on
    lossless codec once over that array so `tabularize`/`dictionary` fold across records.

    Returns `(Applied, raw_array, "")` on success, where `raw_array` is the parsed
    blocks BEFORE lossy — the value the model reconstructs, hence the correct cross-call
    diff base (mirroring `apply`, which bases the diff on the raw parse, not the lossy
    form). Returns `(None, None, reason)` when the join does not apply, so the caller
    falls back to the per-block path and records `reason` in the ledger.

    `raws` MUST have >=2 entries — the single-block shape is `apply`'s job.
    """
    rule = policy.select(tool, server)
    if not policy.join_blocks:
        return None, None, JOIN_REFUSED_OFF
    if not rule.tiers:
        # `tiers: []` is an explicit hands-off passthrough — that includes block shape, so
        # the blocks are forwarded byte-for-byte, never reshaped.
        return None, None, JOIN_REFUSED_PASSTHROUGH

    objs: list[Any] = []
    for raw in raws:
        try:
            objs.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError, RecursionError):
            return None, None, JOIN_REFUSED_NON_JSON
    if not all(isinstance(o, dict) for o in objs):
        # A mixed bag of dicts / scalars / arrays is not a record sequence; joining would
        # assert a relationship that isn't there. Equal key sets are NOT required — a
        # heterogeneous-keyed dict list still unlocks the diff tier, the larger prize.
        return None, None, JOIN_REFUSED_HETEROGENEOUS

    joined = objs
    # Same guards `apply` runs, but on the JOINED array (one level deeper than each block):
    # the codec and diff encoders recurse without a depth argument.
    if transforms.has_terse_marker(joined):
        return None, None, JOIN_REFUSED_MARKER
    if transforms.exceeds_depth(joined):
        return None, None, JOIN_REFUSED_DEPTH

    warnings = _lossy_warnings(rule)
    never_lossy = policy.server_never_lossy(server) or force_lossless
    if never_lossy and rule.lossy_fields():
        warnings.append(f"lossy fields suppressed: server '{server}' is never-lossy "
                        "(credential/personal store) — kept fully lossless")
    if "minify" not in rule.tiers:
        warnings.append("'minify' implied by serialization; added")

    # One origins map across every block: handles are content-addressed over (tool, path,
    # bytes), so two blocks dropping the identical span under the same rule collapse to one
    # handle and one store slot — and therefore to one attribution, which is correct.
    drop_origins: dict[str, tuple[str, str]] = {}
    data = [_lossy_stage(o, rule, tool=tool, never_lossy=never_lossy,
                         drop_sink=drop_sink, warnings=warnings,
                         drop_origins=drop_origins) for o in objs]
    text, tiers = _lossless_stage(data, rule, warnings)
    return (Applied(text=text, tool=tool, tiers=tiers, skipped=False, warnings=warnings,
                    drop_origins=drop_origins),
            objs, "")
