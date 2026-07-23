"""Auto-author a conservative, lossless per-tool policy from a measured corpus (#24).

The maintenance/adoption blocker for terse is that `policy.json` is hand-written: add
a server, see no compression, then manually capture → measure → read the report → edit
JSON. `capture.py`/`measure.py` already produce per-tool, per-tier token savings — this
turns that measurement into a policy directly.

The tier decision is deliberately CONSERVATIVE and 100% lossless (it only ever enables the
round-trip-gated Tier-0/0.5 tiers, never a lossy mode):

  For each tool, score its RESULTS the way the proxy compresses them — a multi-block
  result as one joined record array (#116/#147), not block by block — then:
    1. any round-trip failure                          -> passthrough  (never compress
       a tool we can't losslessly handle). A non-JSON payload does NOT disqualify the
       tool: the runtime passes it through untouched, and its raw tokens stay in the
       denominator, so a mostly-text tool falls below the threshold on its own.
    2. total savings %  < threshold                    -> passthrough  (transform cost
       not worth a marginal gain)
    3. otherwise  ["minify","tabularize"] (+ "dictionary" iff its MARGINAL saving clears
       the threshold — mirrors the hand-authored example dropping dictionary on `kb.*`).

Separately, it SUGGESTS drop-to-retrieve candidates (#47) — fields that are large AND
near-unique, where the lossless tiers are structurally powerless — nothing repeats to fold —
but a huge, rarely-needed value dominates the record. The measured example: a kb
`embedding` field, 77% of tokens, all unique — lossless gets +4%, dropping it gets +77%.
These are emitted as an INACTIVE `_suggested_fields` block the operator opts into by
renaming to `fields`: drop is lossy, so the generator never enables it automatically — it
only surfaces the opportunity the operator would otherwise miss.

The output is a policy DOC (the same shape `policy.load_policy` parses) plus a row per
tool explaining the decision. Pure: no I/O, so the generator is unit-testable and the
CLI owns reading the corpus and writing the file.

It does NOT call a model. The #24 thesis — "prove the model still reads the compressed
form" — stays a separate, explicit `terse fluency` step: gating policy authoring on a
model call would couple a key/latency into what is otherwise a deterministic, lossless
transform. The CLI surfaces that pointer; it is not a hard gate here.
"""

from __future__ import annotations

import fnmatch
import json
import re
from typing import Any

from . import policy as policy_mod
from . import probes
from .capture import LONG_TEXT, classify_shape, find_record_list_with_path
from .lossy import DEFAULT_TEXT_DROP_MIN, TEXT_SELECTOR_CODE_BLOCKS, fenced_spans
from .measure import measure_joined, measure_payload
from .tokenize import count_cl100k

# Tiers are cumulative in the codec (minify ⊂ tabularize ⊂ dictionary). minify is implied
# by re-serialization, so it never ships alone — tabularize always carries it.
_BASE_TIERS = ["minify", "tabularize"]

# Field-role classification steers drop-to-retrieve suggestions away from load-bearing
# fields (the step-1 finding: the size+uniqueness heuristic alone confidently suggested
# dropping kb's `principle` — the very field the model reasons over — because it was the
# biggest unique blob). `identity` = a key/essence field the record needs IN-LINE (never a
# drop candidate); `prose` = supporting free text, the safe candidate (ranked first);
# `unknown` = the name doesn't reveal the role, so it MAY be load-bearing → still suggested
# but flagged for the dropeval gate. A name-only heuristic cannot catch a DOMAIN essence
# field (`principle`, `verdict`, `answer`) — those fall to `unknown` by design, which is
# why the behavioral dropeval gate, not this classifier, is the real safety net.
_IDENTITY_TOKENS = frozenset({
    "id", "ids", "key", "keys", "name", "title", "slug", "uuid", "guid", "hash", "type",
    "kind", "status", "state", "path", "url", "uri", "command", "cmd", "version", "ref",
    "sha", "label", "tag", "count", "size", "timestamp", "date", "time"})
_PROSE_TOKENS = frozenset({
    "evidence", "rationale", "note", "notes", "description", "desc", "summary", "detail",
    "details", "body", "text", "content", "comment", "comments", "explanation", "context",
    "snippet", "message", "msg", "reason", "readme", "abstract", "excerpt", "bio", "blurb"})


def _field_tokens(name: str) -> set[str]:
    """Lowercase word tokens of a field path's leaf key (drop the record path and `[]`,
    split camelCase + snake/space/hyphen). `result[].bodyText` -> {'body', 'text'}."""
    leaf = name.rsplit(".", 1)[-1].replace("[]", "")
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", leaf)
    return {t for t in re.split(r"[\s_\-]+", spaced.lower()) if t}


def classify_field_role(name: str) -> str:
    """Best-effort role of a record field from its NAME: 'identity' | 'prose' | 'unknown'.
    See _IDENTITY_TOKENS / _PROSE_TOKENS. Identity wins over prose on a tie (a field that is
    both a key and prose-shaped is treated as load-bearing — the safer call)."""
    toks = _field_tokens(name)
    if toks & _IDENTITY_TOKENS:
        return "identity"
    if toks & _PROSE_TOKENS:
        return "prose"
    return "unknown"


# drop-to-retrieve candidate thresholds (#47). A field is suggested only when all three hold:
_DROP_MIN_MEAN_TOK = 50.0    # large: ~200 serialized chars, matching lossy.DEFAULT_DROP_MIN
_DROP_MIN_UNIQ_RATIO = 0.9   # near-unique: the dictionary tier can't fold it, so lossless is out
_DROP_MIN_SHARE = 0.10       # worth a retrieve round-trip: >=10% of the record list's tokens


def activate_suggestions(doc: dict) -> dict:
    """Return a deep COPY of a generated policy doc with every entry's INACTIVE
    `_suggested_fields` promoted to active `fields` (merged over any existing fields), and
    the `_suggested_fields_note` dropped. Used by `terse tune --drop-eval` to verify the
    suggested drops AS IF enabled, without mutating the doc written to disk (which stays
    inactive until the operator opts in). Pure — no I/O."""
    import copy

    out = copy.deepcopy(doc)
    for entry in out.get("policies", []):
        sug = entry.pop("_suggested_fields", None)
        entry.pop("_suggested_fields_note", None)
        if sug:
            entry["fields"] = {**entry.get("fields", {}), **sug}
    return out


# Rule keys the CORPUS is allowed to decide. Everything else on an existing rule is the
# operator's and survives a merge verbatim — see `merge_policy`.
CORPUS_OWNED_KEYS = ("tiers", "_comment", "_suggested_fields", "_suggested_fields_note")


def _keep_lossy_inert(entry: dict, before: list[str]) -> dict:
    """Refuse to turn `tiers: []` into a compressing stack when the rule carries a LOSSY
    field selector. `policy.apply` treats `tiers: []` as an explicit hands-off passthrough
    that suppresses the text-drop path entirely (it even warns that a rule carrying both is
    a contradiction), so such a selector is inert today — and flipping tiers on would ACTIVATE
    it. A merge documented as lossless and operator-preserving must not be the thing that
    puts a lossy transform live; the operator opts into that by editing `fields`, gated by
    `terse fluency --drop-eval`. Returns the entry unchanged when the situation doesn't
    arise."""
    if before or not entry.get("tiers"):
        return entry                        # not a []-to-compressing transition
    lossy = [k for k, v in (entry.get("fields") or {}).items()
             if isinstance(v, dict) and v.get("lossy")]
    if not lossy:
        return entry
    note = (f"tiers left at [] by autotune: enabling them would ACTIVATE the lossy "
            f"selector(s) {sorted(lossy)} that 'tiers: []' currently suppresses. "
            f"Enable deliberately, then verify with `terse fluency --drop-eval`.")
    return {**entry, "tiers": [], "_comment": f"{entry.get('_comment', '')} — {note}".strip(" —")}


def merge_policy(existing: dict, generated: dict) -> tuple[dict, list[dict[str, Any]]]:
    """Merge a freshly `generate_policy`'d doc INTO an existing policy (#136). Returns
    `(merged_doc, changes)`; pure, like everything else here.

    `generate_policy` is *total* — it authors a whole fresh doc from the corpus and knows
    nothing about what is already deployed. Writing that over a live policy silently drops
    every decision the corpus cannot see, which `_cmd_policy_generate` already warns about
    for `capture: false` alone. The same is true of `never_lossy_servers`, a `structured`
    override, hand-written active `fields`, any rule for a tool this corpus never saw, and
    rule ORDER (first match wins).

    So the merge is split by what a corpus can possibly know:

      * The corpus decides `tiers` — INCLUDING removing one. That is the whole point: a
        tier decision goes stale the moment a codec change lands (the motivating case is a
        `dictionary` rule disabled by a measurement predating the multi-block join, #116),
        and an additive-only merge could never correct it. Removals are proposed, surfaced
        in `changes`, and — by contract with the CLI — never written without an explicit
        `--apply`.
      * The corpus decides its own `_suggested_fields` block, which stays INACTIVE either
        way (the loader reads `fields`, not `_suggested_fields`).
      * The operator owns EVERYTHING else. `capture`, `structured` and `never_lossy_servers`
        are safety decisions a corpus is structurally incapable of making — a payload
        cannot show that a tool returns a plaintext credential, or that a client validates
        `outputSchema`. terse treats all three as fail-safe elsewhere (#85, #135); a
        regeneration path that quietly reverses them would be the one hole in that.

    Ordering is preserved because it is load-bearing: rules are first-match-wins, so a
    reordered policy is a different policy. Existing rules keep their positions; a rule for
    a tool the corpus didn't see is untouched; new rules are inserted BEFORE the first
    catch-all (`*`) rule, since appending after one would make them dead.

    A duplicate tool glob in the existing doc is merged into the FIRST occurrence only —
    the later ones are already unreachable, and rewriting them would imply otherwise."""
    import copy

    gen_by_tool: dict[str, dict] = {}
    for entry in generated.get("policies", []):
        tool = (entry.get("match") or {}).get("tool", "*")
        gen_by_tool.setdefault(tool, entry)

    merged: list[dict] = []
    changes: list[dict[str, Any]] = []
    seen: set[str] = set()

    for old in existing.get("policies", []):
        tool = (old.get("match") or {}).get("tool", "*")
        gen = gen_by_tool.get(tool)
        if gen is None or tool in seen:
            merged.append(copy.deepcopy(old))
            changes.append({"tool": tool, "kind": "preserved",
                            "why": "unreachable duplicate" if tool in seen
                                   else "not in corpus"})
            seen.add(tool)
            continue
        seen.add(tool)
        new = copy.deepcopy(old)
        for key in CORPUS_OWNED_KEYS:
            new.pop(key, None)
            if key in gen:
                new[key] = copy.deepcopy(gen[key])
        new = _keep_lossy_inert(new, list(old.get("tiers", [])))
        kept = sorted(k for k in old if k not in CORPUS_OWNED_KEYS and k != "match")
        before, after = old.get("tiers", []), new.get("tiers", [])
        if list(before) != list(after):
            changes.append({"tool": tool, "kind": "tiers", "before": list(before),
                            "after": list(after), "preserved": kept})
        elif old.get("_suggested_fields") != new.get("_suggested_fields"):
            changes.append({"tool": tool, "kind": "suggestions", "preserved": kept})
        else:
            changes.append({"tool": tool, "kind": "unchanged", "preserved": kept})
        merged.append(new)

    # Each new rule goes BEFORE the first existing rule that would already match its tool.
    # Appending is wrong and silently so: rules are first-match-wins, so a fresh
    # `kb.read.search` rule placed after an existing `kb.*` is dead on arrival — the policy
    # would look re-tuned and change nothing. The guard is any matching glob, not just a
    # literal `*`: `kb.*` is a catch-all for every kb tool.
    added = [(t, copy.deepcopy(e)) for t, e in gen_by_tool.items() if t not in seen]

    def _shadowing(tool: str) -> tuple[int, dict | None]:
        """Index of the first existing rule that currently governs `tool`, and that rule.

        Uses `Policy`'s own candidate list rather than a bare fnmatch, so a multiproxy
        peer-qualified name resolves the way the loader would. Best-effort on ONE axis:
        the server-qualified candidate needs a server name, and the corpus does not record
        which server a payload came from — so a server-scoped rule (`runecho.*`) against a
        tool captured under its bare name (`structure`) is not detected here. Tracked
        separately; erring this way inserts a rule that the loader may then shadow, which
        is inert, rather than one that silently overrides an operator rule."""
        for i, e in enumerate(merged):
            glob = (e.get("match") or {}).get("tool", "*")
            if any(fnmatch.fnmatch(c, glob)
                   for c in policy_mod.Policy._match_candidates(tool)):
                return i, e
        return len(merged), None

    # Group by insertion point and splice from the back, so earlier indices stay valid and
    # the generator's own order (highest savings first) survives within each group.
    buckets: dict[int, list[dict]] = {}
    for tool, entry in added:
        at, shadowed = _shadowing(tool)
        prior_tiers: list[str] = []
        if shadowed is not None:
            # INHERIT the operator-owned keys of the rule this one displaces. Without this
            # the anti-shadowing insertion above becomes a safety hole: a new
            # `kb.read.search` rule placed ahead of an operator's
            # `kb.* {capture: false, structured: "leave"}` would leave that tool running
            # with capture ON (payloads to disk, reversing the #85 decision) and
            # `structured: "auto"` (rewriting a typed field the operator opted out of).
            # A new rule refines `tiers` for a tool; it must not quietly re-decide
            # anything else about it.
            inherited = {k: copy.deepcopy(v) for k, v in shadowed.items()
                         if k not in CORPUS_OWNED_KEYS and k != "match"}
            entry = {**entry, **inherited}
            prior_tiers = list(shadowed.get("tiers", []))
            if inherited:
                changes.append({"tool": tool, "kind": "inherited",
                                "from": (shadowed.get("match") or {}).get("tool", "*"),
                                "keys": sorted(inherited)})
        entry = _keep_lossy_inert(entry, prior_tiers)
        buckets.setdefault(at, []).append(entry)
        changes.append({"tool": tool, "kind": "added", "before": prior_tiers,
                        "after": list(entry.get("tiers", []))})
    for at in sorted(buckets, reverse=True):
        merged[at:at] = buckets[at]

    out = copy.deepcopy(existing)
    out["policies"] = merged
    return out, changes


# A text tool's drop candidate is judged on the one thing that decides whether the selector
# pays: what share of its tokens sits inside droppable fenced blocks. Deliberately higher
# than the JSON `min_share`— a text drop evicts a contiguous region a reader can see is
# missing, so it should only be proposed where it is the dominant cost, not a trim.
_TEXT_DROP_MIN_SHARE = 0.40
_TEXT_DROP_MIN_PAYLOADS = 2   # one payload is an anecdote; the shape must repeat


def _tok(text: str) -> int:
    """Token count, degrading to the codec-wide length heuristic when tiktoken is absent —
    the ratio this feeds is scale-free, so an approximate count still ranks correctly."""
    n = count_cl100k(text)
    return n if n is not None else len(text) // 4


def _text_drop_candidate(raws: list[str]) -> tuple[dict[str, dict], list[dict[str, Any]]]:
    """Suggest `$text.code_blocks` for a tool whose payloads are long TEXT dominated by
    fenced code — the span-addressed analogue of `_drop_candidates`, and the reason a
    0.0%-saved text tool could never be surfaced by autotune (#136/#139).

    `_drop_candidates` opens with `json.loads(raw)` and `continue`s on failure, so a tool
    that is 100% prose produces zero candidates BY CONSTRUCTION. That is not a threshold
    being missed, it is a shape the generator cannot see: `codegraph_explore` measured
    0.0% across 61 captured payloads and was never proposed anything, while 86.6% of its
    tokens sat in droppable blocks.

    Judged on the aggregate token share of blocks that clear the drop floor — the same
    quantity the runtime would actually evict, so the estimate cannot flatter itself with
    spans the transform would skip. Returns the INACTIVE `_suggested_fields` entry plus a
    report row, never an active rule: drop is lossy and the operator opts in.
    """
    n = spans_tok = total_tok = 0
    max_tok = 0
    for raw in raws:
        if not isinstance(raw, str) or classify_shape(raw) != LONG_TEXT:
            continue
        n += 1
        total_tok += _tok(raw)
        for start, end in fenced_spans(raw):
            if end - start < DEFAULT_TEXT_DROP_MIN:
                continue  # under the floor: the runtime leaves it in place, so don't count it
            t = _tok(raw[start:end])
            spans_tok += t
            max_tok = max(max_tok, t)
    if n < _TEXT_DROP_MIN_PAYLOADS or not total_tok:
        return {}, []
    share = spans_tok / total_tok
    if share < _TEXT_DROP_MIN_SHARE:
        return {}, []
    suggestion = {TEXT_SELECTOR_CODE_BLOCKS: {"lossy": "drop-to-retrieve",
                                              "min": DEFAULT_TEXT_DROP_MIN}}
    # role stays `unknown`, never `prose`: a fenced block is source, and source is exactly
    # the load-bearing case the role split exists to flag for review.
    row = {"path": TEXT_SELECTOR_CODE_BLOCKS, "role": "unknown", "n": n, "distinct": n,
           "uniq_ratio": 1.0, "mean_tok": round(spans_tok / n, 1), "max_tok": max_tok,
           "tok_share": round(share, 4)}
    return suggestion, [row]


def _drop_candidates(
    raws: list[str],
    min_mean_tok: float = _DROP_MIN_MEAN_TOK,
    min_uniq_ratio: float = _DROP_MIN_UNIQ_RATIO,
    min_share: float = _DROP_MIN_SHARE,
) -> tuple[dict[str, dict], list[dict[str, Any]]]:
    """Suggest drop-to-retrieve field paths for one tool from its payloads. Pools records
    across payloads that share the first payload's record path (per-tool shape is assumed
    consistent) and flags fields that are large + near-unique + a meaningful token share.
    Returns `(suggestion, rows)`: `suggestion` is an INACTIVE `{path: {"lossy": ...}}` block
    the operator opts into; `rows` are per-field stats for the report. Empty when a tool has
    no record list or no field clears every threshold.

    Each payload is profiled INDEPENDENTLY and the metrics are averaged: cardinality is a
    within-payload property (the drop runs per result at runtime), so pooling records across
    payloads would wrongly halve `uniq_ratio` when the same result is captured twice."""
    path: str | None = None
    per_payload: list[dict[str, dict[str, Any]]] = []
    for raw in raws:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        records, p = find_record_list_with_path(obj)
        if records is None or p is None:
            continue
        if path is None:
            path = p
        if p == path:
            per_payload.append(probes.field_profiles(records))
    if path is None or not per_payload:
        return {}, []

    def _avg(field: str, key: str) -> float:
        vals = [pp[field][key] for pp in per_payload if field in pp]
        return sum(vals) / len(vals) if vals else 0.0

    fields = {f for pp in per_payload for f in pp}
    agg = {f: {"n": sum(pp[f]["n"] for pp in per_payload if f in pp),
               "distinct": sum(pp[f]["distinct"] for pp in per_payload if f in pp),
               "uniq_ratio": round(_avg(f, "uniq_ratio"), 4),
               "mean_tok": round(_avg(f, "mean_tok"), 1),
               "max_tok": max(pp[f]["max_tok"] for pp in per_payload if f in pp),
               "tok_share": round(_avg(f, "tok_share"), 4)} for f in fields}

    suggestion: dict[str, dict] = {}
    rows: list[dict[str, Any]] = []
    # Order safe-first, then by impact: prose (known-safe) fields lead, unknown-role fields
    # (may be load-bearing) follow, each by descending token-share. `identity` fields are
    # dropped from the candidate list entirely below — the record needs its keys in-line.
    _role_rank = {"prose": 0, "unknown": 1}
    ordered = sorted(
        agg.items(),
        key=lambda kv: (_role_rank.get(classify_field_role(f"{path}.{kv[0]}"), 1),
                        -kv[1]["tok_share"]),
    )
    for field, pr in ordered:
        role = classify_field_role(f"{path}.{field}")
        if role == "identity":
            continue  # a key/essence field is never a drop candidate — the record needs it
        if (pr["mean_tok"] >= min_mean_tok and pr["uniq_ratio"] >= min_uniq_ratio
                and pr["tok_share"] >= min_share):
            fpath = f"{path}.{field}"
            suggestion[fpath] = {"lossy": "drop-to-retrieve"}
            rows.append({"path": fpath, "role": role, **pr})
    return suggestion, rows


def _pct(saved: int, raw: int) -> float:
    """Savings as a percent of the tool's raw token volume (0.0 when raw is 0)."""
    return (saved / raw * 100.0) if raw else 0.0


# Two captured blocks of ONE result are teed back-to-back — a file write apart — so
# consecutive envelopes closer than this belong to the same tool call (#147). Compared
# CONSECUTIVELY, not first-to-last, so a 200-block result groups correctly however long it
# takes overall. Over-grouping is possible in one case — genuinely parallel calls to the
# same tool interleave in time and no window separates them — and is mild: it scores that
# tool as if one larger result arrived, which is still representative of its shape. The
# exact fix would be a result id written into the capture envelope; it is not worth a
# format change until over-grouping is shown to move a decision.
_RESULT_WINDOW_NS = 50_000_000  # 50 ms


def group_results(envelopes: list[dict[str, Any]]) -> dict[str, list[list[str]]]:
    """Reconstruct `tool -> [[block, ...], ...]` from a flat corpus, so a tool's payloads
    can be scored as the RESULTS they arrived as (#147). Envelopes without a `captured_at`
    (predating the field) each become their own single-block group — the pre-#147 behavior,
    which is the safe direction: it under-measures rather than inventing a join.

    Note the corpus is idempotent by sha and preserves a payload's FIRST `captured_at`, so
    this reconstructs first-sightings, not every call. That is the same sample the tier
    decision was always made from; it just stops pretending each block arrived alone."""
    out: dict[str, list[list[str]]] = {}
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for env in envelopes:
        by_tool.setdefault(env.get("tool", "?"), []).append(env)
    for tool, envs in by_tool.items():
        timed = sorted((e for e in envs if isinstance(e.get("captured_at"), int)),
                       key=lambda e: e["captured_at"])
        groups: list[list[str]] = [[e["raw"]] for e in envs
                                   if not isinstance(e.get("captured_at"), int)]
        prev_ts: int | None = None
        for env in timed:
            if prev_ts is not None and env["captured_at"] - prev_ts < _RESULT_WINDOW_NS:
                groups[-1].append(env["raw"])
            else:
                groups.append([env["raw"]])
            prev_ts = env["captured_at"]
        out[tool] = groups
    return out


def _tool_decision(tool: str, groups: list[list[str]], threshold: float,
                   join_blocks: bool = True) -> dict[str, Any]:
    """Aggregate one tool's RESULTS and decide its tiers. Returns a summary row with
    the chosen `tiers`, the measured savings, and a human-readable `reason`.

    Each result is scored the way the proxy would compress it: a multi-block result as one
    joined array, falling back to per-block exactly where `apply_joined` would refuse
    (#147). Scoring blocks individually — which this did until then — under-measures every
    server that returns one record per content block, and produced `passthrough` for tools
    that measurably compress in production."""
    raws = [r for g in groups for r in g]
    rows = []
    joined_results = 0
    for group in groups:
        # `apply_joined`'s FIRST check is `if not policy.join_blocks` — a policy that opted
        # out of #116 must not be tuned on cross-block folding it will never perform.
        joined = measure_joined(group) if join_blocks else None
        if joined is not None:
            rows.append(joined)
            joined_results += 1
        else:
            rows.extend(measure_payload(r) for r in group)
    n = len(raws)

    # Counted over ROWS, so a joined result contributes 1 whatever its block count — see
    # the ratio note where these are rendered. Only `gate_fail` disqualifies the tool;
    # `non_json` is reported, not acted on (#147, and the reasoning at the branch below).
    non_json = sum(1 for r in rows if not r["applicable"])
    gate_fail = sum(1 for r in rows if not r["roundtrip_ok"])

    raw_tok = sum(r["cl100k"]["raw"] or 0 for r in rows)
    minify = sum(r["saved_cl100k"]["minify"] or 0 for r in rows)
    tabularize = sum(r["saved_cl100k"]["tabularize"] or 0 for r in rows)
    dictionary = sum(r["saved_cl100k"]["dictionary"] or 0 for r in rows)
    total = sum(r["saved_cl100k"]["tier_total"] or 0 for r in rows)
    total_pct = _pct(total, raw_tok)
    dict_pct = _pct(dictionary, raw_tok)

    # Drop-to-retrieve candidates are detected INDEPENDENTLY of the lossless-tier decision:
    # the highest-value case (kb `embedding`) is a tool whose lossless savings fall BELOW the
    # threshold — passthrough for tiers — yet is dominated by a huge unique field only drop
    # can shrink. So compute it here and carry it through every return path.
    drop_suggestion, drop_rows = _drop_candidates(raws)
    # The text analogue runs unconditionally alongside it, not as a fallback: the two
    # address disjoint payload shapes (fields of a parsed object vs spans of prose), so a
    # tool that returns both JSON and markdown can legitimately earn one candidate of each.
    text_suggestion, text_rows = _text_drop_candidate(raws)
    drop_suggestion = {**drop_suggestion, **text_suggestion}
    drop_rows = drop_rows + text_rows

    base = {
        "tool": tool, "n": n, "n_results": len(groups), "joined_results": joined_results,
        "raw_tok": raw_tok,
        "saved_pct": round(total_pct, 1), "dict_pct": round(dict_pct, 1),
        "minify": minify, "tabularize": tabularize, "dictionary": dictionary,
        "drop_suggestion": drop_suggestion, "drop_rows": drop_rows,
    }

    # A round-trip FAILURE still disqualifies the whole tool: it says the codec cannot
    # losslessly handle this tool's shape, and the policy matches on tool name, so there is
    # no way to enable a tier for only the payloads that survive.
    if gate_fail:
        return {**base, "tiers": [],
                "reason": f"passthrough — {gate_fail}/{len(rows)} result(s) failed "
                          f"round-trip"}

    # A non-JSON payload does NOT (#147). It used to, and that quietly zeroed the
    # highest-volume tool in a real fleet: `kb.read.search` measured 16.7% saved and was
    # marked passthrough because 4 of its 436 captured payloads were the server's
    # `Error executing tool …` text. The premise was wrong — `policy.apply` passes a
    # non-JSON payload through untouched at runtime, so enabling a tier for a tool that
    # occasionally returns prose costs exactly nothing on those results. Meanwhile a tool
    # that is MOSTLY text is still suppressed, and for the right reason: non-JSON payloads
    # contribute 0 saved while their raw tokens stay in the denominator, so the percentage
    # falls below the threshold on its own (`codegraph_explore`, 61/61 non-JSON, scores
    # 0.0%). The measurement handles it; the disqualifier only ever discarded real savings.
    # Denominator is rows, not `n`: `n` counts BLOCKS and a joined result is one row,
    # so "4/436" would understate by the join factor and argue against the decision
    # it is justifying.
    mixed = f" ({non_json}/{len(rows)} non-JSON, passed through)" if non_json else ""

    if total_pct < threshold:
        return {**base, "tiers": [],
                "reason": f"passthrough — {total_pct:.1f}% < {threshold:.1f}% threshold{mixed}"}

    tiers = list(_BASE_TIERS)
    if dict_pct >= threshold:
        tiers.append("dictionary")
        reason = f"{total_pct:.1f}% saved (dictionary +{dict_pct:.1f}%){mixed}"
    else:
        reason = (f"{total_pct:.1f}% saved (dictionary +{dict_pct:.1f}% below threshold "
                  f"— dropped){mixed}")
    return {**base, "tiers": tiers, "reason": reason}


def generate_policy(
    envelopes: list[dict[str, Any]], threshold: float = 5.0, join_blocks: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Author a conservative lossless policy from captured payloads.

    `threshold` is the minimum total savings (percent of raw tokens) required to compress
    a tool at all, and the minimum marginal saving to add the dictionary tier on top of
    minify+tabularize. Returns `(policy_doc, rows)` — `policy_doc` is loadable by
    `policy.load_policy`; `rows` is one decision summary per tool (sorted by savings desc)
    for a report. Tools are emitted in the same order so the JSON is deterministic."""
    rows = [_tool_decision(tool, groups, threshold, join_blocks)
            for tool, groups in group_results(envelopes).items()]
    # Highest-savings tools first: makes the policy and the report read top-down by value,
    # and keeps output stable regardless of corpus file ordering.
    rows.sort(key=lambda r: (-r["saved_pct"], r["tool"]))

    policies = []
    for r in rows:
        comment = (f"{r['n']} payload(s) in {r['n_results']} result(s)"
                   + (f", {r['joined_results']} scored joined" if r.get("joined_results") else "")
                   + f", {r['reason']}")
        entry: dict[str, Any] = {"_comment": comment, "match": {"tool": r["tool"]},
                                 "tiers": r["tiers"]}
        # Drop-to-retrieve suggestions ride along INACTIVE: the loader reads `fields`, not
        # `_suggested_fields`, so this is a no-op until the operator renames it. Drop is
        # lossy — the human confirms; the generator never enables it.
        if r.get("drop_suggestion"):
            shares = ", ".join(
                f"{dr['path']} ~{dr['tok_share']*100:.0f}% [{dr.get('role', 'unknown')}]"
                for dr in r["drop_rows"])
            has_unknown = any(dr.get("role") == "unknown" for dr in r["drop_rows"])
            caution = (" A field tagged [unknown] may be LOAD-BEARING — the name doesn't "
                       "reveal its role, so dropping it forces the model to call retrieve for "
                       "it; treat [prose] as safer and verify any [unknown] before enabling."
                       if has_unknown else "")
            entry["_suggested_fields"] = r["drop_suggestion"]
            entry["_suggested_fields_note"] = (
                f"LOSSY drop-to-retrieve candidates (large + near-unique; safe-first, [role] "
                f"guessed from the field name): {shares}. Rename '_suggested_fields' -> "
                f"'fields' to enable, then confirm the model still answers with `terse fluency "
                f"--drop-eval`. Off until you do.{caution}")
        policies.append(entry)

    any_drops = any(r.get("drop_suggestion") for r in rows)
    doc = {
        "version": 1,
        "_comment": (f"Auto-generated by `terse policy generate` (threshold {threshold:.1f}%). "
                     f"Conservative + lossless: a tier is enabled only where measured savings "
                     f"clear the threshold and every payload round-trips. Verify the model still "
                     f"reads the compressed form with `terse fluency --corpus <dir>` before relying "
                     f"on it."
                     + (" Some tools carry INACTIVE `_suggested_fields` (lossy drop-to-retrieve "
                        "candidates) — opt in by renaming to `fields`." if any_drops else "")),
        "defaults": {"tiers": ["minify", "tabularize", "dictionary"]},
        "policies": policies,
    }
    return doc, rows
