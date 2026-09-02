"""Corpus capture + shape bucketing.

Any measurement is only as good as the captured tools, so coverage is tracked
explicitly (see report.py) — a thin sample must not masquerade as "nothing to
compress". Shape buckets are the whole point: they expose where each tier is a
no-op (e.g. compact-JSON, single-object) versus where it pays (array-of-records).

Persistence model: one JSON envelope per payload under corpus/, named
`{tool}__{sha8}.json`. The sha of the raw bytes makes capture idempotent (the
same payload re-captured overwrites the same file) and avoids stamping a
nondeterministic timestamp into the corpus (principle #31).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import policy as policy_mod
from ._secure_io import append_restricted, mkdir_restricted, write_restricted
from .transforms import (
    MAX_DEPTH,
    is_tabularizable,  # the canonical "what tabularize folds" rule
)

# Shape buckets. classify_shape returns one of these.
PRETTY_JSON = "pretty-json"
COMPACT_JSON = "compact-json"
ARRAY_OF_RECORDS = "array-of-records"
SINGLE_OBJECT = "single-object"
LONG_TEXT = "long-text"
OTHER = "other"

_LONG_TEXT_CHARS = 2000
_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


# Cap recursion so an adversarially/pathologically nested payload (which json.loads
# will happily parse) can't blow the stack inside the classifier; real tool output is
# shallow, and at absurd depth the tabularizer itself would also bail, so returning
# "no record list" is the safe, mirror-preserving direction (#4). The cap is the codec-
# wide one from transforms (#79) so the classifier and the compression boundaries agree
# on what "too deep" means.
_MAX_SHAPE_DEPTH = MAX_DEPTH


def _find_record_list(obj: Any, _depth: int = 0) -> list[dict] | None:
    """The first list the TABULARIZER would fold, at any depth in obj (depth-first), else
    None.

    Shares `is_tabularizable` with the codec rather than hand-rolling a second "mirror"
    check, which is the bug behind #4 (three such checks disagreed) — and #204, where the
    shared rule was the strict identical-keyset one while the codec had moved on. A payload
    where two thirds of the rows carry `line` bucketed as `compact-json` with no record
    list while the codec compressed it 55.8%, so `classify_shape`'s buckets, `policy_gen`'s
    auto drop-path generation, `dropeval`, `measure`'s coverage and `fluency.questions` all
    under-fired on exactly the traffic union-schema tabularize was built for.

    Records reached this way may have DIFFERING key sets. Callers that index a record by
    another record's columns must intersect first; `extract_records` says so, and
    `fluency._intersection_cols` is the shared helper for it."""
    if _depth > _MAX_SHAPE_DEPTH:
        return None
    if isinstance(obj, list):
        if is_tabularizable(obj):
            return obj
        for x in obj:
            found = _find_record_list(x, _depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, dict):
        for v in obj.values():
            found = _find_record_list(v, _depth + 1)
            if found is not None:
                return found
    return None


def _has_record_list(obj: Any) -> bool:
    """True if obj contains, at ANY depth, a list-of-uniform-dicts (the tabularize shape)."""
    return _find_record_list(obj) is not None


def extract_records(obj: Any) -> list[dict] | None:
    """Return the record list inside obj (at any depth) that the tabularizer would fold,
    else None.

    Mirrors what the codec folds, so the probes reason about the same cells.

    Key sets are NOT guaranteed to be identical, as of #204 — union-schema tabularize folds record lists
    where some rows omit a key, and this follows it. Indexing every record by
    `records[0].keys()` is therefore a KeyError waiting to happen — take the intersection
    (`fluency._intersection_cols`) when a column must exist in every record. `probes` needs no
    change: it iterates `rec.items()` and already counts per-field presence.
    """
    return _find_record_list(obj)


def find_record_list_with_path(obj: Any, _prefix: tuple[str, ...] = ()) -> tuple[list[dict] | None, str | None]:
    """Like `extract_records`, but also return the field-path prefix to the record list in
    `lossy._parse_path` form (e.g. `result[]`, `data.items[]`, or `[]` for a top-level
    list) — so a caller can build a per-field drop path like `result[].embedding` (#47).

    Walks DICT KEYS only, not into intermediate lists: a record list nested inside another
    list has no simple expressible path, so it returns (records, None-path) is avoided —
    such a list yields (None, None). Returns the first record list reached through keys,
    depth-first, using the same `is_tabularizable` rule as `_find_record_list` — so the
    same non-uniform key-set caveat applies to the records it returns."""
    if len(_prefix) > _MAX_SHAPE_DEPTH:
        return None, None
    if isinstance(obj, list):
        if is_tabularizable(obj):
            prefix = ".".join(_prefix)
            return obj, (f"{prefix}[]" if prefix else "[]")
        return None, None  # list-of-non-records / list-of-lists: no simple field path
    if isinstance(obj, dict):
        for k, v in obj.items():
            records, path = find_record_list_with_path(v, (*_prefix, str(k)))
            if records is not None:
                return records, path
    return None, None


def classify_shape(raw: str) -> str:
    """Bucket a raw tool-output string by structural shape.

    Heuristic and deliberately simple — thresholds are refined against the
    real corpus. Distinguishes pretty vs compact JSON by whitespace, and flags
    record-shaped payloads (what tabularize targets) separately from single objects.
    """
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return LONG_TEXT if len(raw) >= _LONG_TEXT_CHARS else OTHER
    except RecursionError:
        # On the 3.11 floor, json.loads itself recurses and overflows on a deeply nested
        # payload (3.12+ parse it iteratively). Too deep to parse == can't be classified
        # or compressed, so bucket it as unparseable rather than crash the measurement.
        return LONG_TEXT if len(raw) >= _LONG_TEXT_CHARS else OTHER

    is_pretty = "\n" in raw.strip()  # indented JSON has interior newlines; a lone
    #                                   trailing newline (e.g. from `jq -c`) is not pretty

    if _has_record_list(obj):
        return ARRAY_OF_RECORDS
    if isinstance(obj, dict):
        return PRETTY_JSON if is_pretty else COMPACT_JSON
    if isinstance(obj, list):
        return PRETTY_JSON if is_pretty else COMPACT_JSON
    # bare scalar JSON (number/string/bool/null)
    return COMPACT_JSON


def envelope_shape(env: dict[str, Any], default: str = "?") -> str:
    """The shape bucket for one captured envelope, classified LIVE from its `raw` (#355).

    `shape` is written onto the envelope at capture time, but it is a pure function of
    `raw` — so the stored field is a CACHE, not a fact, and it goes stale the moment
    `classify_shape` changes. It did: `7be9d41` (#208/#204) relaxed `_find_record_list`
    from an identical-keyset rule to the codec's union-schema `is_tabularizable`, and 36
    of the live corpus's 1524 envelopes kept a `compact-json` bucket for a payload the
    codec tabularizes. That put `terse measure`'s two shape tables 36 payloads apart in
    ONE report (Coverage reads the stored field; the savings table re-classifies), and
    filed the codec's per-`(tool, shape)` verdict under a shape the codec never sees.

    Every consumer of an envelope's bucket reads it through here — that is the WHOLE
    mechanism, deliberately, so there is one place a stale bucket can be stopped and one
    place to change when `classify_shape` next moves. `load_corpus` does NOT rewrite the
    stored field: the read is what re-derives, and the stored value stays as evidence of
    what the capturing version thought. It is used only when there is no `raw` string to
    classify — a hand-built envelope or a foreign corpus — and `default` only when there
    is neither.

    Consumers, verified by grepping every `["shape"]` / `.get("shape"` in `src/` and
    `scripts/`: `coverage`, `codeceval.run_codec_fluency` (the `(tool, shape)` key of the
    codec verdict), and `scripts/bench/text_alias_ceiling.py`. Every OTHER `"shape"` read
    in the tree is on a MEASURED ROW from `measure.measure_payload`, which has always
    classified live — a row's shape and an envelope's are different values and only the
    latter was ever stored."""
    raw = env.get("raw")
    if isinstance(raw, str):
        return classify_shape(raw)
    stored = env.get("shape")
    return stored if isinstance(stored, str) and stored else default


def _sha8(raw: str) -> str:
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


# Retention cap, PER TOOL rather than over the whole corpus. Every consumer of this
# corpus — measure, probes, policy generate/autotune — reasons per tool, so a global
# byte cap (the shape stats.py and history.py use) would let one chatty tool evict the
# only samples another tool ever produced and silently narrow what a generated policy
# can even see. A per-tool cap bounds disk while preserving breadth.
#
# This matters more here than for the other two sinks: envelopes hold RAW tool payloads
# (credentials, PII, private source), so unbounded retention is a widening blast radius,
# not just a disk-space question. Eviction is oldest-first by mtime — cheap, and the
# right axis, since the newest samples are the ones that reflect a tool's current shape.
MAX_SAMPLES_PER_TOOL = 200


def _prune_tool_samples(corpus: Path, safe_tool: str, keep: int) -> None:
    """Drop the oldest envelopes for one tool past `keep`. Best-effort by design: a
    prune failure must never fail the capture that triggered it."""
    try:
        existing = sorted(corpus.glob(f"{safe_tool}__*.json"),
                          key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for stale in existing[:max(0, len(existing) - keep)]:
        try:
            stale.unlink()
        except OSError:
            pass  # a concurrent proxy already pruned it, or it is not ours to remove


def capture_payload(tool: str, raw: str, corpus_dir: str | Path, *,
                    server: str | None = None, result_id: str | None = None,
                    max_per_tool: int | None = MAX_SAMPLES_PER_TOOL) -> Path:
    """Persist one captured payload as a shape-tagged envelope. Idempotent by sha.

    `server` is the downstream's name in the MCP config and `result_id` identifies the
    tool RESULT this payload was one content block of. Both are optional because the
    format is additive — a corpus captured before they existed stays loadable, and every
    consumer treats their absence as "unknown", never as a value (#148, #152).

    `max_per_tool` bounds how many envelopes this tool keeps; the oldest are evicted past
    it. Pass `None` to retain everything (the pre-cap behavior) — appropriate for a
    deliberate one-shot `terse capture` run building a fixed corpus, not for a proxy
    capturing a live session indefinitely.
    """
    corpus = Path(corpus_dir)
    mkdir_restricted(corpus)
    sha = _sha8(raw)
    safe_tool = _SANITIZE.sub("_", tool).strip("_") or "unknown"
    path = corpus / f"{safe_tool}__{sha}.json"
    # `captured_at` records the chronological CAPTURE order (nanoseconds), which is the
    # session/gateway order a cross-call replay (measure --session-dict, #64) must honor —
    # the sha-based filename does NOT preserve it. Preserved on rewrite so the value is
    # stable at a payload's FIRST sighting and re-capturing the same content stays idempotent.
    captured_at = time.time_ns()
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(prior, dict) and isinstance(prior.get("captured_at"), int):
                captured_at = prior["captured_at"]
                # `result_id` travels WITH `captured_at`, never independently: an envelope
                # describes a payload's FIRST sighting, and a later sighting's result id
                # beside an earlier sighting's timestamp would put the grouping key and the
                # clock in disagreement about which call this envelope stands for — the
                # block would join the new result's group but sort by the old result's
                # position in it. So when the prior envelope predates the field, the
                # incoming id is DROPPED rather than adopted: that payload stays legacy
                # (grouped by timing, and reported as such) until a new payload replaces it.
                result_id = prior.get("result_id")
                if not isinstance(result_id, str):
                    result_id = None
        except (json.JSONDecodeError, OSError):
            pass
    envelope: dict[str, Any] = {
        "tool": tool,
        "shape": classify_shape(raw),
        "bytes": len(raw),
        "sha": sha,
        "captured_at": captured_at,
    }
    # Omitted rather than written as null when unknown, so "the field is absent" is the
    # one signal a consumer has to check — an explicit null would make every reader
    # handle two spellings of the same nothing.
    if server is not None:
        envelope["server"] = server
    if result_id is not None:
        envelope["result_id"] = result_id
    envelope["raw"] = raw   # last: keeps the big field at the end of the file
    # Captured payloads are real MCP tool traffic (README/TECHNICAL: "may contain real
    # data") — restrict permissions the same as terse-managed config/secrets (#42).
    write_restricted(path, json.dumps(envelope, ensure_ascii=False, indent=2))
    # AFTER the write, so `keep` counts this payload and a re-capture of an existing sha
    # (which rewrote in place, adding no file) cannot evict anything.
    if max_per_tool is not None and max_per_tool > 0:
        _prune_tool_samples(corpus, safe_tool, max_per_tool)
    return path


def append_audit(record: dict[str, Any], log_path: str | Path) -> None:
    """Append one audit record as a JSON line to log_path (#23).

    A chronological replay trace — unlike capture_payload's idempotent-by-sha corpus,
    order matters here (diff chains are sequence-dependent) so we append, never dedup.
    One open-append-close per call keeps it crash-safe and lock-free across the proxy's
    threads; tool results are low-frequency enough that the syscall cost is irrelevant.
    """
    p = Path(log_path)
    mkdir_restricted(p.parent)
    # Replay records embed raw tool traffic too — same secrets exposure as capture_payload.
    append_restricted(p, json.dumps(record, ensure_ascii=False) + "\n")


def is_sidecar_filename(name: str) -> bool:
    """True for a `_`-prefixed filename in a corpus/capture dir -- never a real capture
    envelope (e.g. `scripts/bench/mcp_servers/mcp_probe.py`'s `_calls.json`, #138).
    Shared so the places that scan such a directory by filename (this module's own
    `load_corpus` shape check already excludes it structurally; `cli.py`'s
    `_corpus_size` and `toon_column.py`'s loader need it explicitly, since both count/
    load by filename before -- or instead of -- opening the file) stay in sync by
    construction rather than by three hand-copied `.startswith("_")` checks drifting
    apart under a future second sidecar type."""
    return name.startswith("_")


def load_corpus(corpus_dir: str | Path) -> list[dict[str, Any]]:
    """Load every captured envelope from corpus/, in CAPTURE order.

    Ordered by `captured_at` (the chronological session/gateway order), so an
    order-dependent replay — `measure --session-dict` (#64), where a value must be defined
    by an EARLIER payload to be elided by a later one — sees the real sequence, not the
    sha-alphabetical filename order the glob yields. Legacy envelopes with no `captured_at`
    sort first (as 0) in filename order, preserving prior behavior for old corpora; every
    order-independent measure is unaffected. Skips the .gitkeep placeholder and non-envelopes.
    """
    corpus = Path(corpus_dir)
    loaded: list[tuple[int, str, dict[str, Any]]] = []
    skipped = 0
    for path in corpus.glob("*.json"):
        try:
            env = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt/torn envelope was previously dropped silently, so it just
            # disappeared from coverage with no signal. Skipping is still correct (one
            # bad file mustn't fail the whole measure), but count and surface it.
            skipped += 1
            continue
        if isinstance(env, dict) and "raw" in env and "tool" in env:
            # The stored `shape` is deliberately left ALONE, stale or not. Re-classification
            # belongs at the READ (`envelope_shape`), not here: rewriting it would mutate a
            # dict the caller owns and would destroy the one signal that a corpus predates a
            # classifier change — which is what a "N envelopes carry a stale bucket, re-capture
            # them" diagnostic would have to read (#355).
            seq = env["captured_at"] if isinstance(env.get("captured_at"), int) else 0
            loaded.append((seq, path.name, env))
    if skipped:
        sys.stderr.write(
            f"[terse] load_corpus: skipped {skipped} unreadable envelope(s) in "
            f"{corpus} (corrupt JSON)\n")
    loaded.sort(key=lambda t: (t[0], t[1]))
    return [env for _, _, env in loaded]


def bare_and_server(env: dict[str, Any]) -> tuple[str, str | None]:
    """The pair `Policy.select` is called with for this payload.

    The bare name is the DOWNSTREAM tool's own name, with multiproxy's peer prefix stripped:
    the proxy selects on that name and only capture sees the peer-qualified one, so a rule
    has to be authored against the former. The server is whatever the envelope recorded, or
    None — an empty string is None, so "unknown" has exactly one spelling."""
    tool = env.get("tool", "?")
    bare = tool.partition(policy_mod.PREFIX_SEP)[2] if policy_mod.PREFIX_SEP in tool else tool
    server = env.get("server")
    return bare, (server if isinstance(server, str) and server else None)


def qualify(bare: str, server: str | None) -> str:
    """`Policy._match_candidates(bare, server)[0]` — the first name `select` looks up, and
    therefore the only name a generated rule can carry and still be reachable.

    `select` iterates CANDIDATE-major: the qualified candidate is tried against every rule
    before the bare one is tried against any. So a bare `structure` rule sits dead behind a
    deployed `runecho.*` no matter where in the file it is placed, and authoring
    `runecho.structure` is what reaches it (#152). Qualification is skipped when the tool
    already carries the server as its own prefix — kb names its tools `kb.read.*`, and
    `kb.kb.read.search` would match nothing (mirrors the same skip in `_match_candidates`).

    Kept here, beside the envelope that feeds it, so the corpus and the runtime cannot drift
    on what a tool is called — the failure behind #4, where three hand-rolled copies of one
    rule disagreed."""
    if server is None or bare.startswith(f"{server}."):
        return bare
    return f"{server}.{bare}"


def qualified_tool(env: dict[str, Any]) -> str:
    """The name a corpus entry's tool is looked up under AT RUNTIME — `qualify` applied to
    this envelope's `bare_and_server` pair."""
    return qualify(*bare_and_server(env))


def coverage(envelopes: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tool and per-shape counts — surfaced in the report so thin samples show.

    Keyed on `qualified_tool` — the SAME name `policy generate` authors and the proxy
    looks a rule up by (`qualify(bare, server)`), not the bare `env["tool"]` (#158). On a
    server-tagged corpus the bare name reported `structure` while the generated policy said
    `runecho.structure`, so an operator cross-checking a rule against its coverage count had
    to know the two named one tool. Legacy envelopes with no server qualify to their bare
    name, unchanged."""
    by_tool: dict[str, int] = {}
    by_shape: dict[str, int] = {}
    for env in envelopes:
        name = qualified_tool(env)
        by_tool[name] = by_tool.get(name, 0) + 1
        shape = envelope_shape(env)
        by_shape[shape] = by_shape.get(shape, 0) + 1
    return {"total": len(envelopes), "by_tool": by_tool, "by_shape": by_shape}
