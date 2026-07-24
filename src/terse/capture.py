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
    _uniform_dict_list,  # the one canonical "what tabularize folds" rule
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
    """The first list-of-uniform-dicts at ANY depth in obj (depth-first), else None.

    This is exactly what `transforms.compress_structure` folds into a table — a list
    of >=2 dicts that share one key set, nested arbitrarily deep — and it reuses the
    canonical `_uniform_dict_list` rule so the shape classifier, the probe/fluency
    record extractor, and the tabularizer can never drift on what counts as
    record-shaped (the bug behind #4: three hand-rolled "mirror" checks disagreed)."""
    if _depth > _MAX_SHAPE_DEPTH:
        return None
    if isinstance(obj, list):
        if _uniform_dict_list(obj):
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
    """Return the list-of-uniform-dicts inside obj (at any depth), else None.

    Mirrors what the tabularizer folds, so the probes reason about the same cells.
    Uniform keys are guaranteed, so callers may index every record by the first
    record's columns without a KeyError.
    """
    return _find_record_list(obj)


def find_record_list_with_path(obj: Any, _prefix: tuple[str, ...] = ()) -> tuple[list[dict] | None, str | None]:
    """Like `extract_records`, but also return the field-path prefix to the record list in
    `lossy._parse_path` form (e.g. `result[]`, `data.items[]`, or `[]` for a top-level
    list) — so a caller can build a per-field drop path like `result[].embedding` (#47).

    Walks DICT KEYS only, not into intermediate lists: a record list nested inside another
    list has no simple expressible path, so it returns (records, None-path) is avoided —
    such a list yields (None, None). Returns the first record list reached through keys,
    depth-first, matching `_find_record_list`'s canonical `_uniform_dict_list` rule."""
    if len(_prefix) > _MAX_SHAPE_DEPTH:
        return None, None
    if isinstance(obj, list):
        if _uniform_dict_list(obj):
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
        by_shape[env.get("shape", "?")] = by_shape.get(env.get("shape", "?"), 0) + 1
    return {"total": len(envelopes), "by_tool": by_tool, "by_shape": by_shape}
