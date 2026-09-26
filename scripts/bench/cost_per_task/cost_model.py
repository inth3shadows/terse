"""TTL-aware cost accounting for the cost_per_task harness.

Extends ab_session.py's published constants (W_INPUT, W_CACHE_READ, W_OUTPUT)
rather than editing them: ab_session.py is a separately-used script and its
single flat `W_CACHE_WRITE = 1.25` was built against traffic that only ever
saw 5-minute cache writes. Anthropic bills a cache write by the TTL it was
created with -- 5-minute writes at 1.25x base input, 1-hour writes at 2.0x --
and this harness's --effort/long-lived-cache runs can hit either, so this
module reads the same transcripts and adds the split ab_session.py doesn't
carry, rather than mutating its constants out from under other callers.

Also handles (see plan blockers #2, #3, #9):
  - locating subagent transcripts (`<session>/subagents/*.jsonl`, sibling to
    the main transcript) so Task-spawned subagent spend is not invisible to
    the harness's own cost accounting (belt-and-suspenders: the harness also
    passes --disallowedTools Task on every arm so this should normally find
    nothing).
  - first-turn cache read/write tokens (the model's very first assistant
    turn), separate from the running total.
  - a rough USD estimate per model, compared against the CLI's own
    `total_cost_usd` to flag rows where the two disagree by more than 5%
    (`cost_gap`) -- catching an accounting bug in either side rather than
    trusting one number blind.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Cache-write multipliers by TTL, relative to base input token price.
# Anthropic's published rates; ab_session.py's W_CACHE_WRITE=1.25 is the
# 5-minute rate only.
W_CACHE_WRITE_5M = 1.25
W_CACHE_WRITE_1H = 2.0
# Fallback multiplier when a usage block has cache_creation_input_tokens but
# no `cache_creation` TTL split (older CLI/API responses). 5 minutes is
# Anthropic's default cache TTL, so treating an unsplit write as 5-minute is
# the conservative (not cost-inflating) assumption.
W_CACHE_WRITE_FALLBACK = W_CACHE_WRITE_5M


# Matches a full Claude model ID with a trailing release-date suffix, e.g.
# "claude-haiku-4-5-20251001" -> family "claude-haiku-4-5". `--model` requires
# a full (often dated) model ID (see runner.py's argparse help), but
# MODEL_USD_PER_MTOK is keyed by family for most entries -- a model whose
# exact dated ID isn't a table key must still resolve to its family's price
# rather than silently reading as "unknown model" (plan Partial #3).
_DATED_MODEL_RE = re.compile(r"^(claude-[a-z0-9]+(?:-[a-z0-9]+)*?)-\d{8}$")


def normalize_model_family(model: str) -> str:
    """Strip a trailing YYYYMMDD release-date suffix from a full model ID.
    Returns `model` unchanged if it doesn't match that shape (already a bare
    family name, or some other ID entirely)."""
    m = _DATED_MODEL_RE.match(model)
    return m.group(1) if m else model


def _price_for_model(model: str | None) -> dict | None:
    """MODEL_USD_PER_MTOK's entry for `model`, trying the exact ID first and
    then its normalized family. None if neither is in the table."""
    if not model:
        return None
    return MODEL_USD_PER_MTOK.get(model) or MODEL_USD_PER_MTOK.get(normalize_model_family(model))


def _cache_read_weight(model: str | None, ab_session_mod) -> float:
    """This model's cache-read weight relative to base input, for the
    unitless `weighted_cost` proxy metric -- NOT a USD estimate. Read
    straight from MODEL_USD_PER_MTOK (cache_read / input) so it can never
    silently disagree with `estimate_usd_cost`'s own numbers: Opus 5.5's
    actual cache-read price is 0.05x of its input price ($0.20 / $4.00), not
    the flat 0.1x every other current model uses (ab_session.W_CACHE_READ)
    -- see MODEL_USD_PER_MTOK's docstring, checked against the claude-api
    skill's pricing table 2026-09-25. Falls back to the flat 0.1x for a
    model absent from the price table -- unlike `estimate_usd_cost`, this
    proxy metric must keep producing a number even for an unpriced/unreleased
    model alias, since every row needs SOME weighted_cost (task_arm_stats
    refuses to treat a None cost as free)."""
    price = _price_for_model(model)
    if price:
        return price["cache_read"] / price["input"]
    return ab_session_mod.W_CACHE_READ


def find_subagent_transcripts(transcript_path: Path) -> list[Path]:
    """`<project-dir>/<session_id>/subagents/*.jsonl`, sibling to
    `<project-dir>/<session_id>.jsonl` -- Claude Code's on-disk layout for
    Task-tool subagent transcripts. Returns [] if no such directory exists
    (the expected case now that every arm passes --disallowedTools Task)."""
    transcript_path = Path(transcript_path)
    subdir = transcript_path.parent / transcript_path.stem / "subagents"
    if not subdir.is_dir():
        return []
    return sorted(subdir.glob("*.jsonl"))


# The CLI's own frozen "no real API call happened" usage sentinel -- grepped
# verbatim (`Object.freeze({input_tokens:0,output_tokens:0,
# cache_read_input_tokens:0,cache_creation_input_tokens:0})`) from the
# installed claude 2.1.283 binary, where it backs a locally-injected
# 'assistant' transcript record (e.g. an "API Error: ..." notice) that never
# reached the API. Every field is exactly zero -- a real API response always
# bills at least one output token, so this is a safe, specific signature.
_ZERO_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                     "cache_creation_input_tokens")


def is_real_turn_usage(usage: dict | None) -> bool:
    """False for the CLI's synthetic all-zero-usage 'assistant' record (see
    `_ZERO_USAGE_KEYS`) -- that record is not a real model turn. Counting it
    as one would (a) mark an infra-failed run as having 'a turn', letting it
    slip past the nonzero-exit-no-turn infra check, and (b) corrupt the
    first-turn cache-read/write numbers if it happens to be the first
    assistant record in the transcript (plan Partial #2)."""
    if not usage:
        return False
    return any(usage.get(k, 0) for k in _ZERO_USAGE_KEYS)


def _iter_assistant_usage(transcript_paths: list[Path]):
    """Yield (usage_dict, model, version) for each de-duplicated, REAL
    assistant API response across ALL given transcripts (a synthetic
    all-zero-usage record -- see `is_real_turn_usage` -- is skipped, never
    yielded). Dedup key matches ab_session.SessionStats: (requestId,
    message.id) -- one response can be split across several transcript
    records that each repeat the full `usage` block."""
    seen: set[tuple] = set()
    for path in transcript_paths:
        with Path(path).open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "assistant":
                    continue
                msg = rec.get("message") or {}
                key = (rec.get("requestId"), msg.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                usage = msg.get("usage")
                if not is_real_turn_usage(usage):
                    continue
                yield usage, msg.get("model"), rec.get("version")


def count_real_turns(transcript_paths: list[Path]) -> int:
    """Count of real (non-synthetic) assistant turns across
    `transcript_paths` -- see `is_real_turn_usage`. The infra-error check
    uses this instead of ab_session.SessionStats.turns (which counts a
    synthetic all-zero-usage error record as a turn), so a run whose ONLY
    assistant record is the CLI's own injected error notice is correctly
    treated as zero real turns, not one."""
    return sum(1 for _ in _iter_assistant_usage(transcript_paths))


def compute_ttl_weighted(transcript_paths: list[Path], ab_session_mod) -> dict:
    """Parse `transcript_paths` (main transcript, plus any subagent
    transcripts) and return TTL-aware token totals, weighted cost, and the
    first assistant turn's cache read/write tokens. `ab_session_mod` is the
    already-loaded ab_session module -- reused for W_INPUT/W_CACHE_READ/
    W_OUTPUT so the non-cache-write weights never drift from the existing
    A/B harness's numbers.

    Returns None-valued cost fields only if given an empty path list; a
    non-empty list with zero usage records still returns real zeros (not
    None), since a transcript that exists and truly recorded no billed usage
    is a fact, not a missing-data case -- callers that need to distinguish
    "no transcript at all" pass no paths."""
    input_tok = cache_read = output_tok = write_5m = write_1h = 0
    turns = 0
    first_read: int | None = None
    first_write: int | None = None
    models: dict[str, int] = {}
    versions: dict[str, int] = {}
    # Accumulated PER RECORD, using that record's own model's cache-read
    # weight (see `_cache_read_weight`) -- not a single flat weight applied
    # to the aggregated total at the end. A mixed-model transcript (a
    # mid-session fallback) would otherwise price every cache-read token at
    # whichever model's ratio happened to be used globally, silently wrong
    # for whichever model that wasn't (plan Partial #3: Opus 5.5's
    # cache-read multiplier is 0.05x, not the 0.1x every other model uses).
    weighted = 0.0
    for usage, model, version in _iter_assistant_usage(transcript_paths):
        it_input = usage.get("input_tokens", 0)
        it_read = usage.get("cache_read_input_tokens", 0)
        it_write_total = usage.get("cache_creation_input_tokens", 0)
        it_output = usage.get("output_tokens", 0)
        cc = usage.get("cache_creation")
        if isinstance(cc, dict) and (
                "ephemeral_5m_input_tokens" in cc or "ephemeral_1h_input_tokens" in cc):
            w5 = cc.get("ephemeral_5m_input_tokens", 0)
            w1 = cc.get("ephemeral_1h_input_tokens", 0)
        else:
            w5 = it_write_total
            w1 = 0
        if turns == 0:
            first_read = it_read
            first_write = it_write_total
        input_tok += it_input
        cache_read += it_read
        write_5m += w5
        write_1h += w1
        output_tok += it_output
        weighted += (it_input * ab_session_mod.W_INPUT
                     + it_read * _cache_read_weight(model, ab_session_mod)
                     + w5 * W_CACHE_WRITE_5M
                     + w1 * W_CACHE_WRITE_1H
                     + it_output * ab_session_mod.W_OUTPUT)
        turns += 1
        if model:
            models[model] = models.get(model, 0) + 1
        if version:
            versions[version] = versions.get(version, 0) + 1

    return {
        "turns": turns,
        "input_tokens": input_tok,
        "cache_read_tokens": cache_read,
        "cache_write_5m_tokens": write_5m,
        "cache_write_1h_tokens": write_1h,
        "output_tokens": output_tok,
        "weighted_cost": weighted,
        "first_turn_cache_read_tokens": first_read,
        "first_turn_cache_write_tokens": first_write,
        "resolved_model": max(models, key=models.get) if models else None,
        "cli_version": max(versions, key=versions.get) if versions else None,
    }


# Rough USD-per-million-token prices, used ONLY to sanity-check the CLI's own
# `total_cost_usd` against this harness's independent token accounting
# (`cost_gap`) -- never used as the harness's own cost metric (that is
# `weighted_cost`, in unitless weighted tokens, comparable across models by
# design). Cache-write rates are the standard 1.25x/2.0x-of-input multipliers
# except where a model's docs state a different cache-read rate explicitly
# (Opus 5.5: $0.20/MTok, not 0.1x of its $4 input rate) -- recorded here
# 2026-09-25 from the claude-api skill's pricing table.
MODEL_USD_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10,
                          "cache_write_5m": 1.25, "cache_write_1h": 2.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "cache_read": 0.10,
                                   "cache_write_5m": 1.25, "cache_write_1h": 2.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20,
                         "cache_write_5m": 2.50, "cache_write_1h": 4.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20,
                         "cache_write_5m": 5.00, "cache_write_1h": 8.00},
}


def estimate_usd_cost(model: str, counts: dict) -> float:
    """Estimated USD cost from token counts (as returned by
    `compute_ttl_weighted`). `model`'s exact ID is tried first, then its
    normalized family (`normalize_model_family`) -- a dated ID like
    'claude-sonnet-5-20260115' must price the same as 'claude-sonnet-5'
    rather than silently missing the table. Raises ValueError if NEITHER
    resolves (plan Partial #3: "fail loudly instead of returning None
    silently" -- a benchmark run with an unpriced model should stop and get
    its price added, not keep going and quietly produce an unjudgeable
    cost_gap on every row)."""
    price = _price_for_model(model)
    if price is None:
        raise ValueError(
            f"no USD price entry for model {model!r} (normalized family "
            f"{normalize_model_family(model)!r}) in MODEL_USD_PER_MTOK -- "
            f"add it there rather than silently estimating an unknown cost")
    usd = (counts["input_tokens"] * price["input"]
           + counts["cache_read_tokens"] * price["cache_read"]
           + counts["cache_write_5m_tokens"] * price["cache_write_5m"]
           + counts["cache_write_1h_tokens"] * price["cache_write_1h"]
           + counts["output_tokens"] * price["output"])
    return usd / 1_000_000


def cost_gap_flag(computed_usd: float | None, cli_usd: float | None, *,
                   rel_tol: float = 0.05) -> bool | None:
    """True if `computed_usd` (this harness's own estimate) differs from
    `cli_usd` (claude -p's own `total_cost_usd`) by more than `rel_tol`
    relative to `cli_usd`. None if either side is unavailable -- an
    unknown-model row can't be judged and must not silently read as
    'no gap'."""
    if computed_usd is None or cli_usd is None:
        return None
    if cli_usd == 0:
        return computed_usd != 0
    return abs(computed_usd - cli_usd) / abs(cli_usd) > rel_tol
