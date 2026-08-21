"""Eval harnesses — paired raw-vs-terse runs and the diff/soak/text-diff variants (#78 split).

Scoring is PAIRED: the same questions, same order, over raw vs terse (+/- a
one-time format primer). A regression is a question the model got right on raw
and wrong on terse — a stronger signal than two independent accuracy rates.
The verdict gates on the WORST model, not the mean (mirrors how `validate`
reports cross-tokenizer divergence rather than averaging it away).
"""

from __future__ import annotations

import json
from typing import Any

from .. import text_diff
from ..capture import LONG_TEXT, OTHER, classify_shape
from ..transforms import compress, diff_wire
from .answerers import Answerer
from .pack import PRIMER
from .questions import _text_diff_questions_from_lines, gen_questions
from .scoring import score


def _user_prompt(prompt: str, instruction: str, data: str) -> str:
    return f"{prompt}\n{instruction}\n\nDATA:\n{data}"


def _safe_ask(answerer: Answerer, system: str, user: str) -> str | None:
    """Call the model, but never let one failed call abort a long multi-model run.

    Returns **None** on a transport failure (connection error, rate limit, bad model id) —
    deliberately distinct from a real reply, including an empty one. This used to return
    `""`, which `score` then counted as a WRONG ANSWER, so an unreachable model reported
    ~0% accuracy and was indistinguishable in the report from a model that genuinely could
    not read terse's compressed form (#263). Because `run_fluency`'s verdict gates on the
    WORST model, a single rate-limited backend did not merely dilute a panel — it decided
    it. A live `gemini-3.6-flash` rate limit would have returned FAIL for the whole
    no-primer question (#249) on a result that measured nothing.

    A second, quieter hazard goes with it: `""` could MATCH a question whose expected
    answer was empty, scoring a total failure as correct. `questions.py` excludes such
    questions on every generation path via `_answerable` for exactly that reason; with
    None those exclusions become belt-and-braces rather than load-bearing.

    Blank replies are normalised to None HERE, not only in `openai_answerer`, because this
    is the choke point every harness funnels through. Putting the invariant in one adapter
    left the package holding two contracts — a future `Answerer`, or the `pack.py` replay
    path, would reopen `#268` with no test failing. "The model produced nothing" and "the
    call did not happen" are the same fact to every consumer downstream: unanswered, so
    counted and never scored."""
    try:
        reply = answerer(system, user)
        # NOT `_answerable(reply)`: that predicate treats any non-None, non-blank-string
        # value as answerable (an int `expected` is legitimate there), so it would let a
        # contract-violating non-str reply through unchanged instead of catching it here.
        # `.strip()` is deliberately called on a value that might not have it — the
        # resulting AttributeError is what turns "answerer returned garbage" into "treat
        # it as unanswered," inside this try.
        return None if reply is None or not reply.strip() else reply
    except Exception:
        return None


def _ask_n(answerer: Answerer, system: str, user: str,
           qtype: str, expected: Any, trials: int) -> tuple[int, int]:
    """Ask the same question `trials` times; return `(correct, unanswered)`.

    Repeating at temperature 0 is not redundant — it surfaces the provider-side
    nondeterminism (batching / MoE routing) behind the ~5pt run-to-run accuracy wobble the
    report's question-clustered bound quantifies.

    An unanswered call is counted SEPARATELY and never scored. It is not a wrong answer:
    either the model was never reached at all (transport failure, #263), or it was reached
    but produced no content (#268). Callers carry the count into the row so the report can
    decline to publish an accuracy for a backend that did not answer.

    Note what the count does NOT do on its own: `_form_stats` removes these from that arm's
    denominator, which is only safe while the losses are uncorrelated with the arm. #268's
    live cause is a token-budget stop that scales with prompt length, so the diff arm —
    whose prompt is strictly longer than its control's — is systematically the one that
    loses calls; this module makes no attempt to protect against that correlation. See
    issue #280 for the open design question."""
    ok = fails = 0
    for _ in range(trials):
        reply = _safe_ask(answerer, system, user)
        if reply is None:
            fails += 1
            continue
        ok += score(qtype, expected, reply)
    return ok, fails


def run_payload(obj: Any, raw_text: str, answerer: Answerer,
                primer: str = PRIMER, trials: int = 1) -> list[dict]:
    """Ask one payload's questions over raw / terse / terse+primer / terse+inline-primer,
    `trials` times each.

    Each returned row carries per-form success COUNTS (0..trials) plus `trials`, not
    booleans. At trials=1 a count is 0 or 1 — truthy/falsy exactly like the old bool —
    so every existing aggregation keeps working unchanged.

    `inline_ok` is the fourth arm and exists to decide #168's lazy primer. As measured, the
    primer shipped in `initialize.instructions` and the client re-read it EVERY turn as
    cache_read, which is why terse measured a 14.0% win at one wrapped server and a loss at
    three. The fix — shipped in #211 for standalone entries, and still not applied to a
    router — attaches it to the FIRST COMPRESSED RESULT instead: paid once per session, not
    once per turn.

    `proxy.py` records a measurement that a *system-level* primer beat an inline per-result
    note, but that compared the primer against a short note, not against the same primer
    delivered inline. This arm closes that gap: identical primer text, identical questions,
    the only difference being that it rides with the DATA rather than in the system slot —
    which is exactly what the proxy could do, since a stdio proxy cannot set a system prompt.
    """
    terse_text = compress(obj)
    out: list[dict] = []
    for q in gen_questions(obj):
        raw_u = _user_prompt(q.prompt, q.instruction, raw_text)
        terse_u = _user_prompt(q.prompt, q.instruction, terse_text)
        # The primer PREFIXES the result body, which is where a lazy primer would put it:
        # the proxy owns the result text and nothing else.
        inline_u = _user_prompt(q.prompt, q.instruction, f"{primer}\n\n{terse_text}")
        raw_ok, raw_fail = _ask_n(answerer, "", raw_u, q.qtype, q.expected, trials)
        terse_ok, terse_fail = _ask_n(answerer, "", terse_u, q.qtype, q.expected, trials)
        primer_ok, primer_fail = _ask_n(answerer, primer, terse_u, q.qtype, q.expected, trials)
        # No system message: the whole point is that the model was never told anything
        # at initialize time.
        inline_ok, inline_fail = _ask_n(answerer, "", inline_u, q.qtype, q.expected, trials)
        out.append({
            "qid": q.qid, "qtype": q.qtype, "transform": q.transform, "trials": trials,
            "raw_ok": raw_ok, "terse_ok": terse_ok,
            "primer_ok": primer_ok, "inline_ok": inline_ok,
            # Per-arm DENOMINATORS excluding the calls that never happened (#263).
            # `_form_stats` already prefers a per-form `<form>_trials` over the shared
            # `trials`, so a transport failure is removed from the denominator rather than
            # scored as a miss. This is what keeps a handful of transient 429s from
            # depressing an otherwise-complete run: the accuracy stays over the calls that
            # actually returned, instead of being diluted by calls that never left.
            "raw_trials": trials - raw_fail,
            "terse_trials": trials - terse_fail,
            "primer_trials": trials - primer_fail,
            "inline_trials": trials - inline_fail,
            # Kept for reporting: how degraded this run was, and out of how much.
            "fails": raw_fail + terse_fail + primer_fail + inline_fail,
            "attempts": trials * 4,
        })
    return out


def run_fluency(envelopes: list[dict], answerers: dict[str, Answerer],
                primer: str = PRIMER, trials: int = 1) -> dict:
    """Run the eval for each named answerer over every record-shaped payload.

    Returns {model_name: [scored_row, ...]} where each row carries tool/sha plus the
    per-form success counts and trial count. Payloads that generate no questions
    (no record list, no nested dict-map, not a flat record) are skipped.
    """
    results: dict[str, list[dict]] = {}
    for name, fn in answerers.items():
        rows: list[dict] = []
        for env in envelopes:
            try:
                obj = json.loads(env["raw"])
            except (json.JSONDecodeError, TypeError):
                continue
            for row in run_payload(obj, env["raw"], fn, primer, trials):
                rows.append({"tool": env["tool"], "sha": env.get("sha", "?"), **row})
        results[name] = rows
    return results


# --------------------------------------------------------------------------- #
# Cross-call diff fluency — does the model read a diff as well as the full result?
# (Issue #1 risk item: the round-trip gate proves the diff reconstructs, NOT that the
# model reads it. This measures the second thing, with NO system primer — only the diff's
# inline note, as the proxy delivers it — so flipping `proxy --diff` on is a measured
# decision under production conditions.)
# --------------------------------------------------------------------------- #
def run_diff_payload(prev_obj: Any, curr_obj: Any, answerer: Answerer,
                     tool: str = "", trials: int = 1) -> list[dict]:
    """Does the model answer questions about the CURRENT result as well from
    (previous full result + diff) as from the full current result?

    Both forms are asked with NO system primer — the proxy can't set one, so the diff's
    inline note is the only format guidance, exactly as in production. This is the
    honest test of #9's shortened note (an earlier version fed a full DIFF_PRIMER the
    proxy can't deliver, overstating comprehension).

    Returns rows carrying full-terse (`terse_ok`) and diff-form (`diff_ok`) success
    counts over the SAME questions. [] if curr is not record-shaped or no lossless diff
    applies (nothing to compare)."""
    questions = gen_questions(curr_obj)
    if not questions:
        return []
    wire = diff_wire(prev_obj, curr_obj, tool)
    if wire is None:
        return []
    curr_terse = compress(curr_obj)
    diff_data = f"PREVIOUS RESULT:\n{compress(prev_obj)}\n\nUPDATE (diff against it):\n{wire}"
    out: list[dict] = []
    for q in questions:
        full_u = _user_prompt(q.prompt, q.instruction, curr_terse)
        diff_u = _user_prompt(q.prompt, q.instruction, diff_data)
        _t_ok, _t_f = _ask_n(answerer, "", full_u, q.qtype, q.expected, trials)
        _d_ok, _d_f = _ask_n(answerer, "", diff_u, q.qtype, q.expected, trials)
        # Failed calls leave the denominator, exactly as in `run_payload`: the
        # defect #263 names applies to every harness, not only the primer one.
        _t_tr, _d_tr = trials - _t_f, trials - _d_f
        out.append({
            "qid": q.qid, "qtype": q.qtype, "transform": q.transform, "trials": trials,
            "terse_ok": _t_ok, "terse_trials": _t_tr,
            "diff_ok": _d_ok, "diff_trials": _d_tr,
            # `attempts` is what `_unmeasured` divides by; without it the report reads the
            # row as predating the counters and publishes a verdict regardless (#264).
            "fails": _t_f + _d_f, "attempts": trials * 2,
        })
    return out


def _iter_consecutive_pairs(envelopes: list[dict]):
    """Group envelopes by tool, sort each group by sha (for determinism — the order the
    proxy would see them), and yield consecutive (tool, sha, prev_raw, curr_raw) tuples.
    Shared pairing scaffold for run_diff_fluency and run_text_diff_fluency; each applies
    its own JSON/text domain filter on top of this."""
    by_tool: dict[str, list[dict]] = {}
    for env in envelopes:
        by_tool.setdefault(env["tool"], []).append(env)
    for tool, envs in by_tool.items():
        envs = sorted(envs, key=lambda e: e.get("sha", ""))
        for prev_env, curr_env in zip(envs, envs[1:], strict=False):  # sliding pairs
            yield tool, curr_env.get("sha", "?"), prev_env["raw"], curr_env["raw"]


def _aggregate_by_model(pairs: list[tuple], answerers: dict[str, Answerer], trials: int,
                        payload_fn) -> dict:
    """Shared per-model aggregation loop for run_diff_fluency/run_text_diff_fluency:
    run `payload_fn` over every pair for every answerer and collect the rows."""
    results: dict[str, list[dict]] = {}
    for name, fn in answerers.items():
        rows: list[dict] = []
        for tool, csha, a, b in pairs:
            for row in payload_fn(a, b, fn, tool, trials=trials):
                rows.append({"tool": tool, "sha": csha, **row})
        results[name] = rows
    return results


def run_diff_fluency(envelopes: list[dict], answerers: dict[str, Answerer],
                     trials: int = 1) -> dict:
    """Run the diff-fluency eval over consecutive same-tool payload PAIRS (sorted by sha
    for determinism — the order the proxy would see them). Returns {model: [rows]}."""
    pairs: list[tuple] = []
    for tool, csha, prev_raw, curr_raw in _iter_consecutive_pairs(envelopes):
        try:
            prev_obj = json.loads(prev_raw)
            curr_obj = json.loads(curr_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        pairs.append((tool, csha, prev_obj, curr_obj))
    return _aggregate_by_model(pairs, answerers, trials, run_diff_payload)


# --------------------------------------------------------------------------- #
# Diff-chain soak — the DEPTH dimension run_diff_fluency can't see (#8/#20 follow-up).
# run_diff_payload tests one hop (full + 1 diff). In production a model reads up to
# `diff_keyframe_interval` (default 5) consecutive diffs off one full anchor before
# the proxy re-anchors — so the open question before any default-flip is whether
# comprehension DRIFTS with chain depth. These functions build real depth-k windows
# from the corpus (chronological order, #67) and score the same final-state questions
# at every depth, so the report can show accuracy as a function of depth.
# --------------------------------------------------------------------------- #
def build_chain_windows(envelopes: list[dict], max_depth: int = 5,
                        per_depth_cap: int = 6) -> list[tuple]:
    """Depth-k windows (k = 1..max_depth) of consecutive same-tool payloads where
    EVERY hop admits a lossless diff and the final payload generates questions —
    the exact between-keyframe chains `proxy --diff` emits. Envelope order is
    preserved (load_corpus replays chronologically, #67), so windows are real
    session sequences, not synthetic evolutions.

    Sampling per depth is round-robin across tools (one window per tool per pass,
    starts spread along each tool's maximal diffable run) until `per_depth_cap` —
    tool diversity beats stacking one chatty tool's windows. Returns
    [(tool, final_sha, depth, [obj_0 .. obj_k])]."""
    runs: dict[str, list[list[tuple]]] = {}   # tool -> maximal runs of (sha, obj)
    by_tool: dict[str, list[dict]] = {}
    for env in envelopes:
        by_tool.setdefault(env["tool"], []).append(env)
    for tool, envs in by_tool.items():
        seq: list[tuple] = []
        for e in envs:
            try:
                seq.append((e.get("sha", "?"), json.loads(e["raw"])))
            except (json.JSONDecodeError, TypeError):
                seq.append((e.get("sha", "?"), None))
        cur: list[tuple] = []
        for (psha, p), (csha, c) in zip(seq, seq[1:], strict=False):  # sliding pairs
            ok = (p is not None and c is not None
                  and diff_wire(p, c, tool) is not None)
            if ok:
                if not cur:
                    cur = [(psha, p)]
                cur.append((csha, c))
            elif cur:
                runs.setdefault(tool, []).append(cur)
                cur = []
        if cur:
            runs.setdefault(tool, []).append(cur)

    windows: list[tuple] = []
    for depth in range(1, max_depth + 1):
        # every eligible start per tool, spread over its runs; then round-robin
        candidates: dict[str, list[list[tuple]]] = {}
        for tool, tool_runs in sorted(runs.items()):
            starts: list[list[tuple]] = []
            for run in tool_runs:
                for i in range(0, len(run) - depth, depth):   # non-overlapping
                    window = run[i:i + depth + 1]
                    if gen_questions(window[-1][1]):
                        starts.append(window)
            if starts:
                candidates[tool] = starts
        picked = 0
        idx = 0
        while picked < per_depth_cap and candidates:
            for tool in sorted(candidates):
                starts = candidates[tool]
                if idx >= len(starts):
                    del candidates[tool]
                    continue
                window = starts[idx]
                windows.append((tool, window[-1][0], depth,
                                [obj for _, obj in window]))
                picked += 1
                if picked >= per_depth_cap:
                    break
            idx += 1
    return windows


def run_chain_payload(objs: list, answerer: Answerer, tool: str = "",
                      trials: int = 1) -> list[dict]:
    """run_diff_payload generalized to a depth-N chain: the same questions about the
    FINAL state, control = full-terse of the final result, form = the base full-terse
    plus every intermediate diff wire in order — exactly the context a model has
    accumulated after N consecutive diffs, with no system primer (production
    condition). [] if any hop stops admitting a lossless diff or no questions."""
    questions = gen_questions(objs[-1])
    if not questions:
        return []
    wires: list[str] = []
    for prev, curr in zip(objs, objs[1:], strict=False):  # sliding pairs
        wire = diff_wire(prev, curr, tool)
        if wire is None:
            return []
        wires.append(wire)
    curr_terse = compress(objs[-1])
    chain_data = f"PREVIOUS RESULT:\n{compress(objs[0])}" + "".join(
        f"\n\nUPDATE (diff against the result above, applied in order):\n{w}"
        for w in wires)
    out: list[dict] = []
    for q in questions:
        full_u = _user_prompt(q.prompt, q.instruction, curr_terse)
        chain_u = _user_prompt(q.prompt, q.instruction, chain_data)
        _t_ok, _t_f = _ask_n(answerer, "", full_u, q.qtype, q.expected, trials)
        _d_ok, _d_f = _ask_n(answerer, "", chain_u, q.qtype, q.expected, trials)
        # Failed calls leave the denominator, exactly as in `run_payload`: the
        # defect #263 names applies to every harness, not only the primer one.
        _t_tr, _d_tr = trials - _t_f, trials - _d_f
        out.append({
            "qid": q.qid, "qtype": q.qtype, "transform": q.transform, "trials": trials,
            "depth": len(wires),
            "terse_ok": _t_ok, "terse_trials": _t_tr,
            "diff_ok": _d_ok, "diff_trials": _d_tr,
            # `attempts` is what `_unmeasured` divides by; without it the report reads the
            # row as predating the counters and publishes a verdict regardless (#264).
            "fails": _t_f + _d_f, "attempts": trials * 2,
        })
    return out


def run_diff_soak(envelopes: list[dict], answerers: dict[str, Answerer],
                  trials: int = 1, max_depth: int = 5,
                  per_depth_cap: int = 6) -> dict:
    """Score every answerer over the same depth-1..max_depth chain windows.
    Returns {model: [row,...]} where each row also carries `depth` — the report
    aggregates by it to show comprehension as a function of chain depth."""
    windows = build_chain_windows(envelopes, max_depth=max_depth,
                                  per_depth_cap=per_depth_cap)
    results: dict[str, list[dict]] = {}
    for name, fn in answerers.items():
        rows: list[dict] = []
        for tool, sha, _depth, objs in windows:
            for row in run_chain_payload(objs, fn, tool, trials=trials):
                rows.append({"tool": tool, "sha": sha, **row})
        results[name] = rows
    return results


# --------------------------------------------------------------------------- #
# Text-diff fluency — the text-payload analogue of the diff eval above.
#
# run_diff_payload/run_diff_fluency answer "does a model read a diff against a
# record-shaped JSON payload as well as the full-terse form?" These functions ask
# the same question for unstructured text (text_diff.py, Tier 0.7): does a model
# reconstruct the CURRENT text as accurately from (previous text + text-diff) as
# from the full current text? No new protocol is needed — it's the same paired
# single-shot Answerer comparison, just with a different question source and a
# different "form" — so these live here next to run_diff_payload rather than in
# their own module. (dropeval.py earns its own module because drop-to-retrieve is
# a genuinely different, stateful 2-turn tool-calling protocol; this isn't.)
# --------------------------------------------------------------------------- #
def run_text_diff_payload(prev: str, curr: str, answerer: Answerer,
                          tool: str = "", trials: int = 1) -> list[dict]:
    """Does the model answer questions about the CURRENT text as well from
    (previous text + text-diff) as from the full current text?

    Unlike run_diff_payload's "terse" form (Tier-0 compressed JSON), the control form
    here is just the raw current text — Tier 0 doesn't touch non-JSON payloads at all.
    Field names stay terse_ok/diff_ok anyway (not raw_ok/diff_ok) so the row shape is
    identical to run_diff_payload's — that identity is what lets diff_gap_rows/
    build_terminal_diff_report be reused UNCHANGED for text-diff-eval (a deliberate
    reuse-over-duplication tradeoff).

    Computes the diff wire exactly ONCE (unlike calling gen_text_diff_questions, whose
    own gate would otherwise recompute the same content-defined-chunking diff a second
    time) — text_diff_encode's chunk+hash+round-trip-proof cost is not free."""
    wire = text_diff.text_diff_wire(prev, curr, tool)
    if wire is None:
        return []
    questions = _text_diff_questions_from_lines(curr)
    diff_data = f"PREVIOUS RESULT:\n{prev}\n\nUPDATE (diff against it):\n{wire}"
    out: list[dict] = []
    for q in questions:
        full_u = _user_prompt(q.prompt, q.instruction, curr)
        diff_u = _user_prompt(q.prompt, q.instruction, diff_data)
        _t_ok, _t_f = _ask_n(answerer, "", full_u, q.qtype, q.expected, trials)
        _d_ok, _d_f = _ask_n(answerer, "", diff_u, q.qtype, q.expected, trials)
        # Failed calls leave the denominator, exactly as in `run_payload`: the
        # defect #263 names applies to every harness, not only the primer one.
        _t_tr, _d_tr = trials - _t_f, trials - _d_f
        out.append({
            "qid": q.qid, "qtype": q.qtype, "transform": q.transform, "trials": trials,
            "terse_ok": _t_ok, "terse_trials": _t_tr,
            "diff_ok": _d_ok, "diff_trials": _d_tr,
            # `attempts` is what `_unmeasured` divides by; without it the report reads the
            # row as predating the counters and publishes a verdict regardless (#264).
            "fails": _t_f + _d_f, "attempts": trials * 2,
        })
    return out


def run_text_diff_fluency(envelopes: list[dict], answerers: dict[str, Answerer],
                          trials: int = 1) -> dict:
    """Same tool-pairing loop as run_diff_fluency, inverted: only pairs envelopes whose
    raw text is NOT valid JSON on EITHER side (text-diff's domain; JSON payloads are
    run_diff_fluency's domain instead) — classify_shape (already used by measure.py)
    catches the Python-3.11 RecursionError on deeply-nested JSON that a bare
    json.loads/except wouldn't."""
    pairs: list[tuple] = []
    for tool, csha, prev_raw, curr_raw in _iter_consecutive_pairs(envelopes):
        if classify_shape(curr_raw) not in (LONG_TEXT, OTHER):
            continue  # curr is JSON-shaped -> run_diff_fluency's domain, not this one's
        if classify_shape(prev_raw) not in (LONG_TEXT, OTHER):
            continue  # prev is JSON-shaped -> not a text-to-text transition
        pairs.append((tool, csha, prev_raw, curr_raw))
    return _aggregate_by_model(pairs, answerers, trials, run_text_diff_payload)
