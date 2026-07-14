"""Format-fluency eval: does a model read terse's compressed form as accurately
as raw JSON?

This answers the proxy's one open question (TECHNICAL.md "Known Limitations"):
*correctness* is test-covered — `decompress(compress(x)) == x` — but *usefulness*
rests on an untested assumption, that a model reading the table/legend form in place
of raw JSON loses no comprehension. We measure it the way this project measures
everything: deterministically, cross-model, and never as a single black-box number.

Method (the honesty bar, principle #24):
  - Ground truth is computed from the parsed records (count / lookup / enumerate /
    aggregate) and checked programmatically — no LLM-as-judge, which would re-import
    the nondeterminism we are trying to measure away.
  - Questions deliberately target the two transforms most likely to cost
    comprehension: `~N` dictionary-alias resolution and column->value mapping over
    wide/long positional tables (the under-enumeration gap the row-count hint exists
    for). Each question is tagged with the transform it stresses, so the report says
    *which* transform costs comprehension, not just an aggregate.
  - Scoring is PAIRED: the same questions, same order, over raw vs terse (+/- a
    one-time format primer). A regression is a question the model got right on raw
    and wrong on terse — a stronger signal than two independent accuracy rates.
  - The verdict gates on the WORST model, not the mean: a format that helps one model
    but breaks the real consumer is a regression (mirrors how `validate` reports
    cross-tokenizer divergence rather than averaging it away).

The answerer is a pluggable `(system, user) -> reply` callable, so the pure core
(question generation + scoring) runs offline with no network or key. The live backend
(`openai_answerer` over stdlib urllib) reaches any OpenAI-compatible endpoint — the
broker pool or a loopback gateway — and adds zero new dependencies.

Package layout (#78 — split from one 958-line module; this facade preserves the
`fluency.X` surface every caller/test already uses):
  questions.py  — the three question sources behind `gen_questions`, + text-diff's
  scoring.py    — programmatic answer checking (`score`), no LLM-as-judge
  answerers.py  — the Answerer protocol + the stdlib-urllib live backend
  harnesses.py  — paired eval runners (plain / diff / chain-soak / text-diff)
  pack.py       — offline eval packs + the one-time format PRIMER
"""

from __future__ import annotations

# Kept importable as `fluency.text_diff` / `fluency.compress`: pre-split these were
# reachable module attributes and tests/monkeypatching rely on them.
from .. import text_diff  # noqa: F401
from ..transforms import compress  # noqa: F401
from .answerers import (  # noqa: F401
    _LOOPBACK_HOSTS,
    Answerer,
    openai_answerer,
)
from .harnesses import (  # noqa: F401
    _aggregate_by_model,
    _ask_n,
    _iter_consecutive_pairs,
    _safe_ask,
    _user_prompt,
    build_chain_windows,
    run_chain_payload,
    run_diff_fluency,
    run_diff_payload,
    run_diff_soak,
    run_fluency,
    run_payload,
    run_text_diff_fluency,
    run_text_diff_payload,
)
from .pack import (  # noqa: F401
    PRIMER,
    build_pack,
    score_pack,
    token_summary,
)

# Private helpers are re-exported too (not just the public API): dropeval.py reuses
# _pick_id_col/_user_prompt/_safe_ask by design (its docstring says why), and the
# tests address the generators through the `fluency.` namespace. The split is a pure
# move — no caller churn.
from .questions import (  # noqa: F401
    Question,
    _aliased_canon,
    _aliased_strings,
    _flat_record_questions,
    _intersection_cols,
    _nested_questions,
    _nested_record_group,
    _pick_id_col,
    _pick_numeric_col,
    _pick_target_col,
    _text_diff_questions_from_lines,
    gen_questions,
    gen_text_diff_questions,
)
from .scoring import (  # noqa: F401
    _NUM,
    _extract_json,
    _is_number,
    _matches_number,
    _norm_scalar,
    _parse_list,
    _score_form,
    score,
)
