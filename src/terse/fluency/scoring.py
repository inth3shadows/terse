"""Scoring — deterministic, lenient on formatting, strict on the value (#78 split).

Ground truth is computed from the parsed records and checked programmatically —
no LLM-as-judge, which would re-import the nondeterminism the eval is trying to
measure away (principle #24).
"""

from __future__ import annotations

import json
import re
from typing import Any

# A standalone number: not glued to a letter/digit/underscore on either side, so the
# expected value can't be spuriously "found" inside an identifier or version string
# (e.g. expected 6 must not match the "6" in "record_6" or "v6.2"). Trailing sentence
# punctuation (". ,)") is fine — only alnum/underscore neighbours disqualify a match.
_NUM = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _norm_scalar(s: str) -> str:
    return s.strip().strip("\"'").strip().lower()


def _parse_list(reply: str) -> list | None:
    """Best-effort: a JSON array if present, else a comma/newline split. We instructed
    a JSON array, so the split is only a courtesy against formatting quirks."""
    start, end = reply.find("["), reply.rfind("]")
    if start != -1 and end > start:
        try:
            v = json.loads(reply[start:end + 1])
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    parts = [p.strip().strip("\"'") for p in re.split(r"[,\n]", reply) if p.strip()]
    return parts or None


def _matches_number(reply: str, expected: Any) -> bool:
    """True iff the expected number appears as a standalone token anywhere in the reply.
    Matching ANY standalone number (not just the first) tolerates prose like "there are 6
    records" without being fooled by a leading incidental number; the standalone rule (see
    _NUM) additionally rejects a value merely embedded in an identifier/version string."""
    return any(abs(float(tok) - float(expected)) < 1e-9 for tok in _NUM.findall(reply))


def _is_sole_number(reply: str, expected: Any) -> bool:
    """True iff the reply IS the expected number and essentially nothing else — the strict
    counterpart to `_matches_number`'s present-anywhere rule. Needed where the correct value
    also appears in the surrounding context the model was shown, so 'contains it' proves
    nothing: the drop-eval's numbered recall injects the retrieved code block (every one of
    its line numbers) into the conversation, and a reply that merely echoes that block would
    contain the target line number without demonstrating the model located the right line.
    Requires a lone number after stripping quotes, whitespace and one trailing period."""
    toks = _NUM.findall(reply.strip().strip("\"'").strip())
    return len(toks) == 1 and abs(float(toks[0]) - float(expected)) < 1e-9


def _extract_json(reply: str) -> Any:
    """Pull the first JSON object/array out of a reply (tolerating surrounding prose).
    Returns a sentinel-free value or raises ValueError if none parses."""
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = reply.find(open_c), reply.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(reply[i:j + 1])
            except json.JSONDecodeError:
                pass
    return json.loads(reply)  # last resort; raises if not JSON


def score(qtype: str, expected: Any, reply: str) -> bool:
    """True iff the reply conveys the expected answer. Tolerates surrounding prose/
    quotes; compares the value exactly (numbers within float epsilon). No blanket
    empty-reply reject — an empty reply only matches an empty expected scalar, which
    each branch already decides correctly."""
    reply = reply.strip()
    if qtype == "sole_number":
        return _is_sole_number(reply, expected)
    if qtype == "count" or (qtype == "aggregate" and _is_number(expected)):
        return _matches_number(reply, expected)
    if qtype == "enumerate":
        got = _parse_list(reply)
        if got is None:
            return False
        return [_norm_scalar(str(x)) for x in got] == [_norm_scalar(str(x)) for x in expected]
    if qtype == "deref":
        try:
            return _extract_json(reply) == expected  # JSON value-equality (dict order-insensitive)
        except (json.JSONDecodeError, ValueError):
            return False
    # lookup / generic scalar
    if _is_number(expected):
        return _matches_number(reply, expected)
    return _norm_scalar(reply) == _norm_scalar(str(expected))


# A form absent from a responses file. NOT the same fact as a reply that came back empty,
# and the whole of #279 is that the two were indistinguishable: `forms.get("raw", "")` fed
# a never-collected form to the scorer as `""`, which then scored as a wrong answer. Pass
# this sentinel for "the key is not in the file" and a real value (including `None`) for
# "the driver recorded something for this form".
MISSING: Any = object()


def _score_form(qtype: str, expected: Any, form_val: Any) -> tuple[int, int, int]:
    """(successes, scored, collected) for one form's collected reply(s).

    A single value is one collected reply; a list is N. `collected` counts every reply the
    driver recorded; `scored` counts only the ones that are actually answers. A reply that
    is `None`, non-str, or blank is a NON-ANSWER and leaves BOTH the numerator and the
    denominator (#279) — it is not evidence the model got the question wrong.

    This is parity with the live path, not a new rule: `answerers.openai_answerer` already
    returns `None` for a blank reply (`questions._answerable`'s docstring names that as the
    premise it defends), and `harnesses._ask_n` already answers it with `fails += 1;
    continue`. Only the offline `score_pack` path scored a non-answer as a miss — executed
    before this fix, `_score_form("lookup", "x", [None, None, "x"])` returned `(1, 3)`: a
    model right on every reply it actually produced, reported at 33%.

    A form that is `MISSING` yields `(0, 0, 0)` — nothing was collected, so there is neither
    an answer nor a lost call to count."""
    if form_val is MISSING:
        return 0, 0, 0
    replies = form_val if isinstance(form_val, list) else [form_val]
    answered = [r for r in replies if isinstance(r, str) and r.strip()]
    successes = sum(score(qtype, expected, r) for r in answered)
    return successes, len(answered), len(replies)
