"""A row with no `tool` key must not fall out of the per-tool tables.

All three report surfaces built their tool list with `r.get("tool", "?")` and then filtered
with `r.get("tool") == tool`. The two disagree on exactly one input: a row where the key is
ABSENT. The set comprehension substitutes `"?"`, the filter compares `None == "?"`, and the
row matches nothing — so its tokens vanish from the per-tool table while still counting in
every other total on the page, and a phantom `?` row is printed at 0/0.

A row with `"tool": None` is affected differently and worse. The key exists, so `None` enters
the set and `sorted()` raises `TypeError: '<' not supported between 'str' and 'NoneType'` —
the whole report dies instead of mis-reporting. That second failure is why the fix normalises
with `or "?"` rather than just giving the filter a matching default: `or` covers absent,
`None` and `""` in one expression, on both lines.

Not reachable from the CLI today: `measure_corpus` sets `tool` unconditionally
(`measure.py`), and `_cmd_measure`/`_cmd_verify` are the only production callers. These
functions are public and take rows as an argument, though, and the `"?"` default is itself
evidence someone expected the key to be missable — the code intends to handle this case and
handled it by silently dropping data. Pinned here so the intent and the implementation
agree.
"""

from __future__ import annotations

import pytest

from terse.html_report import build_html_report
from terse.report import build_report
from terse.terminal_report import build_terminal_report

COVERAGE = {"tools": 2}


def _row(tool, raw, cmp_):
    row = {
        "shape": "record-array", "sha": "abc", "roundtrip_ok": True, "embedded_ok": True,
        "cl100k": {"raw": raw, "minify": raw, "tabularize": cmp_, "dictionary": cmp_,
                   "compressed": cmp_, "embedded": cmp_},
        "saved_cl100k": {"minify": 0, "tabularize": raw - cmp_, "dictionary": 0,
                         "embedded": 0, "tier_total": raw - cmp_},
        "chars": {"raw": raw * 4, "compressed": cmp_ * 4},
    }
    if tool is not None:          # `None` here means "omit the key entirely"
        row["tool"] = tool
    return row


# 1,000 raw tokens under a named tool, 5,000 under a row with no `tool` key at all.
ROWS = [_row("alpha", 1_000, 400), _row(None, 5_000, 500)]


def test_the_markdown_per_tool_table_accounts_for_every_raw_token():
    md = build_report(ROWS, COVERAGE)
    body = [ln for ln in md.splitlines() if ln.startswith("| `")]
    accounted = sum(int(ln.split("|")[3]) for ln in body
                    if ln.split("|")[3].strip().isdigit())
    assert accounted == 6_000, (
        f"per-tool table accounts for {accounted} of 6,000 raw tokens — a row whose "
        f"`tool` key is absent fell out of it:\n" + "\n".join(body))


def test_the_untooled_row_is_not_rendered_as_an_empty_placeholder():
    """The visible symptom: a `?` row at 0/0/n/a beside the real ones, which reads as "a
    tool that saved nothing" when it is in fact the largest payload in the corpus."""
    md = build_report(ROWS, COVERAGE)
    placeholder = [ln for ln in md.splitlines()
                   if ln.startswith("| `?`") and "| 0 | 0 |" in ln]
    assert not placeholder, f"phantom zero row: {placeholder}"


def test_the_html_surface_accounts_for_the_untooled_row():
    """All three surfaces carried the identical mismatch, so fixing one is not fixing it.
    HTML prints raw token counts, so the untooled row's 5,000 has to appear."""
    assert "5000" in build_html_report(ROWS, COVERAGE), (
        "the untooled row's 5,000 raw tokens are absent from the HTML per-tool table")


def test_the_terminal_surface_rates_the_untooled_row_instead_of_zeroing_it():
    """The terminal surface renders percentages and bars rather than counts, so the symptom
    there is a rate, not a missing number: before the fix the `?` bucket matched no rows and
    charted at +0.0% — "a tool that saved nothing" — when it is in fact the best-compressing
    payload in the corpus at 5,000 -> 500."""
    out = build_terminal_report(ROWS, COVERAGE)
    assert "+90.0%" in out, f"untooled row not rated at its real 90% saving:\n{out}"
    assert "+0.0%" not in out


@pytest.mark.parametrize("tool_value", [None, ""], ids=["explicit-none", "empty-string"])
def test_a_falsy_tool_value_does_not_crash_the_report(tool_value):
    """This started as a control asserting `"tool": None` was unaffected. It is not, and it
    is WORSE than the absent-key case: the key exists, so `None` entered the set and
    `sorted()` raised `TypeError: '<' not supported between 'str' and 'NoneType'` — the
    whole report died rather than mis-reporting. Normalising with `or "?"` covers absent,
    `None` and `""` in one expression, which is why the fix is that and not a matching
    default on the filter."""
    rows = [_row("alpha", 1_000, 400), {**_row(None, 5_000, 500), "tool": tool_value}]
    md = build_report(rows, COVERAGE)               # must not raise
    body = [ln for ln in md.splitlines() if ln.startswith("| `")]
    accounted = sum(int(ln.split("|")[3]) for ln in body
                    if ln.split("|")[3].strip().isdigit())
    assert accounted == 6_000
