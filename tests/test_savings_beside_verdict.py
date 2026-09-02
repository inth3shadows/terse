"""Token savings are reported BESIDE the codec verdict, never inside it (#303, #295 DoD 4).

`build_codec_verdict_report` renders two tables over the same `(tool, shape)` groups: the
SAFE/UNSAFE/UNRESOLVED verdict, then — as a sibling section — what each group saves. The
invariant this file exists to hold is a PRESENTATION one, and prose is exactly where such
an invariant drifts: nothing may weight, multiply, or otherwise fold a savings figure into
a verdict, and equally nothing may suppress a savings figure because the verdict was UNSAFE.
Both directions are editorialising. "This shape saves 60% and is UNSAFE" is the true and
useful statement; either half alone is a worse report.

The de-duplication tests are not decoration. `run_codec_fluency` stamps the same per-payload
counts onto every question row that payload produces, once per model that answered it — so
summing rows naively multiplies a payload's tokens by (questions x models). At the fixture's
own numbers that is a 6x overstatement of a savings figure."""
from __future__ import annotations

from terse.codeceval import _payload_tokens
from terse.report import _CODEC_MIN_TRIALS, build_codec_verdict_report

_VERDICTS = ("SAFE", "UNSAFE", "UNRESOLVED")

# Deliberately unlike anything the verdict table can print: its cells carry trial counts
# (20) and accuracy percentages rendered with no decimal ("100%", "80%"). Every token below
# is therefore attributable to the savings table alone, which is what makes the
# "no line carries both" assertion mean something.
_RAW_TOK, _TERSE_TOK = 1000, 400
_SAVED = "+600"
_SAVED_PCT = "+60.0%"


def _row(qid: str, raw_ok: int, terse_ok: int, sha: str = "sha1",
         tokens: bool = True, trials: int = 1) -> dict:
    r = {
        "qid": qid, "qtype": "deref", "transform": "table", "trials": trials,
        "raw_ok": raw_ok, "terse_ok": terse_ok,
        "raw_trials": trials, "terse_trials": trials,
        "fails": 0, "attempts": trials * 2, "sha": sha,
    }
    if tokens:
        r |= {"raw_tokens": _RAW_TOK, "terse_tokens": _TERSE_TOK}
    return r


def _tagged(rows: list[dict], tool: str, shape: str) -> list[dict]:
    return [{"tool": tool, "shape": shape, **r} for r in rows]


def _sections(report: str) -> tuple[str, str]:
    """(verdict section, savings section) — split on the savings heading, so a test can say
    where a number appeared and not merely that it appeared."""
    head, sep, tail = report.partition("## Savings by tool and shape")
    assert sep, "report rendered no savings section"
    return head, sep + tail


# --------------------------------------------------------------------------- #
# An UNSAFE group still publishes its savings
# --------------------------------------------------------------------------- #
def test_an_unsafe_group_still_renders_its_savings_number():
    # One demonstrated excess terse miss -> UNSAFE, full stop. The savings figure is an
    # independent measurement of the same payload and must survive that verdict.
    results = {"m1": _tagged([_row("q1", 1, 0)], "tool-a", "array-of-records")}
    report = build_codec_verdict_report(results)
    assert "**UNSAFE**" in report
    verdict, savings = _sections(report)
    assert "**UNSAFE**" in verdict
    assert _SAVED in savings and _SAVED_PCT in savings
    assert str(_RAW_TOK) in savings and str(_TERSE_TOK) in savings


def test_every_verdict_grade_gets_a_savings_row():
    # SAFE, UNSAFE and UNRESOLVED side by side: the savings table must have one row per
    # group regardless of grade, or "suppressed on UNSAFE" could hide behind a fixture that
    # only ever renders one verdict.
    clean = [_row(f"q{i}", 1, 1, sha="safe-sha") for i in range(_CODEC_MIN_TRIALS)]
    results = {"m1": (_tagged(clean, "tool-safe", "array-of-records")
                      + _tagged([_row("q1", 1, 0, sha="unsafe-sha")],
                                "tool-unsafe", "array-of-records")
                      + _tagged([_row("q1", 1, 1, sha="thin-sha")],
                                "tool-thin", "array-of-records"))}
    report = build_codec_verdict_report(results)
    verdict, savings = _sections(report)
    for grade in _VERDICTS:
        assert f"**{grade}**" in verdict
    for tool in ("tool-safe", "tool-unsafe", "tool-thin"):
        assert f"| `{tool}` |" in savings, f"{tool} lost its savings row"


# --------------------------------------------------------------------------- #
# Nothing combines the two
# --------------------------------------------------------------------------- #
def test_no_rendered_line_carries_both_a_verdict_and_a_savings_figure():
    clean = [_row(f"q{i}", 1, 1, sha="safe-sha") for i in range(_CODEC_MIN_TRIALS)]
    results = {"m1": (_tagged(clean, "tool-safe", "array-of-records")
                      + _tagged([_row("q1", 1, 0, sha="unsafe-sha")],
                                "tool-unsafe", "array-of-records"))}
    report = build_codec_verdict_report(results)
    savings_tokens = (str(_RAW_TOK), str(_TERSE_TOK), _SAVED, _SAVED_PCT)
    offenders = [
        line for line in report.splitlines()
        if any(f"**{v}**" in line for v in _VERDICTS)
        and any(t in line for t in savings_tokens)
    ]
    assert not offenders, f"a cell combined a verdict with a savings figure: {offenders}"


def test_the_savings_table_declares_its_own_independence_in_prose():
    # The narrative around the numbers is half of this invariant: a reader who takes the
    # savings figure as a mitigation of an UNSAFE verdict has read the report wrong, and
    # the report is what has to say so.
    _, savings = _sections(build_codec_verdict_report(
        {"m1": _tagged([_row("q1", 1, 0)], "tool-a", "array-of-records")}))
    lowered = savings.lower()
    assert "never" in lowered and "beside" in lowered
    assert "unsafe group still prints" in lowered


def test_the_verdict_table_comes_first():
    # Ordering IS the argument (#295 DoD 4): correctness is settled before the economics
    # are read, not alongside them.
    report = build_codec_verdict_report(
        {"m1": _tagged([_row("q1", 1, 0)], "tool-a", "array-of-records")})
    assert (report.index("## Verdict by tool and shape")
            < report.index("## Savings by tool and shape"))


# --------------------------------------------------------------------------- #
# Per-payload counts, summed per payload
# --------------------------------------------------------------------------- #
def test_a_payloads_tokens_are_counted_once_across_its_questions_and_models():
    # One payload, three questions, two models = six rows carrying the same two counts.
    # Summing rows would print 6000/2400; the truth is 1000/400 over ONE payload.
    rows = _tagged([_row(f"q{i}", 1, 1, sha="one-sha") for i in range(3)],
                   "tool-a", "array-of-records")
    report = build_codec_verdict_report({"m1": rows, "m2": rows})
    _, savings = _sections(report)
    assert f"| 1 | {_RAW_TOK} | {_TERSE_TOK} | {_SAVED} | {_SAVED_PCT} |" in savings
    assert "6000" not in savings and "2400" not in savings


def test_two_distinct_payloads_in_a_group_are_both_counted():
    # The mirror of the test above: de-duplication must key on `sha`, not collapse the
    # group to a single payload.
    rows = _tagged([_row("q1", 1, 1, sha="sha-a"), _row("q1", 1, 1, sha="sha-b")],
                   "tool-a", "array-of-records")
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert f"| 2 | {2 * _RAW_TOK} | {2 * _TERSE_TOK} | +1200 | {_SAVED_PCT} |" in savings


def test_groups_keep_their_savings_separate_the_way_verdicts_do():
    rows = (_tagged([_row("q1", 1, 1, sha="sha-a")], "tool-a", "array-of-records")
            + _tagged([_row("q1", 1, 1, sha="sha-b")], "tool-a", "compact-json"))
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "| `tool-a` | array-of-records | 1 |" in savings
    assert "| `tool-a` | compact-json | 1 |" in savings


# --------------------------------------------------------------------------- #
# An unmeasured payload is excluded, never read as zero
# --------------------------------------------------------------------------- #
def test_a_row_without_token_counts_is_excluded_not_counted_as_zero_saving():
    # A stored result predating #303 (or a run with no tokenizer) carries no counts. Read
    # as 0/0 those payloads would print a perfect saving off a measurement that never
    # happened — the same "absence read as a result" failure #279 fixed in the scorer.
    rows = _tagged([_row("q1", 1, 1, sha="counted"),
                    _row("q1", 1, 1, sha="uncounted", tokens=False)],
                   "tool-a", "array-of-records")
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert f"| 1 | {_RAW_TOK} | {_TERSE_TOK} |" in savings  # one payload, not two
    assert "1 payload(s) carry no token counts" in savings
    assert "+100.0%" not in savings


def test_a_group_with_no_token_counts_at_all_prints_na_rather_than_a_saving():
    rows = _tagged([_row("q1", 1, 1, sha="s", tokens=False)], "tool-a", "array-of-records")
    report = build_codec_verdict_report({"m1": rows})
    _, savings = _sections(report)
    assert "| `tool-a` | array-of-records | 0 | n/a | n/a | n/a | n/a |" in savings
    assert "**SAFE**" not in savings   # the verdict does not leak into the savings section
    assert "1 payload(s) carry no token counts" in savings
    # No saving of ANY sign is claimed for a group nothing measured.
    assert "%" not in savings.split("|---|---|---|---|---|---|---|")[1].split("\n\n")[0]


def test_a_payload_counted_by_a_later_model_is_not_also_reported_as_uncounted():
    # Reachable by merging result files across runs — one predating #303, one after — where
    # the same `sha` appears both with and without counts. The payload IS measured; listing
    # it in the uncounted note as well would double-count it against itself and understate
    # how much of the group the sums cover. Order matters: the uncounted row is seen FIRST.
    stale = _tagged([_row("q1", 1, 1, sha="shared", tokens=False)],
                    "tool-a", "array-of-records")
    fresh = _tagged([_row("q1", 1, 1, sha="shared")], "tool-a", "array-of-records")
    _, savings = _sections(build_codec_verdict_report({"old-run": stale, "new-run": fresh}))
    assert f"| 1 | {_RAW_TOK} | {_TERSE_TOK} |" in savings
    assert "carry no token counts" not in savings


def test_the_uncounted_note_is_absent_when_every_payload_was_measured():
    rows = _tagged([_row("q1", 1, 1, sha="s")], "tool-a", "array-of-records")
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "carry no token counts" not in savings


# --------------------------------------------------------------------------- #
# The emitter side
# --------------------------------------------------------------------------- #
def test_payload_tokens_measures_the_two_forms_the_model_was_actually_fed():
    import json

    from terse.tokenize import count_cl100k
    from terse.transforms import compress

    obj = [{"a": 1, "b": {"x": [1, 2, 3]}}, {"a": 2, "b": {"x": [4, 5, 6]}}]
    raw = json.dumps(obj)
    toks = _payload_tokens(raw, obj)
    assert toks == {"raw_tokens": count_cl100k(raw),
                    "terse_tokens": count_cl100k(compress(obj))}
    # A real record-shaped payload must actually be smaller compressed, or the fixture is
    # measuring nothing and would pass with the two counts swapped.
    assert toks["terse_tokens"] < toks["raw_tokens"]


def test_payload_tokens_emits_nothing_when_the_tokenizer_is_unavailable(monkeypatch):
    monkeypatch.setattr("terse.codeceval.count_cl100k", lambda _t: None)
    assert _payload_tokens('{"a": 1}', {"a": 1}) == {}


def test_run_codec_fluency_stamps_the_counts_on_every_row_of_a_payload():
    import json

    from terse import codeceval
    from terse.dropeval import ToolCall, Turn

    obj = [{"id": 1, "blob": {"k": [1, 2, 3]}}, {"id": 2, "blob": {"k": [4, 5, 6]}}]
    raw = json.dumps(obj)
    env = {"tool": "t", "shape": "array-of-records", "sha": "abc", "raw": raw}

    def answerer(_messages, **_kw):
        return Turn(content=None,
                    tool_calls=[ToolCall(name=codeceval.RECORD_VALUE_TOOL,
                                         arguments={"value": {"k": [1, 2, 3]}})])

    rows = codeceval.run_codec_fluency([env], {"m": answerer})["m"]
    assert rows, "fixture produced no deref questions — it cannot fail"
    expected = _payload_tokens(raw, obj)
    assert expected, "tokenizer unavailable; this test cannot assert anything"
    for r in rows:
        assert r["raw_tokens"] == expected["raw_tokens"]
        assert r["terse_tokens"] == expected["terse_tokens"]
        assert r["sha"] == "abc"


# --------------------------------------------------------------------------- #
# Mutation catalogue — every entry was applied to the source, the suite re-run, and the
# named test confirmed to redden. Zero SURVIVED. Entry 13 is why
# `test_a_payload_counted_by_a_later_model_is_not_also_reported_as_uncounted` exists: the
# first cut of this file did not have it, and 13 survived all 14 other tests.
#
# The renderer (report.py `_codec_savings_section`):
#   1. never call `_codec_savings_section` -> reddens 10 tests, including the two that hold
#      #303's actual requirement.
#   2. `continue` past any group with a demonstrated excess terse miss (i.e. suppress
#      savings on UNSAFE) -> reddens `test_an_unsafe_group_still_renders_its_savings_number`
#      and `test_every_verdict_grade_gets_a_savings_row`. This is the failure #303 forbids
#      in the OTHER direction from the one people expect.
#   3. append the payload's token counts to the verdict table's `Why` cell -> reddens
#      `test_no_rendered_line_carries_both_a_verdict_and_a_savings_figure` alone. That test
#      is the whole presentation invariant; nothing else catches this.
#   4. `counted[sha + str(len(counted))] = ...` (sum rows instead of de-duplicating by sha)
#      -> reddens `test_a_payloads_tokens_are_counted_once_across_its_questions_and_models`.
#      At the fixture's numbers this prints 6000 raw tokens for a 1000-token payload.
#   5. `counted.setdefault("_", ...)` (collapse the group to one payload) -> reddens
#      `test_two_distinct_payloads_in_a_group_are_both_counted`. The opposite direction from
#      4, and a fixture with one payload per group would miss it.
#   6. treat a missing/None count as 0 -> reddens the two n/a tests. Read as 0/0 an
#      unmeasured payload prints a perfect saving.
#  12. emit the uncounted note unconditionally -> reddens
#      `test_the_uncounted_note_is_absent_when_every_payload_was_measured`.
#  13. drop `uncounted -= set(counted)` -> reddens
#      `test_a_payload_counted_by_a_later_model_is_not_also_reported_as_uncounted` ALONE.
#      SURVIVED every other test here: the ordering it depends on (a sha's uncounted row
#      seen before its counted one) only arises from result files merged across runs.
#
# Ordering (report.py `build_codec_verdict_report`):
#  10. render the savings section BEFORE the verdict table -> reddens
#      `test_the_verdict_table_comes_first` plus the two `_sections`-splitting tests.
#  11. replace the independence prose with a bare "Token savings for each group." ->
#      reddens `test_the_savings_table_declares_its_own_independence_in_prose`. The prose
#      is load-bearing: the numbers alone do not tell a reader not to trade them off.
#
# The emitter (codeceval.py `_payload_tokens` / `run_codec_fluency`):
#   7. return `{"raw_tokens": 0, "terse_tokens": 0}` when the tokenizer is unavailable ->
#      reddens `test_payload_tokens_emits_nothing_when_the_tokenizer_is_unavailable`.
#   8. swap the two counts -> reddens
#      `test_payload_tokens_measures_the_two_forms_the_model_was_actually_fed`, and only
#      because that test asserts `terse < raw` on a genuinely record-shaped fixture. A
#      fixture that did not compress would make the swap undetectable.
#   9. drop `**toks` from the emitted row -> reddens
#      `test_run_codec_fluency_stamps_the_counts_on_every_row_of_a_payload`.
