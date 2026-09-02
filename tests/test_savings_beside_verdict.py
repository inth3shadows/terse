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
    verdict, savings = _sections(build_codec_verdict_report(results))
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


# The two sentences that carry the invariant in prose, pinned VERBATIM. Substring probes
# for "never" / "beside" / "unsafe group still prints" are not enough: review found that
# "Savings are never reported beside an UNSAFE verdict; an UNSAFE group still prints
# nothing." satisfies all three while asserting the exact negation. A negation is built by
# inserting words, so only an exact match excludes one.
_INDEPENDENCE_PROSE = (
    "Reported BESIDE the verdict above, never folded into it: no figure here is\n"
    "weighted by, multiplied into, or gated on a SAFE/UNSAFE/UNRESOLVED result, and an\n"
    "UNSAFE group still prints what it saves."
)


def test_the_savings_table_declares_its_own_independence_in_prose():
    # The narrative around the numbers is half of this invariant: a reader who takes the
    # savings figure as a mitigation of an UNSAFE verdict has read the report wrong, and
    # the report is what has to say so.
    _, savings = _sections(build_codec_verdict_report(
        {"m1": _tagged([_row("q1", 1, 0)], "tool-a", "array-of-records")}))
    assert _INDEPENDENCE_PROSE in savings


def test_the_savings_heading_is_a_sibling_not_a_subsection():
    # `###` would nest the economics UNDER the verdict, which is as close to "folded into
    # it" as markdown gets — and it survives every substring probe, because "### Savings"
    # contains "## Savings". The heading LEVEL is the structural half of "sibling", so it
    # is pinned line-anchored and against the verdict heading it must be a peer of.
    report = build_codec_verdict_report(
        {"m1": _tagged([_row("q1", 1, 0)], "tool-a", "array-of-records")})
    headings = [ln for ln in report.splitlines() if ln.startswith("#")]
    assert "## Savings by tool and shape" in headings
    assert "### Savings by tool and shape" not in headings
    verdict_h = "## Verdict by tool and shape"
    assert verdict_h in headings
    assert headings.index("## Savings by tool and shape") == headings.index(verdict_h) + 1


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
    assert "1 payload(s)" in savings and "carry no token counts" in savings
    assert "+100.0%" not in savings


def test_a_group_with_no_token_counts_at_all_prints_na_rather_than_a_saving():
    rows = _tagged([_row("q1", 1, 1, sha="s", tokens=False)], "tool-a", "array-of-records")
    report = build_codec_verdict_report({"m1": rows})
    _, savings = _sections(report)
    assert "| `tool-a` | array-of-records | 0 | n/a | n/a | n/a | n/a |" in savings
    assert "**SAFE**" not in savings   # the verdict does not leak into the savings section
    assert "1 payload(s)" in savings and "carry no token counts" in savings
    # No saving of ANY sign is claimed for a group nothing measured. Scoped to the table
    # BODY: the header row carries a literal "%", and the disclosure note below it may
    # carry digits. `[1]` raises IndexError if the column count ever changes — a loud
    # failure rather than a silently widened scan.
    body = savings.split("|---|---|---|---|---|---|---|")[1].split("\n\n")[0]
    assert "%" not in body and "+" not in body


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
# A row that cannot be attributed to a payload (#303 review, Q1)
# --------------------------------------------------------------------------- #
def test_rows_without_a_sha_are_excluded_and_disclosed_not_collapsed_into_one():
    # The defect review found: `str(r.get("sha", "?"))` mapped every sha-less row to one
    # shared key, so N distinct payloads collapsed into whichever was seen first — wrong
    # count, wrong sums, wrong percentage — and because "?" landed in `counted`, the
    # disclosure note stayed silent about it. All five wrong at once, in silence.
    rows = _tagged([_row("q1", 1, 1), _row("q1", 1, 1)], "tool-a", "array-of-records")
    for r in rows:
        del r["sha"]
    rows[1] |= {"raw_tokens": 50, "terse_tokens": 45}
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "| `tool-a` | array-of-records | 0 | n/a | n/a | n/a | n/a |" in savings
    assert "2 row(s) carry no `sha`" in savings
    # Specifically NOT the first row's numbers standing in for both payloads.
    assert _SAVED not in savings and str(_RAW_TOK) not in savings


def test_a_sha_that_is_not_a_usable_string_is_treated_the_same_way():
    rows = _tagged([_row("q1", 1, 1, sha="ok"), _row("q2", 1, 1), _row("q3", 1, 1)],
                   "tool-a", "array-of-records")
    rows[1]["sha"] = ""      # present but empty
    rows[2]["sha"] = 12345   # present but not a string
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert f"| 1 | {_RAW_TOK} | {_TERSE_TOK} |" in savings   # only the usable one
    assert "2 row(s) carry no `sha`" in savings


# --------------------------------------------------------------------------- #
# What counts as a token count
# --------------------------------------------------------------------------- #
def test_a_string_token_count_is_uncounted_rather_than_summed():
    # A stored "1000" would reach sum() and the {:+d} format and raise. Never shipped
    # broken — the first cut's bare `isinstance(v, int)` already excluded it — but pinned
    # so a later widening of the check cannot reintroduce it. Excluded, and the payload is
    # disclosed: the same treatment as an absent count.
    rows = _tagged([_row("q1", 1, 1, sha="s")], "tool-a", "array-of-records")
    rows[0] |= {"raw_tokens": "1000", "terse_tokens": "400"}
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "| `tool-a` | array-of-records | 0 | n/a | n/a | n/a | n/a |" in savings
    assert "1 payload(s)" in savings


def test_one_usable_count_beside_one_unusable_one_counts_as_neither():
    # The mutation that survived all 26 tests of the previous round: `and` -> `or` at the
    # token-count check. Every fixture set BOTH counts or NEITHER, so the operator was
    # unpinned — and under `or` this row reaches `counted` as `(1000, None)`, then `sum()`
    # raises TypeError. A half-measured payload is an unmeasured payload.
    for present, absent in (("raw_tokens", "terse_tokens"), ("terse_tokens", "raw_tokens")):
        rows = _tagged([_row("q1", 1, 1, sha="s")], "tool-a", "array-of-records")
        del rows[0][absent]
        rows[0][present] = 1000
        _, savings = _sections(build_codec_verdict_report({"m1": rows}))
        assert "| `tool-a` | array-of-records | 0 | n/a | n/a | n/a | n/a |" in savings, absent
        assert "1000" not in savings, present
        assert "1 payload(s)" in savings


def test_a_negative_token_count_cannot_render_a_saving():
    # A token count cannot be negative, and `_pct` guards a zero base but not a negative
    # one. Executed on the unguarded code: raw=-1, terse=1 renders -2 saved at +200.0% —
    # a saving reported for a payload that expanded. The guard belongs at the gate, not in
    # `_pct`, which six other renderers share.
    for raw_t, terse_t in ((-1, 1), (-100, 50), (1000, -400)):
        rows = _tagged([_row("q1", 1, 1, sha="s")], "tool-a", "array-of-records")
        rows[0] |= {"raw_tokens": raw_t, "terse_tokens": terse_t}
        _, savings = _sections(build_codec_verdict_report({"m1": rows}))
        assert "| `tool-a` | array-of-records | 0 | n/a | n/a | n/a | n/a |" in savings
        assert "%" not in savings.split("|---|---|---|---|---|---|---|")[1].split("\n\n")[0]


def test_a_boolean_token_count_does_not_sum_as_one_token():
    # `isinstance(True, int)` is True in Python, so a JSON `true` would otherwise be
    # counted as a 1-token payload and print a saving.
    rows = _tagged([_row("q1", 1, 1, sha="s")], "tool-a", "array-of-records")
    rows[0] |= {"raw_tokens": True, "terse_tokens": True}
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "| `tool-a` | array-of-records | 0 | n/a | n/a | n/a | n/a |" in savings


# --------------------------------------------------------------------------- #
# Signs, zeroes, ordering, and cross-group accounting
# --------------------------------------------------------------------------- #
def test_a_payload_that_terse_expands_reports_a_negative_saving():
    # The sign path was never exercised: every fixture compressed. A codec that grew a
    # payload must print that, not hide behind an unsigned number.
    rows = _tagged([_row("q1", 1, 1, sha="s")], "tool-a", "compact-json")
    rows[0] |= {"raw_tokens": 400, "terse_tokens": 1000}
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "| `tool-a` | compact-json | 1 | 400 | 1000 | -600 | -150.0% |" in savings


def test_a_zero_token_payload_prints_na_percent_rather_than_dividing_by_zero():
    rows = _tagged([_row("q1", 1, 1, sha="s")], "tool-a", "other")
    rows[0] |= {"raw_tokens": 0, "terse_tokens": 0}
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "| `tool-a` | other | 1 | 0 | 0 | +0 | n/a |" in savings


def test_both_tables_list_their_groups_in_the_same_order():
    # The report asks a reader to line row X of one table up against row X of the other.
    # Ordering the two loops independently survives every presence-only assertion.
    rows: list[dict] = []
    for tool in ("z-tool", "a-tool", "m-tool"):
        rows += _tagged([_row("q1", 1, 1, sha=f"sha-{tool}")], tool, "array-of-records")
    report = build_codec_verdict_report({"m1": rows})
    verdict, savings = _sections(report)

    def order(section: str) -> list[str]:
        return [ln.split("|")[1].strip() for ln in section.splitlines()
                if ln.startswith("| `")]

    assert order(verdict) == order(savings) == ["`a-tool`", "`m-tool`", "`z-tool`"]


def test_uncounted_payloads_accumulate_across_groups():
    # `uncounted_total += len(uncounted)` was only ever exercised with a single group, so
    # `+=` vs `=` was unpinned — a report with uncounted payloads in three groups would
    # have disclosed only the last one.
    rows: list[dict] = []
    for tool in ("t1", "t2", "t3"):
        rows += _tagged([_row("q1", 1, 1, sha=f"sha-{tool}", tokens=False)],
                        tool, "array-of-records")
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "3 payload(s)" in savings


def test_merged_runs_that_disagree_keep_the_first_reading_of_a_payload():
    # `setdefault` is first-wins, and which arm wins is only visible for a merged result
    # set whose runs measured the same sha differently. Pinned because it is a decision,
    # not an accident: the earlier run's number is the one already cited in whatever report
    # it produced, so re-reading it keeps the two agreeing.
    first = _tagged([_row("q1", 1, 1, sha="shared")], "tool-a", "array-of-records")
    second = _tagged([_row("q1", 1, 1, sha="shared")], "tool-a", "array-of-records")
    second[0] |= {"raw_tokens": 77, "terse_tokens": 11}
    _, savings = _sections(build_codec_verdict_report({"run-a": first, "run-b": second}))
    assert f"| 1 | {_RAW_TOK} | {_TERSE_TOK} | {_SAVED} | {_SAVED_PCT} |" in savings
    assert "77" not in savings


def test_the_note_says_the_count_is_per_group_not_per_payload():
    # A payload counted in one group can still be uncounted in another — reachable via the
    # stored-`shape` drift of #355. The note must not claim a payload-level total it does
    # not have.
    rows = (_tagged([_row("q1", 1, 1, sha="shared")], "tool-a", "array-of-records")
            + _tagged([_row("q1", 1, 1, sha="shared", tokens=False)],
                      "tool-a", "compact-json"))
    _, savings = _sections(build_codec_verdict_report({"m1": rows}))
    assert "1 payload(s)" in savings
    assert "counted once per `(tool, shape)` group" in savings


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


def test_run_codec_fluency_omits_sha_rather_than_defaulting_it_to_a_placeholder():
    # The defect the SECOND review found: the renderer stopped defaulting a missing sha to
    # "?", but this emitter did not — and "?" is a non-empty str, so it walked past the
    # renderer's guard and collapsed every sha-less payload in a group into the first one.
    # `capture.load_corpus` does not require `sha`, so a foreign corpus reaches here.
    import json

    from terse import codeceval
    from terse.dropeval import ToolCall, Turn

    def answerer(_messages, **_kw):
        return Turn(content=None,
                    tool_calls=[ToolCall(name=codeceval.RECORD_VALUE_TOOL,
                                         arguments={"value": {"k": [1, 2, 3]}})])

    def env_for(blob, **extra):
        obj = [{"id": 1, "blob": blob}, {"id": 2, "blob": {"k": [9]}}]
        return {"tool": "t", "shape": "array-of-records", "raw": json.dumps(obj), **extra}

    rows = codeceval.run_codec_fluency([env_for({"k": [1, 2, 3]})], {"m": answerer})["m"]
    assert rows, "fixture produced no deref questions — it cannot fail"
    assert all("sha" not in r for r in rows), "a placeholder sha was emitted"

    # And the end-to-end consequence: two distinct sha-less payloads stay two, disclosed as
    # rows, rather than collapsing into one payload's token counts.
    envs = [env_for({"k": [1, 2, 3]}), env_for({"k": [7, 7, 7]})]
    allrows = codeceval.run_codec_fluency(envs, {"m": answerer})["m"]
    _, savings = _sections(build_codec_verdict_report({"m": allrows}))
    assert "| `t` | array-of-records | 0 | n/a | n/a | n/a | n/a |" in savings
    assert f"{len(allrows)} row(s) carry no `sha`" in savings


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
# Mutation catalogue — every entry was applied to the source, this file re-run, and the
# named test confirmed to redden. 26 mutations, zero SURVIVED. Entries 14-22 came from an
# adversarial review of the first cut, which found FOUR of them surviving all 15 tests it
# then had; 23-26 came from a second review OF THAT FIX, which found three more plus the
# defect at entry 25. That is the record this file should be read against: the mutations
# you think of yourself are the ones your fixtures were already shaped around.
#
# The renderer (report.py `_codec_savings_section`):
#   1. never call `_codec_savings_section` -> reddens 10 tests.
#   2. `continue` past any group with a demonstrated excess terse miss (suppress savings on
#      UNSAFE) -> `test_an_unsafe_group_still_renders_its_savings_number`,
#      `test_every_verdict_grade_gets_a_savings_row`. The failure #303 forbids in the
#      direction people do not expect.
#   3. append the payload's token counts to the verdict table's `Why` cell -> reddens
#      `test_no_rendered_line_carries_both_a_verdict_and_a_savings_figure` ALONE.
#   4. `counted[sha + str(len(counted))]` (sum rows instead of de-duplicating by sha) ->
#      `test_a_payloads_tokens_are_counted_once_across_its_questions_and_models`. At the
#      fixture's numbers this prints 6000 raw tokens for a 1000-token payload.
#   5. `counted.setdefault("_", ...)` (collapse the group to one payload) ->
#      `test_two_distinct_payloads_in_a_group_are_both_counted`.
#   6. treat a missing/None count as 0 -> the two n/a tests.
#  12. emit the uncounted note unconditionally ->
#      `test_the_uncounted_note_is_absent_when_every_payload_was_measured`.
#  13. drop `uncounted -= set(counted)` ->
#      `test_a_payload_counted_by_a_later_model_is_not_also_reported_as_uncounted` alone.
#      SURVIVED the other fourteen tests of the first cut: the ordering it needs (a sha's
#      uncounted row seen before its counted one) only arises from merged result files.
#
# Ordering and structure (report.py `build_codec_verdict_report`):
#  10. render the savings section BEFORE the verdict table ->
#      `test_the_verdict_table_comes_first` plus the `_sections`-splitting tests.
#  11. replace the independence prose with a bare "Token savings for each group." ->
#      `test_the_savings_table_declares_its_own_independence_in_prose`.
#  14. `## Savings...` -> `### Savings...`, nesting the economics UNDER the verdict —
#      as close to "folded into it" as markdown gets. SURVIVED the first cut entirely,
#      because every check was a substring match and "### Savings" CONTAINS "## Savings".
#      -> `test_the_savings_heading_is_a_sibling_not_a_subsection`.
#  15. invert the prose while keeping its words ("Savings are never reported beside an
#      UNSAFE verdict... an UNSAFE group still prints nothing") -> reddens the prose test
#      only since it pins the sentences VERBATIM. SURVIVED the first cut, whose substring
#      probes for "never"/"beside"/"unsafe group still prints" all passed on the negation.
#  17. iterate the savings groups in the reverse order of the verdict table's ->
#      `test_both_tables_list_their_groups_in_the_same_order`. SURVIVED the first cut: every
#      assertion tested presence, in a report whose premise is reading row X against row X.
#
# Attribution and accounting (added after review question Q1):
#  16. `uncounted_total = len(uncounted)` instead of `+=` ->
#      `test_uncounted_payloads_accumulate_across_groups`. SURVIVED the first cut, which
#      never had uncounted payloads in more than one group.
#  18. `counted[sha] = ...` (last-wins) ->
#      `test_merged_runs_that_disagree_keep_the_first_reading_of_a_payload`. SURVIVED the
#      first cut, whose every fixture gave identical counts per sha.
#  19. `_is_token_count` -> `v is not None` -> the string- and bool-count tests. A stored
#      "1000" would reach `sum()` and `{:+d}` and raise.
#  20. `sha = str(r.get("sha", "?"))` — the first cut's actual line. Every sha-less payload
#      in a group collapses onto one key AND is excluded from the disclosure note: wrong
#      count, wrong sums, wrong saved, wrong percentage, in silence. ->
#      `test_rows_without_a_sha_are_excluded_and_disclosed_not_collapsed_into_one`.
#  21. exclude sha-less rows but never disclose them -> same test. Excluding without saying
#      so presents a subset as a total, which is the failure the whole section guards.
#  22. `isinstance(v, int)` without the bool guard -> `test_a_boolean_token_count_does_not_
#      sum_as_one_token`. `isinstance(True, int)` is True, so a JSON `true` renders as a
#      1-token payload with a saving.
#
# The emitter (codeceval.py `_payload_tokens` / `run_codec_fluency`):
#   7. return zeros when the tokenizer is unavailable ->
#      `test_payload_tokens_emits_nothing_when_the_tokenizer_is_unavailable`.
#   8. swap the two counts -> `test_payload_tokens_measures_the_two_forms_the_model_was_
#      actually_fed`, and only because it asserts `terse < raw` on a genuinely
#      record-shaped fixture. A fixture that did not compress would hide the swap.
#   9. drop `**toks` from the emitted row ->
#      `test_run_codec_fluency_stamps_the_counts_on_every_row_of_a_payload`.
#
# Added after the SECOND review, which found a defect inside the first round's fix — the
# outcome this repo keeps producing, and the reason a fix round gets its own review:
#  23. `_is_token_count(raw_t) and ...` -> `or`. SURVIVED all 26 tests of the previous
#      round: every fixture set BOTH counts or NEITHER, so the operator was never pinned.
#      Under `or` a half-measured row reaches `counted` as `(1000, None)` and `sum()`
#      raises. -> `test_one_usable_count_beside_one_unusable_one_counts_as_neither`.
#  24. drop `v >= 0` from `_is_token_count`. `_pct` guards a zero base, not a negative one:
#      executed, `raw=-1, terse=1` renders `-2` saved at `+200.0%`, a saving for a payload
#      that expanded. -> `test_a_negative_token_count_cannot_render_a_saving`.
#  25. restore `"sha": env.get("sha", "?")` in `run_codec_fluency`. THE defect of the second
#      review: the renderer's guard was hardened while the emitter one call upstream still
#      wrote the placeholder, and `"?"` is a non-empty `str`, so it walked straight past.
#      On the only production path this made the whole sha fix inert. ->
#      `test_run_codec_fluency_omits_sha_rather_than_defaulting_it_to_a_placeholder`.
#  26. omit `sha` from the emitted row even when the envelope has one -> the stamping test.
#      The opposite direction from 25, and without it 25's fix could be "never emit a sha".
#
# Known EQUIVALENT mutant, recorded rather than fixed: none. The first cut had one — the
# `elif sha not in counted` guard, redundant given `uncounted -= set(counted)` two lines
# below (executed on every input, but with no effect the subtraction did not already have)
# — and it was deleted rather than catalogued, since a redundant guard that looks
# load-bearing is worse than either a test or an honest note.
