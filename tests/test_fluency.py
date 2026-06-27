"""Tests for the format-fluency eval — the harness that answers the proxy's open
question. The pure core (question generation, scoring, pack scoring) is exercised
offline with no network or key; live model backends are thin and not unit-tested.
"""

from __future__ import annotations

from terse import fluency

# A record-shaped payload that exercises both stressed transforms:
#  - `id` is the unique scalar identifier column
#  - `status` repeats enough (6x) to be dict-coded -> a lookup on it stresses `~N`
#    (a `~0` alias costs ~4 tokens quoted, so aliasing only pays with real repetition)
#  - `score` is numeric and distinct from `id` -> aggregate (max), a non-trivial check
PAYLOAD = [
    {"id": 1, "status": "active-long-status-string-value", "score": 10},
    {"id": 2, "status": "active-long-status-string-value", "score": 30},
    {"id": 3, "status": "active-long-status-string-value", "score": 20},
    {"id": 4, "status": "active-long-status-string-value", "score": 5},
    {"id": 5, "status": "active-long-status-string-value", "score": 15},
    {"id": 6, "status": "active-long-status-string-value", "score": 25},
]


def _qmap(obj):
    return {q.qtype: q for q in fluency.gen_questions(obj)}


def test_gen_questions_cover_all_types_with_correct_ground_truth():
    qs = _qmap(PAYLOAD)
    assert set(qs) == {"count", "lookup", "enumerate", "aggregate"}
    assert qs["count"].expected == 6
    assert qs["enumerate"].expected == [1, 2, 3, 4, 5, 6]
    assert qs["aggregate"].expected == 30  # max of score, not id
    # middle record (index 6//2 == 3) -> id 4, status the repeated/aliased string
    assert qs["lookup"].expected == "active-long-status-string-value"


def test_lookup_targets_a_dict_coded_field():
    # the repeated status string is folded into the legend, so the lookup must be
    # tagged as stressing alias resolution
    qs = _qmap(PAYLOAD)
    assert qs["lookup"].transform == "table+dict"


BLOB_PAYLOAD = [
    {"id": i + 1, "config": [{"region": "us-east-1", "tier": "gold", "flags": ["a", "b"]},
                             {"zone": "eu-west-1", "tier": "silver", "extra": 1}][i % 2]}
    for i in range(8)
]


def test_deref_question_targets_an_object_valued_alias():
    # whole-subtree aliasing folds the repeated config objects; the deref question must
    # exist, target that column, and be tagged as stressing alias resolution
    qs = {q.qtype: q for q in fluency.gen_questions(BLOB_PAYLOAD)}
    assert "deref" in qs
    assert isinstance(qs["deref"].expected, dict)
    assert qs["deref"].transform == "table+dict"  # the object is dict-coded


def test_aliased_helpers_survive_unhashable_subtree_legend():
    # regression: _aliased_strings/_aliased_canon must not choke on object legend values
    # (set(legend.values()) would raise TypeError once subtrees can be aliased)
    assert isinstance(fluency._aliased_strings(BLOB_PAYLOAD), set)
    assert isinstance(fluency._aliased_canon(BLOB_PAYLOAD), set)


def test_score_deref_json_value_equality():
    assert fluency.score("deref", {"a": 1, "b": 2}, '{"b": 2, "a": 1}')  # order-insensitive
    assert fluency.score("deref", {"a": 1}, 'The value is {"a": 1}.')    # prose-tolerant
    assert fluency.score("deref", [1, 2, 3], "[1, 2, 3]")
    assert not fluency.score("deref", {"a": 1}, '{"a": 2}')
    assert not fluency.score("deref", {"a": 1}, "not json")


def test_no_questions_for_non_record_payloads():
    assert fluency.gen_questions({"just": "an object"}) == []
    assert fluency.gen_questions([1, 2, 3]) == []
    assert fluency.gen_questions("a string") == []


def test_score_count_and_aggregate_tolerate_prose_and_check_value():
    assert fluency.score("count", 4, "There are 4 records.")
    assert fluency.score("count", 4, "4")
    assert not fluency.score("count", 4, "5")
    assert not fluency.score("count", 4, "")
    assert fluency.score("aggregate", 30, "The maximum is 30")
    assert not fluency.score("aggregate", 30, "20")


def test_score_lookup_strips_quotes_and_case():
    assert fluency.score("lookup", "active", '"Active"')
    assert fluency.score("lookup", "active", "active")
    assert not fluency.score("lookup", "active", "inactive")
    # numeric lookup answer
    assert fluency.score("lookup", 42, "the value is 42")


def test_score_enumerate_json_array_exact_and_lenient():
    assert fluency.score("enumerate", [1, 2, 3], "[1, 2, 3]")
    assert fluency.score("enumerate", [1, 2, 3], "Here you go: [1,2,3]")
    assert not fluency.score("enumerate", [1, 2, 3], "[1, 2]")  # under-enumeration fails
    assert not fluency.score("enumerate", [1, 2, 3], "[1, 2, 3, 4]")
    # comma fallback when the model ignores the JSON-array instruction
    assert fluency.score("enumerate", ["a", "b"], "a, b")


def test_score_empty_expected_scalar_matches_empty_reply():
    # a legitimately-empty field value must score correct when the model returns nothing
    assert fluency.score("lookup", "", "")
    assert not fluency.score("lookup", "x", "")  # empty reply, non-empty expected -> wrong


def test_score_number_matches_anywhere_not_just_first():
    # prose with a leading incidental number must not fool numeric scoring
    assert fluency.score("count", 6, "I see 2 columns and 6 records")
    assert fluency.score("aggregate", 30, "the values range up to 30")
    assert not fluency.score("count", 6, "I see 2 columns and 5 records")


def test_run_payload_structure_with_constant_answerer():
    # a model that always says "6" gets count right, the rest wrong — proves the
    # harness scores each form independently and returns one row per question
    rows = fluency.run_payload(PAYLOAD, fluency.compress(PAYLOAD), lambda s, u: "6")
    assert {r["qid"] for r in rows} == {"count", "lookup", "enumerate", "aggregate"}
    count_row = next(r for r in rows if r["qid"] == "count")
    assert count_row["raw_ok"] and count_row["terse_ok"] and count_row["primer_ok"]
    lookup_row = next(r for r in rows if r["qid"] == "lookup")
    assert not lookup_row["terse_ok"]


def test_run_payload_trials_count_partial_successes():
    # a flaky answerer: right, wrong, right -> count question scores 2/3, and every row
    # carries trials=3. Proves multi-trial records counts, not a single boolean.
    replies = iter(["6", "nope", "6"] * 10)  # enough for all questions x 3 forms
    rows = fluency.run_payload(PAYLOAD, fluency.compress(PAYLOAD), lambda s, u: next(replies), trials=3)
    assert all(r["trials"] == 3 for r in rows)
    count_row = next(r for r in rows if r["qid"] == "count")
    assert count_row["raw_ok"] == 2  # right, wrong, right
    assert 0 <= count_row["terse_ok"] <= 3


def test_score_pack_accepts_multi_trial_lists():
    pack = fluency.build_pack([{"tool": "demo", "sha": "abc123", "raw": fluency_raw()}], trials=2)
    assert pack["trials"] == 2
    sha = pack["payloads"][0]["sha"]
    # two replies per form: first correct, second wrong -> 1/2 each form
    resp = {q["qid"]: {"raw": [_gt(q), "wrong"], "terse": [_gt(q), "wrong"],
                       "primer": [_gt(q), "wrong"]}
            for q in pack["payloads"][0]["questions"]}
    rows = fluency.score_pack(pack, {"m": {sha: resp}})["m"]
    assert rows and all(r["trials"] == 2 for r in rows)
    assert all(r["raw_ok"] == 1 for r in rows)  # exactly one of two correct


def test_multi_trial_report_shows_bound():
    from terse.report import build_fluency_report
    # 10 questions, 4 trials each; terse a touch noisier than raw
    rows = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
             "trials": 4, "raw_ok": 4, "terse_ok": 3, "primer_ok": 4} for i in range(10)]
    report = build_fluency_report({"m": rows}, [])
    assert "Trials per question: **4**" in report
    assert "±" in report
    verdict = report.split("## Verdict", 1)[1]
    assert "pts)" in verdict  # the gap carries a confidence interval


def test_build_pack_then_score_pack_roundtrips_through_ground_truth():
    pack = fluency.build_pack([{"tool": "demo", "sha": "abc123", "raw": fluency_raw()}])
    assert len(pack["payloads"]) == 1
    sha = pack["payloads"][0]["sha"]
    # perfect answers for every question, all three forms
    perfect = {q["qid"]: {"raw": _gt(q), "terse": _gt(q), "primer": _gt(q)}
               for q in pack["payloads"][0]["questions"]}
    results = fluency.score_pack(pack, {"oracle": {sha: perfect}})
    rows = results["oracle"]
    assert rows and all(r["raw_ok"] and r["terse_ok"] and r["primer_ok"] for r in rows)


def _rows(raw, terse, primer, n=20, transform="table"):
    """n scored rows with the given number of correct raw/terse/primer answers."""
    return [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": transform,
             "raw_ok": i < raw, "terse_ok": i < terse, "primer_ok": i < primer}
            for i in range(n)]


def test_verdict_passes_at_exactly_tolerance_boundary():
    from terse.report import build_fluency_report
    # raw 100%, best terse-form 95% -> gap exactly -5% must be PASS (not a float FAIL)
    report = build_fluency_report({"m": _rows(20, 19, 19)}, [])
    verdict = report.split("## Verdict", 1)[1]
    assert "PASS" in verdict and "FAIL" not in verdict


def test_verdict_excludes_models_that_fail_the_raw_control():
    from terse.report import build_fluency_report
    # a model at 0% on raw is a backend error, not comprehension — excluded from the
    # gate, and the good model still drives a PASS
    results = {"broken": _rows(0, 0, 0), "good": _rows(20, 20, 20)}
    report = build_fluency_report(results, [])
    verdict = report.split("## Verdict", 1)[1]
    assert "Excluded" in verdict and "`broken`" in verdict
    assert "PASS" in verdict


DIFF_PREV = [{"id": i, "status": "active-long-status-string-value", "score": i} for i in range(8)]
DIFF_CURR = ([{"id": i, "status": "active-long-status-string-value", "score": i} for i in range(8)]
             + [{"id": 8, "status": "active-long-status-string-value", "score": 99}])


def test_run_diff_payload_structure_and_forms():
    # a perfect answerer scores both forms right; rows carry terse_ok AND diff_ok counts
    def oracle(system, user):
        import json as _j
        # answer count questions with the current count (9); others may be wrong — we only
        # assert structure here, not full correctness
        return "9"
    rows = fluency.run_diff_payload(DIFF_PREV, DIFF_CURR, oracle, tool="demo", trials=2)
    assert rows, "a record-shaped curr with a representable diff yields rows"
    assert all("terse_ok" in r and "diff_ok" in r and r["trials"] == 2 for r in rows)
    count_row = next(r for r in rows if r["qid"] == "count")
    assert count_row["terse_ok"] == 2 and count_row["diff_ok"] == 2  # 9 is the new count


def test_run_diff_payload_empty_when_no_diff_applies():
    # identical-shape but non-record curr -> no questions -> no rows
    assert fluency.run_diff_payload({"a": 1}, {"a": 2}, lambda s, u: "x") == []


def test_run_diff_fluency_pairs_same_tool_payloads():
    import json
    envs = [{"tool": "demo", "sha": "aaa", "raw": json.dumps(DIFF_PREV)},
            {"tool": "demo", "sha": "bbb", "raw": json.dumps(DIFF_CURR)}]
    results = fluency.run_diff_fluency(envs, {"m": lambda s, u: "9"}, trials=1)
    assert results["m"], "the one same-tool pair produces rows"
    assert all(r["tool"] == "demo" for r in results["m"])


def test_build_diff_report_verdict_and_empty():
    from terse.report import build_diff_report
    assert "No model answers" in build_diff_report({})
    rows = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
             "trials": 1, "terse_ok": 1, "diff_ok": 1} for i in range(10)]
    report = build_diff_report({"m": rows})
    verdict = report.split("## Verdict", 1)[1]
    assert "PASS" in verdict and "FAIL" not in verdict


def _gt(q: dict) -> str:
    """Render a question's expected answer the way a perfect model would."""
    import json
    if q["qtype"] == "enumerate":
        return json.dumps(q["expected"])
    return str(q["expected"])


def fluency_raw() -> str:
    import json
    return json.dumps(PAYLOAD)
