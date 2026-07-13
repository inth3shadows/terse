"""Tests for the format-fluency eval — the harness that answers the proxy's open
question. The pure core (question generation, scoring, pack scoring) is exercised
offline with no network or key; live model backends are thin and not unit-tested.
"""

from __future__ import annotations

import json

import pytest

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


# A runecho.structure-shaped payload (#71): `files` is a dict-map of file records, each
# holding a NON-UNIFORM `symbols` list — imports carry only name/kind, functions also carry
# line/hash. terse's strict identical-keyset extractor skips this, so gen_questions must
# fall through to _nested_questions, scoping to the first file and its intersection columns.
STRUCTURE = {
    "detail": "symbols",
    "file_count": 2,
    "files": {
        "a/first.py": {
            "hash": "h0",
            "symbols": [
                {"name": "alpha", "kind": "function", "line": 10, "hash": "x1"},
                {"name": "beta", "kind": "function", "line": 20, "hash": "x2"},
                {"name": "os", "kind": "import"},  # non-uniform: no line/hash
            ],
        },
        "b/second.py": {
            "hash": "h1",
            "symbols": [
                {"name": "gamma", "kind": "class", "line": 5, "hash": "y1"},
                {"name": "delta", "kind": "function", "line": 8, "hash": "y2"},
            ],
        },
    },
    "repo": "demo",
}


def test_structure_uses_group_scoped_nested_questions():
    # Even though b/second.py's symbols ARE uniform (so extract_records finds a list),
    # a dict-map payload must use GROUP-SCOPED questions — an unscoped count would be
    # ambiguous across files. Group-scoping wins over the uniform extractor (#71).
    qs = {q.qtype: q for q in fluency.gen_questions(STRUCTURE)}
    assert set(qs) == {"count", "enumerate", "lookup"}
    # scoped to the first file (map order), its 3 non-uniform symbols
    assert qs["count"].expected == 3
    assert qs["enumerate"].expected == ["alpha", "beta", "os"]  # 'name' = most-distinct id col
    assert qs["lookup"].expected == "function"                  # alpha's kind
    # no aggregate: 'line' is absent from the import symbol, so it isn't an intersection col
    assert "aggregate" not in qs
    for q in qs.values():
        assert 'files["a/first.py"]' in q.prompt


def test_nested_group_uses_intersection_columns_only():
    grp = fluency._nested_record_group(STRUCTURE)
    assert grp is not None
    label, records, cols = grp
    assert label == 'files["a/first.py"]'
    assert cols == ["kind", "name"]  # sorted intersection; line/hash excluded (non-uniform)
    assert len(records) == 3


def test_nested_questions_are_self_consistent():
    for q in fluency.gen_questions(STRUCTURE):
        reply = q.expected if isinstance(q.expected, str) else json.dumps(q.expected)
        assert fluency.score(q.qtype, q.expected, reply)


def test_nested_lookup_skipped_when_no_column_uniquely_addresses_a_record():
    # Both `name` and `kind` repeat within the file, so no column uniquely identifies a
    # record — a lookup prompt would be ambiguous and a truthful answer about a different
    # matching record would score as a false-negative regression. lookup must be OMITTED;
    # count + enumerate (which tolerate duplicates) still fire.
    obj = {"files": {"x.py": {"symbols": [
        {"name": "X", "kind": "function"},
        {"name": "Y", "kind": "function"},
        {"name": "X", "kind": "class"},
        {"name": "Y", "kind": "class"}]}}}
    qs = {q.qtype: q for q in fluency.gen_questions(obj)}
    assert "lookup" not in qs
    assert qs["count"].expected == 4
    assert "enumerate" in qs
    # the enumerate ground truth stays exactly checkable even with repeated values
    assert fluency.score("enumerate", qs["enumerate"].expected,
                         json.dumps(qs["enumerate"].expected))


def test_nested_aggregate_appears_when_a_numeric_col_is_shared():
    # every symbol carries `line` (only `hash` varies) -> line is an intersection col -> aggregate
    obj = {"files": {"f": {"symbols": [
        {"name": "a", "kind": "fn", "line": 3, "hash": "h"},
        {"name": "b", "kind": "fn", "line": 9},          # no hash -> still non-uniform
        {"name": "c", "kind": "var", "line": 1, "hash": "h2"}]}}}
    from terse.capture import extract_records
    assert extract_records(obj) is None
    qs = {q.qtype: q for q in fluency.gen_questions(obj)}
    assert qs["aggregate"].expected == 9


def test_run_diff_payload_now_exercises_structure_pairs():
    # the #71 payoff: a structure diff yields the same questions in both forms, so
    # `terse fluency --diff` can finally measure structure comprehension.
    curr = json.loads(json.dumps(STRUCTURE))
    curr["files"]["a/first.py"]["symbols"].append(
        {"name": "epsilon", "kind": "function", "line": 40, "hash": "x9"})
    rows = fluency.run_diff_payload(STRUCTURE, curr, lambda s, u: "",
                                    tool="runecho.structure", trials=1)
    assert rows  # non-empty: structure now generates questions -> diff is testable
    assert {r["qid"] for r in rows} >= {"count", "enumerate"}
    assert all("terse_ok" in r and "diff_ok" in r for r in rows)


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


def test_diff_gap_rows_matches_build_diff_report_verdict():
    # diff_gap_rows feeds the bar-chart renderers (html/terminal) — its per-model
    # accuracy/gap must agree with what the markdown verdict already gates on.
    from terse.report import diff_gap_rows

    rows = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "table",
             "trials": 1, "terse_ok": 1, "diff_ok": 1 if i < 8 else 0} for i in range(10)]
    gap_rows = diff_gap_rows({"m": rows})
    facc, _, cacc, _ = gap_rows["m"]
    assert facc == 0.8 and cacc == 1.0


def test_diff_gap_rows_skips_empty_models():
    from terse.report import diff_gap_rows
    assert diff_gap_rows({"empty": []}) == {}


def test_fluency_gap_rows_best_of_terse_or_primer_vs_raw():
    from terse.report import fluency_gap_rows
    # raw 100%, terse 95%, primer 100% -> best form is primer (100%), gap 0
    gap_rows, broken = fluency_gap_rows({"m": _rows(20, 19, 20)})
    facc, _, cacc, _ = gap_rows["m"]
    assert facc == 1.0 and cacc == 1.0
    assert broken == []


def test_fluency_gap_rows_excludes_broken_raw_control():
    from terse.report import fluency_gap_rows
    gap_rows, broken = fluency_gap_rows({"broken": _rows(0, 0, 0), "good": _rows(20, 20, 20)})
    assert "broken" not in gap_rows and broken == ["broken"]
    assert "good" in gap_rows


def _gt(q: dict) -> str:
    """Render a question's expected answer the way a perfect model would."""
    import json
    if q["qtype"] == "enumerate":
        return json.dumps(q["expected"])
    return str(q["expected"])


def fluency_raw() -> str:
    import json
    return json.dumps(PAYLOAD)


# --------------------------------------------------------------------------- #
# Text-diff fluency — the text-payload analogue of the diff tests above.
# --------------------------------------------------------------------------- #
TEXT_PREV = "\n".join(f"line {i}: some repeated filler content for chunking" for i in range(20))
TEXT_CURR = TEXT_PREV + "\na brand new appended line at the very end"


def test_gen_text_diff_questions_returns_deterministic_questions():
    qs = {q.qid: q for q in fluency.gen_text_diff_questions(TEXT_PREV, TEXT_CURR, tool="demo")}
    assert set(qs) == {"line-count", "last-line", "mid-line"}
    assert qs["line-count"].expected == len(TEXT_CURR.splitlines())
    assert qs["last-line"].expected == "a brand new appended line at the very end"
    # mid-line stresses a REFERENCED (unchanged) chunk, not the edited tail
    mid = len(TEXT_CURR.splitlines()) // 2
    assert qs["mid-line"].expected == TEXT_CURR.splitlines()[mid]


def test_gen_text_diff_questions_empty_when_no_diff_applies():
    # prev="" -> text_diff_encode's `if not prev_chunks: return None` -> no diff applies
    assert fluency.gen_text_diff_questions("", "some new text") == []


def test_gen_text_diff_questions_omits_last_line_when_blank():
    # A blank final line would give expected="", indistinguishable from _safe_ask's
    # empty-string return on a total answerer failure — so it must not be asked.
    curr = TEXT_PREV + "\n\n"
    qs = {q.qid: q for q in fluency.gen_text_diff_questions(TEXT_PREV, curr, tool="demo")}
    assert "last-line" not in qs
    assert "line-count" in qs


def test_run_text_diff_payload_structure_and_forms():
    def oracle(system, user):
        return "21"  # the new line count; other questions may score wrong here
    rows = fluency.run_text_diff_payload(TEXT_PREV, TEXT_CURR, oracle, tool="demo", trials=2)
    assert rows, "a text pair with a representable diff yields rows"
    assert all("terse_ok" in r and "diff_ok" in r and r["trials"] == 2 for r in rows)
    count_row = next(r for r in rows if r["qid"] == "line-count")
    assert count_row["terse_ok"] == 2 and count_row["diff_ok"] == 2


def test_run_text_diff_payload_empty_when_no_diff_applies():
    assert fluency.run_text_diff_payload("", "some new text", lambda s, u: "x") == []


def test_run_text_diff_payload_computes_wire_exactly_once(monkeypatch):
    calls = []
    original = fluency.text_diff.text_diff_wire

    def counting_wire(prev, curr, tool=""):
        calls.append(1)
        return original(prev, curr, tool)

    monkeypatch.setattr(fluency.text_diff, "text_diff_wire", counting_wire)
    fluency.run_text_diff_payload(TEXT_PREV, TEXT_CURR, lambda s, u: "x", tool="demo", trials=1)
    assert len(calls) == 1


def test_run_text_diff_fluency_only_pairs_non_json_payloads():
    import json
    envs = [
        {"tool": "json-tool", "sha": "aaa", "raw": json.dumps(DIFF_PREV)},
        {"tool": "json-tool", "sha": "bbb", "raw": json.dumps(DIFF_CURR)},
        {"tool": "text-tool", "sha": "aaa", "raw": TEXT_PREV},
        {"tool": "text-tool", "sha": "bbb", "raw": TEXT_CURR},
    ]
    results = fluency.run_text_diff_fluency(envs, {"m": lambda s, u: "21"}, trials=1)
    assert results["m"], "the one non-JSON pair produces rows"
    assert all(r["tool"] == "text-tool" for r in results["m"])


def test_run_text_diff_fluency_excludes_json_prev_text_curr_pair():
    # Both sides must be non-JSON — a JSON prev paired with a text curr is not a
    # text-to-text transition the proxy would ever emit a text-diff for.
    import json
    envs = [
        {"tool": "mixed-tool", "sha": "aaa", "raw": json.dumps(DIFF_PREV)},
        {"tool": "mixed-tool", "sha": "bbb", "raw": TEXT_CURR},
    ]
    results = fluency.run_text_diff_fluency(envs, {"m": lambda s, u: "21"}, trials=1)
    assert results["m"] == []


def test_build_text_diff_report_verdict_and_empty():
    from terse.report import build_text_diff_report
    assert "No model answers" in build_text_diff_report({})
    rows = [{"tool": "t", "sha": "s", "qid": f"q{i}", "qtype": "count", "transform": "text-diff",
             "trials": 1, "terse_ok": 1, "diff_ok": 1} for i in range(10)]
    report = build_text_diff_report({"m": rows})
    assert "raw text" in report
    verdict = report.split("## Verdict", 1)[1]
    assert "PASS" in verdict and "FAIL" not in verdict


def test_build_diff_report_unchanged_by_the_refactor():
    # The _build_diff_style_report extraction must not change build_diff_report's own
    # output for its existing callers — re-run the two tests that already pin its
    # behavior to prove the refactor is a no-op for JSON diff-eval.
    test_build_diff_report_verdict_and_empty()
    test_diff_gap_rows_matches_build_diff_report_verdict()


def test_build_diff_report_empty_hint_preserves_two_line_wrap():
    # Pin build_diff_report's original two-physical-line empty-corpus hint — the
    # _build_diff_style_report extraction must not collapse it into one long line.
    from terse.report import build_diff_report
    report = build_diff_report({})
    assert "Capture a tool\n2+ times (an agent loop)" in report


# --- openai_answerer TLS guard: never send an API key over cleartext http to a
#     remote host (a local loopback gateway over http is fine — never leaves the box) ---

def test_openai_answerer_refuses_cleartext_key_to_remote_host():
    with pytest.raises(ValueError, match="cleartext http"):
        fluency.openai_answerer("http://api.example.com/v1", "sk-secret", "gpt-x")


def test_openai_answerer_allows_http_to_loopback_with_key():
    # a local LiteLLM/CCR gateway over loopback http carries no wire-exposure risk
    assert callable(fluency.openai_answerer("http://127.0.0.1:3456/v1", "sk-secret", "gpt-x"))
    assert callable(fluency.openai_answerer("http://localhost:1234/v1", "sk-secret", "gpt-x"))


def test_openai_answerer_allows_https_with_key():
    assert callable(fluency.openai_answerer("https://api.example.com/v1", "sk-secret", "gpt-x"))


def test_openai_answerer_allows_http_without_key():
    # no key set -> nothing secret to leak, so plain http is permitted
    assert callable(fluency.openai_answerer("http://api.example.com/v1", "", "gpt-x"))


# --- diff-chain soak (#8/#20 follow-up): depth-k windows + chained-diff form ---

def _soak_envs(tool="gh.items", n=8, start_id=0):
    """n consecutive envelopes for one tool, each a small mutation of the last —
    every hop admits a lossless row diff (update-in-place, append at the end)."""
    envs = []
    rows = [{"id": i, "status": "active", "score": i % 7} for i in range(start_id, start_id + 20)]
    for k in range(n):
        if k:
            rows[k % len(rows)]["status"] = f"state-{k}"
            rows.append({"id": start_id + 20 + k, "status": "new", "score": k})
        envs.append({"tool": tool, "sha": f"s{k:02d}",
                     "raw": json.dumps({"result": [dict(r) for r in rows]})})
    return envs


def test_build_chain_windows_yields_every_depth_and_valid_hops():
    from terse.transforms import diff_wire

    windows = fluency.build_chain_windows(_soak_envs(n=8), max_depth=3, per_depth_cap=4)
    depths = {d for _, _, d, _ in windows}
    assert depths == {1, 2, 3}
    for tool, sha, depth, objs in windows:
        assert len(objs) == depth + 1
        for prev, curr in zip(objs, objs[1:]):
            assert diff_wire(prev, curr, tool) is not None   # every hop truly chains
        assert fluency.gen_questions(objs[-1])               # final state is askable


def test_build_chain_windows_never_spans_a_diff_break():
    # an unrelated payload mid-run splits it: no window may bridge the break, because
    # in production that hop would have re-anchored as a full, ending the chain.
    envs = _soak_envs(n=4)
    envs.insert(2, {"tool": "gh.items", "sha": "sXX",
                    "raw": json.dumps({"totally": "different", "shape": [1, 2, 3]})})
    windows = fluency.build_chain_windows(envs, max_depth=3, per_depth_cap=8)
    from terse.transforms import diff_wire
    for tool, _sha, _depth, objs in windows:
        for prev, curr in zip(objs, objs[1:]):
            assert diff_wire(prev, curr, tool) is not None


def test_run_chain_payload_context_carries_one_full_plus_depth_wires():
    envs = _soak_envs(n=4)
    objs = [json.loads(e["raw"]) for e in envs]
    seen: list[str] = []

    def spy(system, user):
        seen.append(user)
        return ""

    rows = fluency.run_chain_payload(objs, spy, tool="gh.items", trials=1)
    assert rows and all(r["depth"] == 3 for r in rows)
    chain_prompts = [u for u in seen if "UPDATE (diff against the result above" in u]
    assert chain_prompts                                     # the chain form was asked
    assert all(u.count("UPDATE (diff against the result above") == 3
               for u in chain_prompts)                       # exactly depth wires
    assert all(u.count("PREVIOUS RESULT:") == 1 for u in chain_prompts)  # one anchor


def test_run_chain_payload_empty_when_a_hop_stops_diffing():
    objs = [json.loads(e["raw"]) for e in _soak_envs(n=3)]
    objs.append({"totally": "different", "shape": [1, 2, 3]})  # last hop can't diff
    assert fluency.run_chain_payload(objs, lambda s, u: "", tool="gh.items") == []


def test_run_diff_soak_rows_carry_depth_per_model():
    results = fluency.run_diff_soak(_soak_envs(n=8), {"m1": lambda s, u: ""},
                                    trials=1, max_depth=3, per_depth_cap=2)
    assert set(results) == {"m1"}
    assert {r["depth"] for r in results["m1"]} == {1, 2, 3}
    assert all({"tool", "sha", "qid", "terse_ok", "diff_ok"} <= set(r)
               for r in results["m1"])


def test_build_diff_soak_report_by_depth_table_and_verdicts():
    from terse.report import build_diff_soak_report

    rows = [{"tool": "gh.items", "sha": "s", "qid": "count", "qtype": "count",
             "transform": "tabularize", "trials": 1, "depth": d,
             "terse_ok": 1, "diff_ok": 1} for d in (1, 2, 3) for _ in range(4)]
    report = build_diff_soak_report({"m1": rows})
    assert "## Accuracy by chain depth" in report
    assert "At the deepest tested depth (3)" in report
    assert "**PASS**" in report and "No depth-correlated comprehension drift" in report

    # a depth-correlated slide beyond tolerance must FAIL the deepest-depth gate
    bad = [dict(r, diff_ok=0 if r["depth"] == 3 else 1) for r in rows]
    report = build_diff_soak_report({"m1": bad})
    assert "**FAIL**" in report

    # empty results explain how to get soakable data
    assert "diffable RUNS" in build_diff_soak_report({})


def test_flat_record_questions_cover_single_record_payloads():
    # a single flat record (search hit, status receipt, one KB row): keys-count plus
    # deterministic numeric/string lookups — the diff surface the soak was blind to.
    obj = {"type": "note", "id": 42, "title": "diff soak", "snippet": "x" * 200,
           "score": 0.87, "nested": {"skip": True}}
    qs = fluency.gen_questions(obj)
    by_qid = {q.qid: q for q in qs}
    assert by_qid["keys-count"].expected == 6            # counts ALL keys
    assert by_qid["field-0"].expected == 42              # first numeric by sorted key
    assert by_qid["field-1"].expected == "diff soak"     # long/nested values not asked
    assert all(q.transform == "flat-record" for q in qs)


def test_flat_record_questions_stay_silent_when_underqualified():
    assert fluency.gen_questions({"just": "an object", "two": 2}) == []   # <3 lookable
    assert fluency.gen_questions({"__terse_dict__": 1, "a": 1, "b": 2, "c": 3}) == []
    assert fluency.gen_questions({"a": "", "b": "y" * 999, "c": None, "d": True}) == []
