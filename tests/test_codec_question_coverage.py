"""#403 Blocker 2: which corpus payloads the codec verdict can ask a question about.

Measured on the live corpus: deref-only questions covered tools carrying 11.9% of the
codec's 30-day savings. Two defects kept the rest dark — a KB metadata trailer record that
emptied `_intersection_cols`, and `enumerate` being excluded although its answer is a
container carried verbatim into a tool argument.
"""

from __future__ import annotations

import json

from terse import capture, codeceval, fluency
from terse.dropeval import ToolCall, Turn
from terse.report import build_codec_verdict_report
from terse.transforms import minify

# The live `kb.read.list_principles` shape: uniform records, then a trailer that shares no
# key with them. Wide enough records that the codec still folds the list (it refuses a
# table under half full), which every test below asserts rather than assumes.
RECORDS = [{"id": 35, "category": "ops", "principle": "p1", "evidence": "e1", "status": "a"},
           {"id": 55, "category": "ops", "principle": "p2", "evidence": "e2", "status": "a"},
           {"id": 12, "category": "dev", "principle": "p3", "evidence": "e3", "status": "b"}]
TRAILER = {"_categories": ["ops", "dev"], "_note": "truncated", "_truncated": True}


def _by_type(qs):
    return {q.qtype: q for q in qs}


def test_a_metadata_trailer_no_longer_hides_every_question():
    assert capture.extract_records(RECORDS + [TRAILER]) is not None   # the codec folds it
    qs = _by_type(fluency.gen_questions(RECORDS + [TRAILER]))
    assert qs["count"].expected == 3
    assert qs["enumerate"].expected == [35, 55, 12]
    # Questions over EVERY record say the trailer is not one; both forms show it.
    note = "The final entry, holding only '_categories', '_note', '_truncated', is metadata"
    assert all(note in qs[t].prompt for t in ("count", "enumerate", "aggregate"))
    assert note not in qs["lookup"].prompt   # addressed by id: no scope needed


def test_the_trailer_payload_is_codec_eligible():
    qs = codeceval.gen_codec_questions(RECORDS + [TRAILER])
    assert [(q.qtype, q.expected) for q in qs] == [("enumerate", [35, 55, 12])]


def test_a_final_row_sharing_a_key_is_still_a_record():
    last = {"id": 99, "_note": "x"}
    qs = _by_type(fluency.gen_questions(RECORDS + [last]))
    assert qs["count"].expected == 4
    assert "metadata" not in qs["count"].prompt


def test_one_record_and_a_trailer_is_not_split():
    # Splitting would leave a single "record" — not a list worth scoping. Asserted on the
    # helper: through `gen_questions` it is unreachable, because two key-disjoint rows are
    # exactly half full and the codec never folds them (so `extract_records` returns None).
    from terse.fluency.questions import _split_metadata_trailer

    pair = [RECORDS[0], {"_note": "x"}]
    assert capture.extract_records(pair) is None
    assert _split_metadata_trailer(pair) == (pair, None)
    assert _split_metadata_trailer(RECORDS + [TRAILER]) == (RECORDS, TRAILER)


def test_a_payload_without_a_trailer_keeps_its_exact_prompts():
    qs = _by_type(fluency.gen_questions(RECORDS))
    assert qs["count"].prompt == "How many records does the dataset contain?"
    assert qs["enumerate"].prompt == "List the 'id' of every record, in order."
    assert qs["aggregate"].prompt == "What is the maximum value of 'id' across all records?"


def test_a_payload_the_codec_leaves_alone_is_skipped_not_scored():
    # Both arms would carry the same JSON, so every trial is a free match counting toward
    # the SAFE floor. `enumerate` reaches such payloads; `deref` never did.
    from terse.transforms import has_terse_marker

    untouched = {"result": [{"id": 1}, {"id": 2}]}
    encoded = [{"id": i, "meta": {"o": f"n{i}"}} for i in range(8)]
    assert codeceval.gen_codec_questions(untouched)                      # askable...
    assert not has_terse_marker(json.loads(fluency.compress(untouched)))  # ...but not encoded
    assert has_terse_marker(json.loads(fluency.compress(encoded)))

    def never_calls(messages):
        return Turn(text="", tool_calls=[])

    envs = [{"tool": "t", "sha": "untouched", "raw": json.dumps(untouched)},
            {"tool": "t", "sha": "encoded", "raw": json.dumps(encoded)}]
    lines: list[str] = []
    rows = codeceval.run_codec_fluency(envs, {"m": never_calls}, preflight=False,
                                       progress=lines.append)["m"]
    assert rows and {r["sha"] for r in rows} == {"encoded"}
    assert sum("(skipped)" in ln for ln in lines) == 1


def test_the_verdict_table_names_the_question_types_each_cell_was_scored_on():
    # TWO payloads and TWO models, so the column cannot pass by counting per-payload,
    # per-question, or across every model's rows: it counts the rows `n` sums over, which
    # are the WORST model's.
    payloads = [{"result": [{"id": i, "meta": {"a": i}} for i in range(base, base + 8)]}
                for base in (1, 9)]

    def correct(messages):
        content = messages[-1]["content"]
        q = next(q for p in payloads for q in codeceval.gen_codec_questions(p)
                 if q.prompt in content)
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c1", name=codeceval.RECORD_VALUE_TOOL,
                     arguments={"value": q.expected})])

    envs = [{"tool": "demo.get", "sha": f"sha{i}", "raw": json.dumps(p)}
            for i, p in enumerate(payloads)]
    results = codeceval.run_codec_fluency(envs, {"m1": correct, "m2": correct}, trials=2,
                                          preflight=False)
    report = build_codec_verdict_report(results)
    assert "| Tool | Shape | Questions | n | Verdict |" in report
    assert "| `demo.get` | array-of-records | deref 2, enumerate 2 | 8 | **UNRESOLVED** |" in report


def test_an_empty_final_entry_is_not_metadata():
    # `{}` shares no key with anything, so the disjointness test alone would call it a
    # trailer and describe it as "holding only " — nothing.
    payload = [{**r, "id": i} for i, r in enumerate(RECORDS * 3)] + [{}]
    assert capture.extract_records(payload) is not None
    assert fluency.gen_questions(payload) == []


def test_a_trailer_is_judged_against_every_record_not_just_the_first():
    # The head is drifted: record 0 carries 'a', the rest carry 'b'. The final entry shares
    # 'b' with records 1..9, so it is a record, and the list has no column in common.
    payload = ([{"id": 0, "a": "x"}] + [{"id": i, "b": "y"} for i in range(1, 10)]
               + [{"b": "z"}])
    assert capture.extract_records(payload) is not None
    assert fluency.gen_questions(payload) == []


def test_codec_changes_reads_a_passthrough_as_unchanged_even_with_a_reserved_marker():
    # The codec hands a payload carrying a reserved marker straight through, so a
    # marker-in-the-output test would call it encoded and score free matches.
    passthrough = {"result": [{"id": 1, "note": "__terse_absent__"}, {"id": 2, "note": "x"}]}
    assert codeceval.gen_codec_questions(passthrough)          # askable...
    assert fluency.compress(passthrough) == minify(passthrough)  # ...but untouched
    assert codeceval.codec_changes(passthrough) is False
    assert codeceval.codec_changes([{"id": i, "meta": {"o": f"n{i}"}} for i in range(8)])
