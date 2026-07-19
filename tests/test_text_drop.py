"""drop-to-retrieve over a TEXT payload, addressed by span (`$text.code_blocks`).

The invariant under test throughout: whatever is emitted must splice back to the ORIGINAL
payload byte for byte using only the session store. Because a text payload has no field
structure, that whole-string reconstruction IS the gate — so these tests lean on it hard,
including on the shapes designed to break the fence scanner.
"""

from __future__ import annotations

import json

from terse import dropeval, lossy
from terse.policy import Policy, Rule, _lossy_warnings, apply

CODE = "\n".join(f"line {i} of source with enough text to clear the floor" for i in range(20))
DOC = f"""## Exploration

Found 3 symbols across 2 files.

### Source Code

#### src/a.py

```python
{CODE}
```

Trailing prose.
"""


def _rule(fields, tiers=("minify", "tabularize", "dictionary")):
    return Rule(tool_glob="*", tiers=tiers, fields=fields)


def _drop_rule(**spec):
    return _rule({lossy.TEXT_SELECTOR_CODE_BLOCKS: {"lossy": "drop-to-retrieve", **spec}})


def _apply(raw, rule, tool="codegraph_explore", server=None, policy=None):
    store: dict[str, object] = {}
    pol = policy or Policy(rules=[rule])
    applied = apply(raw, tool, pol, drop_sink=store.__setitem__, server=server)
    return applied, store


# --------------------------------------------------------------------------- #
# fence scanning
# --------------------------------------------------------------------------- #
def test_fenced_spans_are_exact_substrings():
    spans = lossy.fenced_spans(DOC)
    assert len(spans) == 1
    start, end = spans[0]
    assert DOC[start:end] == f"```python\n{CODE}\n```\n"


def test_fenced_spans_handles_multiple_and_tilde_fences():
    text = "a\n```\none\n```\nb\n~~~js\ntwo\n~~~\nc\n"
    assert [text[s:e] for s, e in lossy.fenced_spans(text)] == ["```\none\n```\n", "~~~js\ntwo\n~~~\n"]


def test_fenced_spans_unterminated_fence_runs_to_end():
    text = "intro\n```py\ncut off mid-block"
    assert [text[s:e] for s, e in lossy.fenced_spans(text)] == ["```py\ncut off mid-block"]


def test_fenced_spans_inner_backticks_are_content_not_closers():
    # A shorter run inside a longer fence is content; only >= the opener closes.
    text = "````\n```\ninner\n```\n````\n"
    assert [text[s:e] for s, e in lossy.fenced_spans(text)] == [text]


# --------------------------------------------------------------------------- #
# the transform + its gate
# --------------------------------------------------------------------------- #
def test_drop_replaces_block_and_restores_byte_exact():
    applied, store = _apply(DOC, _drop_rule())
    assert applied.text != DOC
    assert "line 10 of source" not in applied.text
    assert "Found 3 symbols across 2 files." in applied.text  # prose retained
    assert len(store) == 1
    assert lossy.restore_text_drops(applied.text, store.__getitem__) == DOC


def test_marker_is_the_same_wire_form_the_json_path_emits():
    applied, store = _apply(DOC, _drop_rule())
    line = next(ln for ln in applied.text.splitlines() if transforms_marker(ln))
    marker = json.loads(line)
    handle = marker[lossy.DROP_KEY]
    assert marker["retrieve"] == lossy.RETRIEVE_TOOL
    assert marker["bytes"] == len(store[handle])
    assert set(marker) == {lossy.DROP_KEY, "bytes", "retrieve"}


def transforms_marker(line: str) -> bool:
    return line.startswith('{"' + lossy.DROP_KEY + '"')


def test_span_under_the_floor_is_left_in_place():
    text = "intro\n\n```\ntiny\n```\n"
    applied, store = _apply(text, _drop_rule())
    assert applied.text == text
    assert store == {}


def test_min_is_honoured_from_the_spec():
    text = "intro\n\n```\n" + ("x" * 50) + "\n```\n"
    assert _apply(text, _drop_rule(min=500))[0].text == text
    assert _apply(text, _drop_rule(min=10))[0].text != text


def test_gate_fails_closed_when_a_handle_cannot_be_restored():
    # A payload that ALREADY contains a marker line: restoring it would need a handle that
    # was never stored, so the gate must reject and the original text must survive intact.
    poisoned = (DOC + '\n{"' + lossy.DROP_KEY
                + '":"deadbeefdeadbeef","bytes":10,"retrieve":"terse.retrieve"}\n')
    applied, store = _apply(poisoned, _drop_rule())
    assert applied.text == poisoned
    assert store == {}
    assert any("droppable-loss gate failed" in w for w in applied.warnings)


def test_multiple_blocks_all_drop_and_restore():
    text = DOC + "\n#### src/b.py\n\n```go\n" + CODE + "\n```\n"
    applied, store = _apply(text, _drop_rule())
    assert len(store) == 2
    assert lossy.restore_text_drops(applied.text, store.__getitem__) == text


def test_identical_blocks_share_one_handle():
    text = f"```\n{CODE}\n```\n\nmiddle\n\n```\n{CODE}\n```\n"
    applied, store = _apply(text, _drop_rule())
    assert len(store) == 1  # content-addressed: same bytes, same slot
    assert lossy.restore_text_drops(applied.text, store.__getitem__) == text


def test_block_without_trailing_newline_at_eof_restores():
    text = f"intro\n\n```\n{CODE}\n```"
    applied, store = _apply(text, _drop_rule())
    assert applied.text != text
    assert lossy.restore_text_drops(applied.text, store.__getitem__) == text


# --------------------------------------------------------------------------- #
# policy wiring / fail-closed contracts
# --------------------------------------------------------------------------- #
def test_json_payload_is_untouched_by_a_text_selector():
    from terse import transforms

    obj = {"result": [{"id": 1, "body": "x" * 900}]}
    applied, store = _apply(json.dumps(obj), _drop_rule())
    assert store == {}
    assert lossy.DROP_KEY not in applied.text
    assert transforms.decompress(applied.text) == obj  # still fully lossless


def test_never_lossy_server_suppresses_text_drops():
    pol = Policy(rules=[_drop_rule()], never_lossy_servers=frozenset({"vault"}))
    applied, store = _apply(DOC, _drop_rule(), server="vault", policy=pol)
    assert applied.text == DOC
    assert store == {}


def test_no_drop_sink_keeps_it_lossless():
    applied = apply(DOC, "t", Policy(rules=[_drop_rule()]), drop_sink=None)
    assert applied.text == DOC
    assert any("needs the proxy store" in w for w in applied.warnings)


def test_critical_selector_is_never_dropped():
    rule = _rule({lossy.TEXT_SELECTOR_CODE_BLOCKS:
                  {"lossy": "drop-to-retrieve", "critical": True}})
    applied, store = _apply(DOC, rule)
    assert applied.text == DOC
    assert store == {}


def test_passthrough_tiers_suppress_and_warn():
    rule = _rule({lossy.TEXT_SELECTOR_CODE_BLOCKS: {"lossy": "drop-to-retrieve"}}, tiers=())
    applied, store = _apply(DOC, rule)
    assert applied.text == DOC
    assert store == {}
    assert any("'tiers': []" in w for w in applied.warnings)


def test_unknown_text_selector_warns_and_does_nothing():
    rule = _rule({"$text.codeblocks": {"lossy": "drop-to-retrieve"}})
    applied, store = _apply(DOC, rule)
    assert applied.text == DOC
    assert store == {}
    assert any("unknown text selector" in w for w in _lossy_warnings(rule))


def test_has_drop_sees_a_text_selector_so_retrieve_is_advertised():
    assert Policy(rules=[_drop_rule()]).has_drop()


# --------------------------------------------------------------------------- #
# the fluency gate's question generation
# --------------------------------------------------------------------------- #
def test_text_drop_questions_are_grounded_in_the_real_store():
    rule = _drop_rule()
    qs = dropeval.gen_text_drop_questions(DOC, rule, "codegraph_explore")
    assert [q.kind for q in qs] == ["recall", "precision"]
    recall, precision = qs
    _, store = _apply(DOC, rule)
    assert recall.expected_handle in store
    assert recall.expected in store[recall.expected_handle]  # the line really is in the span
    assert recall.needs_retrieve and not precision.needs_retrieve
    assert precision.expected == sum(len(v) for v in store.values())


def test_no_questions_when_nothing_was_dropped():
    assert dropeval.gen_text_drop_questions("no fences here", _drop_rule(), "t") == []
    assert dropeval.gen_text_drop_questions(DOC, _rule({}), "t") == []


# --------------------------------------------------------------------------- #
# Regressions from the post-merge review of #115
# --------------------------------------------------------------------------- #
def test_dollar_prefixed_json_key_is_still_a_json_field_path():
    """`$schema`/`$ref`/`$id` are ordinary JSON keys. Reserving the whole `$` sigil for
    text selectors silently disabled drop-to-retrieve on them — only `$text.` is ours."""
    obj = {"$schema": "x" * 900, "name": "t"}
    rule = _rule({"$schema": {"lossy": "drop-to-retrieve"}})
    applied, store = _apply(json.dumps(obj), rule)
    assert len(store) == 1
    assert lossy.DROP_KEY in applied.text
    assert store[next(iter(store))] == "x" * 900
    assert not any("unknown text selector" in w for w in _lossy_warnings(rule))


def test_known_selector_with_unsupported_mode_warns_instead_of_going_silent():
    rule = _rule({lossy.TEXT_SELECTOR_CODE_BLOCKS: {"lossy": "truncate", "max": 100}})
    applied, store = _apply(DOC, rule)
    assert applied.text == DOC and store == {}
    assert any("not span-addressable" in w for w in _lossy_warnings(rule))


def test_inline_code_prose_line_does_not_open_a_phantom_fence():
    """CommonMark 4.5: a backtick fence's info string may not contain backticks. Allowing
    it made ```` ```py``` ```` open a fence and swallow the prose that followed."""
    prose = "\n".join(f"real prose line {i} that must stay visible" for i in range(15))
    text = f"intro\n```py```\n{prose}\n```\nafter\n"
    applied, store = _apply(text, _drop_rule(min=100))
    assert applied.text == text and store == {}
    # The trailing bare ``` legitimately opens an unterminated fence (CommonMark), but it
    # starts AFTER the prose — the bug was the prose itself landing inside a span.
    assert all(start > text.index("real prose line 0") for start, _ in lossy.fenced_spans(text))
    assert "real prose line 7 that must stay visible" in applied.text


def test_tilde_fence_may_still_carry_backticks_in_its_info_string():
    text = f"~~~`weird`\n{CODE}\n~~~\n"
    assert [text[s:e] for s, e in lossy.fenced_spans(text)] == [text]


def test_error_results_are_never_evicted_to_a_handle():
    """An isError payload is what the model must READ to recover; a lossy transform must
    not put a retrieve round-trip in front of it."""
    applied, store = _apply(DOC, _drop_rule(), policy=Policy(rules=[_drop_rule()]))
    assert applied.text != DOC  # sanity: it WOULD drop without the override
    forced = apply(DOC, "codegraph_explore", Policy(rules=[_drop_rule()]),
                   drop_sink={}.__setitem__, force_lossless=True)
    assert forced.text == DOC


def test_drop_marker_shape_is_shared_by_both_paths():
    """One constructor for the wire form, so the text regex can never drift from what the
    JSON path emits."""
    marker = lossy.drop_marker("abc", 12)
    assert marker == {lossy.DROP_KEY: "abc", "bytes": 12, "retrieve": lossy.RETRIEVE_TOOL}
    wire = json.dumps(lossy.drop_marker("a" * 16, 12), separators=(",", ":"))
    assert lossy._TEXT_MARKER_RE.match(wire)


def test_precision_question_requires_reading_the_payload():
    """A marker COUNT is 1 on a single-drop payload — guessable. The byte total is not."""
    rule = _drop_rule()
    text = DOC + "\n#### src/b.py\n\n```go\n" + CODE + "\n```\n"
    _, precision = dropeval.gen_text_drop_questions(text, rule, "codegraph_explore")
    _, store = _apply(text, rule)
    assert precision.expected == sum(len(v) for v in store.values())
    assert precision.expected > len(store)  # not the count


def test_run_drop_fluency_scores_a_text_payload():
    """The text branch of the live-model harness, end to end with a scripted answerer."""
    rule = _drop_rule()
    envelopes = [{"tool": "codegraph_explore", "raw": DOC, "sha": "deadbeef"}]
    recall, _ = dropeval.gen_text_drop_questions(DOC, rule, "codegraph_explore")

    def answerer(messages):
        last = messages[-1]
        if last["role"] == "tool":                      # retrieved: answer from the span
            return dropeval.Turn(text=json.dumps(recall.expected))
        return dropeval.Turn(text="", tool_calls=[dropeval.ToolCall(
            call_id="c1", name=lossy.RETRIEVE_TOOL,
            arguments={"handle": recall.expected_handle})])

    rows = dropeval.run_drop_fluency(envelopes, lambda _t: rule, {"m": answerer})["m"]
    recall_row = next(r for r in rows if r["kind"] == "recall")
    assert recall_row["retrieve_ok"] == 1 and recall_row["handle_ok"] == 1
    assert recall_row["answer_ok"] == 1
    assert recall_row["tool"] == "codegraph_explore" and recall_row["sha"] == "deadbeef"


def test_run_drop_text_payload_is_the_single_payload_entry_point():
    rule = _drop_rule()
    rows = dropeval.run_drop_text_payload(
        DOC, rule, "codegraph_explore", lambda _m: dropeval.Turn(text="nope"))
    assert [r["kind"] for r in rows] == ["recall", "precision"]
