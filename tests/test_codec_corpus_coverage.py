"""Tests for #403 Blocker 4 — the codec eval must not score a payload the model cannot fit.

The defect had two halves. The obvious one: an oversized request is answered off a
TRUNCATED payload (measured live — the gateway returned 200 with `err=0` for a 35,707-token
request against a 32,768 limit), so it cannot be caught by watching for transport errors.

The serious one is an arm asymmetry that is structural rather than incidental. terse's whole
purpose is that the compressed arm is smaller, so there is always a band where the RAW arm
exceeds the limit and the TERSE arm does not. Scored rather than excluded, the CONTROL arm
reads a truncated payload while the TREATMENT arm reads a whole one — which flatters terse
and can manufacture a false SAFE. Most of this file exists to pin that band shut.
"""

from __future__ import annotations

import json

import pytest

from terse import capture, cli, codeceval, fluency
from terse.dropeval import ToolCall, Turn
from terse.report import build_codec_verdict_report

# Wide enough that the codec tabularizes it and both a `deref` and an `enumerate` question
# exist, with a `blob` column whose cells are containers.
PAYLOAD = {"result": [{"id": i, "blob": {"owner": f"owner-{i}", "tags": ["x", "y", "z"]}}
                      for i in range(1, 13)]}
RAW_TEXT = json.dumps(PAYLOAD)


def _env(**extra) -> dict:
    return {"tool": "demo.list", "sha": "a" * 40, "raw": RAW_TEXT, **extra}


def _answers_correctly(seen: list[str] | None = None):
    """A model that always answers correctly, optionally recording every payload it saw.

    Answers the PRE-FLIGHT questions too, so this stub can be driven through `cli.main`,
    which always pre-flights. Matched on prompt AND payload text, like `preflight_stub` —
    each pre-flight payload carries one question per codec qtype, so the prompt alone is
    ambiguous between them."""
    expected = [(q.prompt, text, q.expected)
                for q, text in codeceval.preflight_questions()]
    expected += [(q.prompt, RAW_TEXT, q.expected)
                 for q in codeceval.gen_codec_questions(PAYLOAD)]
    expected += [(q.prompt, fluency.compress(PAYLOAD), q.expected)
                 for q in codeceval.gen_codec_questions(PAYLOAD)]

    def ask(messages):
        content = messages[-1]["content"]
        if seen is not None:
            seen.append(content)
        value = next((e for prompt, text, e in expected
                      if prompt in content and text in content), None)
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c1", name=codeceval.RECORD_VALUE_TOOL,
                     arguments={"value": value})])
    return ask


def _arm_tokens() -> tuple[int, int]:
    """The largest request each arm would send, in cl100k."""
    qs = codeceval.gen_codec_questions(PAYLOAD)
    raw = max(codeceval.request_tokens(q, RAW_TEXT) or 0 for q in qs)
    terse = max(codeceval.request_tokens(q, fluency.compress(PAYLOAD)) or 0 for q in qs)
    return raw, terse


@pytest.fixture(autouse=True)
def _needs_a_tokenizer():
    if codeceval.request_tokens(codeceval.gen_codec_questions(PAYLOAD)[0], RAW_TEXT) is None:
        pytest.skip("no tiktoken: the limit check cannot be exercised")


# --------------------------------------------------------------------------- #
# request_tokens — the WHOLE request, not the payload
# --------------------------------------------------------------------------- #
def test_request_tokens_counts_more_than_the_payload():
    """`b0e5f862`'s terse arm is 32,510 tokens against a 32,768 limit — it fits by payload
    and does not fit once the prompt, instruction and tool schema are added. A payload-only
    check would miss exactly the case that motivated this."""
    from terse.tokenize import count_cl100k

    q = codeceval.gen_codec_questions(PAYLOAD)[0]
    payload_only = count_cl100k(RAW_TEXT)
    assert payload_only is not None
    assert codeceval.request_tokens(q, RAW_TEXT) > payload_only


def test_request_tokens_includes_the_tool_schema():
    q = codeceval.gen_codec_questions(PAYLOAD)[0]
    bare = codeceval.request_tokens(q, RAW_TEXT, tool_defs=[])
    with_tool = codeceval.request_tokens(q, RAW_TEXT)
    assert bare is not None and with_tool is not None and with_tool > bare


# --------------------------------------------------------------------------- #
# oversized_arms — which arms do not fit
# --------------------------------------------------------------------------- #
def test_a_payload_that_fits_reports_no_oversized_arm():
    raw, _ = _arm_tokens()
    assert codeceval.oversized_arms(PAYLOAD, RAW_TEXT, raw) == []


def test_the_band_where_only_the_RAW_arm_is_over_is_detected():
    """The asymmetry this whole issue is about. terse is smaller, so a limit can land
    between the arms; there the control is truncated and the treatment is not."""
    raw, terse = _arm_tokens()
    assert terse < raw, "fixture does not compress — it cannot express the band"
    between = (raw + terse) // 2
    assert [a for a, _ in codeceval.oversized_arms(PAYLOAD, RAW_TEXT, between)] == ["raw"]


def test_both_arms_are_reported_when_both_are_over():
    _, terse = _arm_tokens()
    arms = codeceval.oversized_arms(PAYLOAD, RAW_TEXT, terse - 1)
    assert [a for a, _ in arms] == ["raw", "terse"]


def test_oversized_arms_reports_the_LARGEST_question_not_the_first():
    """One oversized question makes the payload unaskable, so the check is the max over the
    payload's questions. Using the first would let question order decide the verdict."""
    qs = codeceval.gen_codec_questions(PAYLOAD)
    sizes = [codeceval.request_tokens(q, RAW_TEXT) for q in qs]
    # The premise is about GENERATION ORDER, not merely that the sizes differ: what kills a
    # "first instead of max" mutation is that the FIRST question is not the largest. Sorting
    # first destroys exactly the property the test needs, and the mutation then dies only by
    # luck of `gen_codec_questions`' emission order.
    assert sizes[0] != max(sizes), (
        "the first question generated is already the largest — this fixture cannot "
        "distinguish max-over-questions from first-question")
    # A limit at the first question's size must still exclude, because a larger one follows.
    assert codeceval.oversized_arms(PAYLOAD, RAW_TEXT, sizes[0])


# --------------------------------------------------------------------------- #
# run_codec_fluency — excluded BEFORE the call, per model, both arms together
# --------------------------------------------------------------------------- #
def test_an_oversized_payload_is_never_asked():
    """Excluded before the question is put: the answerer must not see it at all. Scoring it
    and discarding the rows afterwards would still spend the call and, worse, would be an
    exclusion by outcome rather than by input."""
    seen: list[str] = []
    raw, _ = _arm_tokens()
    run = codeceval.run_codec_fluency(
        [_env()], {"m": _answers_correctly(seen)}, trials=2, preflight=False,
        limits={"m": raw - 1})
    assert run.rows["m"] == []
    assert seen == [], "the model was asked a payload the run had already excluded"


def test_the_exclusion_names_the_model_tool_arm_and_numbers():
    raw, terse = _arm_tokens()
    between = (raw + terse) // 2
    run = codeceval.run_codec_fluency(
        [_env()], {"m": _answers_correctly()}, trials=1, preflight=False,
        limits={"m": between})
    assert len(run.excluded) == 1
    e = run.excluded[0]
    assert (e.model, e.tool, e.arm, e.limit) == ("m", "demo.list", "raw", between)
    assert e.tokens == raw and e.sha == "a" * 40


def test_both_arms_go_together_so_a_models_corpus_never_splits_by_arm():
    """The invariant that makes the fix a fix. If only the oversized ARM were dropped, the
    payload would still contribute rows — scored on one arm against nothing — which is the
    asymmetry restated, not removed."""
    raw, terse = _arm_tokens()
    run = codeceval.run_codec_fluency(
        [_env()], {"m": _answers_correctly()}, trials=3, preflight=False,
        limits={"m": (raw + terse) // 2})   # only RAW is over
    assert run.rows["m"] == [], "the fitting terse arm was scored without its control"


def test_the_limit_is_per_model_not_global():
    """A global rule would let the smallest-context model shrink the corpus every other
    model is scored on — a different distortion, not a fix."""
    raw, _ = _arm_tokens()
    run = codeceval.run_codec_fluency(
        [_env()], {"small": _answers_correctly(), "big": _answers_correctly()},
        trials=1, preflight=False, limits={"small": raw - 1, "big": raw + 1_000})
    assert run.rows["small"] == []
    assert run.rows["big"], "a limit on one model removed the payload from another"
    assert {e.model for e in run.excluded} == {"small"}


def test_a_model_with_no_known_limit_is_scored_over_the_whole_corpus():
    """No default is invented: a guessed limit would silently drop payloads against a
    backend that publishes nothing."""
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"other": 1})
    assert run.rows["m"] and run.excluded == []


def test_no_limits_at_all_behaves_exactly_as_before():
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False)
    assert run.rows["m"] and run.excluded == []


def test_the_progress_stream_says_what_was_not_asked():
    raw, _ = _arm_tokens()
    lines: list[str] = []
    codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                preflight=False, limits={"m": raw - 1},
                                progress=lines.append)
    assert sum("(over limit)" in ln for ln in lines) == 1


def test_an_oversized_payload_cannot_produce_a_verdict_either_way():
    """End to end, and the reason the whole change exists: in the band where only RAW is
    over, a scored run compares a truncated control against a whole treatment. The payload
    must contribute nothing — no SAFE, no UNSAFE."""
    raw, terse = _arm_tokens()
    run = codeceval.run_codec_fluency(
        [_env()], {"m": _answers_correctly()}, trials=7, preflight=False,
        limits={"m": (raw + terse) // 2})
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": (raw + terse) // 2}, models=["m"])
    assert "**SAFE**" not in report and "**UNSAFE**" not in report
    # ...and the same corpus WITHOUT a limit does produce rows, so the assertion above is
    # about the limit and not about a fixture that never scores anything.
    unlimited = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()},
                                            trials=7, preflight=False)
    assert unlimited.rows["m"]


# --------------------------------------------------------------------------- #
# The report's Corpus coverage section
# --------------------------------------------------------------------------- #
def test_the_report_names_the_excluded_payload():
    raw, terse = _arm_tokens()
    between = (raw + terse) // 2
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"m": between})
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": between}, models=["m"])
    assert "## Corpus coverage" in report
    assert "`demo.list`" in report and "raw" in report
    assert f"{raw:,}" in report and f"{between:,}" in report


def test_the_report_names_a_model_whose_limit_is_unknown():
    """"No exclusions" means something different for a model that was checked than for one
    that could not be, so the report must not let the two read alike."""
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"m": 10 ** 9})
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": 10 ** 9}, models=["m", "unchecked"])
    assert "Limit unknown for" in report and "`unchecked`" in report
    assert "`m`" not in report.split("Limit unknown for")[1].split("\n")[0]


def test_the_coverage_section_is_absent_when_nothing_was_checked():
    """A run with no limits at all is the pre-#403 world; printing an empty coverage section
    there would imply a check that did not happen."""
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False)
    assert "## Corpus coverage" not in build_codec_verdict_report(run.rows)


def test_the_coverage_section_is_a_sibling_of_the_verdict_not_a_subsection():
    # Same reasoning as the savings section: it qualifies WHICH payloads the verdict covers,
    # which a reader needs before reading the verdict, so it cannot be nested under it.
    raw, _ = _arm_tokens()
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"m": raw - 1})
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": raw - 1}, models=["m"])
    assert "\n## Corpus coverage" in report and "\n### Corpus coverage" not in report


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec,expected", [
    (None, {}),
    ("", {}),
    ("m=10", {"m": 10}),
    (" a=1 , b=2 ", {"a": 1, "b": 2}),
])
def test_parse_model_limits_accepts(spec, expected):
    assert cli._parse_model_limits(spec) == expected


@pytest.mark.parametrize("spec", ["m", "m=", "=10", "m=abc", "m=0", "m=-5"])
def test_parse_model_limits_refuses_a_malformed_entry(spec):
    """Refused, never skipped: a typo'd model name would leave that model unchecked, which
    is the pre-#403 behaviour the flag exists to replace."""
    with pytest.raises(ValueError):
        cli._parse_model_limits(spec)


def test_discover_model_limits_reads_the_litellm_shape(monkeypatch):
    _stub_http(monkeypatch, {"/model/info": {"data": [
        {"model_name": "m", "model_info": {"max_input_tokens": 4096}},
        {"model_name": "other", "model_info": {"max_input_tokens": 8}},
    ]}})
    assert cli._discover_model_limits("http://gw/v1", "k", ["m"]) == {"m": 4096}


def test_discover_model_limits_falls_back_to_context_length(monkeypatch):
    _stub_http(monkeypatch, {"/models": {"data": [{"id": "m", "context_length": 2048}]}})
    assert cli._discover_model_limits("http://gw/v1", "k", ["m"]) == {"m": 2048}


def test_discover_model_limits_is_silent_when_the_gateway_says_nothing(monkeypatch):
    """A discovery error must leave the model unlimited and the run working — a non-LiteLLM
    backend has no `/model/info`, and that is not a reason to fail an eval."""
    _stub_http(monkeypatch, {})
    assert cli._discover_model_limits("http://gw/v1", "k", ["m"]) == {}


def _stub_http(monkeypatch, by_path: dict[str, dict], seen: list[str] | None = None):
    import urllib.request

    class _Resp:
        def __init__(self, body): self._b = body
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        if seen is not None:
            seen.append(req.full_url)
        for path, payload in by_path.items():
            if req.full_url.endswith(path):
                return _Resp(json.dumps(payload).encode())
        raise OSError("no such endpoint")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_envelope_shape_still_classifies_an_excluded_payloads_neighbours(monkeypatch):
    """Exclusion is per (model, payload): the rest of the corpus is untouched, including the
    live shape classification a neighbouring payload gets."""
    raw, _ = _arm_tokens()
    small = {"tool": "demo.list", "sha": "b" * 40,
             "raw": json.dumps({"result": [{"id": i, "blob": {"k": [i]}}
                                           for i in range(1, 13)]})}
    run = codeceval.run_codec_fluency([_env(), small], {"m": _answers_correctly()},
                                      trials=1, preflight=False, limits={"m": raw - 1})
    assert {r["sha"] for r in run.rows["m"]} == {"b" * 40}
    assert all(r["shape"] == capture.envelope_shape(small) for r in run.rows["m"])


def test_an_all_excluded_run_says_so_instead_of_claiming_the_corpus_has_no_questions():
    """The branch that used to swallow the disclosure. When every payload is excluded there
    are no rows, and the "no payload yields a question" message is FALSE about a corpus full
    of payloads that do — it points the reader at the wrong remedy."""
    raw, _ = _arm_tokens()
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"m": raw - 1})
    assert run.rows["m"] == []
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": raw - 1}, models=["m"])
    assert "excluded by a model's input limit" in report
    assert "no payload in the corpus yields" not in report
    assert "## Corpus coverage" in report


def test_a_genuinely_empty_run_still_says_the_corpus_has_no_questions():
    # The control for the test above: without exclusions the original message must survive.
    report = build_codec_verdict_report({"m": []})
    assert "whole object/array column" in report
    assert "excluded by a model's input limit" not in report


# --------------------------------------------------------------------------- #
# The CLI seam — driven through `main`, not grepped out of `cli.py`
# --------------------------------------------------------------------------- #
def _drive_cli(tmp_path, monkeypatch, argv_extra: list[str], discovered: dict[str, int]):
    """Run `terse fluency --codec-verdict` over a one-payload corpus, returning the `limits`
    the CLI actually handed the harness."""
    from terse.cli import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "demo.list__aaaa.json").write_text(json.dumps(
        {"tool": "demo.list", "sha": "a" * 40, "raw": RAW_TEXT}))
    monkeypatch.setattr(cli, "_build_answerers",
                        lambda args, make, **kw: {"m": _answers_correctly()})
    monkeypatch.setattr(cli, "_discover_model_limits",
                        lambda base, key, models: dict(discovered))
    captured: dict = {}
    real = codeceval.run_codec_fluency

    def spy(*a, **kw):
        captured["limits"] = kw.get("limits")
        captured["tool_defs"] = kw.get("tool_defs")
        return real(*a, **kw)

    monkeypatch.setattr(codeceval, "run_codec_fluency", spy)
    rc = main(["fluency", "--corpus", str(corpus), "--out", str(tmp_path / "r.md"),
               "--codec-verdict", *argv_extra])
    return rc, captured, (tmp_path / "r.md")


def test_the_cli_passes_discovered_limits_to_the_harness(tmp_path, monkeypatch):
    rc, captured, _ = _drive_cli(tmp_path, monkeypatch, [], {"m": 4096})
    assert rc == 0
    assert captured["limits"] == {"m": 4096}
    # ...and the real tool schema, or `request_tokens` would under-count every request.
    assert captured["tool_defs"] == [codeceval.RECORD_VALUE_TOOL_DEF]


def test_the_explicit_flag_beats_what_the_gateway_publishes(tmp_path, monkeypatch):
    """A gateway can publish a limit that is wrong for the route actually served — an alias
    resolves through a fallback chain at request time — so the operator needs the last word.
    """
    rc, captured, _ = _drive_cli(tmp_path, monkeypatch, ["--max-input-tokens", "m=999"],
                                 {"m": 4096})
    assert rc == 0 and captured["limits"] == {"m": 999}


def test_the_cli_refuses_a_malformed_limit_instead_of_running_unchecked(tmp_path,
                                                                       monkeypatch, capsys):
    from terse.cli import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "demo.list__aaaa.json").write_text(json.dumps(
        {"tool": "demo.list", "sha": "a" * 40, "raw": RAW_TEXT}))
    monkeypatch.setattr(cli, "_build_answerers",
                        lambda args, make, **kw: {"m": _answers_correctly()})
    report = tmp_path / "r.md"
    rc = main(["fluency", "--corpus", str(corpus), "--out", str(report),
               "--codec-verdict", "--max-input-tokens", "m=lots"])
    assert rc == 2
    _, err = capsys.readouterr()
    assert "--max-input-tokens" in err
    assert not report.exists(), "a report would read as a measured run"


def test_the_cli_writes_the_coverage_section_into_the_report(tmp_path, monkeypatch):
    raw, _ = _arm_tokens()
    rc, _, report = _drive_cli(tmp_path, monkeypatch, [], {"m": raw - 1})
    assert rc == 0
    assert "## Corpus coverage" in report.read_text()
    assert "excluded by a model's input limit" in report.read_text()


# --------------------------------------------------------------------------- #
# Adversarial-review findings — the exclusion must not decide the verdict
# --------------------------------------------------------------------------- #
SMALL = {"result": [{"id": i, "blob": {"k": [i]}} for i in range(1, 13)]}


def _fails_terse_on(target):
    """Correct on raw always; wrong on the terse arm of `target` only."""
    def ask(messages):
        content = messages[-1]["content"]
        for obj in (PAYLOAD, SMALL):
            for q in codeceval.gen_codec_questions(obj):
                if q.prompt not in content:
                    continue
                if json.dumps(obj) in content:
                    value = q.expected
                elif fluency.compress(obj) in content:
                    value = {"wrong": True} if obj is target else q.expected
                else:
                    continue
                return Turn(text="", tool_calls=[
                    ToolCall(call_id="c", name=codeceval.RECORD_VALUE_TOOL,
                             arguments={"value": value})])
        return Turn(text="", tool_calls=[])
    return ask


def _two_payload_corpus() -> list[dict]:
    return [{"tool": "demo.list", "sha": "b" * 40, "raw": RAW_TEXT},
            {"tool": "demo.list", "sha": "s" * 40, "raw": json.dumps(SMALL)}]


def test_excluding_the_payload_that_carries_the_failure_cannot_print_SAFE():
    """The finding that falsified this change's first design.

    "Excluding by a predeclared property of the INPUT is categorically safe, unlike
    excluding by OUTCOME" is true of the SELECTION and false of the CONSEQUENCE. Payload
    SIZE is not independent of what the eval measures: the largest payloads are the widest
    tables, which is exactly where a positional row lookup fails. Executed before the gate,
    on this fixture: a cell reading UNSAFE (raw 100%, terse 50%) read **SAFE** once the
    payload carrying the failure was excluded."""
    raw, _ = _arm_tokens()
    models = {"m": _fails_terse_on(PAYLOAD)}

    unlimited = codeceval.run_codec_fluency(_two_payload_corpus(), models, trials=20,
                                            preflight=False)
    assert "**UNSAFE**" in build_codec_verdict_report(unlimited.rows), (
        "fixture does not demonstrate corruption — it cannot express the flip")

    limited = codeceval.run_codec_fluency(_two_payload_corpus(), models, trials=20,
                                          preflight=False, limits={"m": raw - 1})
    assert limited.rows["m"], "the surviving payload must still be scored"
    report = build_codec_verdict_report(limited.rows, excluded=limited.excluded,
                                        limits={"m": raw - 1}, models=list(models))
    # The surviving payload is clean, so without the gate this prints SAFE/UNRESOLVED-by-
    # trial-count and the demonstrated corruption is simply gone.
    assert "| **SAFE** |" not in report
    assert "**UNRESOLVED**" in report
    assert "trimmed corpus" in report


def test_an_exclusion_never_suppresses_an_UNSAFE_that_survived():
    """Gates SAFE only, in the same direction as `_CODEC_MIN_CALL_RATE`: evidence that
    survived the trim is still evidence."""
    raw, _ = _arm_tokens()
    models = {"m": _fails_terse_on(SMALL)}      # the failure is on the payload that FITS
    run = codeceval.run_codec_fluency(_two_payload_corpus(), models, trials=20,
                                      preflight=False, limits={"m": raw - 1})
    assert run.excluded
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": raw - 1}, models=["m"])
    assert "**UNSAFE**" in report


def test_a_cell_whose_every_payload_was_excluded_still_gets_a_row():
    """A model that loses every payload of a cell contributes no rows, so the cell vanishes
    from the groups built out of rows — and a cell that silently disappears is
    indistinguishable from a corpus that never had it."""
    raw, _ = _arm_tokens()
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=5,
                                      preflight=False, limits={"m": raw - 1})
    assert run.rows["m"] == []
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": raw - 1}, models=["m"])
    assert "| `demo.list` |" in report
    assert "every payload exceeded the input limit" in report


def test_the_exclusion_records_the_shape_the_verdict_is_keyed_by():
    # The coverage table is keyed (model, tool, sha); the verdict is keyed (tool, shape).
    # Without shape, a tool spanning two shapes cannot say which cell lost coverage.
    raw, _ = _arm_tokens()
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"m": raw - 1})
    assert run.excluded[0].shape == capture.envelope_shape(_env())


# --------------------------------------------------------------------------- #
# Disclosure: a check that did not run must not read as a check that passed
# --------------------------------------------------------------------------- #
def test_without_a_tokenizer_the_report_says_the_check_did_not_run(monkeypatch):
    """`oversized_arms` returns `[]` both for a payload that FITS and for one it could not
    measure. With no tokenizer every payload passes vacuously, and the first cut printed
    "Every payload fitted every model that declared a limit" about a 504-token request
    against a 10-token limit."""
    monkeypatch.setattr(codeceval, "count_cl100k", lambda _t: None)
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"m": 10})
    assert run.excluded == [] and run.limit_check_ran is False
    report = build_codec_verdict_report(run.rows, excluded=run.excluded, limits={"m": 10},
                                        models=["m"], limit_check_ran=run.limit_check_ran)
    assert "The limit check did not run" in report
    assert "Every payload fitted" not in report


def test_with_a_tokenizer_the_check_is_reported_as_having_run():
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"m": 10 ** 9})
    assert run.limit_check_ran is True
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": 10 ** 9}, models=["m"],
                                        limit_check_ran=run.limit_check_ran)
    assert "Every payload fitted" in report


def test_an_empty_run_with_BOTH_causes_names_both():
    """On a real corpus most envelopes yield no question at all, and the ones that do are
    wide record tables — exactly what blows a small limit. Naming only the limit sends the
    reader after a bigger model when the remedy is mixed."""
    raw, _ = _arm_tokens()
    unaskable = {"tool": "demo.list", "sha": "n" * 40, "raw": json.dumps({"k": "scalar"})}
    run = codeceval.run_codec_fluency([_env(), unaskable], {"m": _answers_correctly()},
                                      trials=1, preflight=False, limits={"m": raw - 1})
    assert run.rows["m"] == [] and run.excluded and run.skipped_unaskable == 1
    report = build_codec_verdict_report(
        run.rows, excluded=run.excluded, limits={"m": raw - 1}, models=["m"],
        skipped_unaskable=run.skipped_unaskable)
    assert "TWO reasons" in report
    assert "no `deref`/`enumerate` question" in report


def test_the_savings_table_discloses_payloads_no_model_was_asked():
    """The savings header promises "whatever the sums could not cover is disclosed beneath
    the table, never quietly dropped". A payload excluded for every model contributes no
    row, so it left no trace — and the drop is systematically downward, because the excluded
    payloads are the largest."""
    raw, _ = _arm_tokens()
    run = codeceval.run_codec_fluency(_two_payload_corpus(), {"m": _answers_correctly()},
                                      trials=1, preflight=False, limits={"m": raw - 1})
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": raw - 1}, models=["m"])
    assert "absent from the sums above entirely" in report
    assert "understates what the codec saves" in report


def test_an_unknown_limit_is_disclosed_even_when_NO_model_has_one():
    """Discovery failing for the whole fleet is the common real case. The first cut returned
    early on empty `limits`, so the artifact carried no trace that no check ran — the only
    warning was an ephemeral stderr line."""
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False)
    report = build_codec_verdict_report(run.rows, excluded=run.excluded, limits={},
                                        models=["m"])
    assert "Limit unknown for" in report and "`m`" in report


# --------------------------------------------------------------------------- #
# Discovery and flag parsing, after review
# --------------------------------------------------------------------------- #
def test_discovery_takes_the_SMALLEST_of_duplicate_deployments(monkeypatch):
    """LiteLLM returns one entry per DEPLOYMENT, so a model group with two deployments
    yields two entries under one name. First-wins would let a request route to the smaller
    deployment and be answered off a truncated payload — the exact defect being checked
    for."""
    _stub_http(monkeypatch, {"/model/info": {"data": [
        {"model_name": "m", "model_info": {"max_input_tokens": 262144}},
        {"model_name": "m", "model_info": {"max_input_tokens": 32768}},
    ]}})
    assert cli._discover_model_limits("http://gw/v1", "k", ["m"]) == {"m": 32768}


def test_discovery_reaches_an_openrouter_style_models_endpoint_under_v1(monkeypatch):
    """OpenRouter's listing is at `<base>/v1/models`. Stripping `/v1` for every probe made
    the fallback unreachable for the only backend the docstring names."""
    seen: list[str] = []
    _stub_http(monkeypatch, {"/v1/models": {"data": [{"id": "m", "context_length": 2048}]}},
               seen=seen)
    assert cli._discover_model_limits("https://openrouter.ai/api/v1", "k", ["m"]) == {"m": 2048}
    assert any(u == "https://openrouter.ai/api/v1/models" for u in seen), seen


def test_parse_model_limits_refuses_a_duplicate_model():
    with pytest.raises(ValueError, match="given twice"):
        cli._parse_model_limits("m=10,m=20")


def test_a_declared_limit_of_zero_means_nothing_fits():
    """`if limit` read 0 as "no limit known" and disabled the check for the one value that
    most obviously asks for it."""
    run = codeceval.run_codec_fluency([_env()], {"m": _answers_correctly()}, trials=1,
                                      preflight=False, limits={"m": 0})
    assert run.rows["m"] == [] and run.excluded
