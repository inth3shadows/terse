"""Tests for #403 Blocker 3 — one tool must be graded as one verdict cell.

multiproxy qualifies a peer's tool as `<peer>__<tool>`, so the same tool is captured under
two spellings depending on which topology served it. The codec verdict keyed cells on the
captured name verbatim and graded them apart: `kb.read.list_nodes` n=12 beside
`kb__kb.read.list_nodes` n=12, instead of one n=24 cell.

The fix keys cells on `codeceval.cell_tool` — the policy name (`capture.qualify`), reached by
stripping a `<peer>__` prefix only when it is VERIFIED against the recorded server. On every
envelope a router writes that equals `capture.qualified_tool`; it diverges only where that
function guesses, because a guess that pools two different tools into one verdict is worse
than the split. Merging also opens a trap: a payload captured under BOTH spellings would be
asked twice and counted twice in one cell, padding `n` toward the SAFE floor.
"""

from __future__ import annotations

import json

from terse import capture, codeceval, fluency
from terse.dropeval import ToolCall, Turn
from terse.report import _CODEC_MIN_TRIALS, build_codec_verdict_report

PAYLOAD = {"result": [{"id": i, "blob": {"owner": f"o-{i}", "tags": ["x", "y"]}}
                      for i in range(1, 13)]}
OTHER = {"result": [{"id": i, "blob": {"owner": f"p-{i}", "tags": ["z"]}}
                    for i in range(1, 13)]}


def _env(tool: str, obj: dict, sha: str, server: str | None = "kb") -> dict:
    env = {"tool": tool, "sha": sha, "raw": json.dumps(obj)}
    if server is not None:
        env["server"] = server
    return env


def _correct():
    """Always right, on both arms, for either payload."""
    answers = []
    for obj in (PAYLOAD, OTHER):
        for q in codeceval.gen_codec_questions(obj):
            answers.append((q.prompt, json.dumps(obj), q.expected))
            answers.append((q.prompt, fluency.compress(obj), q.expected))

    def ask(messages):
        content = messages[-1]["content"]
        value = next((e for p, t, e in answers if p in content and t in content), None)
        return Turn(text="", tool_calls=[
            ToolCall(call_id="c", name=codeceval.RECORD_VALUE_TOOL, arguments={"value": value})])
    return ask


def _counting(calls: list[str]):
    inner = _correct()

    def ask(messages):
        calls.append(messages[-1]["content"])
        return inner(messages)
    return ask


def _cells(report: str) -> list[str]:
    return [ln for ln in report.splitlines()
            if ln.startswith("| `") and ("**SAFE**" in ln or "**UNSAFE**" in ln
                                         or "**UNRESOLVED**" in ln)]


# --------------------------------------------------------------------------- #
# One tool, one cell
# --------------------------------------------------------------------------- #
def test_the_router_and_bare_spellings_are_one_cell():
    """The defect as reported: two cells where there is one tool."""
    envs = [_env("kb.read.list_nodes", PAYLOAD, "a" * 40),
            _env("kb__kb.read.list_nodes", OTHER, "b" * 40)]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=1, preflight=False)
    assert {r["tool"] for r in run.rows["m"]} == {"kb.read.list_nodes"}
    cells = _cells(build_codec_verdict_report(run.rows))
    assert len(cells) == 1, cells
    assert "kb__" not in cells[0]


def test_the_merged_cell_pools_BOTH_spellings_trials():
    """The whole point: sample size decides whether a verdict exists. Each spelling ALONE is
    below the SAFE floor; together they reach it — and that is only true if the report
    grades them as one cell.

    The first cut of this test summed every row of the model regardless of tool and never
    rendered a verdict, so it stayed green with the cell key reverted to the raw name. Both
    reviews caught it. This one reads the verdict row."""
    questions = len(codeceval.gen_codec_questions(PAYLOAD))
    assert questions == len(codeceval.gen_codec_questions(OTHER))
    trials = _CODEC_MIN_TRIALS // questions // 2 + 1
    # Premise, checked rather than assumed: one spelling is under the floor, two are not.
    assert trials * questions < _CODEC_MIN_TRIALS <= trials * questions * 2

    alone = codeceval.run_codec_fluency([_env("kb.read.list_nodes", PAYLOAD, "a" * 40)],
                                        {"m": _correct()}, trials=trials, preflight=False)
    assert "trial(s), need" in build_codec_verdict_report(alone.rows)

    envs = [_env("kb.read.list_nodes", PAYLOAD, "a" * 40),
            _env("kb__kb.read.list_nodes", OTHER, "b" * 40)]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=trials,
                                      preflight=False)
    cells = _cells(build_codec_verdict_report(run.rows))
    assert len(cells) == 1, cells
    # Pooled, the TRIAL floor clears. The cell can still be UNRESOLVED on the separate
    # question floor (`_CODEC_MIN_QUESTIONS`: two payloads give 4 questions), which is not
    # what this test pins.
    assert "trial(s), need" not in cells[0], "the two spellings did not pool into one cell"


def test_a_genuine_bare_runecho_tool_is_qualified_the_way_policy_names_it():
    """`runecho__structure` and `structure` (server=runecho) are one tool. The conservative
    `stats.canonical_tool` would leave them apart — it strips only a redundant `runecho__`
    and never re-qualifies the bare name — which is why the policy name is used instead."""
    envs = [_env("structure", PAYLOAD, "a" * 40, server="runecho"),
            _env("runecho__structure", OTHER, "b" * 40, server="runecho")]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=1, preflight=False)
    assert {r["tool"] for r in run.rows["m"]} == {"runecho.structure"}


def test_the_cell_name_is_the_name_a_policy_rule_is_authored_under():
    """#158: the verdict answers a POLICY question, so its cell must carry the name the
    policy file uses. Pinned against the real helper rather than a literal, so the two
    cannot drift apart."""
    env = _env("kb__kb.read.list_nodes", PAYLOAD, "a" * 40)
    run = codeceval.run_codec_fluency([env], {"m": _correct()}, trials=1, preflight=False)
    assert {r["tool"] for r in run.rows["m"]} == {capture.qualified_tool(env)}


def test_a_different_peer_is_not_merged_into_this_one():
    """A genuine cross-peer name must stay distinct. Capture records the PEER as `server`
    for a router-qualified name — `kb__kb.read.x` carries `server=kb` — so re-qualifying
    with `server` keeps two peers apart. If capture ever recorded the router instead, this
    is the test that notices."""
    envs = [_env("kb__search", PAYLOAD, "a" * 40, server="kb"),
            _env("gh__search", OTHER, "b" * 40, server="gh")]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=1, preflight=False)
    assert {r["tool"] for r in run.rows["m"]} == {"kb.search", "gh.search"}
    assert len(_cells(build_codec_verdict_report(run.rows))) == 2


def test_an_envelope_with_no_server_keeps_its_captured_name():
    # Legacy envelopes predate the `server` field; `qualified_tool` leaves them bare.
    run = codeceval.run_codec_fluency([_env("structure", PAYLOAD, "a" * 40, server=None)],
                                      {"m": _correct()}, trials=1, preflight=False)
    assert {r["tool"] for r in run.rows["m"]} == {"structure"}


def test_an_exclusion_is_keyed_to_the_same_cell_the_verdict_uses():
    """Blocker 4 joins exclusions to verdict cells by `(tool, shape)`. If the exclusion kept
    the raw spelling while the verdict used the policy name, the join would miss, and a cell
    that lost a payload would print SAFE."""
    env = _env("kb__kb.read.list_nodes", PAYLOAD, "a" * 40)
    run = codeceval.run_codec_fluency([env], {"m": _correct()}, trials=1, preflight=False,
                                      limits={"m": 1})
    assert run.excluded and run.excluded[0].tool == "kb.read.list_nodes"


# --------------------------------------------------------------------------- #
# The trap: the same payload under both spellings
# --------------------------------------------------------------------------- #
def test_the_same_payload_under_both_spellings_is_asked_ONCE():
    """Merge the two spellings without deduping and one payload is asked twice. (The live
    corpus's two real duplicates are error text and so unaskable either way — this pins the
    mechanism for an askable one, which is the case that could pad a cell.)"""
    sha = "c" * 40
    calls: list[str] = []
    envs = [_env("kb.read.get", PAYLOAD, sha), _env("kb__kb.read.get", PAYLOAD, sha)]
    run = codeceval.run_codec_fluency(envs, {"m": _counting(calls)}, trials=3,
                                      preflight=False)
    once = codeceval.run_codec_fluency([envs[0]], {"m": _counting([])}, trials=3,
                                       preflight=False)
    assert len(run.rows["m"]) == len(once.rows["m"])
    questions = len(codeceval.gen_codec_questions(PAYLOAD))
    assert len(calls) == questions * 3 * 2, "the duplicate payload was asked a second time"


def test_deduplication_cannot_pad_a_cell_to_the_SAFE_floor():
    """The consequence that makes the dedupe load-bearing. One payload, below the floor,
    duplicated under two spellings: without the dedupe its trials double and the cell
    crosses `_CODEC_MIN_TRIALS` on evidence it never had."""
    questions = len(codeceval.gen_codec_questions(PAYLOAD))
    trials = _CODEC_MIN_TRIALS // questions // 2 + 1      # one copy is below the floor...
    assert trials * questions < _CODEC_MIN_TRIALS <= trials * questions * 2  # ...two are not
    sha = "d" * 40
    envs = [_env("kb.read.get", PAYLOAD, sha), _env("kb__kb.read.get", PAYLOAD, sha)]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=trials,
                                      preflight=False)
    report = build_codec_verdict_report(run.rows, merged_duplicates=run.merged_duplicates)
    assert "**SAFE**" not in report
    assert "**UNRESOLVED**" in report


def test_the_merge_is_counted_and_disclosed():
    sha = "e" * 40
    envs = [_env("kb.read.get", PAYLOAD, sha), _env("kb__kb.read.get", PAYLOAD, sha)]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=1, preflight=False)
    assert run.merged_duplicates == {"kb.read.get": 1}
    report = build_codec_verdict_report(run.rows, merged_duplicates=run.merged_duplicates)
    assert "## Tool name spellings merged" in report
    assert "| `kb.read.get` | 1 |" in report


def test_different_payloads_under_the_two_spellings_are_both_kept():
    # Deduplication is by payload identity, not by tool: two DISTINCT payloads of one tool
    # are two pieces of evidence and both belong in the cell.
    envs = [_env("kb.read.get", PAYLOAD, "f" * 40), _env("kb__kb.read.get", OTHER, "g" * 40)]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=1, preflight=False)
    assert run.merged_duplicates == {}
    assert {r["sha"] for r in run.rows["m"]} == {"f" * 40, "g" * 40}


def test_the_same_sha_under_two_DIFFERENT_tools_is_not_merged():
    """The key is `(tool, sha)`, not sha alone. Two tools returning byte-identical payloads
    are two cells, and each is entitled to that payload's evidence."""
    sha = "h" * 40
    envs = [_env("kb.read.get", PAYLOAD, sha), _env("kb.read.search", PAYLOAD, sha)]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=1, preflight=False)
    assert run.merged_duplicates == {}
    assert {r["tool"] for r in run.rows["m"]} == {"kb.read.get", "kb.read.search"}


def test_envelopes_without_a_sha_are_never_merged():
    """Without a payload identity nothing proves two envelopes are the same payload, and
    collapsing every sha-less one into the first is the placeholder-identity defect
    `run_codec_fluency` already refuses for `sha` itself."""
    envs = [{"tool": "kb.read.get", "server": "kb", "raw": json.dumps(PAYLOAD)},
            {"tool": "kb__kb.read.get", "server": "kb", "raw": json.dumps(PAYLOAD)}]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=1, preflight=False)
    assert run.merged_duplicates == {}
    per_payload = len(codeceval.gen_codec_questions(PAYLOAD))
    assert len(run.rows["m"]) == per_payload * 2


def test_the_first_spelling_by_load_order_is_the_one_asked():
    # Deterministic: `capture.load_corpus` sorts by (captured_at, filename).
    sha = "i" * 40
    calls: list[str] = []
    first, second = _env("kb.read.get", PAYLOAD, sha), _env("kb__kb.read.get", PAYLOAD, sha)
    run = codeceval.run_codec_fluency([first, second], {"m": _counting(calls)}, trials=1,
                                      preflight=False)
    assert run.merged_duplicates == {"kb.read.get": 1}
    assert len(calls) == len(codeceval.gen_codec_questions(PAYLOAD)) * 2   # asked once


def test_an_UNASKABLE_duplicate_is_not_counted_as_a_merge():
    """Both reviews of the first cut. Deduplication ran before the askability skip, so it
    counted duplicates that would never have been asked — and on the live corpus BOTH real
    duplicates are error text (`Error executing tool kb.read.get: ...`). The report said
    "each was asked once" about payloads that were never asked at all."""
    sha = "k" * 40
    error_text = "Error executing tool kb.read.get: Unknown target_table 'sessions'"
    envs = [{"tool": "kb.read.get", "server": "kb", "sha": sha, "raw": error_text},
            {"tool": "kb__kb.read.get", "server": "kb", "sha": sha, "raw": error_text}]
    calls: list[str] = []
    run = codeceval.run_codec_fluency(envs, {"m": _counting(calls)}, trials=1,
                                      preflight=False)
    assert calls == [] and run.rows["m"] == []
    assert run.merged_duplicates == {}, "an unaskable duplicate was reported as merged"
    assert run.skipped_unaskable == 2, "both copies are unaskable and must say so"
    assert "## Tool name spellings merged" not in build_codec_verdict_report(
        run.rows, merged_duplicates=run.merged_duplicates)


def test_the_merge_note_is_not_filed_under_the_input_limit_section():
    """That section's intro explains payloads declined for not FITTING — the wrong reason
    for a merge. The first cut rendered the merge line there."""
    sha = "l" * 40
    envs = [_env("kb.read.get", PAYLOAD, sha), _env("kb__kb.read.get", PAYLOAD, sha)]
    run = codeceval.run_codec_fluency(envs, {"m": _correct()}, trials=1, preflight=False,
                                      limits={"m": 10 ** 9})
    report = build_codec_verdict_report(run.rows, excluded=run.excluded,
                                        limits={"m": 10 ** 9}, models=["m"],
                                        merged_duplicates=run.merged_duplicates)
    coverage = report.split("## Corpus coverage")[1].split("\n## ")[0]
    assert "kb.read.get" not in coverage
    assert "## Tool name spellings merged" in report


# --------------------------------------------------------------------------- #
# Wrong merges — the cell key must never pool two DIFFERENT tools
# --------------------------------------------------------------------------- #
def test_a_legacy_envelope_with_no_server_does_not_pool_two_peers():
    """`capture.qualified_tool` strips ANY `__` prefix unverified, so with no `server`
    `gh__search` and `kb__search` both become `search` — one verdict pooling two tools'
    evidence, which is worse than the split this change fixes."""
    assert codeceval.cell_tool({"tool": "gh__search"}) == "gh__search"
    assert codeceval.cell_tool({"tool": "kb__search"}) == "kb__search"
    assert codeceval.cell_tool({"tool": "gh__search", "server": ""}) == "gh__search"


def test_a_tool_whose_own_name_contains_a_double_underscore_is_not_truncated():
    """`issues__list` under `server=gh` is not a router qualification — the prefix is not the
    server. Unverified stripping filed it in the cell of gh's real `list` tool."""
    assert codeceval.cell_tool({"tool": "issues__list", "server": "gh"}) == "gh.issues__list"
    assert codeceval.cell_tool({"tool": "list", "server": "gh"}) == "gh.list"


def test_on_every_router_produced_envelope_the_cell_is_the_policy_name():
    """The divergence from `capture.qualified_tool` is confined to envelopes that function
    would be guessing about. multiproxy records the PEER as server (`server_name=spec.name`),
    so for everything it writes — prefix == server, or bare with a server — the verdict cell
    and the policy name are the same string."""
    router_shaped = [
        {"tool": "kb__kb.read.list_nodes", "server": "kb"},
        {"tool": "kb.read.list_nodes", "server": "kb"},
        {"tool": "runecho__structure", "server": "runecho"},
        {"tool": "structure", "server": "runecho"},
        {"tool": "codegraph__codegraph_explore", "server": "codegraph"},
        {"tool": "codegraph_explore", "server": "codegraph"},
        {"tool": "get_by_path", "server": "shot-mcp"},
    ]
    for env in router_shaped:
        assert codeceval.cell_tool(env) == capture.qualified_tool(env), env


def test_a_non_string_tool_does_not_crash_the_sweep():
    # `capture.load_corpus` only requires the key; the pre-Blocker-3 code tolerated any type.
    assert codeceval.cell_tool({"tool": 5, "server": "kb"}) == "?"
    assert codeceval.cell_tool({"server": "kb"}) == "?"


def test_the_cli_writes_the_merge_into_the_report(tmp_path, monkeypatch):
    """Driven through `main`: a merge the harness counts but the CLI never hands the report
    is invisible in the only artifact anyone reads."""
    from terse import cli
    from terse.cli import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    sha = "j" * 40
    for tool in ("kb.read.get", "kb__kb.read.get"):
        (corpus / f"{tool}__{sha[:8]}.json").write_text(json.dumps(_env(tool, PAYLOAD, sha)))
    ask = _correct()

    def answerer(messages):
        content = messages[-1]["content"]
        pre = next((q.expected for q, text in codeceval.preflight_questions()
                    if q.prompt in content and text in content), None)
        if pre is not None:
            return Turn(text="", tool_calls=[ToolCall(
                call_id="c", name=codeceval.RECORD_VALUE_TOOL, arguments={"value": pre})])
        return ask(messages)

    monkeypatch.setattr(cli, "_build_answerers", lambda args, make, **kw: {"m": answerer})
    monkeypatch.setattr(cli, "_discover_model_limits", lambda *a: {})
    report = tmp_path / "r.md"
    assert main(["fluency", "--corpus", str(corpus), "--out", str(report),
                 "--codec-verdict"]) == 0
    text = report.read_text()
    assert "## Tool name spellings merged" in text
    assert "kb__" not in "\n".join(_cells(text))
