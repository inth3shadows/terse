"""#267: every `terse fluency` mode buffered its whole report and printed on completion, so
a 45-minute run and a wedged one looked identical for the whole duration -- #263's own
report names a 66-minute run with zero output. #264 fixed the false verdict at the end;
this is the silence before it.

Each live harness takes `progress=None` and, when given, calls it once per (model,
payload) with a cumulative line -- questions so far, failed calls so far over attempts,
elapsed seconds. The CLI passes `_stderr_progress`; the harnesses themselves print nothing,
so the tests that drive them directly stay quiet. Tested from both ends: the harness
contract (one line per unit of work, `done` reaching `total`, silence when not asked) and
the CLI seam (lines on stderr, stdout untouched).
"""

from __future__ import annotations

import json

import pytest
from test_fluency import DIFF_CURR, DIFF_PREV, PAYLOAD, TEXT_CURR, TEXT_PREV, _soak_envs

from terse import dropeval, fluency
from terse.fluency.harnesses import progress_line


def _lines_for(model, lines):
    return [ln for ln in lines if f" {model} " in ln]


def _questions(line):
    return int(line.split(" question(s)")[0].split()[-1])


def _done_total(line):
    """`3/9` out of `... 3/9 payload(s) ...`."""
    token = line.split(" payload(s)")[0].split()[-1]
    done, total = token.split("/")
    return int(done), int(total)


# --------------------------------------------------------------------------- #
# The line
# --------------------------------------------------------------------------- #


def test_progress_line_is_cumulative_and_reads_both_failure_counter_names():
    """Fluency rows say `fails`, dropeval rows say `errors` (#264 / #299 named the same
    counter twice); the line reads either, over `attempts`, which both emit."""
    fl = [{"fails": 1, "attempts": 4}, {"fails": 0, "attempts": 4}]
    de = [{"errors": 2, "attempts": 2}]
    assert (progress_line("fluency", "m", 2, 9, fl, started=100.0, now=148.0)
            == "[fluency] m                        2/9 payload(s)  2 question(s)  "
               "1/8 call(s) failed  (48s)")
    assert "2/2 call(s) failed" in progress_line("drop-eval", "m", 1, 1, de, started=0.0,
                                                 now=1.0)
    assert "0/0 call(s) failed  (0s)" in progress_line("x", "m", 0, 3, [], started=5.0,
                                                       now=5.0)


# --------------------------------------------------------------------------- #
# The harnesses
# --------------------------------------------------------------------------- #


def test_run_fluency_reports_every_model_and_payload_including_skipped_ones():
    """`done` must reach `total`, or the last line of a run that ends on a non-JSON
    payload reads as unfinished -- the exact ambiguity #267 is about."""
    envs = [{"tool": "t", "sha": "a", "raw": json.dumps(PAYLOAD)},
            {"tool": "t", "sha": "b", "raw": "not json at all"},
            {"tool": "t", "sha": "c", "raw": json.dumps(PAYLOAD)}]
    lines: list[str] = []
    results = fluency.run_fluency(envs, {"m1": lambda s, u: "", "m2": lambda s, u: ""},
                                  trials=1, progress=lines.append)
    assert len(lines) == 6
    for model in ("m1", "m2"):
        mine = _lines_for(model, lines)
        assert [_done_total(ln) for ln in mine] == [(1, 3), (2, 3), (3, 3)]
        assert all(ln.startswith("[fluency]") for ln in mine)
        # Cumulative: the question count on the last line is the model's whole result.
        assert f"{len(results[model])} question(s)" in mine[-1]
        # The skipped payload moved `done`, not the counts.
        assert _questions(mine[1]) == _questions(mine[0]) > 0
        assert _questions(mine[2]) == 2 * _questions(mine[0])


def test_the_harnesses_are_silent_unless_asked(capsys):
    envs = [{"tool": "t", "sha": "a", "raw": json.dumps(PAYLOAD)}]
    fluency.run_fluency(envs, {"m": lambda s, u: ""}, trials=1)
    out, err = capsys.readouterr()
    assert out == "" and err == ""


def test_run_fluency_surfaces_transport_failures_as_they_accumulate():
    """The counters #264 added are what to watch live: a run visibly losing calls at
    payload 1 is one to kill at payload 1."""
    def dead(system, user):
        raise RuntimeError("backend down")

    envs = [{"tool": "t", "sha": "a", "raw": json.dumps(PAYLOAD)}]
    lines: list[str] = []
    fluency.run_fluency(envs, {"m": dead}, trials=1, progress=lines.append)
    (line,) = lines
    fails, attempts = line.split(" call(s) failed")[0].split()[-1].split("/")
    assert int(fails) == int(attempts) > 0


def test_diff_harnesses_report_with_their_own_labels():
    envs = [{"tool": "demo", "sha": "aaa", "raw": json.dumps(DIFF_PREV)},
            {"tool": "demo", "sha": "bbb", "raw": json.dumps(DIFF_CURR)}]
    lines: list[str] = []
    fluency.run_diff_fluency(envs, {"m": lambda s, u: "9"}, trials=1, progress=lines.append)
    assert lines and all(ln.startswith("[fluency --diff]") for ln in lines)
    assert _done_total(lines[-1]) == (1, 1)

    text = [{"tool": "tt", "sha": "aaa", "raw": TEXT_PREV},
            {"tool": "tt", "sha": "bbb", "raw": TEXT_CURR}]
    lines = []
    fluency.run_text_diff_fluency(text, {"m": lambda s, u: "21"}, trials=1,
                                  progress=lines.append)
    assert lines and all(ln.startswith("[fluency --text-diff]") for ln in lines)

    lines = []
    fluency.run_diff_soak(_soak_envs(n=8), {"m1": lambda s, u: ""}, trials=1, max_depth=3,
                          per_depth_cap=2, progress=lines.append)
    assert lines and all(ln.startswith("[fluency --diff-soak]") for ln in lines)
    done, total = _done_total(lines[-1])
    assert done == total == len(lines)


def test_run_drop_fluency_reports_per_model_per_payload_and_counts_skips_as_done(tmp_path):
    from conftest import drop_eval_envelope, drop_eval_policy_doc

    from terse import policy as policy_mod

    pol_path = tmp_path / "policy.json"
    pol_path.write_text(json.dumps(
        drop_eval_policy_doc(["minify", "tabularize", "dictionary"], suggested=False)))
    pol = policy_mod.load_policy(pol_path)
    envs = [drop_eval_envelope(),
            {"tool": "nothing.here", "sha": "zz", "raw": json.dumps({"ok": True})}]
    answerers = {"m1": lambda messages: dropeval.Turn(text="no"),
                 "m2": lambda messages: dropeval.Turn(text="no")}
    lines: list[str] = []
    dropeval.run_drop_fluency(envs, pol.select, answerers, trials=1, control=True,
                              progress=lines.append)
    for model in ("m1", "m2"):
        mine = _lines_for(model, lines)
        assert [_done_total(ln) for ln in mine] == [(1, 2), (2, 2)], mine
        assert all(ln.startswith("[drop-eval]") for ln in mine)


# --------------------------------------------------------------------------- #
# The CLI seam
# --------------------------------------------------------------------------- #


_CLI_MODES = {
    # flag(s)            label                    envelopes written to the corpus
    "base": ([], "[fluency]", "json"),
    "diff": (["--diff"], "[fluency --diff]", "json"),
    "diff-soak": (["--diff-soak"], "[fluency --diff-soak]", "soak"),
    "text-diff": (["--text-diff-eval"], "[fluency --text-diff]", "text"),
}


def _write_corpus(corpus, kind):
    corpus.mkdir()
    if kind == "json":
        envs = [{"tool": "demo", "sha": sha, "raw": json.dumps(obj)}
                for sha, obj in (("aaa", DIFF_PREV), ("bbb", DIFF_CURR))]
    elif kind == "soak":
        envs = _soak_envs(n=8)
    else:
        envs = [{"tool": "tt", "sha": "aaa", "raw": TEXT_PREV},
                {"tool": "tt", "sha": "bbb", "raw": TEXT_CURR}]
    for e in envs:
        (corpus / f"{e['tool']}__{e['sha']}.json").write_text(json.dumps(e))


@pytest.mark.parametrize("mode", sorted(_CLI_MODES))
def test_the_cli_streams_progress_to_stderr_and_keeps_stdout_clean(tmp_path, monkeypatch,
                                                                    capsys, mode):
    """Unit-pinned harnesses and an unwired call site is how mutations survive; every
    CLI mode gets its own witness. stderr, so `--out` and stdout piping are untouched."""
    from terse import cli
    from terse.cli import main

    flags, label, kind = _CLI_MODES[mode]
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, kind)
    monkeypatch.setattr(cli, "_build_answerers",
                        lambda args, make: {"stub-model": lambda s, u: "9"})
    argv = ["fluency", "--corpus", str(corpus), "--out", str(tmp_path / "rep.md"), *flags]
    assert main(argv) == 0
    out, err = capsys.readouterr()
    progress = [ln for ln in err.splitlines() if ln.startswith(label)]
    assert progress, err
    assert all("stub-model" in ln for ln in progress)
    assert "payload(s)" not in out


@pytest.mark.parametrize("seam", ["tune", "fluency"])
def test_both_drop_eval_seams_stream_progress(tmp_path, monkeypatch, capsys, seam):
    """`tune --drop-eval` (`_tune_drop_eval`) and `fluency --drop-eval` (`_cmd_fluency`)
    are two call sites of `run_drop_fluency`; each is wired separately."""
    import argparse

    from conftest import (
        drop_eval_envelope,
        drop_eval_policy_doc,
        fluency_drop_eval_args,
    )

    from terse import cli

    monkeypatch.setattr(cli, "_build_answerers",
                        lambda args, make: {"stub-model": lambda m: dropeval.Turn(text="no")})
    doc = drop_eval_policy_doc(["minify", "tabularize", "dictionary"], suggested=False)
    env = drop_eval_envelope()
    if seam == "tune":
        args = argparse.Namespace(trials=1, no_control=True, accept_degraded=False)
        cli._tune_drop_eval(args, doc, [env])
    else:
        pol_path = tmp_path / "policy.json"
        pol_path.write_text(json.dumps(doc))
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.json").write_text(json.dumps({k: env[k] for k in ("tool", "sha", "raw")}))
        cli._cmd_fluency(fluency_drop_eval_args(corpus=corpus, policy=pol_path,
                                                out=tmp_path / "rep.md"))
    out, err = capsys.readouterr()
    progress = [ln for ln in err.splitlines() if ln.startswith("[drop-eval]")]
    assert progress, err
    assert all("stub-model" in ln for ln in progress)
    assert "payload(s)" not in out
