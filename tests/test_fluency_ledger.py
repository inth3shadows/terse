"""Fluency ledger (ledger.py): one JSONL row per live model call, written at `_ask_n`'s
choke point. Mirrors the pattern `test_stats.py` uses for the savings ledger — an env var
resolved at call time (here `TERSE_FLUENCY_LEDGER`, there `XDG_STATE_HOME`), monkeypatched
per test rather than mocking the module.

These tests call through `_ask_n`/`run_payload`, not `log_call` directly, wherever the
point is "a real harness run logs a row" — a test that only calls `log_call` in isolation
would keep passing even if the `_ask_n` call site were deleted (#263/#268's own lesson,
applied to this file: assert through the real seam, not the primitive alone)."""

from __future__ import annotations

import hashlib
import json

import terse
from terse.fluency.harnesses import _ask_n, run_payload
from terse.fluency.ledger import default_ledger_path

PAYLOAD = [{"id": i, "state": "open", "repo": "acme/widgets"} for i in range(6)]


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- path resolution ---

def test_default_ledger_path_honors_xdg_state_home(monkeypatch, tmp_path):
    # Mirrors test_stats.py's test_default_stats_log_honors_xdg_state_home — same
    # convention, same fixture shape.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("TERSE_FLUENCY_LEDGER", raising=False)
    assert default_ledger_path() == tmp_path / "terse" / "fluency-ledger.jsonl"


def test_empty_env_value_disables_the_ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setenv("TERSE_FLUENCY_LEDGER", "")
    ok, fails = _ask_n(lambda s, u: "6", "", "q", "count", 6, 1,
                       model="m", arm="raw", harness="h")
    assert (ok, fails) == (1, 0)
    assert not (tmp_path / "terse" / "fluency-ledger.jsonl").exists()


# --- row contents, via the real _ask_n seam ---

def test_ask_n_logs_an_answered_correct_call(monkeypatch, tmp_path):
    log = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TERSE_FLUENCY_LEDGER", str(log))
    ok, fails = _ask_n(lambda s, u: "6", "sys", "usr", "count", 6, 1,
                       model="gpt-x", arm="raw", harness="run_payload")
    assert (ok, fails) == (1, 0)
    rows = _lines(log)
    assert len(rows) == 1
    row = rows[0]
    assert row["harness"] == "run_payload"
    assert row["model"] == "gpt-x"
    assert row["arm"] == "raw"
    assert row["qtype"] == "count"
    assert row["expected"] == 6
    assert row["reply"] == "6"
    assert row["correct"] is True
    assert row["unanswered"] is False
    assert row["trial"] == 0
    assert row["terse_version"] == terse.__version__
    assert row["prompt_chars"] == len("sys\x00usr")
    assert row["prompt_sha256"] == hashlib.sha256(b"sys\x00usr").hexdigest()
    assert "ts" in row and row["ts"].endswith("Z")


def test_ask_n_logs_a_wrong_answer_as_correct_false(monkeypatch, tmp_path):
    log = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TERSE_FLUENCY_LEDGER", str(log))
    _ask_n(lambda s, u: "not six", "", "usr", "count", 6, 1,
          model="m", arm="terse", harness="run_payload")
    row = _lines(log)[0]
    assert row["reply"] == "not six"
    assert row["correct"] is False
    assert row["unanswered"] is False


def test_ask_n_logs_an_unanswered_call_distinctly_from_wrong(monkeypatch, tmp_path):
    """The repo's standing rule (#263/#268): a transport failure or blank reply is NOT a
    wrong answer. The ledger must preserve that distinction, not collapse it into
    `correct: false`."""
    log = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TERSE_FLUENCY_LEDGER", str(log))

    def dead(system, user):
        raise ConnectionError("rate limited")

    _ask_n(dead, "", "usr", "count", 6, 1, model="m", arm="terse", harness="run_payload")
    row = _lines(log)[0]
    assert row["reply"] is None
    assert row["correct"] is None
    assert row["unanswered"] is True


def test_ask_n_writes_one_row_per_trial_with_increasing_trial_index(monkeypatch, tmp_path):
    log = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TERSE_FLUENCY_LEDGER", str(log))
    _ask_n(lambda s, u: "6", "", "usr", "count", 6, 3,
          model="m", arm="terse", harness="run_payload")
    rows = _lines(log)
    assert [r["trial"] for r in rows] == [0, 1, 2]


def test_run_payload_logs_every_arm_through_the_real_harness(monkeypatch, tmp_path):
    """The end-to-end path a live `terse fluency` run actually takes — exercises the
    model/arm/harness threading through `run_payload` into `_ask_n`, not just `_ask_n`
    called directly."""
    log = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TERSE_FLUENCY_LEDGER", str(log))
    rows = run_payload(PAYLOAD, "[]", lambda s, u: "6", primer="P", trials=1,
                       model="my-model")
    assert rows, "the corpus must generate questions or this pins nothing"
    logged = _lines(log)
    arms = {r["arm"] for r in logged}
    assert arms == {"raw", "terse", "primer", "inline"}
    assert all(r["model"] == "my-model" for r in logged)
    assert all(r["harness"] == "run_payload" for r in logged)


# --- fail-open: a write failure never changes the run's result ---

def test_write_failure_never_changes_the_ask_n_result(monkeypatch, tmp_path, capsys):
    import terse.fluency.ledger as ledger_mod
    ledger_mod._warned = False  # test isolation: the module-level warn-once flag
    # Point the ledger at a path whose PARENT is a regular file — mkdir_restricted then
    # raises FileExistsError/NotADirectoryError trying to create it as a directory.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("TERSE_FLUENCY_LEDGER", str(blocker / "ledger.jsonl"))
    ok, fails = _ask_n(lambda s, u: "6", "", "usr", "count", 6, 2,
                       model="m", arm="terse", harness="run_payload")
    # The scored result is IDENTICAL to a run with a working ledger.
    assert (ok, fails) == (2, 0)
    err = capsys.readouterr().err
    assert "[terse fluency]" in err
    assert not (blocker / "ledger.jsonl").exists()


def test_write_failure_warns_only_once(monkeypatch, tmp_path, capsys):
    import terse.fluency.ledger as ledger_mod
    ledger_mod._warned = False
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("TERSE_FLUENCY_LEDGER", str(blocker / "ledger.jsonl"))
    _ask_n(lambda s, u: "6", "", "usr", "count", 6, 3,
          model="m", arm="terse", harness="run_payload")
    err = capsys.readouterr().err
    assert err.count("[terse fluency]") == 1, err
    ledger_mod._warned = False  # leave clean for the rest of the suite
