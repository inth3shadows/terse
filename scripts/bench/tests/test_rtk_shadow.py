"""Tests for the RTK shadow trial hook (scripts/bench/rtk_shadow.py)."""
from __future__ import annotations

import datetime
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rtk_shadow as rk


def test_simple_words_takes_one_plain_command_with_quoting():
    assert rk.simple_words("git log --oneline -5") == ["git", "log", "--oneline", "-5"]
    assert rk.simple_words("grep -rn 'a.*b|c' src") == ["grep", "-rn", "a.*b|c", "src"]
    assert rk.simple_words('find . -name "*.py"') == ["find", ".", "-name", "*.py"]


def test_simple_words_refuses_anything_the_shell_would_expand_chain_or_redirect():
    for cmd in ("ls | head", "ls && pwd", "ls; pwd", "cat f > g", "sort < f", "ls &",
                "echo $(date)", "echo `date`", 'grep "$X" f', "ls *.py", "ls ~/x",
                "cat f?", "ls [ab]", "ls {a,b}", "a\nb", "ls \\\n -l", "FOO=1 ls",
                "grep 'unclosed f", 'grep "a`id`" f', "ls # note", "", "   ",
                "ls\r-la", "ls\x0b-la", "ls\x0c-la"):
        assert rk.simple_words(cmd) is None, cmd


def test_peel_takes_an_absolute_cd_and_a_stderr_redirect_only():
    assert rk.peel("git status") == ("git status", None, False)
    assert rk.peel("cd /repo && git log -3 2>&1") == ("git log -3", "/repo", False)
    assert rk.peel("cd '/my dir' && grep -rn x src 2>/dev/null") == \
        ("grep -rn x src", "/my dir", True)
    assert rk.peel("cd sub && ls") == ("cd sub && ls", None, False)
    # The shell's directory here is the relative name `"/tmp"`, quotes included.
    assert rk.peel("cd '\"/tmp\"' && ls")[1] is None
    assert rk.simple_words(rk.peel("cd sub && ls")[0]) is None
    assert rk.simple_words(rk.peel("cd /a && cd /b && ls")[0]) is None
    assert rk.simple_words(rk.peel("cd /a && ls | head")[0]) is None


def test_read_only_list_and_the_options_that_write_or_run():
    yes = ("git log -3", "git diff --stat", "git status", "git show HEAD", "grep -rn x .",
           "rg x", "cat f", "head -5 f", "tail -5 f", "ls -la", "find . -name x")
    no = ("git push", "git -C /x log", "git -c a=b log", "git branch -D x", "git",
          "git diff --output=f", "git diff --output f", "git log --ext-diff",
          "find . -delete", "find . -exec rm {} +", "find . -fprint f", "rg --pre cat x",
          "rg --pre=cat x", "tree", "tree -ao f", "rm f", "pytest", "sed -n 1p f",
          "python x.py", "git log --help", "git diff -h", "grep --help", "ls --help")
    for cmd in yes:
        assert rk.read_only(cmd.split()), cmd
    for cmd in no:
        assert not rk.read_only(cmd.split()), cmd


def test_command_word_never_carries_an_argument_or_an_assignment():
    assert rk.command_word("git diff --stat") == "git diff"
    assert rk.command_word("cd /repo && git log -3") == "git log"
    assert rk.command_word("/usr/bin/grep -rn secret .") == "grep"
    assert rk.command_word("TOKEN=abc123 curl x") == "(other)"
    assert rk.command_word("git -C /x log") == "git"
    assert rk.command_word("git hunter2hunter2") == "git"
    assert rk.command_word("/home/me/clients/acme/rotate-prod-keys.sh --now") == "(other)"
    assert rk.command_word("hunter2hunter2") == "(other)"
    assert rk.command_word("$(evil) x") == "(other)"
    assert rk.command_word("") == "(other)"


def test_shown_joins_stderr_and_trims_newlines():
    assert rk.shown("a\n", "") == "a"
    assert rk.shown("a\n", "b\n") == "a\nb"
    assert rk.shown("", "b") == "b"


def test_tokens_counts_text_spelling_a_special_token():
    assert rk.tokens("a <|endoftext|> b") > 0


def test_active_needs_a_config_within_its_date_and_no_off_file(tmp_path):
    day = datetime.date(2026, 10, 8)
    assert not rk.active(tmp_path, day)
    (tmp_path / "config.json").write_text('{"until": "2026-10-08"}')
    assert rk.active(tmp_path, day)
    assert not rk.active(tmp_path, datetime.date(2026, 10, 9))
    (tmp_path / "OFF").touch()
    assert not rk.active(tmp_path, day)
    (tmp_path / "OFF").unlink()
    (tmp_path / "config.json").write_text("not json")
    assert not rk.active(tmp_path, day)


def fake_rtk(calls, rewrite="rtk git status", rc=0, out="short", err="", version_rc=0,
             rewrite_rc=0):
    """Stands in for `run_isolated`; `calls` gets every run after the version check."""
    def run(argv, cwd, env):
        if argv[1] == "--version":
            return version_rc, "", ""
        calls.append((argv, cwd, env))
        if argv[1] == "rewrite":
            return (rewrite_rc, rewrite + "\n", "") if rewrite else (1, "", "")
        return rc, out, err
    return run


@pytest.fixture
def base(tmp_path):
    (tmp_path / "base").mkdir()
    return tmp_path / "base"


def test_measure_runs_the_rtk_form_in_the_commands_directory(tmp_path, base, monkeypatch):
    calls = []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls, out="M a.py\n", err="warn"))
    row = rk.measure(f"cd {tmp_path} && git status 2>&1", "/nonexistent", base)
    assert set(row) == {"status", "rtk_exit", "rtk_chars", "rtk_tokens", "rtk_ms"}
    assert row["status"] == "ran" and row["rtk_exit"] == 0
    assert row["rtk_chars"] == len("M a.py\nwarn")
    rtk = str(base / "rtk")
    assert calls[0][0] == [rtk, "rewrite", "git status"]
    assert calls[1][0] == [rtk, "git", "status"] and calls[1][1] == str(tmp_path)
    env = calls[1][2]
    assert env["HOME"] != str(Path.home()) and not Path(env["HOME"]).exists()
    assert env["GIT_OPTIONAL_LOCKS"] == "0" and env["RTK_TELEMETRY_DISABLED"] == "1"


def test_measure_drops_stderr_when_the_command_sent_it_to_dev_null(tmp_path, base,
                                                                   monkeypatch):
    monkeypatch.setattr(rk, "run_isolated", fake_rtk([], out="o", err="e"))
    assert rk.measure("git status 2>/dev/null", str(tmp_path), base)["rtk_chars"] == 1


def test_measure_reports_a_broken_install_as_an_error_not_as_no_rtk_form(tmp_path, base,
                                                                         monkeypatch):
    calls = []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls, version_rc=127))
    assert rk.measure("git status", str(tmp_path), base) == \
        {"status": "rtk_error", "error": "preflight exit 127"}
    assert calls == []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls, rewrite_rc=2))
    assert rk.measure("git status", str(tmp_path), base) == \
        {"status": "rtk_error", "error": "rewrite exit 2"}


def test_measure_keeps_rtks_exit_code_and_does_not_read_it_as_failure(tmp_path, base,
                                                                      monkeypatch):
    monkeypatch.setattr(rk, "run_isolated", fake_rtk([], rc=1, out="diff text"))
    row = rk.measure("git diff --exit-code", str(tmp_path), base)
    assert row["status"] == "ran" and row["rtk_exit"] == 1 and row["rtk_tokens"] > 0


def test_measure_skips_when_every_worker_slot_is_taken(tmp_path, base, monkeypatch):
    calls = []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls))
    held = [rk.take_slot(base) for _ in range(rk.MAX_WORKERS)]
    assert None not in held and rk.take_slot(base) is None
    assert rk.measure("git status", str(tmp_path), base) == {"status": "skip_busy"}
    assert calls == []
    for fd in held:
        rk.os.close(fd)
    assert rk.measure("git status", str(tmp_path), base)["status"] == "ran"


def test_measure_leaves_no_temporary_home_and_sweeps_a_dead_workers(tmp_path, base,
                                                                   monkeypatch):
    root = tmp_path / "homes"
    root.mkdir()
    monkeypatch.setattr(rk, "home_root", lambda base: root)
    dead, live = root / "rtk-shadow-dead", root / "rtk-shadow-live"
    for d in (dead, live):
        d.mkdir()
        (d / "history.db").write_text("grep hunter2 f")
    old = rk.time.time() - rk.STALE_HOME_SECONDS - 5
    rk.os.utime(dead, (old, old))
    monkeypatch.setattr(rk, "run_isolated", fake_rtk([]))
    assert rk.measure("git status", str(tmp_path), base)["status"] == "ran"
    assert [p.name for p in root.iterdir()] == ["rtk-shadow-live"]


def test_measure_skips_and_never_runs_what_is_not_plain_and_read_only(tmp_path, base,
                                                                      monkeypatch):
    calls = []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls))
    cwd = str(tmp_path)
    assert rk.measure("git status & ls", cwd, base) == {"status": "skip_compound"}
    assert rk.measure("(git status)", cwd, base) == {"status": "skip_compound"}
    assert rk.measure("FOO=1 git status && ls", cwd, base) == {"status": "skip_compound"}
    assert rk.measure("git status |& head", cwd, base) == {"status": "skip_compound"}
    assert rk.measure("git status;; ls", cwd, base) == {"status": "skip_compound"}
    assert rk.measure("git status &&", cwd, base) == {"status": "skip_compound"}
    assert rk.measure("ls *.py | head", cwd, base) == {"status": "skip_expansion"}
    assert rk.measure('git status && echo "$HOME"', cwd, base) == \
        {"status": "skip_expansion"}
    assert rk.measure("cat $(ls) | head", cwd, base) == {"status": "skip_expansion"}
    assert rk.measure("git log -3 > out.txt", cwd, base) == {"status": "skip_redirect"}
    assert rk.measure("git status && sort < f", cwd, base) == {"status": "skip_redirect"}
    assert rk.measure("git status 2>err.txt | head", cwd, base) == \
        {"status": "skip_redirect"}
    # A redirect in the middle of a command is not one of the two forms taken.
    assert rk.measure("git status 2>&1 -s | head", cwd, base) == \
        {"status": "skip_compound"}
    for chain in ("git status && rm -rf x", "git status; git push", "ls | tee out",
                  "git status && cd sub && ls", "git status && cd /tmp && ls",
                  "git log | sort -o out",
                  "git log | sort out", "git log | uniq a b", "git log | sort --output=o",
                  "ls | xargs rm", "git status && rtk git diff", "echo hi && python3 x.py",
                  "git status | find . -delete", "git log | sort -T /tmp"):
        assert rk.measure(chain, cwd, base) == {"status": "skip_not_listed"}, chain
    # One command stays on the single-command list, with or without a new line after it.
    assert rk.measure("echo hi", cwd, base) == {"status": "skip_not_listed"}
    assert rk.measure("echo hi\n", cwd, base) == {"status": "skip_not_listed"}
    # bash splits words on space and tab only: these ran some other program.
    assert rk.measure("ls | cat\x0b", cwd, base) == {"status": "skip_compound"}
    assert rk.measure("\xa0ls | cat", cwd, base) == {"status": "skip_not_listed"}
    assert rk.measure("ls | cat\x1f", cwd, base) == {"status": "skip_not_listed"}
    assert rk.measure("rm -rf x", cwd, base) == {"status": "skip_not_listed"}
    assert rk.measure("git push", cwd, base) == {"status": "skip_not_listed"}
    assert rk.measure("tree -ao out", cwd, base) == {"status": "skip_not_listed"}
    assert rk.measure("git status", "/nonexistent", base) == {"status": "skip_no_cwd"}
    assert calls == []


def test_measure_refuses_a_rewrite_that_is_not_a_plain_rtk_command(tmp_path, base,
                                                                   monkeypatch):
    for rewrite in (None, "git status", "rtk", "rtk git status && rm -rf x",
                    "rtk git status | cat", "FOO=1 rtk git status"):
        calls = []
        monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls, rewrite=rewrite))
        assert rk.measure("git status", str(tmp_path), base) == \
            {"status": "skip_no_rtk_form"}, rewrite
        assert len(calls) == 1


def test_measure_reports_a_timeout(tmp_path, base, monkeypatch):
    def run(argv, cwd, env):
        if argv[1] == "--version":
            return 0, "", ""
        if argv[1] == "rewrite":
            return 0, "rtk tail -f x\n", ""
        raise subprocess.TimeoutExpired(argv, 1)
    monkeypatch.setattr(rk, "run_isolated", run)
    assert rk.measure("tail -f x", str(tmp_path), base) == {"status": "rtk_timeout"}


def run_hook(tmp_path, monkeypatch, event, detach=True):
    monkeypatch.setattr(rk, "detach", lambda: detach)
    monkeypatch.setattr(rk, "active", lambda base, today: True)
    monkeypatch.setattr(rk.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    try:
        rk.hook(tmp_path)
    except SystemExit:
        pass
    path = tmp_path / "shadow.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_hook_row_holds_sizes_and_a_word_and_nothing_from_the_command(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "run_isolated", fake_rtk([], rewrite="rtk grep -rn hunter2 ."))
    rows = run_hook(tmp_path, monkeypatch, {
        "tool_name": "Bash", "cwd": str(tmp_path),
        "tool_input": {"command": "grep -rn hunter2 ."},
        "tool_response": {"stdout": "a.py:1:password = hunter2\n", "stderr": ""}})
    assert len(rows) == 1 and rows[0]["status"] == "ran" and rows[0]["word"] == "grep"
    assert set(rows[0]) == {"ts", "word", "raw_chars", "raw_tokens", "status",
                            "rtk_exit", "rtk_chars", "rtk_tokens", "rtk_ms"}
    assert "hunter2" not in (tmp_path / "shadow.jsonl").read_text()
    assert (tmp_path / "shadow.jsonl").stat().st_mode & 0o777 == 0o600


def test_hook_logs_a_skipped_command_and_ignores_other_tools(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls))
    bash = {"tool_name": "Bash", "cwd": str(tmp_path),
            "tool_input": {"command": "pytest -q"},
            "tool_response": {"stdout": "3 passed", "stderr": ""}}
    rows = run_hook(tmp_path, monkeypatch, bash)
    assert rows[0]["status"] == "skip_not_listed" and rows[0]["raw_tokens"] > 0
    rows = run_hook(tmp_path, monkeypatch, {**bash, "tool_response": {
        "stdout": "x", "stderr": "", "interrupted": True}, "tool_input": {"command": "ls"}})
    assert rows[-1]["status"] == "skip_interrupted"
    for event in ({**bash, "tool_name": "Read"}, {**bash, "tool_response": "text"},
                  {**bash, "tool_input": {}}, {**bash, "tool_input": "ls"}, [1], "x", None):
        assert len(run_hook(tmp_path, monkeypatch, event)) == 2
    assert calls == []


def test_hook_does_not_measure_a_result_it_cannot_compare(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls))
    ok = {"stdout": "x", "stderr": ""}
    cases = [
        ({"command": "ls", "run_in_background": True}, ok, "skip_background"),
        ({"command": "ls"}, {**ok, "backgroundTaskId": "b1"}, "skip_background"),
        ({"command": "ls"}, {**ok, "stdout": "x" * (rk.SHOWN_CAP + 1)}, "skip_over_cap"),
        ({"command": "ls"}, {**ok, "persistedOutputPath": "/p"}, "skip_over_cap"),
        ({"command": "ls"}, {**ok, "interrupted": True}, "skip_interrupted"),
    ]
    for tool_input, resp, status in cases:
        rows = run_hook(tmp_path, monkeypatch, {
            "tool_name": "Bash", "cwd": str(tmp_path), "tool_input": tool_input,
            "tool_response": resp})
        assert rows[-1]["status"] == status and rows[-1]["raw_tokens"] > 0
    assert calls == []


def test_a_result_saved_to_a_file_is_sized_at_its_preview(tmp_path, monkeypatch):
    """The hook is handed 30,000 characters of it; the session was shown about 2,000."""
    monkeypatch.setattr(rk, "run_isolated", fake_rtk([]))
    event = {"tool_name": "Bash", "cwd": str(tmp_path), "tool_input": {"command": "cat f"},
             "tool_response": {"stdout": "word " * 6000, "stderr": "",
                               "persistedOutputPath": "/p"}}
    row = run_hook(tmp_path, monkeypatch, event)[-1]
    assert row["status"] == "skip_over_cap" and row["raw_chars"] == 30_000
    assert row["raw_tokens"] == rk.tokens(("word " * 6000)[:rk.PERSIST_PREVIEW])


def test_hook_mode_never_fails_the_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rk, "active", lambda base, today: True)
    for stdin in ("not json", "[1]", '{"tool_name": "Bash", "tool_input": "ls"}', ""):
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        assert rk.main(["hook", str(tmp_path)]) == 0
    assert rk.main(["hook"]) == 0 and rk.main(["hook", "a", "b"]) == 0
    monkeypatch.setattr(rk, "detach", lambda: (_ for _ in ()).throw(OSError("fork")))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "tool_name": "Bash", "tool_input": {"command": "ls"},
        "tool_response": {"stdout": "x", "stderr": ""}})))
    assert rk.main(["hook", str(tmp_path)]) == 0
    assert capsys.readouterr() == ("", "")


def test_hook_parent_returns_without_writing(tmp_path, monkeypatch):
    rows = run_hook(tmp_path, monkeypatch, {
        "tool_name": "Bash", "cwd": str(tmp_path), "tool_input": {"command": "ls"},
        "tool_response": {"stdout": "x", "stderr": ""}}, detach=False)
    assert rows == []


def test_summarize_counts_the_saving_with_and_without_non_zero_exits():
    """A run that failed with a one-line message must not read as a saving."""
    rows = [
        {"ts": 1, "word": "grep", "raw_tokens": 100, "status": "ran", "rtk_exit": 0,
         "rtk_tokens": 40},
        {"ts": 1, "word": "tail", "raw_tokens": 400, "status": "ran", "rtk_exit": 1,
         "rtk_tokens": 8},
        {"ts": 1, "word": "ls", "raw_tokens": 50, "status": "ran", "rtk_exit": 0,
         "rtk_tokens": 0},
        {"ts": 1, "word": "cat", "raw_tokens": 50, "status": "ran", "rtk_exit": 137,
         "rtk_tokens": 2},
        {"ts": 1, "word": "pytest", "raw_tokens": 400, "status": "skip_not_listed"},
        {"ts": 1, "status": "worker_error", "error": "X"},
    ]
    s = rk.summarize(rows)
    assert (s["calls"], s["bash_tokens"], s["ran"], s["ran_raw"]) == (5, 1000, 4, 600)
    assert (s["suspect"], s["nonzero"], s["crashed"], s["worker_errors"]) == (1, 2, 1, 1)
    assert (s["saved"], s["saved_any"]) == (60, 452)
    text = rk.render(rows)
    all_row = next(x for x in text.splitlines() if x.startswith("ALL"))
    # calls, errors, bash tokens, covered, saved on covered, any exit, of logged Bash, of bill
    assert all_row.split() == ["ALL", "5", "1", "1,000", "60.00%", "10.00%", "75.33%",
                               "6.00%", "0.97%"]
    assert [x.split()[0] for x in text.splitlines() if x.startswith("  ")] == ["grep"]


def test_an_empty_rtk_output_for_an_empty_result_is_not_suspect():
    assert not rk.suspect({"raw_tokens": 3, "rtk_tokens": 0})
    assert rk.suspect({"raw_tokens": 21, "rtk_tokens": 0})


needs_unshare = pytest.mark.skipif(
    subprocess.run(["unshare", "-rn", "--pid", "--fork", "--kill-child", "--", "true"],
                   capture_output=True).returncode != 0,
    reason="user, network and pid namespaces are not available here")


@needs_unshare
def test_run_isolated_returns_output_and_exit_code(tmp_path):
    rc, out, err = rk.run_isolated(["sh", "-c", "echo o; echo e >&2; exit 3"],
                                   str(tmp_path), {"PATH": rk.os.environ["PATH"]})
    assert (rc, out, err) == (3, "o\n", "e\n")


@needs_unshare
def test_run_isolated_has_no_network_interface_but_loopback(tmp_path):
    probe = "import socket; print(*[name for _, name in socket.if_nameindex()])"
    rc, out, _ = rk.run_isolated([sys.executable, "-c", probe], str(tmp_path),
                                 {"PATH": rk.os.environ["PATH"]})
    assert rc == 0 and out.split() == ["lo"]


@needs_unshare
def test_run_isolated_kills_a_child_that_left_the_process_group(tmp_path, monkeypatch):
    """A daemon started by the wrapped command (`setsid`, holding the output pipe)
    must neither hang the worker nor outlive the run."""
    monkeypatch.setattr(rk, "RUN_TIMEOUT", 1)
    marker = tmp_path / "alive"
    script = f"setsid sh -c 'sleep 4; touch {marker}' & sleep 30"
    started = rk.time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        rk.run_isolated(["sh", "-c", script], str(tmp_path), {"PATH": rk.os.environ["PATH"]})
    assert rk.time.monotonic() - started < 1 + rk.REAP_TIMEOUT + 1
    rk.time.sleep(4.5)
    assert not marker.exists()


def test_render_handles_no_rows():
    assert "n/a" in rk.render([])


def test_main_rejects_bad_arguments(capsys):
    assert rk.main([]) == 2 and rk.main(["run", "x"]) == 2 and rk.main(["report"]) == 2


@pytest.mark.parametrize("cmd", ["git log -3", "cd /repo && git log -3 2>&1"])
def test_peel_then_simple_words_round_trip(cmd):
    assert rk.simple_words(rk.peel(cmd)[0]) == ["git", "log", "-3"]


def test_parse_chain_splits_on_operators_outside_quotes_and_keeps_stderr_redirects():
    assert rk.parse_chain("git status && git diff") == \
        [("", ["git", "status"], ""), ("&&", ["git", "diff"], "")]
    assert rk.parse_chain("grep -n 'a|b; c && d' f 2>/dev/null || echo none") == \
        [("", ["grep", "-n", "a|b; c && d", "f"], " 2>/dev/null"),
         ("||", ["echo", "none"], "")]
    assert rk.parse_chain('git status 2>&1 | tail -5;echo "x 2>&1 y"\nls') == \
        [("", ["git", "status"], " 2>&1"), ("|", ["tail", "-5"], ""),
         (";", ["echo", "x 2>&1 y"], ""), ("\n", ["ls"], "")]
    # `2>&1` is a redirect only where a word starts.
    assert rk.parse_chain("grep x2>&1 | head") == "skip_redirect"
    assert rk.parse_chain("ls 2>&1x | head") == "skip_redirect"
    assert rk.parse_chain("grep 'open | head") == "skip_compound"
    assert rk.parse_chain("ls 2>&1&& ls 2>&1|cat") == \
        [("", ["ls"], " 2>&1"), ("&&", ["ls"], " 2>&1"), ("|", ["cat"], "")]
    assert rk.parse_chain("ls \x1f 2>&1 | cat")[0] == ("", ["ls", "\x1f"], " 2>&1")


def test_rerunnable_takes_filters_without_file_operands_and_no_cd():
    for words in (["echo", "-n", "a b"], ["pwd"], ["true"], ["wc", "-l"],
                  ["sort"], ["sort", "-rn"], ["sort", "-k2,2nr", "-t,"], ["uniq", "-c"],
                  ["git", "log", "-3"], ["head", "-5"]):
        assert rk.rerunnable(words), words
    for words in (["cd", "sub"], ["cd", "/tmp"], ["sort", "f"],
                  ["sort", "-o", "f"], ["sort", "-ro"], ["sort", "-k", "2"],
                  ["sort", "--compress-program=x"], ["uniq", "a", "b"], ["uniq", "-f1"],
                  ["tee", "f"], ["sed", "-n", "1p"], ["git", "push"]):
        assert not rk.rerunnable(words), words


def test_chain_script_is_built_from_checked_words_and_only_for_the_same_chain():
    head = ("exec 3>'/h/my exits'\n"
            'r() { "$@"; local s=$?; echo "$s" >&3; return "$s"; }\n')
    chain = rk.parse_chain("pwd && git status 2>&1 | head -3; echo 'a b'")
    assert rk.chain_script(chain, "pwd && rtk git status 2>&1 | head -3; echo 'a b'\n",
                           "/b/rtk", "/h/my exits") == \
        head + "pwd && r /b/rtk git status 2>&1 | head -3 ; echo 'a b'"
    two = rk.parse_chain("head -5 a.py && ls")
    assert rk.chain_script(two, "rtk read a.py --head-lines 5 && rtk ls", "/b/rtk",
                           "/h/my exits") == \
        head + "r /b/rtk read a.py --head-lines 5 && r /b/rtk ls"
    for rewritten in ("head -5 a.py && ls",                      # nothing in RTK form
                      "rtk read a.py && rtk ls && rm -rf x",      # one command more
                      "rtk read a.py; rtk ls",                    # another operator
                      "rtk read a.py 2>&1 && rtk ls",             # another redirect
                      "rtk read a.py && rm -rf x",                # a command swapped
                      "rtk read a.py && rtk",                     # RTK with nothing to run
                      "rtk read a.py && rtk ls > out",            # a redirect to a file
                      "rtk read $(id) && rtk ls",                 # an expansion
                      "rtk read a.py && FOO=1 rtk ls", ""):
        assert rk.chain_script(two, rewritten, "/b/rtk", "/h/e") is None, rewritten


def test_measure_runs_a_chain_once_under_bash_with_its_redirects(tmp_path, base, monkeypatch):
    calls = []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(
        calls, rewrite="rtk git status 2>/dev/null | head -3", out="M a.py", err="warn"))
    row = rk.measure(f"cd {tmp_path} && git status 2>/dev/null | head -3", "/nonexistent",
                     base)
    assert set(row) == {"status", "rtk_exit", "rtk_chars", "rtk_tokens", "rtk_ms", "segs",
                        "seg_exits"}
    # The stand-in ran no script, so no RTK command left an exit code.
    assert row["status"] == "ran" and row["segs"] == 2 and row["seg_exits"] == []
    # stderr is whatever the script let through: the chain's own redirect did the dropping.
    assert row["rtk_chars"] == len("M a.py\nwarn")
    rtk = str(base / "rtk")
    assert calls[0][0] == [rtk, "rewrite", "git status 2>/dev/null | head -3"]
    assert calls[1][0][:4] == ["/bin/bash", "--noprofile", "--norc", "-c"]
    script = calls[1][0][4].splitlines()
    assert script[0] == f"exec 3>{calls[1][2]['HOME']}/rtk-shadow-exits"
    assert script[2] == f"r {rtk} git status 2>/dev/null | head -3"
    assert calls[1][1] == str(tmp_path) and len(calls) == 2
    assert calls[1][2]["TMPDIR"] == calls[1][2]["HOME"]


def test_measure_takes_one_listed_command_with_a_new_line_after_it(tmp_path, base,
                                                                   monkeypatch):
    calls = []
    monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls, rewrite="rtk git status"))
    row = rk.measure("git status\n", str(tmp_path), base)
    assert row["status"] == "ran" and row["segs"] == 1


def test_measure_never_runs_a_chain_rtk_rewrote_into_something_else(tmp_path, base,
                                                                    monkeypatch):
    for rewrite in (None, "git status && ls", "rtk git status && rm -rf x",
                    "rtk git status; rtk ls", "rtk git status && rtk ls && id"):
        calls = []
        monkeypatch.setattr(rk, "run_isolated", fake_rtk(calls, rewrite=rewrite))
        assert rk.measure("git status && ls", str(tmp_path), base) == \
            {"status": "skip_no_rtk_form"}, rewrite
        assert len(calls) == 1


@needs_unshare
def test_a_chain_runs_for_real_under_the_isolation(tmp_path):
    (tmp_path / "f").write_text("b\na\nb\n")
    chain = rk.parse_chain("echo 'x; y' && cat f | sort | uniq -c 2>&1; cat nope 2>/dev/null")
    assert all(rk.rerunnable(w) for _, w, _ in chain)
    # `rtk` stands in as /bin/cat here: only the first `cat` is given an RTK form.
    exits = str(tmp_path / "my exits")
    script = rk.chain_script(
        chain, "echo 'x; y' && rtk f | sort | uniq -c 2>&1; rtk nope 2>/dev/null",
        "/bin/cat", exits)
    rc, out, err = rk.run_isolated(["/bin/bash", "--noprofile", "--norc", "-c", script],
                                   str(tmp_path), {"PATH": "/usr/bin:/bin"})
    assert (rc, err) == (1, "")
    assert out.split() == ["x;", "y", "1", "a", "2", "b"]
    # The first RTK command is in front of a pipe, the second is the chain's last.
    assert rk.read_exits(exits) == [0, 1]
    assert rk.read_exits(str(tmp_path / "none")) == []


def test_a_chain_counts_as_saved_only_when_every_rtk_command_exited_zero():
    def row(**kw):
        return {"ts": 0, "word": "echo", "status": "ran", "raw_tokens": 2000,
                "rtk_tokens": 10, "rtk_exit": 0, "segs": 2, **kw}
    good = row(seg_exits=[0, rk.PIPE_CLOSED])
    # `rtk git log | head`: RTK failed with a one-line message, `head` exited 0.
    hidden = row(seg_exits=[128])
    never = row(seg_exits=[])
    killed = row(seg_exits=[0, 137])
    assert [rk.failed(r) for r in (good, hidden, never, killed)] == [False, True, True, True]
    assert [rk.crashed(r) for r in (good, hidden, never, killed)] == \
        [False, True, False, True]
    s = rk.summarize([good, hidden, never, killed])
    assert (s["saved"], s["nonzero"], s["crashed"]) == (1990, 3, 2)
    # Face value for a non-zero exit; a crash counts as no saving in both columns.
    assert s["saved_any"] == 2 * 1990
    single = {"ts": 0, "word": "echo", "status": "ran", "raw_tokens": 2000,
              "rtk_tokens": 10, "rtk_exit": rk.PIPE_CLOSED}
    assert rk.failed(single) and rk.crashed(single)


def test_render_files_a_chain_under_its_own_label_not_its_first_command():
    rows = [{"ts": 0, "word": "echo", "status": "ran", "raw_tokens": 100, "rtk_tokens": 40,
             "rtk_exit": 0, "segs": 3, "seg_exits": [0]},
            {"ts": 0, "word": "git log", "status": "ran", "raw_tokens": 50,
             "rtk_tokens": 50, "rtk_exit": 0}]
    text = rk.render(rows)
    assert "(chain)      n=    1" in text and "git log      n=    1" in text
    assert "  echo " not in text


def test_a_rebuilt_chain_does_what_the_original_line_did(tmp_path):
    """Random chains: the script built from checked words against the line as typed."""
    import random
    rng = random.Random(7)
    (tmp_path / "f").write_text("b\na\nb\n")
    cmds = ["echo a", "echo 'b  c'", 'echo "d;e"', "true", "cat f", "cat nope", "wc -l",
            "sort -r", "uniq -c", "echo x 2>&1", "cat nope 2>/dev/null", "cat nope 2>&1"]
    seps = [" && ", " || ", "; ", " | ", "\n", "&&", "|", " ;"]
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    taken = 0
    for _ in range(150):
        parts = [rng.choice(cmds) for _ in range(rng.randint(2, 5))]
        line = parts[0] + "".join(rng.choice(seps) + c for c in parts[1:])
        chain = rk.parse_chain(line)
        assert not isinstance(chain, str), line
        # `rtk` stands in as /bin/echo for every `echo`, so the output must not change.
        rewritten = "".join(
            f"{op} {'rtk' if w[0] == 'echo' else w[0]} "
            f"{' '.join(map(rk.shlex.quote, w[1:]))}{redirect}" for op, w, redirect in chain)
        script = rk.chain_script(chain, rewritten, "/bin/echo", str(tmp_path / "exits"))
        if script is None:
            continue
        taken += 1
        want, got = (subprocess.run(["/bin/bash", "-c", text], cwd=tmp_path, env=env,
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True)
                     for text in (line, script))
        assert (got.returncode, got.stdout, got.stderr) == \
            (want.returncode, want.stdout, want.stderr), line
    assert taken > 100
