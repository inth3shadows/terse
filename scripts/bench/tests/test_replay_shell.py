"""Tests for the shell-output replay (scripts/bench/replay_shell.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import replay_shell as sh


def test_plain_command_strips_a_leading_cd_and_a_stderr_redirect():
    assert sh.plain_command("git status") == "git status"
    assert sh.plain_command("cd /repo && git log -3 2>&1") == "git log -3"
    assert sh.plain_command("cd 'my dir' && grep -rn x src 2>/dev/null") == "grep -rn x src"


def test_plain_command_rejects_pipelines_and_compound_commands():
    for cmd in ("grep x f | head -5", "git status && git log", "a; b", "cat f > g",
                "echo $(date)", "echo `date`", "sort < f", "make &", "a\nb",
                "cd /x && cd /y && ls", ""):
        assert sh.plain_command(cmd) is None, cmd


def test_command_word_never_carries_more_than_a_program_name():
    assert sh.command_word("cd /repo && git diff HEAD~1") == "git diff"
    assert sh.command_word("/usr/bin/sed -n 1,5p f") == "sed"
    assert sh.command_word("TOKEN=abc123 curl https://x") == "(other)"
    assert sh.command_word("'my tool' --flag") == "(other)"
    assert sh.command_word("git -C /x status") == "git"
    assert sh.command_word("   ") == "(empty)"


def test_method_admits_only_filters_checked_against_the_live_command():
    assert sh.method("uv run rtk pytest tests -q") == "pytest"
    assert sh.method("FOO=1 rtk pytest -q") == "pytest"
    assert sh.method("rtk read foo.py --head-lines 50") == "read"
    assert sh.method("rtk read -n foo.py") is None and sh.method("rtk read f --level=minimal") is None
    # Pipe mode is not what the live command prints for these: they cannot be replayed.
    for form in ("rtk grep -rn foo src/", "rtk git log --oneline -5", "rtk git diff --stat",
                 "rtk git status", "rtk find . -name x", "rtk ruff check .",
                 "rtk git show abc", "rtk ls -la", "rtk 'unterminated"):
        assert sh.method(form) is None, form


def test_classify_separates_never_seen_unchanged_measured_and_unmeasured():
    assert sh.classify(None, None) == ("not_rewritten", None)
    assert sh.classify("rtk read f", "rtk read f") == ("unchanged", "read")
    assert sh.classify("rtk pytest -q", "rtk pytest -q") == ("measured", "pytest")
    assert sh.classify("rtk grep x f", "rtk grep x f") == ("unmeasured", None)
    assert sh.classify("rtk pytest -q | tail", None) == ("unmeasured", None)


def test_summarize_reports_reach_the_unchanged_share_and_a_ceiling():
    def row(tool="Bash", tokens=1000, cost=1000.0, saved=0.0, status="not_rewritten",
            method="", word="sed"):
        return {"tool": tool, "tokens": tokens, "cost": cost, "saved": saved,
                "status": status, "method": method, "word": word, "error": False}

    rows = [row(cost=5000.0),
            row(cost=1000.0, saved=600.0, status="measured", method="pytest", word="pytest"),
            row(cost=1000.0, status="unchanged", method="read", word="cat"),
            row(cost=3000.0, status="unmeasured", word="grep"),
            row(tool="Read", tokens=9000, cost=10000.0, status="", word="")]
    got = sh.summarize(rows, 100000.0)
    assert got["bash"] == {"results": 4, "cost": 10000.0, "share_of_bill_pct": 10.0}
    assert got["read"]["share_of_bill_pct"] == 10.0
    rtk = got["rtk"]
    assert rtk["by_status"]["not_rewritten"]["share_of_bash_pct"] == 50.0
    assert rtk["by_status"]["unchanged"] == {"results": 1, "share_of_bash_pct": 10.0,
                                             "share_of_bill_pct": 1.0}
    assert rtk["pytest_saved_pct"] == 60.0 and rtk["pytest_saved_share_of_bill_pct"] == 0.6
    # Measured plus unmeasured, as if every token went: 4,000 of 10,000. Not the cat row.
    assert rtk["ceiling_share_of_bash_pct"] == 40.0 and rtk["ceiling_share_of_bill_pct"] == 4.0
    assert got["rtk_takes_by_command"]["sed"] == {"takes_pct": 0.0, "unchanged_pct": 0.0}
    assert got["rtk_takes_by_command"]["cat"] == {"takes_pct": 100.0, "unchanged_pct": 100.0}
    assert got["size_shares"]["Read"]["over 8,000"] == 100.0
    text = sh.render(got)
    assert "ceiling" in text and "was not measured" in text


def test_render_survives_a_command_nothing_was_billed_for():
    row = {"tool": "Bash", "tokens": 5, "cost": 0.0, "saved": 0.0, "status": "measured",
           "method": "pytest", "word": "pytest", "error": False}
    text = sh.render(sh.summarize([row], 0.0))
    assert "pytest" in text and " -" in text
