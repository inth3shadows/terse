"""Offline tests for checkers.py -- no network, no claude -p, no MCP."""
from __future__ import annotations

import checkers


def test_exact_match_pass_plain():
    ok, detail = checkers.check_exact_match("kb-mcp", {"expected": "kb-mcp"})
    assert ok
    assert "kb-mcp" in detail


def test_exact_match_pass_with_noise():
    # backticks, surrounding whitespace, different case -- all normalized away
    ok, _ = checkers.check_exact_match("  `KB-MCP`\n", {"expected": "kb-mcp"})
    assert ok


def test_exact_match_rejects_when_buried_in_a_sentence():
    # The task prompts demand "ONLY the answer, no explanation" -- a model
    # that ignores that and answers in a sentence must FAIL, not pass on a
    # lenient buried-substring match.
    ok, _ = checkers.check_exact_match("The hostname is kb-mcp.", {"expected": "kb-mcp"})
    assert not ok


def test_exact_match_rejects_substring_collision():
    # "kb-mcp" must not match inside "kb-mcp-reviewer" -- whole-word only.
    ok, _ = checkers.check_exact_match("kb-mcp-reviewer", {"expected": "kb-mcp"})
    assert not ok


def test_exact_match_rejects_negation():
    # "not kb-mcp" contains "kb-mcp" as a bare whole word -- a checker doing
    # whole-word substring search (rather than whole-answer equality) would
    # wrongly PASS this.
    ok, _ = checkers.check_exact_match("not kb-mcp", {"expected": "kb-mcp"})
    assert not ok


def test_exact_match_rejects_a_list_of_candidates():
    ok, _ = checkers.check_exact_match("kb-mcp, caddy-edge, or router-1",
                                        {"expected": "kb-mcp"})
    assert not ok


def test_exact_match_fail_wrong_answer():
    ok, _ = checkers.check_exact_match("caddy-edge", {"expected": "kb-mcp"})
    assert not ok


def test_regex_match_extracts_and_compares_numerically():
    ok, detail = checkers.check_regex_match(
        "The multiplier is 1.25.", {"pattern": r"(\d+\.\d+)", "expected": "1.25"})
    assert ok
    assert "1.25" in detail


def test_regex_match_numeric_tolerance():
    # 1.250 vs 1.25 -- same float, must still pass.
    ok, _ = checkers.check_regex_match(
        "1.250", {"pattern": r"(\d+\.\d+)", "expected": "1.25"})
    assert ok


def test_regex_match_wrong_number_fails():
    ok, _ = checkers.check_regex_match(
        "2.0", {"pattern": r"(\d+\.\d+)", "expected": "1.25"})
    assert not ok


def test_regex_match_no_match_at_all():
    ok, detail = checkers.check_regex_match(
        "no numbers here", {"pattern": r"(\d+\.\d+)", "expected": "1.25"})
    assert not ok
    assert "no match" in detail


def test_command_check_pass(tmp_path):
    (tmp_path / "ok.py").write_text("import sys; sys.exit(0)\n")
    ok, detail = checkers.check_command(
        "", {"command": "python3 ok.py", "expect_exit": 0}, tmp_path)
    assert ok
    assert "exit=0" in detail


def test_command_check_fail_nonzero_exit(tmp_path):
    (tmp_path / "bad.py").write_text("import sys; sys.exit(1)\n")
    ok, detail = checkers.check_command(
        "", {"command": "python3 bad.py", "expect_exit": 0}, tmp_path)
    assert not ok
    assert "exit=1" in detail


def test_command_check_timeout(tmp_path):
    ok, detail = checkers.check_command(
        "", {"command": "sleep 5", "timeout": 0.2}, tmp_path)
    assert not ok
    assert "timed out" in detail


def test_run_check_dispatches_on_type(tmp_path):
    task = {"id": "t1", "check": {"type": "exact_match", "expected": "kb-mcp"}}
    ok, _ = checkers.run_check(task, "kb-mcp", tmp_path)
    assert ok


def test_run_check_unknown_type_raises(tmp_path):
    task = {"id": "t1", "check": {"type": "nonsense"}}
    try:
        checkers.run_check(task, "anything", tmp_path)
    except ValueError as e:
        assert "unknown check type" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_command_check_protected_file_hash_mismatch_fails_before_running_pytest(tmp_path):
    import hashlib
    original = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    (tmp_path / "test_calc.py").write_text(original)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")  # a correct fix
    # tamper with the protected test file -- e.g. a model "fixing" the test
    # instead of the bug
    (tmp_path / "test_calc.py").write_text(original.replace("== 5", "== 999"))
    check = {"type": "command", "command": "pytest -q", "expect_exit": 0,
             "protect_file": "test_calc.py",
             "protect_file_sha256": hashlib.sha256(original.encode()).hexdigest()}
    ok, detail = checkers.check_command("", check, tmp_path)
    assert not ok
    assert "protected" in detail.lower() or "modified" in detail.lower()


def test_command_check_protected_file_hash_match_runs_pytest_normally(tmp_path):
    import hashlib
    original = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    (tmp_path / "test_calc.py").write_text(original)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    check = {"type": "command", "command": "pytest -q", "expect_exit": 0,
             "protect_file": "test_calc.py",
             "protect_file_sha256": hashlib.sha256(original.encode()).hexdigest()}
    ok, detail = checkers.check_command("", check, tmp_path)
    assert ok, detail


def test_control_task_fixture_end_to_end(tmp_path):
    """The exact fixture shipped in tasks.json's control task: an unfixed
    calc.py must fail pytest, and the fix must pass it -- proves the checker
    actually exercises the test, not just parses text."""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    check = {"type": "command", "command": "pytest -q", "expect_exit": 0}
    ok, _ = checkers.check_command("", check, tmp_path)
    assert not ok, "buggy calc.py must NOT pass yet"

    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    ok, detail = checkers.check_command("", check, tmp_path)
    assert ok, detail


def test_whole_answer_markdown_emphasis_is_stripped_for_every_arm():
    """The minimal output style (B/C) bolds the answer; A's unstyled answer does not.
    Without this a correct B/C answer failed: a bias against the operator's setup."""
    chk = {"expected": "2026-06-27"}
    for got in ("**2026-06-27**", "__2026-06-27__", "*2026-06-27*", "`**2026-06-27**`",
                " **2026-06-27** \n", "2026-06-27"):
        assert checkers.check_exact_match(got, chk)[0], got


def test_partial_emphasis_does_not_rescue_a_wrong_answer():
    chk = {"expected": "kb-mcp"}
    for got in ("not **kb-mcp**", "**kb-mcp** or caddy", "**not kb-mcp**"):
        assert not checkers.check_exact_match(got, chk)[0], got


def test_emphasis_plus_anything_else_still_fails():
    chk = {"expected": "kb-mcp"}
    for got in ("**kb-mcp**.", "**kb-mcp**\nverified", "*kb-mcp**"):
        assert not checkers.check_exact_match(got, chk)[0], got


def test_no_task_expects_a_value_that_emphasis_stripping_would_alter():
    """`_norm` strips whole-value `_x_`/`*x*` from the EXPECTED value too, so a task whose
    answer is `__init__` would accept `init`. Refuse such a task rather than score it wrongly."""
    import json
    from pathlib import Path
    doc = json.loads((Path(checkers.__file__).parent / "tasks.json").read_text())
    for task in doc["tasks"]:
        exp = (task.get("check") or {}).get("expected")
        if isinstance(exp, str):
            assert checkers._norm(exp) == exp.strip().lower(), task["id"]
