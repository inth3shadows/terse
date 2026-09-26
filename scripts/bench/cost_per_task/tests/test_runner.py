"""Offline tests for runner.py's pure pieces (command building, JSON output
parsing, transcript lookup, workdir management). NEVER invokes `claude -p` --
that is the one thing this test module must not do."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import runner as r


def _write_transcript(path, records):
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _usage_record(req_id="r1", msg_id="m1", input_tokens=10, output_tokens=1):
    return {"type": "assistant", "requestId": req_id, "version": "2.1.283",
            "message": {"id": msg_id, "model": "claude-haiku-4-5", "usage": {
                "input_tokens": input_tokens, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0, "output_tokens": output_tokens}}}


def _base_task(**overrides):
    task = {"id": "t1", "prompt": "hi", "workdir": "scratch", "timeout": 300,
            "check": {"type": "exact_match", "expected": "ok"}}
    task.update(overrides)
    return task


def _fake_process(returncode=0, stdout="", stderr="", timed_out=False):
    """A `_run_claude_process`-shaped fake: (returncode, stdout, stderr,
    timed_out). `_run_claude_process` itself is what actually launches the
    subprocess/handles the timeout+process-group kill now (plan F5) -- it is
    tested directly (against a REAL child process tree) further down;
    run_one's own tests fake it out at this boundary instead of
    subprocess.run/Popen, matching the new call shape."""
    def _fn(cmd, *, cwd, env, timeout):
        return returncode, stdout, stderr, timed_out
    return _fn


def _run_one(monkeypatch, tmp_path, *, run_side_effect, transcript_records=None,
             session_id_holder=None, task=None, **run_one_kwargs):
    """Call runner.run_one with `_run_claude_process` replaced by
    `run_side_effect` (a `(cmd, *, cwd, env, timeout) -> (returncode, stdout,
    stderr, timed_out)` callable -- see `_fake_process`) and, if given, a
    transcript pre-written under tmp_path/search_root using the session_id
    run_one generates internally (captured via monkeypatching uuid.uuid4 to
    a fixed value so the test can name the transcript file ahead of time)."""
    import uuid as uuid_mod
    fixed_session = "11111111-1111-1111-1111-111111111111"
    ids = iter([fixed_session, "22222222-2222-2222-2222-222222222222"])
    monkeypatch.setattr(uuid_mod, "uuid4", lambda: next(ids))

    search_root = tmp_path / "projects"
    proj_dir = search_root / "-fake-project"
    proj_dir.mkdir(parents=True)
    if transcript_records is not None:
        _write_transcript(proj_dir / f"{fixed_session}.jsonl", transcript_records)

    monkeypatch.setattr(r, "_run_claude_process", run_side_effect)
    out_path = tmp_path / "out.jsonl"
    with out_path.open("a") as out_fh:
        return r.run_one(task=task or _base_task(), arm="B", rep=1, model="claude-haiku-4-5",
                          arm_config_path=Path("/tmp/cfg.json"), workdir=tmp_path,
                          setting_sources=None, out_fh=out_fh,
                          transcript_search_root=search_root, **run_one_kwargs)


def test_build_claude_command_basic_shape():
    cmd = r.build_claude_command(prompt="hello", model="haiku",
                                  mcp_config_path=Path("/tmp/x.json"),
                                  session_id="abc-123", setting_sources=None)
    assert cmd[:3] == ["claude", "-p", "hello"]
    assert cmd[cmd.index("--model") + 1] == "haiku"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == "/tmp/x.json"
    assert cmd[cmd.index("--session-id") + 1] == "abc-123"
    assert "--setting-sources" not in cmd  # None means "omit the flag"


def test_build_claude_command_uses_dontask_not_bypass_permissions():
    # bypassPermissions gave every arm unrestricted tool access for the
    # whole run; dontAsk auto-denies anything not on --allowedTools instead.
    cmd = r.build_claude_command(prompt="hi", model="haiku",
                                  mcp_config_path=Path("/tmp/x.json"),
                                  session_id="s1", setting_sources=None)
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert "bypassPermissions" not in cmd


def test_build_claude_command_always_disallows_task_webfetch_websearch():
    cmd = r.build_claude_command(prompt="hi", model="haiku",
                                  mcp_config_path=Path("/tmp/x.json"),
                                  session_id="s1", setting_sources=None)
    assert "--disallowedTools" in cmd
    disallowed = cmd[cmd.index("--disallowedTools") + 1]
    assert set(disallowed.split(",")) == {"Task", "WebFetch", "WebSearch"}


def test_build_claude_command_includes_allowed_tools_when_given():
    cmd = r.build_claude_command(prompt="hi", model="haiku",
                                  mcp_config_path=Path("/tmp/x.json"),
                                  session_id="s1", setting_sources=None,
                                  allowed_tools=["Read(./**)", "Bash(pytest:*)"])
    assert "--allowedTools" in cmd
    assert cmd[cmd.index("--allowedTools") + 1] == "Read(./**),Bash(pytest:*)"


def test_build_claude_command_no_allowed_tools_flag_when_empty():
    cmd = r.build_claude_command(prompt="hi", model="haiku",
                                  mcp_config_path=Path("/tmp/x.json"),
                                  session_id="s1", setting_sources=None,
                                  allowed_tools=[])
    assert "--allowedTools" not in cmd


def test_build_claude_command_includes_effort_flag():
    cmd = r.build_claude_command(prompt="hi", model="haiku",
                                  mcp_config_path=Path("/tmp/x.json"),
                                  session_id="s1", setting_sources=None,
                                  effort="high")
    assert cmd[cmd.index("--effort") + 1] == "high"


def test_build_claude_command_includes_settings_path():
    cmd = r.build_claude_command(prompt="hi", model="haiku",
                                  mcp_config_path=Path("/tmp/x.json"),
                                  session_id="s1", setting_sources=None,
                                  settings_path=Path("/tmp/settings.json"))
    assert cmd[cmd.index("--settings") + 1] == "/tmp/settings.json"


def test_build_claude_command_safe_mode_flag():
    cmd_off = r.build_claude_command(prompt="hi", model="haiku",
                                      mcp_config_path=Path("/tmp/x.json"),
                                      session_id="s1", setting_sources=None)
    cmd_on = r.build_claude_command(prompt="hi", model="haiku",
                                     mcp_config_path=Path("/tmp/x.json"),
                                     session_id="s1", setting_sources="",
                                     safe_mode=True)
    assert "--safe-mode" not in cmd_off
    assert "--safe-mode" in cmd_on


def test_build_claude_command_arm_a_passes_empty_setting_sources():
    cmd = r.build_claude_command(prompt="hi", model="haiku",
                                  mcp_config_path=Path("/tmp/a.json"),
                                  session_id="s1", setting_sources="")
    assert "--setting-sources" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""


def test_parse_cli_json_valid():
    doc = r.parse_cli_json('{"result": "kb-mcp", "total_cost_usd": 0.01}')
    assert doc["result"] == "kb-mcp"


def test_parse_cli_json_invalid_raises_with_context():
    try:
        r.parse_cli_json("not json at all")
    except ValueError as e:
        assert "not json at all" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_find_transcript_locates_by_session_id(tmp_path):
    proj_dir = tmp_path / "-some-escaped-cwd"
    proj_dir.mkdir()
    sid = "11111111-1111-1111-1111-111111111111"
    (proj_dir / f"{sid}.jsonl").write_text("{}\n")
    found = r.find_transcript(sid, search_root=tmp_path)
    assert found == proj_dir / f"{sid}.jsonl"


def test_find_transcript_returns_none_when_absent(tmp_path):
    assert r.find_transcript("nope-here", search_root=tmp_path) is None


def test_make_and_remove_workdir_scratch_with_fixture(tmp_path):
    task = {"workdir": "scratch", "fixture": {"a.txt": "hello"}}
    wd = r.make_workdir(task, tmp_path, repo=tmp_path)
    assert (wd / "a.txt").read_text() == "hello"
    r.remove_workdir(task, wd, repo=tmp_path)
    assert not wd.exists()


def test_make_and_remove_workdir_scratch_without_fixture(tmp_path):
    task = {"workdir": "scratch"}
    wd = r.make_workdir(task, tmp_path, repo=tmp_path)
    assert wd.is_dir()
    r.remove_workdir(task, wd, repo=tmp_path)
    assert not wd.exists()


def test_make_and_remove_workdir_repo_creates_real_worktree(tmp_path):
    """Integration check against the ACTUAL repo (git only -- no claude -p,
    no network): a `workdir: "repo"` task must get a real, working git
    worktree, and cleanup must remove it."""
    repo = r.repo_root()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                           text=True, check=True).stdout.strip()
    task = {"workdir": "repo"}
    wd = r.make_workdir(task, tmp_path, repo, pinned_commit=head)
    try:
        assert (wd / ".git").exists()
        assert (wd / "scripts" / "bench" / "ab_session.py").exists()
    finally:
        r.remove_workdir(task, wd, repo)
    assert not wd.exists()


def test_make_workdir_repo_uses_pinned_commit_not_moving_head(tmp_path):
    # A worktree for a 'repo' task must check out the PINNED commit given by
    # the caller (tasks.json's pinned_worktree_commit), never whatever HEAD
    # happens to be at run time -- otherwise, once this harness is itself
    # committed to the branch, a worktree off "current HEAD" could contain
    # the harness's own tasks.json (the answer key) inside the tree a coding
    # agent's Read tool can reach.
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=fake_repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=fake_repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=fake_repo, check=True)
    (fake_repo / "a.txt").write_text("first\n")
    subprocess.run(["git", "add", "a.txt"], cwd=fake_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=fake_repo, check=True)
    first_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fake_repo,
                                capture_output=True, text=True, check=True).stdout.strip()
    (fake_repo / "b.txt").write_text("second\n")
    subprocess.run(["git", "add", "b.txt"], cwd=fake_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=fake_repo, check=True)

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    task = {"workdir": "repo"}
    wd = r.make_workdir(task, base_dir, fake_repo, pinned_commit=first_sha)
    try:
        assert (wd / "a.txt").exists()
        assert not (wd / "b.txt").exists()  # 'second' commit must be absent
        wd_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wd,
                                  capture_output=True, text=True, check=True).stdout.strip()
        assert wd_head == first_sha
    finally:
        r.remove_workdir(task, wd, fake_repo)


def test_make_workdir_repo_requires_pinned_commit():
    try:
        r.make_workdir({"workdir": "repo"}, Path("/tmp"), Path("/tmp"))
    except ValueError as e:
        assert "pinned_commit" in str(e)
    else:
        raise AssertionError("expected ValueError when pinned_commit is omitted")


def test_load_tasks_reads_shipped_tasks_json():
    tasks = r.load_tasks(r.HERE / "tasks.json")
    ids = [t["id"] for t in tasks]
    assert len(tasks) == 5
    assert len(set(ids)) == 5  # unique ids
    categories = {t["category"] for t in tasks}
    assert categories == {"kb", "code", "control"}
    assert sum(1 for t in tasks if t["category"] == "kb") == 2
    assert sum(1 for t in tasks if t["category"] == "code") == 2
    assert sum(1 for t in tasks if t["category"] == "control") == 1
    for t in tasks:
        assert t.get("why"), f"{t['id']} is missing its 'why' justification"
        assert t.get("check", {}).get("type") in {"exact_match", "regex_match", "command"}


# --------------------------------------------------------------- run_one / infra errors

def test_looks_like_rate_limit_detects_common_markers():
    assert r.looks_like_rate_limit("Error: rate_limit_error occurred")
    assert r.looks_like_rate_limit("HTTP 429 Too Many Requests")
    assert not r.looks_like_rate_limit("some unrelated crash")


def test_stderr_digest_never_stores_raw_text():
    d = r.stderr_digest("super secret trace\nwith stuff")
    assert "stderr" not in json.dumps(d) or d.get("stderr_len") is not None
    assert d["stderr_len"] == len("super secret trace\nwith stuff")
    assert isinstance(d["stderr_sha256"], str) and len(d["stderr_sha256"]) == 64
    # the digest dict itself must not contain the raw text anywhere
    assert "secret" not in json.dumps(d)


def test_run_one_timeout_still_loads_transcript_and_marks_infra_error(monkeypatch, tmp_path):
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=None, timed_out=True),
                    transcript_records=[_usage_record(input_tokens=100, output_tokens=10)])
    assert row["infra_error"] == "timeout"
    assert row["success"] is False
    # the transcript existed on disk despite the timeout -- its cost must be
    # counted, not thrown away
    assert row["weighted_cost"] is not None
    assert row["weighted_cost"] > 0
    assert row["transcript_path"] is not None


def test_run_one_nonzero_exit_with_no_transcript_is_infra_not_task_failure(monkeypatch, tmp_path):
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=1, stderr="boom, unexpected crash"),
                    transcript_records=None)
    assert row["infra_error"] == "nonzero_exit_no_turn"
    assert row["success"] is False
    assert row["weighted_cost"] is None  # no transcript -- refuse to fabricate a cost


def test_run_one_rate_limit_detected_from_stderr(monkeypatch, tmp_path):
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(
                        returncode=1, stderr="anthropic.RateLimitError: rate_limit_error"),
                    transcript_records=None)
    assert row["infra_error"] == "rate_limit"


def test_run_one_unparseable_output_still_loads_transcript(monkeypatch, tmp_path):
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=0, stdout="not valid json at all"),
                    transcript_records=[_usage_record(input_tokens=50, output_tokens=5)])
    assert row["infra_error"] == "unparseable_output"
    assert row["weighted_cost"] is not None
    assert row["weighted_cost"] > 0


def test_run_one_nonzero_exit_with_only_fake_zero_usage_record_is_infra_no_turn(
        monkeypatch, tmp_path):
    # Plan Partial #2: "don't count the fake assistant record as a turn".
    # The OLD behavior (ab_session.SessionStats.turns, which counts any
    # assistant record with a truthy usage dict) would see this transcript
    # as having 1 turn and NOT flag it as nonzero_exit_no_turn -- letting a
    # genuine infra failure slip through and get scored as a task result.
    fake_usage_record = {
        "type": "assistant", "requestId": "r1", "version": "2.1.283",
        "message": {"id": "m1", "model": "claude-haiku-4-5", "usage": {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}}
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=1, stderr="API Error"),
                    transcript_records=[fake_usage_record])
    assert row["infra_error"] == "nonzero_exit_no_turn"
    assert row["turns"] == 0


def test_run_one_cli_is_error_during_execution_marks_infra_rate_limit(monkeypatch, tmp_path):
    cli_json = json.dumps({"is_error": True, "subtype": "error_during_execution",
                            "result": "API Error: 529 overloaded"})
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=0, stdout=cli_json),
                    transcript_records=[_usage_record()])
    assert row["infra_error"] == "rate_limit"


def test_run_one_cli_is_error_other_subtype_marks_infra_api_error(monkeypatch, tmp_path):
    cli_json = json.dumps({"is_error": True, "subtype": "error_something_else",
                            "result": "an unrecognized failure"})
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=0, stdout=cli_json),
                    transcript_records=[_usage_record()])
    assert row["infra_error"] == "api_error"


def test_run_one_cli_error_max_turns_is_not_infra():
    # error_max_turns means the model genuinely ran out of turns on the
    # task -- a task result, not an infrastructure failure.
    assert r.classify_cli_error({"is_error": True, "subtype": "error_max_turns"}, "") is None


def test_run_one_cli_is_error_false_is_not_infra():
    assert r.classify_cli_error({"is_error": False, "result": "ok"}, "") is None
    assert r.classify_cli_error({}, "") is None


def test_run_one_kb_task_success_with_zero_tool_calls_fails_pass_gate(monkeypatch, tmp_path):
    # Plan Partial #8: a pass on a kb/code task also requires
    # tool_calls_total > 0 -- a correct answer with zero tool calls means
    # the checker got lucky on a guess (or memorized training data), not
    # that the arm's tool surface earned the answer.
    task = _base_task(category="kb", check={"type": "exact_match", "expected": "ok"})
    cli_json = json.dumps({"result": "ok"})
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=0, stdout=cli_json),
                    transcript_records=[_usage_record()],  # a real turn, but no tool_use block
                    task=task)
    assert row["tool_calls_total"] == 0
    assert row["success"] is False
    assert "REJECTED" in row["check_detail"]


def test_run_one_kb_task_success_with_a_tool_call_passes(monkeypatch, tmp_path):
    task = _base_task(category="kb", check={"type": "exact_match", "expected": "ok"})
    cli_json = json.dumps({"result": "ok"})
    record = _usage_record()
    record["message"]["content"] = [{"type": "tool_use", "id": "t1",
                                      "name": "mcp__kb__kb_read_get"}]
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=0, stdout=cli_json),
                    transcript_records=[record], task=task)
    assert row["tool_calls_total"] == 1
    assert row["success"] is True


def test_run_one_control_task_success_with_zero_tool_calls_is_not_gated(monkeypatch, tmp_path):
    # The pass gate is scoped to kb/code tasks only (plan Partial #8) -- the
    # control task is a plain file-edit checked by a real pytest run, and
    # legitimately has no MCP tool involved by design.
    task = _base_task(category="control", check={"type": "exact_match", "expected": "ok"})
    cli_json = json.dumps({"result": "ok"})
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=0, stdout=cli_json),
                    transcript_records=[_usage_record()], task=task)
    assert row["tool_calls_total"] == 0
    assert row["success"] is True


def test_resolve_allowed_tools_reads_per_arm_tools_field():
    task = {"tools": {"A": [], "B": ["mcp__kb__x"], "C": ["mcp__terse__x"]}}
    assert r.resolve_allowed_tools(task, "A") == []
    assert r.resolve_allowed_tools(task, "B") == ["mcp__kb__x"]
    assert r.resolve_allowed_tools(task, "C") == ["mcp__terse__x"]


def test_resolve_allowed_tools_defaults_to_empty_when_absent():
    assert r.resolve_allowed_tools({}, "B") == []


def test_build_settings_file_disables_all_hooks_mode_600(tmp_path):
    path = r.build_settings_file(tmp_path)
    assert json.loads(path.read_text())["disableAllHooks"] is True
    assert (path.stat().st_mode & 0o777) == 0o600


def test_refuse_out_inside_repo_raises(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    inside = repo / "results" / "out.jsonl"
    try:
        r._refuse_out_inside_repo(inside, repo)
    except SystemExit as e:
        assert "inside the repo" in str(e)
    else:
        raise AssertionError("expected SystemExit for an --out path inside the repo")


def test_refuse_out_inside_repo_allows_outside_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "scratch" / "out.jsonl"
    r._refuse_out_inside_repo(outside, repo)  # must not raise


def test_plan_run_combos_always_produces_warmup_now():
    # Plan Partial #9: warm-up is no longer opt-in behind a flag -- it
    # ALWAYS happens, one rep zero per arm, on the first task.
    tasks = [{"id": "t1"}, {"id": "t2"}]
    warmup, real = r.plan_run_combos(tasks, ["A", "B"], reps=2, seed=1)
    assert len(warmup) == 2  # one per arm
    assert all(rep == 0 for _, _, rep in warmup)
    assert all(t["id"] == "t1" for t, _, _ in warmup)  # first task only
    assert all(rep != 0 for _, _, rep in real)


def test_plan_run_combos_no_tasks_has_no_warmup():
    warmup, real = r.plan_run_combos([], ["A"], reps=3, seed=1)
    assert warmup == []
    assert real == []


def test_run_one_success_path_has_no_infra_error(monkeypatch, tmp_path):
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(
                        returncode=0,
                        stdout=json.dumps({"result": "ok", "total_cost_usd": 0.001})),
                    transcript_records=[_usage_record(input_tokens=50, output_tokens=5)])
    assert row["infra_error"] is None
    assert row["success"] is True
    assert row["weighted_cost"] is not None
    assert row["cli_cost_usd"] == 0.001


def test_run_one_records_first_turn_cache_and_resolved_model_version(monkeypatch, tmp_path):
    first = {"type": "assistant", "requestId": "r1", "version": "2.1.283",
             "message": {"id": "m1", "model": "claude-haiku-4-5", "usage": {
                 "input_tokens": 1, "cache_read_input_tokens": 7,
                 "cache_creation_input_tokens": 9, "output_tokens": 1}}}
    second = {"type": "assistant", "requestId": "r2", "version": "2.1.283",
              "message": {"id": "m2", "model": "claude-haiku-4-5", "usage": {
                  "input_tokens": 1, "cache_read_input_tokens": 999,
                  "cache_creation_input_tokens": 999, "output_tokens": 1}}}
    row = _run_one(monkeypatch, tmp_path,
                    run_side_effect=_fake_process(returncode=0,
                                                   stdout=json.dumps({"result": "ok"})),
                    transcript_records=[first, second])
    assert row["first_turn_cache_read_tokens"] == 7
    assert row["first_turn_cache_write_tokens"] == 9
    assert row["resolved_model"] == "claude-haiku-4-5"
    assert row["cli_version"] == "2.1.283"


# --------------------------------------------------------------- F5: process-group kill

def test_run_claude_process_returns_normally_on_success(tmp_path):
    returncode, stdout, stderr, timed_out = r._run_claude_process(
        [sys.executable, "-c", "print('hi')"], cwd=tmp_path, env=os.environ.copy(), timeout=10)
    assert timed_out is False
    assert returncode == 0
    assert stdout.strip() == "hi"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_run_claude_process_kills_whole_process_group_on_timeout(tmp_path):
    # A dummy process that spawns a CHILD of its own (mimicking claude
    # spawning terse's router, which itself spawns MCP server children) --
    # the child's pid is written to a file so the test can verify it's
    # actually dead afterward, not merely orphaned (plan F5: "kill the whole
    # process group ... so the router and MCP children don't linger").
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        "import subprocess, sys\n"
        "child = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "child.wait()\n"
    )
    returncode, stdout, stderr, timed_out = r._run_claude_process(
        [sys.executable, str(script)], cwd=tmp_path, env=os.environ.copy(), timeout=1)
    assert timed_out is True

    deadline = time.time() + 5
    child_pid = int(pid_file.read_text().strip())
    while time.time() < deadline and _pid_alive(child_pid):
        time.sleep(0.1)
    assert not _pid_alive(child_pid), (
        "the child process ('sleep 60') survived the timeout -- killpg did not "
        "reach the whole process group")


# --------------------------------------------------------------- F2: --tools

def test_resolve_builtin_tools_matches_category():
    assert r.resolve_builtin_tools({"category": "kb"}) == "Read,Grep"
    assert r.resolve_builtin_tools({"category": "code"}) == "Read,Grep,Glob"
    assert r.resolve_builtin_tools({"category": "control"}) == "Read,Edit,Bash"


def test_resolve_builtin_tools_raises_for_unknown_category():
    try:
        r.resolve_builtin_tools({"id": "x", "category": "mystery"})
    except KeyError as e:
        assert "mystery" in str(e)
    else:
        raise AssertionError("expected KeyError for an unrecognized category")


def test_build_claude_command_includes_tools_flag_including_empty_string():
    cmd_kb = r.build_claude_command(prompt="hi", model="haiku",
                                     mcp_config_path=Path("/tmp/x.json"),
                                     session_id="s1", setting_sources=None, tools="")
    assert "--tools" in cmd_kb
    assert cmd_kb[cmd_kb.index("--tools") + 1] == ""

    cmd_code = r.build_claude_command(prompt="hi", model="haiku",
                                       mcp_config_path=Path("/tmp/x.json"),
                                       session_id="s1", setting_sources=None,
                                       tools="Read,Grep,Glob")
    assert cmd_code[cmd_code.index("--tools") + 1] == "Read,Grep,Glob"


def test_build_claude_command_omits_tools_flag_when_none():
    cmd = r.build_claude_command(prompt="hi", model="haiku",
                                  mcp_config_path=Path("/tmp/x.json"),
                                  session_id="s1", setting_sources=None)
    assert "--tools" not in cmd


def test_builtin_tools_identical_across_arms_for_every_shipped_task():
    # Plan F2: --tools is a per-CATEGORY constant, never per-arm -- verify
    # every task in the shipped tasks.json resolves to the SAME --tools
    # value for A, B, and C.
    tasks = r.load_tasks(r.HERE / "tasks.json")
    for task in tasks:
        value = r.resolve_builtin_tools(task)
        for arm in ("A", "B", "C"):
            cmd = r.build_claude_command(
                prompt="x", model="m", mcp_config_path=Path("/tmp/c.json"),
                session_id="s", setting_sources=None,
                allowed_tools=r.resolve_allowed_tools(task, arm), tools=value)
            assert cmd[cmd.index("--tools") + 1] == value, \
                f"{task['id']} arm {arm} --tools diverged from {value!r}"


# --------------------------------------------------------------- F1: shared codegraph worktree

def test_sqlite_has_tables_true_for_a_real_table(tmp_path):
    import sqlite3
    db = tmp_path / "codegraph.db"
    conn = sqlite3.connect(db)
    conn.execute("create table symbols (id integer)")
    conn.commit()
    conn.close()
    assert r._sqlite_has_tables(db) is True


def test_sqlite_has_tables_false_for_schema_less_db(tmp_path):
    import sqlite3
    db = tmp_path / "codegraph.db"
    # A schema-less db (what a failed/interrupted `init` leaves behind) --
    # created then immediately has nothing, matching the gate's own
    # "create-then-drop empty one" description.
    sqlite3.connect(db).close()
    assert r._sqlite_has_tables(db) is False


def test_sqlite_has_tables_fails_safe_on_unreadable_file(tmp_path):
    not_a_db = tmp_path / "codegraph.db"
    not_a_db.write_text("not a sqlite file at all")
    assert r._sqlite_has_tables(not_a_db) is True  # fail safe, like the bash gate


def test_codegraph_gate_would_serve_real_true_with_indexed_dot_codegraph(tmp_path):
    import sqlite3
    project = tmp_path / "proj"
    sub = project / "a" / "b"
    sub.mkdir(parents=True)
    dotcg = project / ".codegraph"
    dotcg.mkdir()
    conn = sqlite3.connect(dotcg / "codegraph.db")
    conn.execute("create table symbols (id integer)")
    conn.commit()
    conn.close()
    assert r.codegraph_gate_would_serve_real(sub) is True


def test_codegraph_gate_would_serve_real_true_when_dot_codegraph_has_no_db_yet(tmp_path):
    # A fresh .codegraph/ with no db file yet is still 'real' to the gate --
    # the OLD behavior (before `codegraph init` runs) that this harness must
    # move past by actually indexing, not something the preflight itself
    # should treat as a failure.
    project = tmp_path / "proj"
    (project / ".codegraph").mkdir(parents=True)
    assert r.codegraph_gate_would_serve_real(project) is True


def test_codegraph_gate_would_serve_real_false_with_no_dot_codegraph_anywhere(tmp_path):
    # This is the bug F1 fixes: a fresh git worktree has NO .codegraph/
    # anywhere in its ancestry (up to this tmp_path root), so the gate would
    # serve the null (zero-tool) stub -- exactly what B/C's code tasks were
    # silently hitting every rep before the shared, indexed worktree.
    project = tmp_path / "proj"
    project.mkdir()
    assert r.codegraph_gate_would_serve_real(project) is False


def test_codegraph_gate_would_serve_real_skips_home_itself(tmp_path, monkeypatch):
    # $HOME/.codegraph is codegraph's own app-state dir, not a project --
    # the gate must not treat it as a real index just because it exists.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".codegraph").mkdir()  # app-state dir, schema-less/absent db
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    assert r.codegraph_gate_would_serve_real(fake_home) is False


def test_index_codegraph_invokes_codegraph_init_and_raises_on_failure(monkeypatch, tmp_path):
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "boom-binary":
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(r.subprocess, "run", _fake_run)
    r.index_codegraph(tmp_path, codegraph_bin="real-codegraph")
    assert calls == [["real-codegraph", "init", str(tmp_path)]]

    try:
        r.index_codegraph(tmp_path, codegraph_bin="boom-binary")
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("expected CalledProcessError to propagate")


def test_make_shared_code_worktree_creates_a_real_worktree_at_pinned_commit(tmp_path):
    repo = r.repo_root()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                           text=True, check=True).stdout.strip()
    base_dir = tmp_path / "config-dir"
    base_dir.mkdir()
    wt = r.make_shared_code_worktree(base_dir, repo, head)
    try:
        assert wt == base_dir / "shared-code-worktree"
        assert (wt / ".git").exists()
        wt_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, capture_output=True,
                                  text=True, check=True).stdout.strip()
        assert wt_head == head
    finally:
        r.remove_workdir({"workdir": "repo"}, wt, repo)


def test_cleanup_runecho_enrollment_removes_matching_entry(monkeypatch, tmp_path):
    wt = tmp_path / "shared-code-worktree"
    wt.mkdir()
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["runecho-ir", "repo", "list"]:
            return subprocess.CompletedProcess(argv, 0, stdout=f"{wt.name}\nother-repo\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(r.subprocess, "run", _fake_run)
    r.cleanup_runecho_enrollment(wt)
    assert ["runecho-ir", "repo", "rm", wt.name] in calls


def test_cleanup_runecho_enrollment_does_nothing_when_not_enrolled(monkeypatch, tmp_path):
    wt = tmp_path / "shared-code-worktree"
    wt.mkdir()
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="some-other-repo\n")

    monkeypatch.setattr(r.subprocess, "run", _fake_run)
    r.cleanup_runecho_enrollment(wt)
    assert all(c[:3] != ["runecho-ir", "repo", "rm"] for c in calls)


def test_cleanup_runecho_enrollment_never_raises_when_binary_missing(monkeypatch, tmp_path):
    wt = tmp_path / "shared-code-worktree"
    wt.mkdir()

    def _raise(*a, **kw):
        raise FileNotFoundError("no such file: runecho-ir")

    monkeypatch.setattr(r.subprocess, "run", _raise)
    r.cleanup_runecho_enrollment(wt)  # must not raise


# ------------------------------------------- offloaded tool results (2026-09-26 smoke bias)

def test_offload_read_rule_is_scoped_to_this_session_only(tmp_path):
    """A large MCP result is written to <projects>/<slug>/<session>/tool-results/. The rule
    must reach THAT session's file and nothing else: an earlier run's offloaded kb result
    holds the answer."""
    rule = r.offload_read_rule("abc-123", projects_dir=tmp_path)
    assert rule == f"Read(/{tmp_path}/*/abc-123/tool-results/**)"
    assert "abc-123" in rule and rule.count("*") == 3        # slug glob + recursive tail only


def test_every_run_gets_the_offload_rule_on_top_of_its_task_tools(monkeypatch, tmp_path):
    seen = {}
    def fake_build(**kw):
        seen.update(kw)
        raise RuntimeError("stop after argv")
    monkeypatch.setattr(r, "build_claude_command", fake_build)
    try:
        r.run_one(task={"id": "t", "prompt": "p", "category": "kb"}, arm="B", rep=1,
                  model="m", arm_config_path=tmp_path / "c.json", workdir=tmp_path,
                  setting_sources="", out_fh=None, allowed_tools=["mcp__kb__x"],
                  effort="high", settings_path=None, safe_mode=False, builtin_tools="Read,Grep")
    except RuntimeError:
        pass
    tools = seen["allowed_tools"]
    assert tools[0] == "mcp__kb__x"
    assert tools[-1].startswith("Read(/") and f"/{seen['session_id']}/tool-results/**)" in tools[-1]


def test_setup_arms_load_the_operator_settings_and_A_does_not():
    """B/C are "the user's setup": `--setting-sources ""` would also drop CLAUDE.md, rules,
    output style, skills and agents (Opus review 2026-09-26), so only A drops them."""
    import inspect
    src = inspect.getsource(r.main)
    assert 'setting_sources = {"A": "", "B": None, "C": None}' in src


def test_every_arm_disables_hooks_and_fences_reads_to_the_working_dir(tmp_path):
    """B/C's settings.json grants a blanket Read (a ~/.claude leak arm A lacks); the
    generated settings neutralise it for every arm."""
    for name in ("settings-no-hooks.json", "settings-setup.json"):
        doc = json.loads(r.build_settings_file(tmp_path, name=name).read_text())
        assert doc == {"disableAllHooks": True,
                       "permissions": {"blockReadsOutsideWorkingDirectories": True}}
