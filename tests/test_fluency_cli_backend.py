"""The `cli:` answerer — `claude -p` on the OAuth subscription (#249).

This backend exists because the loopback LiteLLM gateway aliases every `claude-*` model id
onto DeepSeek, so a panel that asked for `claude-sonnet-5` measured DeepSeek and published
it as Anthropic. The tests below pin the properties that keep that class of silent
mismeasurement from coming back — the argv the child is launched with, the environment it
is NOT allowed to inherit, and the refusal to turn any failure into a scored wrong answer.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from terse.fluency import CLI_PREFIX, cli_answerer


class _FakeProc:
    """Stand-in for Popen: records what it was constructed with, replays a canned result."""

    def __init__(self, out: str = "", err: str = "", code: int = 0, *, timeout: bool = False,
                raises: BaseException | None = None):
        self._out, self._err, self.returncode, self._timeout = out, err, code, timeout
        self._raises = raises
        self.pid = 4242
        self.killed = False
        self._finished = False  # poll() mirrors this — None until a call "reaps" it

    def communicate(self, _input=None, timeout=None):
        if self._raises is not None:
            raise self._raises
        if self._timeout:
            self._timeout = False  # the post-kill drain call must succeed
            raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)
        self._finished = True
        return self._out, self._err

    def poll(self):
        return self.returncode if self._finished else None

    def kill(self):
        self.killed = True


def _spy(monkeypatch, proc: _FakeProc) -> dict:
    """Patch Popen and return the dict that captures the launch arguments."""
    seen: dict = {}

    def fake_popen(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return seen


def _reply(text: str, **extra) -> str:
    return json.dumps({"is_error": False, "result": text, **extra})


def test_the_prefix_matches_modelbenchs_so_both_harnesses_name_it_alike():
    assert CLI_PREFIX == "cli:"


# --------------------------------------------------------------------------- #
# The launch contract. Each of these, if it regressed, would silently measure
# something other than the model the report names.
# --------------------------------------------------------------------------- #


def test_the_system_prompt_is_always_passed_even_when_empty(monkeypatch):
    """The no-primer arm must send an EMPTY system prompt, not omit the flag.

    Omitting it lets Claude Code load its own multi-thousand-token default preamble for
    that arm only — an arm-correlated confound in the exact comparison #249 turns on.
    """
    seen = _spy(monkeypatch, _FakeProc(out=_reply("4")))
    cli_answerer("opus")("", "What is 2+2?")
    argv = seen["argv"]
    assert "--system-prompt" in argv
    assert argv[argv.index("--system-prompt") + 1] == ""


def test_the_primer_arm_sends_the_primer_in_the_system_slot(monkeypatch):
    seen = _spy(monkeypatch, _FakeProc(out=_reply("4")))
    cli_answerer("opus")("PRIMER TEXT", "What is 2+2?")
    argv = seen["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "PRIMER TEXT"


@pytest.mark.parametrize(
    "var", ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"]
)
def test_the_gateway_env_is_stripped_from_the_child(monkeypatch, var):
    """A session launched via claude-gw/claude-alt exports these.

    Inheriting them routes `claude -p` straight back through the aliasing gateway this
    backend exists to bypass — i.e. it would measure DeepSeek and label it `cli:opus`,
    which is the original #249 defect wearing the fix as a disguise.
    """
    monkeypatch.setenv(var, "http://127.0.0.1:4000")
    seen = _spy(monkeypatch, _FakeProc(out=_reply("4")))
    cli_answerer("opus")("", "q")
    assert var not in seen["kw"]["env"]


def test_the_named_alias_reaches_the_model_flag(monkeypatch):
    seen = _spy(monkeypatch, _FakeProc(out=_reply("4")))
    cli_answerer("haiku")("", "q")
    argv = seen["argv"]
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--model") + 1] == "haiku"


def test_builtin_tools_are_also_disabled_not_just_mcp(monkeypatch):
    """`--strict-mcp-config` only pins off MCP servers — Read/Write/Bash/Glob are still on
    by default. `openai_answerer` sends no tools at all, so the two backends the panel
    compares must answer under equivalent conditions or a tool-call failure ("I can't run
    that") comes back as a non-None string and gets scored as a WRONG answer, not a
    non-answer — the exact contract this file exists to protect."""
    seen = _spy(monkeypatch, _FakeProc(out=_reply("4")))
    cli_answerer("opus")("", "q")
    argv = seen["argv"]
    assert argv[argv.index("--tools") + 1] == ""


def test_a_missing_claude_binary_is_reported_not_silently_swallowed(monkeypatch, capsys):
    """Every other failure path here prints a stderr diagnostic; a missing/unresolvable
    `claude` binary must not be the one silent exception `_safe_ask`'s catch-all absorbs."""
    def raise_enoent(argv, **kw):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(subprocess, "Popen", raise_enoent)
    assert cli_answerer("opus")("", "q") is None
    assert "could not launch" in capsys.readouterr().err


def test_mcp_is_pinned_off_so_no_tool_definitions_enter_the_prompt(monkeypatch):
    seen = _spy(monkeypatch, _FakeProc(out=_reply("4")))
    cli_answerer("opus")("", "q")
    argv, kw = seen["argv"], seen["kw"]
    assert "--strict-mcp-config" in argv
    cfg = argv[argv.index("--mcp-config") + 1]
    with open(cfg) as fh:
        assert json.load(fh) == {"mcpServers": {}}
    # cwd is the scratch dir holding that config — never the repo, which carries a
    # CLAUDE.md the model would otherwise read as instructions.
    assert kw["cwd"] == cfg.rsplit("/", 1)[0]


def test_the_child_gets_its_own_process_group(monkeypatch):
    """Orphaned `claude` children keep generating and burn subscription quota unseen."""
    seen = _spy(monkeypatch, _FakeProc(out=_reply("4")))
    cli_answerer("opus")("", "q")
    assert seen["kw"]["start_new_session"] is True


# --------------------------------------------------------------------------- #
# Every failure is a NON-ANSWER, never a wrong answer (#263/#268 contract).
# --------------------------------------------------------------------------- #


def test_a_good_reply_comes_back_verbatim(monkeypatch):
    _spy(monkeypatch, _FakeProc(out=_reply("  BANANA  ")))
    assert cli_answerer("opus")("", "q") == "  BANANA  "


def test_the_user_prompt_actually_reaches_the_child(monkeypatch):
    """Nothing else in this file asserts on communicate()'s stdin argument — a regression
    that stopped sending the question (e.g. `proc.communicate(timeout=timeout)` losing its
    positional `user` argument) would still pass every other test here."""
    sent: list = []
    proc = _FakeProc(out=_reply("4"))
    real_communicate = proc.communicate

    def spying_communicate(_input=None, timeout=None):
        sent.append(_input)
        return real_communicate(_input, timeout=timeout)

    proc.communicate = spying_communicate
    _spy(monkeypatch, proc)
    cli_answerer("opus")("", "What is 2+2?")
    assert sent == ["What is 2+2?"]


def test_a_quota_wall_is_a_non_answer_not_a_wrong_answer(monkeypatch):
    """The wall lands partway through a run; scoring it wrong invalidates the remainder."""
    _spy(monkeypatch, _FakeProc(err="Claude usage limit reached", code=1))
    assert cli_answerer("opus")("", "q") is None


def test_a_confirmed_quota_wall_trips_a_breaker_for_the_rest_of_this_models_calls(monkeypatch):
    """Once THIS model's window is confirmed exhausted, remaining calls for it must not
    spawn another doomed `claude -p` — at 100+ calls/model that's hundreds of launches
    that can only time out or quota-fail while `_unmeasured` withholds the report anyway."""
    launches: list = []

    def fake_popen(argv, **kw):
        launches.append(argv)
        return _FakeProc(err="Claude usage limit reached", code=1)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    answerer = cli_answerer("opus")
    assert answerer("", "q1") is None
    assert answerer("", "q2") is None
    assert len(launches) == 1, "second call should have short-circuited, not re-launched"


def test_a_nonzero_exit_is_a_non_answer(monkeypatch):
    _spy(monkeypatch, _FakeProc(err="boom", code=2))
    assert cli_answerer("opus")("", "q") is None


def test_a_timeout_kills_the_group_and_returns_a_non_answer(monkeypatch):
    proc = _FakeProc(timeout=True)
    _spy(monkeypatch, proc)
    killed: list = []
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: killed.append(pgid))
    assert cli_answerer("opus", timeout=1)("", "q") is None
    assert killed == [proc.pid]


def test_a_permission_error_killing_the_group_is_reported_not_swallowed(monkeypatch, capsys):
    """If we can't signal the group we ourselves created, that's surprising and worth
    a loud stderr line — silently falling back to killing only the direct child would
    reintroduce the orphaned-descendants quota leak this code exists to prevent."""
    proc = _FakeProc(timeout=True)
    _spy(monkeypatch, proc)
    monkeypatch.setattr("os.getpgid", lambda pid: pid)

    def raise_perm(pgid, sig):
        raise PermissionError()

    monkeypatch.setattr("os.killpg", raise_perm)
    assert cli_answerer("opus", timeout=1)("", "q") is None
    assert proc.killed
    assert "permission denied" in capsys.readouterr().err.lower()


def test_a_non_timeout_exception_still_kills_the_group_before_propagating(monkeypatch):
    """`start_new_session=True` detaches the child; only the TimeoutExpired branch used to
    kill the group. A KeyboardInterrupt (or MemoryError) during communicate() is not a
    timeout and is not caught by `_safe_ask`'s `except Exception` upstream (BaseException) —
    without a safety net the group is left running, generating and burning subscription
    quota with nothing left to stop it. This is the modelbench incident #249 already hit."""
    proc = _FakeProc(raises=KeyboardInterrupt())
    _spy(monkeypatch, proc)
    killed: list = []
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: killed.append(pgid))
    with pytest.raises(KeyboardInterrupt):
        cli_answerer("opus")("", "q")
    assert killed == [proc.pid]


def test_is_error_is_a_non_answer_even_on_a_clean_exit(monkeypatch):
    _spy(monkeypatch, _FakeProc(out=json.dumps({"is_error": True, "result": "nope"})))
    assert cli_answerer("opus")("", "q") is None


def test_non_json_stdout_is_a_non_answer(monkeypatch):
    _spy(monkeypatch, _FakeProc(out="not json at all"))
    assert cli_answerer("opus")("", "q") is None


def test_valid_json_that_is_not_an_object_is_a_non_answer_not_a_crash(monkeypatch):
    """`json.loads` can succeed on a bare `null`/array/string — `.get()` on that would
    raise AttributeError and skip the stderr diagnostic every other failure path prints."""
    _spy(monkeypatch, _FakeProc(out="null"))
    assert cli_answerer("opus")("", "q") is None


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_an_empty_result_is_a_non_answer_not_a_wrong_answer(monkeypatch, blank):
    """Same contract `openai_answerer` carries: no content and no call are one fact."""
    _spy(monkeypatch, _FakeProc(out=_reply(blank)))
    assert cli_answerer("opus")("", "q") is None


def test_a_non_string_result_is_a_non_answer(monkeypatch):
    _spy(monkeypatch, _FakeProc(out=json.dumps({"is_error": False, "result": 42})))
    assert cli_answerer("opus")("", "q") is None
