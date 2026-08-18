"""The answerer transport — pluggable `(system, user) -> reply` callables (#78 split).

The pure core (question generation + scoring) runs offline with no network or key;
the live backend (`openai_answerer` over stdlib urllib) reaches any OpenAI-compatible
endpoint — the broker pool or a loopback gateway — and adds zero new dependencies.

`cli_answerer` is the second backend, and it exists because the OpenAI-compatible path
CANNOT reach a real Anthropic model in this setup — see its docstring for the trap that
cost issue #249 a whole panel.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable

# The cleartext-credential rule lives in ONE place (transport.py) so every
# credential-bearing caller inherits it. dropeval's tool-calling answerer had no such
# check at all while this module did — the parity gap that motivated centralizing it.
# `_LOOPBACK_HOSTS` is re-exported (not redefined) to keep `fluency._LOOPBACK_HOSTS`
# importable for the existing tests while there is still only one definition.
from ..transport import _LOOPBACK_HOSTS, guard_cleartext_credential  # noqa: F401

# An answerer takes (system_prompt, user_prompt) and returns the model's reply text, or
# **None** when the backend produced no content at all. Empty system_prompt means "no
# system message".
#
# `None` is the same channel `_safe_ask` uses for a transport failure, and it means the
# same thing to the report: the question was not answered, so it is counted and never
# scored (#263/#264). A 200 carrying `content: null` is not an exception and so never
# reached that channel — see `openai_answerer` for why that produced a false PASS (#268).
Answerer = Callable[[str, str], "str | None"]


def openai_answerer(base_url: str, api_key: str, model: str,
                    temperature: float = 0.0, timeout: int = 60) -> Answerer:
    """OpenAI-compatible /chat/completions answerer over stdlib urllib. Covers the
    broker pool (OpenRouter et al.) without an SDK dependency. temperature 0 for
    reproducibility."""
    # An http:// base URL sends `Authorization: Bearer <key>` in cleartext — refuse it for
    # a non-loopback host rather than silently leak the key on the wire.
    guard_cleartext_credential(base_url, bool(api_key), what="terse fluency")
    url = base_url.rstrip("/") + "/chat/completions"

    def ask(system: str, user: str) -> str | None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body = json.dumps({"model": model, "messages": messages,
                           "temperature": temperature}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # Some OpenAI-compatible gateways return 200 with an error body (no choices);
        # surface a clear message instead of a bare KeyError.
        if "choices" not in data:
            raise RuntimeError(f"{model}: no choices in response: {data.get('error', data)}")
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content")
        # `content or ""` collapsed "the model produced NO content" into "the model
        # answered, with nothing" — erasing, one layer down, exactly the distinction #264
        # built. A 200 with `content: null` is not an exception, so it never reached
        # `_safe_ask`'s failure channel: it was scored as a WRONG ANSWER and was invisible
        # to `_unmeasured`. With every arm at 0% the gap is 0 and the diff report printed
        # PASS — a model that answered nothing green-lighting the `proxy --diff` flip
        # (#268). Observed live: gemini-3.6-flash returning null when reasoning consumed
        # the token budget.
        #
        # Blank is treated the same as null deliberately. No question has an empty expected
        # answer (`questions.py` excludes them), so an empty reply can never be *correct* —
        # scoring it wrong would charge terse for a backend quirk, while counting it as a
        # non-answer makes `_unmeasured` decline to publish. Refusing to answer is the safe
        # direction; a false PASS is not.
        if content is None or not content.strip():
            # `finish_reason` is the actionable half — `length` means raise max_tokens,
            # `content_filter` means the payload tripped a safety filter, and each calls
            # for a different fix than "the backend was unreachable" does. Said once per
            # occurrence on stderr so a long panel run cannot swallow it silently.
            print(f"terse fluency: {model} returned no content "
                  f"(finish_reason={choice.get('finish_reason')!r}) — counted as a "
                  f"non-answer, not scored", file=sys.stderr)
            return None
        return content

    return ask


# Model ids carrying this prefix are served by `cli_answerer`, not by the OpenAI path.
# Mirrors modelbench's `cli:` convention so the two harnesses name the same thing the
# same way.
CLI_PREFIX = "cli:"

# Substrings that mean "the subscription hit its window limit", not "the model was wrong".
# Scoring a quota wall as a wrong answer is what invalidated a whole modelbench run: the
# wall hits partway through and every remaining call writes a confident-looking failure.
_QUOTA_MARKERS = ("usage limit", "rate limit", "quota", "too many requests", "429")


def _looks_like_quota(*texts: str) -> bool:
    blob = " ".join(t for t in texts if t).lower()
    return any(m in blob for m in _QUOTA_MARKERS)


def cli_answerer(alias: str, timeout: int = 180) -> Answerer:
    """Real Anthropic models via the `claude -p` OAuth subscription (no API key).

    **Why this backend exists at all.** No OpenAI-compatible endpoint in this setup serves
    a real Anthropic model. The local LiteLLM gateway defines `claude-sonnet-5`,
    `claude-fable-5` and `claude-haiku-4-*` as ALIASES onto DeepSeek — its config says so
    outright ("route Anthropic model IDs to DeepSeek direct ... without touching real
    Claude"), because they exist to exercise Claude Code's `/v1/messages` path against a
    cheap backend. That is a fine thing to have and a disastrous thing to point an eval at:
    #249 ran a four-model "frontier panel" that was actually two DeepSeek models measured
    twice under Anthropic names, and reported it as multi-vendor. `modelbench` hit the same
    wall and records it at `runner.py:423`. This backend is the only path to the real thing.

    **The system slot is passed explicitly, always.** `--system-prompt` replaces the
    *system prompt* only — NOT, as an earlier draft of this claimed, Claude Code's whole
    default preamble. Measured directly: `""` costs 19,816 input tokens, `"You are a
    calculator."` costs 19,821 — the ~19.8k of tool-definition/context preamble is present
    in BOTH arms, constant, and so not a confound. What the flag does buy is real: the
    primer arm and the no-primer arm still differ by exactly the primer content and
    nothing else, because omitting the flag for the no-primer arm would hand that arm its
    own multi-thousand-token system prompt on top of the shared preamble — an
    arm-correlated confound in the one comparison #249 turns on. Treat this as a validity
    caveat on absolute accuracy and on any cross-backend comparison, not on the
    primer/no-primer contrast itself.

    **The environment is scrubbed on purpose.** `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`
    / `ANTHROPIC_API_KEY` are dropped from the child env; a session running under
    `claude-gw`/`claude-alt` exports them, which would route this backend straight back
    through the aliasing gateway it exists to bypass — measuring DeepSeek while the report
    says `cli:opus`.

    Not bit-comparable to the gateway path: each `claude -p` is a fresh process, so there is
    a per-call cached preamble (~10k tokens) no gateway model pays, and the OAuth
    subscription enforces a rolling window limit. Treat cross-backend comparisons as
    directional; run one Anthropic model per window.
    """
    # One scratch dir per answerer, not per call: a cwd with no CLAUDE.md and an empty MCP
    # config, so no project instructions or tool definitions leak into the prompt under test.
    # Cleaned up at interpreter exit (atexit, not a `with`) because the returned `ask`
    # closure is called many times across a whole panel run — there is no single point
    # to `rmtree` it right after creation.
    workdir = tempfile.mkdtemp(prefix="terse-fluency-cli-")
    atexit.register(shutil.rmtree, workdir, ignore_errors=True)
    mcp_cfg = os.path.join(workdir, "empty-mcp.json")
    with open(mcp_cfg, "w") as fh:
        fh.write('{"mcpServers":{}}')
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")}
    # Per-answerer circuit breaker: once THIS model's subscription window is confirmed
    # exhausted, every remaining call for it returns immediately instead of spawning
    # another doomed `claude -p` process that can only time out or quota-fail.
    quota_hit = False

    def _kill_group(proc: subprocess.Popen, *, why: str) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # group already exited — nothing left to orphan
        except PermissionError:
            # Can't signal the group we created ourselves — should not happen for a
            # same-user start_new_session child, but if it does, say so loudly rather
            # than silently leaving descendants running past the timeout.
            print(f"terse fluency: {alias}: permission denied killing its process "
                  f"group ({why}) — descendants may keep running and burn "
                  f"subscription quota", file=sys.stderr)
            proc.kill()

    def ask(system: str, user: str) -> str | None:
        nonlocal quota_hit
        if quota_hit:
            return None
        argv = ["claude", "-p", "--model", alias, "--output-format", "json",
                "--strict-mcp-config", "--mcp-config", mcp_cfg,
                # No built-in tools either (Bash/Read/Write/...) — openai_answerer sends
                # none, so the two backends must answer under equivalent conditions. A
                # model electing a tool call here, failing in the scratch dir, and
                # replying with prose like "I can't run that" would otherwise be a
                # non-None string scored as a WRONG answer, not a non-answer.
                "--tools", "",
                # Empty string is deliberate and load-bearing — see docstring.
                "--system-prompt", system]
        # start_new_session puts claude and everything it spawns in a fresh process group.
        # Killing only the direct child on timeout orphans that tree, and the orphans keep
        # generating — burning subscription quota invisibly for the rest of the run.
        try:
            proc = subprocess.Popen(
                argv, cwd=workdir, env=env, text=True, errors="replace",
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as e:
            print(f"terse fluency: {alias}: could not launch claude ({e}) — counted as a "
                  f"non-answer, not scored", file=sys.stderr)
            return None
        try:
            try:
                out, err = proc.communicate(user, timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc, why="timeout")
                try:
                    proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    pass  # best-effort drain — the group is already dead or unkillable
                print(f"terse fluency: {alias} timed out after {timeout}s — counted as a "
                      f"non-answer, not scored", file=sys.stderr)
                return None
        finally:
            # Safety net for anything that unwinds through here WITHOUT going through the
            # timeout branch above — KeyboardInterrupt or MemoryError during
            # communicate() is a BaseException/Exception `_safe_ask` upstream may not
            # stop from propagating. Leaving the group running past this point is the
            # exact orphaned-quota-burn incident modelbench already hit once.
            if proc.poll() is None:
                _kill_group(proc, why="cleanup")
        if proc.returncode != 0:
            why = "subscription window limit" if _looks_like_quota(err, out) else "cli error"
            if why == "subscription window limit":
                quota_hit = True
            print(f"terse fluency: {alias} {why} (exit {proc.returncode}): "
                  f"{(err or out)[:200]} — counted as a non-answer, not scored",
                  file=sys.stderr)
            return None
        try:
            body = json.loads(out)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            print(f"terse fluency: {alias} returned non-JSON: {out[:200]} — counted as a "
                  f"non-answer, not scored", file=sys.stderr)
            return None
        # `is_error` is the CLI's own flag for "this turn failed"; the quota wall arrives
        # here as well as via a nonzero exit, depending on where it lands.
        if body.get("is_error"):
            error_text = str(body.get("result"))
            if _looks_like_quota(error_text):
                quota_hit = True
            print(f"terse fluency: {alias} reported is_error "
                  f"({error_text[:200]}) — counted as a non-answer, not "
                  f"scored", file=sys.stderr)
            return None
        result = body.get("result")
        # Same contract as `openai_answerer`: no content and no call are one fact to every
        # consumer downstream — unanswered, so counted and never scored (#263/#268).
        if not isinstance(result, str) or not result.strip():
            print(f"terse fluency: {alias} returned no content "
                  f"(stop_reason={body.get('stop_reason')!r}) — counted as a non-answer, "
                  f"not scored", file=sys.stderr)
            return None
        return result

    return ask
