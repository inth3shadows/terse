"""The answerer transport — pluggable `(system, user) -> reply` callables (#78 split).

The pure core (question generation + scoring) runs offline with no network or key;
the live backend (`openai_answerer` over stdlib urllib) reaches any OpenAI-compatible
endpoint — the broker pool or a loopback gateway — and adds zero new dependencies.
"""

from __future__ import annotations

import json
import sys
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
