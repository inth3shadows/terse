"""The answerer transport — pluggable `(system, user) -> reply` callables (#78 split).

The pure core (question generation + scoring) runs offline with no network or key;
the live backend (`openai_answerer` over stdlib urllib) reaches any OpenAI-compatible
endpoint — the broker pool or a loopback gateway — and adds zero new dependencies.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable

# The cleartext-credential rule lives in ONE place (transport.py) so every
# credential-bearing caller inherits it. dropeval's tool-calling answerer had no such
# check at all while this module did — the parity gap that motivated centralizing it.
# `_LOOPBACK_HOSTS` is re-exported (not redefined) to keep `fluency._LOOPBACK_HOSTS`
# importable for the existing tests while there is still only one definition.
from ..transport import _LOOPBACK_HOSTS, guard_cleartext_credential  # noqa: F401

# An answerer takes (system_prompt, user_prompt) and returns the model's reply text.
# Empty system_prompt means "no system message".
Answerer = Callable[[str, str], str]


def openai_answerer(base_url: str, api_key: str, model: str,
                    temperature: float = 0.0, timeout: int = 60) -> Answerer:
    """OpenAI-compatible /chat/completions answerer over stdlib urllib. Covers the
    broker pool (OpenRouter et al.) without an SDK dependency. temperature 0 for
    reproducibility."""
    # An http:// base URL sends `Authorization: Bearer <key>` in cleartext — refuse it for
    # a non-loopback host rather than silently leak the key on the wire.
    guard_cleartext_credential(base_url, bool(api_key), what="terse fluency")
    url = base_url.rstrip("/") + "/chat/completions"

    def ask(system: str, user: str) -> str:
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
        return data["choices"][0]["message"]["content"] or ""

    return ask
