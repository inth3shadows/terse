"""The answerer transport — pluggable `(system, user) -> reply` callables (#78 split).

The pure core (question generation + scoring) runs offline with no network or key;
the live backend (`openai_answerer` over stdlib urllib) reaches any OpenAI-compatible
endpoint — the broker pool or a loopback gateway — and adds zero new dependencies.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit

# Loopback hosts where cleartext http is safe (never leaves the machine), so a Bearer
# key over http to one of these is fine — a local LiteLLM/CCR gateway is a common setup.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# An answerer takes (system_prompt, user_prompt) and returns the model's reply text.
# Empty system_prompt means "no system message".
Answerer = Callable[[str, str], str]


def openai_answerer(base_url: str, api_key: str, model: str,
                    temperature: float = 0.0, timeout: int = 60) -> Answerer:
    """OpenAI-compatible /chat/completions answerer over stdlib urllib. Covers the
    broker pool (OpenRouter et al.) without an SDK dependency. temperature 0 for
    reproducibility."""
    parts = urlsplit(base_url)
    if api_key and parts.scheme == "http" and (parts.hostname or "").lower() not in _LOOPBACK_HOSTS:
        # An http:// base URL sends `Authorization: Bearer <key>` in cleartext — refuse
        # it for a non-loopback host rather than silently leak the key on the wire. A
        # loopback host (localhost LiteLLM/CCR) never leaves the machine, so it's allowed.
        raise ValueError(
            f"terse fluency: refusing to send an API key over cleartext http to "
            f"{parts.hostname!r} — use https, or a loopback host for a local gateway")
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
