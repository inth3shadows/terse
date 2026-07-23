"""Guards for the #137 ceiling measurement (`scripts/bench/text_alias_ceiling.py`).

The script's whole value is that a reported saving is a saving on a LOSSLESS encoding.
That property is easy to break silently — the block-then-phrase encoder originally aliased
spans that CONTAINED an earlier alias, nesting aliases inside legend values and making
decoding order-dependent. It corrupted 10 of 296 real files while still reporting cheerful
percentages, because the round-trip check ran per payload and nothing pinned the encoders
themselves. These tests pin them.

The script is loaded by path: `scripts/` is not an importable package, and putting the
research encoders in `src/terse` would ship measurement code in the wheel.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "text_alias_ceiling.py"


def _load():
    spec = importlib.util.spec_from_file_location("text_alias_ceiling", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tac = _load()

pytestmark = pytest.mark.skipif(tac.TOK("probe") is None,
                                reason="tiktoken absent; every assertion here is a token count")

# Repetition dense enough that every encoder finds something, and structured so the block
# pass fires first and the phrase pass then has alias units sitting in its stream — the
# exact arrangement that produced nested aliases.
BLOCKY = ("""\
def handler(request):
    validate(request.headers, schema=STRICT, allow_extra=False)
    record = build_record(request.body, request.user, request.trace_id)
    return respond(record, status=200)

""" * 6) + ("""\
def other(request):
    validate(request.headers, schema=STRICT, allow_extra=False)
    return respond(None, status=204)

""" * 4)

def _b36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out or "0"


CASES = {
    "blocky": BLOCKY,
    "prose": "the quick brown fox jumps over the lazy dog. " * 40,
    # Genuinely repetition-free: every line is a distinct base-36 run, so there is no
    # shared phrase for an encoder to fold. ("line N holds ..." is NOT repetition-free —
    # the scaffolding around N repeats on every line, which the encoder duly found.)
    "no-repeats": "".join(f"{_b36(i * 7919 + 104729)}\n" for i in range(120)),
    "unicode": ("⚠️ no covering tests found — src/resolution/frameworks/cargo-workspace.ts\n"
                * 30),
    "trailing-newlines": "alpha beta gamma delta\n\n\n" * 30,
    "tabs-and-gutter": "".join(f"{i}\t\t\texports = append(exports, extract(x)...)\n"
                               for i in range(60)),
    "empty-lines-only": "\n" * 50,
    "single-line": "a" * 5000,
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("encoder", [n for n, _ in tac.ENCODERS])
def test_every_encoder_round_trips(name: str, encoder: str) -> None:
    """Lossless on every shape, or the saving it reports is meaningless."""
    text = CASES[name]
    body, legend = dict(tac.ENCODERS)[encoder](text)
    assert tac.decode(body, legend) == text


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_alias_nests_inside_a_legend_value(name: str) -> None:
    """The regression that made decoding order-dependent.

    Fails without the `blocked` guard in `_alias_ngrams`: the phrase pass aliases a span
    covering an alias emitted by the block pass, so a legend value contains `~K` and
    decode order decides the result.
    """
    _, legend = tac.encode_blocks_then_phrases(CASES[name])
    for alias, value in legend.items():
        others = set(legend) - {alias}
        assert not (others & set(_aliases_in(value, others))), \
            f"legend[{alias!r}] contains a nested alias: {value!r}"


def _aliases_in(value: str, candidates: set[str]) -> set[str]:
    return {a for a in candidates if a in value}


def test_repetition_is_actually_found() -> None:
    """A guard against the tests above passing because the encoders do nothing."""
    assert tac.score(BLOCKY, "phrase") > 5.0
    assert tac.best(BLOCKY)[1] in {"line", "phrase", "both"}


def test_declines_when_there_is_nothing_to_gain() -> None:
    """No repetition -> no legend -> exactly 0.0%, never a negative from legend overhead."""
    assert tac.score(CASES["no-repeats"], "phrase") == 0.0
    assert tac.best(CASES["no-repeats"]) == (0.0, "-")


def test_saving_is_charged_the_legend_cost() -> None:
    """The reported number must include the legend, not just the shrunken body."""
    body, legend = tac.encode_phrases(BLOCKY)
    assert legend
    raw = tac.TOK(BLOCKY)
    body_only = 100.0 * (raw - tac.TOK(body)) / raw
    assert tac.score(BLOCKY, "phrase") < body_only
    assert tac.legend_cost(legend) > 0
    assert tac.legend_cost({}) == 0


def test_a_lossy_encoder_cannot_report_a_saving(monkeypatch: pytest.MonkeyPatch) -> None:
    """`score` asserts the round-trip itself — the property the whole script rests on."""

    def lossy(text: str) -> tuple[str, dict[str, str]]:
        return "totally different", {"~0": "unrelated"}

    monkeypatch.setattr(tac, "ENCODERS", (("phrase", lossy),))
    with pytest.raises(AssertionError, match="not lossless"):
        tac.score(BLOCKY, "phrase")


def test_decode_handles_two_digit_aliases() -> None:
    """`~1` must not eat the `~1` inside `~10`."""
    legend = {"~1": "ONE", "~10": "TEN"}
    assert tac.decode("~10|~1", legend) == "TEN|ONE"


def test_production_falls_back_to_raw_for_non_json() -> None:
    """The marginal column models what the proxy sends, which for text is the payload."""
    text = "## Blast radius\n\nnot json at all\n"
    assert tac.production(text) == text


def test_production_is_the_shipped_tier_path_for_json() -> None:
    """Enough records that tabularize clears its own header cost.

    A 2-record payload comes out LARGER — terse has no emit-only-if-smaller guard on the
    lossless stage — and `production` deliberately reproduces that rather than papering
    over it, because the marginal column must model what the proxy really sends.
    """
    raw = json.dumps({"rows": [{"alpha": i, "beta": "constant", "gamma": i * 3}
                               for i in range(40)]}, indent=2)
    out = tac.production(raw)
    assert out != raw
    assert tac.TOK(out) < tac.TOK(raw)
