"""Guards for the #137 ceiling measurement (`scripts/bench/text_alias_ceiling.py`).

The script's value is that a reported saving is a saving on a LOSSLESS encoding, and that
nothing silently biases the number toward the conclusion being drawn. Both properties broke
once already:

* the block-then-phrase encoder aliased spans that CONTAINED an earlier alias, nesting
  aliases inside legend values and making decoding order-dependent — 10 of 296 real files
  corrupted while still reporting cheerful percentages;
* the alias namespace was hard-coded to `~K`, which is the SAME namespace `transforms`
  mints from, so on any payload terse's dictionary tier had touched the collision guard
  scored 0.0% — silently zeroing 84 of 626 live payloads, all of them the one tool #137 is
  about, in the direction of "no headroom".

Neither was catchable by reading the output. These tests pin the encoders, the sigil
choice, and the reporting arithmetic that turns them into a headline.

The script is loaded by path: `scripts/` is not an importable package, and putting the
research encoders in `src/terse` would ship measurement code in the wheel.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
    # Already carries terse's OWN alias namespace, on top of real redundancy — the
    # regression case. A terse-compressed payload looks exactly like this: a legend using
    # `~K`, then content. The aliaser must still be able to work on the content.
    "terse-aliased": ('{"__terse_dict__":1,"legend":{"~0":"src/resolution/frameworks/'
                      'cargo-workspace.ts","~1":"no covering tests found"}}\n' + BLOCKY),
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
    covering an alias emitted by the block pass, so a legend value contains an alias and
    decode order decides the result.
    """
    _, legend = tac.encode_blocks_then_phrases(CASES[name])
    for alias, value in legend.items():
        others = set(legend) - {alias}
        assert not (others & {a for a in others if a in value}), \
            f"legend[{alias!r}] contains a nested alias: {value!r}"


# ---------------------------------------------------------------- sigil choice


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_chosen_sigil_is_absent_from_the_payload(name: str) -> None:
    """Absence of the sigil is absence of collisions, for the whole legend at once."""
    assert tac.pick_sigil(CASES[name]) not in CASES[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_alias_collides_with_real_content(name: str) -> None:
    text = CASES[name]
    for encoder, _ in tac.ENCODERS:
        _, legend = dict(tac.ENCODERS)[encoder](text)
        assert not [a for a in legend if a in text], f"{encoder} minted a colliding alias"


def test_a_payload_carrying_terse_aliases_is_still_scored() -> None:
    """THE bias regression.

    `transforms` mints `~0, ~1, ...` from `ALIAS_SIGIL`, the same namespace this script
    used to hard-code. Any payload the dictionary tier had touched therefore tripped the
    collision guard and was scored 0.0% — 84 of 626 live payloads, all `codegraph_explore`,
    every one of them in the direction of the conclusion. A payload that already contains
    `~0` must still get a real, non-zero, round-tripping score.
    """
    text = CASES["terse-aliased"]
    assert "~0" in text and "~1" in text
    pct = tac.score(text, "phrase")
    assert pct > 1.0, "a terse-aliased payload scored as if it had no redundancy"
    body, legend = tac.encode_phrases(text)
    assert tac.decode(body, legend) == text


def test_a_payload_containing_every_sigil_falls_back_and_is_counted() -> None:
    text = "".join(tac.SIGILS) + "\n" + BLOCKY
    sigil = tac.pick_sigil(text)
    assert sigil not in text
    assert tac.LIMITS["no_free_sigil"] >= 1


# ---------------------------------------------------------------- encoder behavior


def test_repetition_is_actually_found() -> None:
    """A guard against the tests above passing because the encoders do nothing.

    Pinned near the real value, not at a floor a crippled encoder clears. The verdict this
    script feeds is "the number is small", so a silent weakening of the encoder is the one
    mutation that must never pass — and `> 5.0` did not catch it: with `max_aliases`
    lowered from 300 to 1, BLOCKY still scored 34%.
    """
    assert tac.score(BLOCKY, "phrase") == pytest.approx(43.0, abs=1.5)
    assert tac.best(BLOCKY)[1] in {"line", "phrase", "both"}


@pytest.mark.parametrize("knob,value", [("max_aliases", 1), ("budget", 3)])
def test_weakening_an_encoder_knob_is_visible(knob: str, value: int) -> None:
    """Every strength knob was mutable without failing a test. Each one moves the number
    the verdict is quoted in, so each needs to move a test."""
    weakened = tac.encode_phrases(BLOCKY, **{knob: value})[1]
    assert len(weakened) < len(tac.encode_phrases(BLOCKY)[1])


def test_the_reported_saving_literally_includes_the_legend() -> None:
    """The accounting, checked against the arithmetic rather than a proxy for it."""
    for name, _ in tac.ENCODERS:
        body, legend = dict(tac.ENCODERS)[name](BLOCKY)
        raw = tac.TOK(BLOCKY)
        expected = 100.0 * (raw - (tac.legend_cost(legend) + tac.TOK(body))) / raw
        assert tac.score(BLOCKY, name) == pytest.approx(expected)
        assert tac.legend_cost(legend) > 0


def test_legend_overhead_is_load_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LEGEND_ENTRY_OVERHEAD = 0` shipped green — the accept test could be neutered with
    no test noticing.

    The direction is worth recording, because it is not the obvious one: zeroing it makes
    the reported saving go DOWN (43.1% -> 42.5% on BLOCKY), because the greedy then accepts
    marginal candidates that consume spans a better later candidate would have used. So it
    is not simply a "ceiling comes out too high" knob — it changes what the greedy picks,
    which is exactly why it needs pinning rather than reasoning about.
    """
    charged = tac.score(BLOCKY, "phrase")
    monkeypatch.setattr(tac, "LEGEND_ENTRY_OVERHEAD", 0)
    assert tac.score(BLOCKY, "phrase") != pytest.approx(charged)


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
    """`score` re-checks the round-trip itself — and raises, so `python -O` cannot strip it."""

    def lossy(text: str, sigil: str | None = None) -> tuple[str, dict[str, str]]:
        return "totally different", {"@0": "unrelated"}

    monkeypatch.setattr(tac, "ENCODERS", (("phrase", lossy),))
    with pytest.raises(tac.NotLossless):
        tac.score(BLOCKY, "phrase")
    assert issubclass(tac.NotLossless, AssertionError)


def test_decode_handles_two_digit_aliases() -> None:
    """`~1` must not eat the `~1` inside `~10`."""
    legend = {"~1": "ONE", "~10": "TEN"}
    assert tac.decode("~10|~1", legend) == "TEN|ONE"


# ---------------------------------------------------------------- baseline


def test_baseline_is_the_shipped_policy_path_not_a_reimplementation(tmp_path: Path) -> None:
    """`policy.apply`, so passthrough / marker / depth / minify-fallback are shipped
    behavior rather than re-derived here. A `tiers: []` rule must yield the payload
    untouched — the case a direct `transforms.compress` call gets wrong, and the reason
    "what terse actually emits" has to go through the real entry point."""
    raw = json.dumps({"rows": [{"alpha": i, "beta": "constant"} for i in range(40)]},
                     indent=2)
    default = tac.make_baseline(tac.policy_mod.default_policy())
    assert tac.TOK(default(raw, "any.tool")) < tac.TOK(raw)

    doc = tmp_path / "p.json"
    doc.write_text(json.dumps({
        "version": 1,
        "defaults": {"tiers": ["minify", "tabularize", "dictionary"]},
        "policies": [{"match": {"tool": "quiet.*"}, "tiers": []}],
    }))
    passthrough = tac.make_baseline(tac.policy_mod.load_policy(doc))
    assert passthrough(raw, "quiet.thing") == raw
    assert tac.TOK(passthrough(raw, "loud.thing")) < tac.TOK(raw)


def test_baseline_leaves_non_json_alone() -> None:
    text = "## Blast radius\n\nnot json at all\n"
    base = tac.make_baseline(tac.policy_mod.default_policy())
    assert base(text, "codegraph_explore") == text


def _drop_policy(tmp_path: Path) -> Any:
    doc = tmp_path / "drop.json"
    doc.write_text(json.dumps({
        "version": 1,
        "defaults": {"tiers": ["minify", "tabularize", "dictionary"]},
        "policies": [{
            "match": {"tool": "*explore"},
            "tiers": ["minify", "tabularize", "dictionary"],
            "fields": {"$text.code_blocks": {"lossy": "drop-to-retrieve", "min": 100}},
        }],
    }))
    return tac.policy_mod.load_policy(doc)


LONG_TEXT_WITH_FENCE = (
    "## Exploration\n\nFound 3 symbols.\n\n### Source Code\n\n```go\n"
    + "".join(f"{i}\tfunc handler{i}(w http.ResponseWriter, r *http.Request) {{}}\n"
             for i in range(40))
    + "```\n\n### Blast radius\n\n- handler0 (a.go:1) — 4 callers\n")


def test_baseline_drop_sink_is_load_bearing(tmp_path: Path) -> None:
    """Deleting `drop_sink=` from `make_baseline` failed NO test, and silently turned the
    whole `--policy` table into the default-policy table (live long-text 86.6% -> 0.0%,
    fleet 25.1% -> 3.0%). The single number the verdict is quoted in rode on one keyword
    argument that nothing verified."""
    base = tac.make_baseline(_drop_policy(tmp_path))
    out = base(LONG_TEXT_WITH_FENCE, "codegraph_explore")
    assert tac.TOK(out) < tac.TOK(LONG_TEXT_WITH_FENCE) * 0.6, \
        "the drop rule did not fire — is the drop-sink still wired?"
    assert "__terse_dropped__" in out
    assert "### Blast radius" in out          # the prose survives; only the fence leaves


def test_authored_rule_counter_distinguishes_a_real_rule_from_the_default(
        tmp_path: Path) -> None:
    """The previous counter incremented on "the codec ran", which is true for the
    synthesized `*` default too — identical under a policy with zero authored rules and
    one with eight, and blind to every drop-path payload (`tiers == ()`)."""
    raw = json.dumps({"rows": [{"a": i} for i in range(30)]})
    tac.LIMITS.clear()
    tac.make_baseline(tac.policy_mod.default_policy())(raw, "anything")
    assert tac.LIMITS["authored_rule"] == 0

    tac.LIMITS.clear()
    base = tac.make_baseline(_drop_policy(tmp_path))
    base(LONG_TEXT_WITH_FENCE, "codegraph_explore")     # drop path: tiers == ()
    assert tac.LIMITS["authored_rule"] == 1, "the drop path must still count as a match"
    base(raw, "unmatched.tool")
    assert tac.LIMITS["authored_rule"] == 1
    tac.LIMITS.clear()


# ---------------------------------------------------------------- reporting math


def _cells(row: str) -> list[float]:
    return [float(p.rstrip("%")) for p in row.split() if p.endswith("%")]


def test_row_arithmetic() -> None:
    """The formulas that produce the headline. A sign flip or a wrong denominator here
    ships green otherwise — nothing else covers `_row`."""
    c = Counter({"n": 10, "raw": 1000, "prod": 800, "alias_raw": 100,
                 "alias_post": 80, "gzip_bytes": 600, "bytes": 1000})
    terse, alias_raw, alias_post, combined, gz = _cells(tac._row("x", c))
    assert terse == pytest.approx(20.0)          # 1000 -> 800
    assert alias_raw == pytest.approx(10.0)      # 100/1000, measured on RAW
    assert alias_post == pytest.approx(10.0)     # 80/800, measured on terse's OUTPUT
    assert combined == pytest.approx(28.0)       # 1000 -> (800 - 80)
    assert gz == pytest.approx(60.0)             # byte-weighted, byte domain


def test_row_gzip_is_byte_weighted_not_token_weighted() -> None:
    """gzip is a byte ratio; dividing it by a token count would silently mix domains."""
    c = Counter({"n": 1, "raw": 100, "prod": 100, "alias_raw": 0, "alias_post": 0,
                 "gzip_bytes": 500, "bytes": 1000})
    assert _cells(tac._row("x", c))[-1] == pytest.approx(50.0)


def test_combined_equals_terse_when_the_aliaser_finds_nothing() -> None:
    c = Counter({"n": 1, "raw": 1000, "prod": 700, "alias_raw": 0, "alias_post": 0,
                 "gzip_bytes": 0, "bytes": 1000})
    cells = _cells(tac._row("x", c))
    assert cells[0] == pytest.approx(cells[3])      # terse == combined


def test_marginal_column_is_fleet_points_not_a_percent_of_terses_output() -> None:
    """The verdict is quoted in fleet POINTS. `alias:post` is a percent of terse's OUTPUT;
    the two only coincide when terse's own saving is near zero, which it is not under the
    live policy. Printing the wrong one would misstate the headline by ~4x."""
    c = Counter({"n": 1, "raw": 1000, "prod": 200, "alias_raw": 0, "alias_post": 40,
                 "gzip_bytes": 0, "bytes": 1000})
    row = tac._row("x", c)
    assert _cells(row)[2] == pytest.approx(20.0)   # alias:post — 40/200, a % of terse's out
    marginal = float(next(p for p in row.split() if p.endswith("p")).rstrip("p"))
    assert marginal == pytest.approx(4.0)          # MARGINAL — 40/1000, fleet points


# ---------------------------------------------------------------- corpus population


def test_inflation_counter_sees_below_floor_payloads_but_the_guard_prevents_inflation(
        tmp_path: Path) -> None:
    """Two properties in one fixture, both about the below-floor population:

    1. The inflation counter's denominator includes payloads under the scoring floor —
       the two-population split, which a counter reusing the scoring filter would be blind
       to (the bug PR #153's counter fixed).
    2. Post-#154, the emit-only-if-smaller guard means such a payload — the classic
       can't-amortize-the-table-header case — is emitted at plain-minify size, so it is
       NOT counted as inflated. Before #154 this same fixture inflated; the guard is what
       makes the tripwire read zero on the live corpus.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Two records: too small to pay for the table header, and far below the scoring floor.
    tiny = json.dumps({"rows": [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]})
    (corpus / "tiny__aaaa.json").write_text(json.dumps(
        {"tool": "t.list", "shape": "pretty-json", "raw": tiny, "sha": "aaaa"}))

    res = tac.run_corpus(corpus, tac.policy_mod.default_policy())
    infl = res["inflation"]
    assert infl["n"] == 1                                  # the counter SAW it...
    assert infl["below_floor"] == 1, "fixture must sit below the scoring floor"
    assert infl["inflated_n"] == 0, "the #154 guard emits plain minify, so nothing inflates"
    assert infl["inflated_tok"] == 0
    assert res["by_shape"] == {}, "nothing was scored (below floor), yet it was still counted"
