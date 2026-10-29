"""`policy generate` reports, and decides on, only what the generated rule delivers (#562).

The tool's total used to be the sum of every row's `tier_total`, which includes the saving
of an opt-in tier the decision then leaves out. That saving could lift a tool over the
passthrough threshold and inflate its reported percentage. A Python-literal fold also
counted the dictionary tier's work as its own, while the runtime fold runs without
`dictionary` when the rule has none.
"""

from __future__ import annotations

import json

import pytest

from terse import policy, policy_gen
from terse.policy import Policy, Rule
from terse.policy_gen import generate_policy
from terse.tokenize import count_cl100k


def _row(*, minify=0, tabularize=0, dictionary=0, embedded=0, pyliteral=0, raw=100,
         applicable=True, embedded_ok=True):
    saved = {"minify": minify, "tabularize": tabularize, "dictionary": dictionary,
             "embedded": embedded, "pyliteral": pyliteral,
             "tier_total": minify + tabularize + dictionary + embedded + pyliteral}
    return {"shape": "records", "applicable": applicable, "roundtrip_ok": True,
            "embedded_ok": embedded_ok,
            "cl100k": {"raw": raw, "minified": raw, "tabular": raw, "compressed": raw,
                       "embedded": raw},
            "saved_cl100k": saved}


def _decide(monkeypatch, rows, threshold=5.0):
    it = iter(rows)
    monkeypatch.setattr(policy_gen, "measure_payload", lambda raw: next(it))
    envs = [{"tool": "t", "raw": f"payload {i}"} for i in range(len(rows))]
    (row,) = generate_policy(envs, threshold=threshold, join_blocks=False)[1]
    return row


@pytest.mark.parametrize("tier", ["dictionary", "embedded", "pyliteral"])
def test_a_tier_below_its_bar_does_not_lift_the_tool(monkeypatch, tier):
    """Tabularize 3% plus a dropped tier's 3% used to read as 6.0% saved at threshold 5;
    the rule delivers 3%, so the tool is passthrough."""
    row = _decide(monkeypatch, [_row(tabularize=3, **{tier: 3})])
    assert row["tiers"] == [], row["reason"]
    assert "3.0% < 5.0%" in row["reason"]


@pytest.mark.parametrize("tier", ["dictionary", "embedded"])
def test_a_tier_below_its_bar_is_not_in_the_reported_saving(monkeypatch, tier):
    row = _decide(monkeypatch, [_row(tabularize=10, **{tier: 3})])
    assert row["tiers"] == ["minify", "tabularize"]
    assert row["saved_pct"] == 10.0


def test_enabled_tiers_still_add_up(monkeypatch):
    row = _decide(monkeypatch, [_row(minify=2, tabularize=10, dictionary=6, embedded=7)])
    assert row["tiers"] == ["minify", "tabularize", "dictionary", "embedded"]
    assert row["saved_pct"] == 25.0


def test_an_embedded_round_trip_failure_still_drops_its_saving(monkeypatch):
    row = _decide(monkeypatch, [_row(tabularize=10, embedded=20, embedded_ok=False)])
    assert "embedded" not in row["tiers"] and row["saved_pct"] == 10.0


def test_a_python_literal_rows_dictionary_share_ships_only_with_pyliteral(monkeypatch):
    """The dictionary saving measured inside a fold is delivered only by a rule with BOTH
    tiers. With pyliteral below its bar, that share must not enable `dictionary`."""
    lit = _row(applicable=False, pyliteral=3, dictionary=8)
    row = _decide(monkeypatch, [lit])
    assert row["tiers"] == [], row["reason"]
    row = _decide(monkeypatch, [_row(applicable=False, pyliteral=6, dictionary=8)])
    assert row["tiers"] == ["minify", "tabularize", "dictionary", "pyliteral"]
    assert row["saved_pct"] == 14.0


def test_a_python_literal_row_left_unread_is_counted_as_passed_through(monkeypatch):
    rows = [_row(tabularize=40), _row(applicable=False, pyliteral=2)]
    row = _decide(monkeypatch, rows)
    assert "pyliteral" not in row["tiers"]
    assert "(1/2 non-JSON, passed through)" in row["reason"]
    rows = [_row(tabularize=40), _row(applicable=False, pyliteral=20)]
    assert "non-JSON" not in _decide(monkeypatch, rows)["reason"]


# --- end to end: reported equals delivered ---

URLS = [f"https://example.com/a/very/long/path/to/resource/number/{k}/index.html"
        for k in range(3)]


def _literal_rows(n=60):
    return repr([{"id": i, "url": URLS[i % 3],
                  "desc": "the same long description sentence repeated on every row"}
                 for i in range(n)])


def _delivered_pct(raws, tiers):
    pol = Policy(rules=[Rule("t", tuple(tiers))])
    raw_tok = sum(count_cl100k(r) for r in raws)
    out_tok = sum(count_cl100k(policy.apply(r, "t", pol).text) for r in raws)
    return (raw_tok - out_tok) / raw_tok * 100


@pytest.mark.parametrize("threshold", [5.0, 30.0])
def test_a_python_literal_tool_reports_what_its_rule_delivers(threshold):
    """The review repro: reported 75.1% for a rule that delivered 23.2%."""
    raws = [_literal_rows(60), _literal_rows(45)]
    (row,) = generate_policy([{"tool": "t", "raw": r} for r in raws],
                             threshold=threshold)[1]
    if not row["tiers"]:
        assert row["saved_pct"] < threshold
        return
    assert row["saved_pct"] == pytest.approx(_delivered_pct(raws, row["tiers"]), abs=0.15)


def test_a_json_tool_reports_what_its_rule_delivers():
    raws = [json.dumps({"items": [{"id": i, "state": "open", "url": URLS[i % 3]}
                                  for i in range(n)]}) for n in (20, 30)]
    (row,) = generate_policy([{"tool": "t", "raw": r} for r in raws], threshold=5.0)[1]
    assert row["tiers"]
    assert row["saved_pct"] == pytest.approx(_delivered_pct(raws, row["tiers"]), abs=0.15)
