"""policy generate (#24): conservative, lossless auto-authoring from a corpus."""
from __future__ import annotations

import json

from terse.policy import load_policy
from terse.policy_gen import generate_policy


def _env(tool: str, obj_or_text):
    raw = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
    return {"tool": tool, "raw": raw}


# A verbose record list (repeated keys + repeated values) — compresses well on both
# tabularize and dictionary.
def _records(n=20):
    return {"result": [{"id": i, "status": "active",
                        "url": "https://x.example/api/items"} for i in range(n)]}


# Repeated keys but (almost) unique values — tabularize pays, dictionary barely does.
def _unique_value_records(n=20):
    return {"result": [{"id": i, "name": f"item-name-number-{i}",
                        "score": i * 7 + 1} for i in range(n)]}


def test_high_savings_tool_gets_lossless_tiers():
    doc, rows = generate_policy([_env("gh.items", _records()) for _ in range(3)])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "gh.items")
    assert rule["tiers"][:2] == ["minify", "tabularize"]
    assert "dictionary" in rule["tiers"]                 # repeated values pay for it
    row = next(r for r in rows if r["tool"] == "gh.items")
    assert row["saved_pct"] > 5.0


def test_compact_object_tool_is_passthrough():
    doc, rows = generate_policy([_env("status.ping", {"ok": True})])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "status.ping")
    assert rule["tiers"] == []
    assert "threshold" in next(r for r in rows if r["tool"] == "status.ping")["reason"]


def test_non_json_payload_disqualifies_the_tool():
    # even one non-JSON result among a tool's payloads forces passthrough — the policy
    # matches by tool name, so we can't compress only "most" of its results.
    doc, _ = generate_policy([_env("logs.tail", _records()),
                              _env("logs.tail", "2026-06-30 12:00:00 INFO started\n...")])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "logs.tail")
    assert rule["tiers"] == []


def test_dictionary_dropped_when_marginal_below_threshold():
    doc, rows = generate_policy([_env("rc.syms", _unique_value_records()) for _ in range(3)])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "rc.syms")
    row = next(r for r in rows if r["tool"] == "rc.syms")
    if row["tiers"]:                                     # cleared the total threshold
        assert "dictionary" not in rule["tiers"]
        assert row["dict_pct"] < 5.0


def test_threshold_is_respected():
    payloads = [_env("gh.items", _records()) for _ in range(2)]
    # An absurdly high bar makes even a well-compressing tool passthrough.
    doc, _ = generate_policy(payloads, threshold=99.0)
    assert next(p for p in doc["policies"] if p["match"]["tool"] == "gh.items")["tiers"] == []


def test_rows_sorted_by_savings_desc():
    doc, rows = generate_policy([
        _env("gh.items", _records()),
        _env("status.ping", {"ok": True}),
    ])
    assert [r["tool"] for r in rows] == sorted(
        [r["tool"] for r in rows], key=lambda t: -next(x["saved_pct"] for x in rows if x["tool"] == t))
    assert rows[0]["saved_pct"] >= rows[-1]["saved_pct"]


def test_generated_policy_loads_back(tmp_path):
    doc, _ = generate_policy([
        _env("gh.items", _records()),
        _env("status.ping", {"ok": True}),
    ])
    p = tmp_path / "gen.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    policy = load_policy(p)                              # must not raise
    # the high-savings tool resolves to a compressing rule; the compact one to passthrough
    assert policy.select("gh.items").tiers
    assert policy.select("status.ping").tiers == ()
