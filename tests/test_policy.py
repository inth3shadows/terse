"""Policy shell: selection, fail-closed defaults, lossless guarantee, round-trip."""

from __future__ import annotations

import json

import pytest

from terse import transforms
from terse.policy import Policy, Rule, apply, default_policy, load_policy

RECORDS = json.dumps({"result": [{"id": i, "url": "https://x.example/api/items", "ok": True}
                                 for i in range(20)]})


def _policy():
    return Policy(rules=[
        Rule(tool_glob="gh.*", tiers=("minify", "tabularize", "dictionary")),
        Rule(tool_glob="kb.*", tiers=("minify", "tabularize")),
        Rule(tool_glob="*.rate_limit", tiers=()),
    ])


def test_select_first_match_wins_else_default():
    p = _policy()
    assert p.select("gh.api.repos").tiers == ("minify", "tabularize", "dictionary")
    assert p.select("kb.read.search").tiers == ("minify", "tabularize")
    assert p.select("gh.api.rate_limit").tiers == ("minify", "tabularize", "dictionary")  # gh.* first
    # unmatched -> lossless default, never empty
    assert p.select("unknown.tool").tiers == ("minify", "tabularize", "dictionary")


def test_apply_is_lossless_for_every_tier_combo():
    p = _policy()
    for tool in ("gh.api.repos", "kb.read.search", "totally.unknown"):
        result = apply(RECORDS, tool, p)
        assert transforms.decompress(result.text) == json.loads(RECORDS)


def test_skip_passes_through_unchanged():
    p = Policy(rules=[Rule(tool_glob="x.rate_limit", tiers=())])
    result = apply(RECORDS, "x.rate_limit", p)
    assert result.skipped is True
    assert result.text == RECORDS


def test_non_json_passes_through():
    p = default_policy()
    result = apply("not json at all", "any.tool", p)
    assert result.skipped is True
    assert result.text == "not json at all"


def test_lossy_field_is_warned_not_executed():
    p = Policy(rules=[Rule(tool_glob="gh.*", tiers=("minify", "tabularize"),
                           fields={"result[].body": {"lossy": "truncate"}})])
    result = apply(RECORDS, "gh.api.x", p)
    assert any("lossy" in w for w in result.warnings)
    # still lossless despite the lossy request
    assert transforms.decompress(result.text) == json.loads(RECORDS)


def test_tabularize_only_smaller_than_passthrough_but_lossless():
    p = Policy(rules=[Rule(tool_glob="*", tiers=("minify", "tabularize"))])
    result = apply(RECORDS, "any", p)
    assert len(result.text) < len(RECORDS)
    assert transforms.decompress(result.text) == json.loads(RECORDS)


def test_load_example_policy_validates(tmp_path):
    # The shipped example must parse and select sensibly.
    import pathlib
    example = pathlib.Path(__file__).resolve().parents[1] / "policy.example.json"
    p = load_policy(example)
    assert p.select("gh.api.repos").tiers == ("minify", "tabularize", "dictionary")
    assert p.select("kb.read.list_nodes").tiers == ("minify", "tabularize")
    assert p.select("gh.api.rate_limit").tiers == ("minify", "tabularize", "dictionary")  # gh.* before *.rate_limit
    assert p.select("ci.api.rate_limit").tiers == ()


def test_invalid_tier_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 1, "policies": [{"match": {"tool": "*"}, "tiers": ["bogus"]}]}))
    with pytest.raises(ValueError):
        load_policy(bad)


def test_unsupported_version_rejected(tmp_path):
    bad = tmp_path / "v2.json"
    bad.write_text(json.dumps({"version": 2}))
    with pytest.raises(ValueError):
        load_policy(bad)
