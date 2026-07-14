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


def test_select_falls_back_to_bare_name_for_multiproxy_peer_qualified_tool():
    # Regression: a corpus captured through multiproxy stores each payload under a
    # peer-qualified name (e.g. "gh__gh.api.repos") to avoid same-named-tool
    # collisions across peers, but a policy rule is authored against the downstream
    # tool's own bare name — select() must still find it, not silently fall through
    # to the lossless default (which made fluency --drop-eval report zero signal for
    # any multiproxy-captured corpus).
    p = _policy()
    assert p.select("gh__gh.api.repos").tiers == ("minify", "tabularize", "dictionary")
    assert p.select("gh__kb.read.search").tiers == ("minify", "tabularize")
    # a rule authored for the QUALIFIED name still wins outright when one exists
    qualified = Policy(rules=[Rule(tool_glob="gh__special.*", tiers=())])
    assert qualified.select("gh__special.tool").tiers == ()
    # single-proxy corpora (no "__") are completely unaffected
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


def test_deferred_lossy_mode_is_warned_not_executed():
    # summarize is still deferred — warned and left lossless (truncate + drop-to-retrieve
    # are now implemented; see test_lossy.py / test_drop.py)
    p = Policy(rules=[Rule(tool_glob="gh.*", tiers=("minify", "tabularize"),
                           fields={"result[].body": {"lossy": "summarize"}})])
    result = apply(RECORDS, "gh.api.x", p)
    assert any("not implemented" in w for w in result.warnings)
    assert transforms.decompress(result.text) == json.loads(RECORDS)


def test_truncate_on_absent_field_is_lossless_noop():
    # RECORDS has no 'body' field, so truncate finds nothing to cut -> stays lossless
    p = Policy(rules=[Rule(tool_glob="gh.*", tiers=("minify", "tabularize"),
                           fields={"result[].body": {"lossy": "truncate"}})])
    result = apply(RECORDS, "gh.api.x", p)
    assert transforms.decompress(result.text) == json.loads(RECORDS)


def test_has_terse_marker_detects_reserved_keys_at_any_depth():
    assert transforms.has_terse_marker({"__terse_table__": 1, "cols": [], "rows": []})
    assert transforms.has_terse_marker({"a": [{"__terse_dict__": 1}]})        # nested in a list
    assert transforms.has_terse_marker({"x": {"y": {"__terse_diff__": 1}}})   # deeply nested
    assert not transforms.has_terse_marker({"result": [{"id": 1}, {"id": 2}]})
    assert not transforms.has_terse_marker({"terse_table": 1, "~0": "not a marker"})


def test_marker_collision_payload_passes_through_uncompressed():
    # A payload that already carries a reserved marker can't be compressed: the consumer
    # reads the marker per the primer and would mis-reconstruct the user's own dict. The
    # guard leaves it verbatim and warns, rather than silently corrupting it (#6).
    collide = json.dumps({"__terse_table__": 1, "cols": ["a"], "rows": [[1]]})
    p = _policy()
    result = apply(collide, "gh.api.x", p)
    assert result.skipped is True
    assert result.text == collide                                  # emitted verbatim
    assert any("reserved terse marker" in w for w in result.warnings)
    # the danger the guard averts: compressing then decompressing this payload mangles it
    assert transforms.decompress(transforms.compress(json.loads(collide))) != json.loads(collide)


def test_marker_collision_guard_does_not_touch_normal_payloads():
    p = _policy()
    result = apply(RECORDS, "gh.api.x", p)
    assert result.skipped is False
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


def test_load_policy_rejects_unknown_keys_at_every_level(tmp_path):
    # A typo'd key silently reverting to default behavior is a trap — the loader
    # rejects unknown keys loudly at every level (audit fix #3). "_"-prefixed
    # annotation keys (policy_gen's _comment/_suggested_fields*) stay exempt.
    import pytest

    from terse.policy import load_policy

    def _write(doc):
        p = tmp_path / "p.json"
        p.write_text(json.dumps(doc))
        return p

    base = {"version": 1, "defaults": {"tiers": ["minify"]}, "policies": []}

    for doc, needle in [
        ({**base, "polices": []}, "polices"),                        # top-level typo
        ({**base, "diff_keyframe_intervall": 3}, "intervall"),       # top-level typo
        ({**base, "defaults": {"tiers": ["minify"], "teirs": []}}, "teirs"),
        ({**base, "policies": [{"match": {"tool": "x"}, "tiers": [], "feilds": {}}]}, "feilds"),
        ({**base, "policies": [{"match": {"tool": "x", "name": "y"}, "tiers": []}]}, "name"),
    ]:
        with pytest.raises(ValueError, match=needle):
            load_policy(_write(doc))

    # underscore-prefixed annotations pass at every level (the policy_gen convention)
    ok = {"version": 1, "_comment": "hi",
          "defaults": {"tiers": ["minify"], "_note": "x"},
          "policies": [{"_comment": "c", "match": {"tool": "x", "_why": "w"},
                        "tiers": [], "_suggested_fields": {"a": {}}}]}
    pol = load_policy(_write(ok))
    assert pol.rules[0].tool_glob == "x"
