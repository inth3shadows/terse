"""Tier-1 lossy (truncate-first slice): the path resolver, truncation, the critical
denylist, and the acceptable-loss gate that replaces the round-trip gate once data is
intentionally dropped."""

from __future__ import annotations

import json

from terse import lossy, transforms
from terse.policy import Policy, Rule, apply


def _rule(fields):
    return Rule(tool_glob="*", tiers=("minify", "tabularize", "dictionary"), fields=fields)


# --- path resolver ---
def test_parse_path_subset():
    assert lossy._parse_path("a") == ["a"]
    assert lossy._parse_path("a.b") == ["a", "b"]
    assert lossy._parse_path("result[].body") == ["result", "[]", "body"]
    assert lossy._parse_path("[].body") == ["[]", "body"]


# --- truncation ---
def test_truncate_string_and_list_annotate_loss():
    assert lossy._truncate("x" * 10, 4) == "xxxx…⟨+6 chars⟩"
    assert lossy._truncate([1, 2, 3, 4, 5], 2) == [1, 2, "…⟨+3 items⟩"]
    assert lossy._truncate("short", 100) == "short"          # under cap -> untouched
    assert lossy._truncate(42, 1) == 42                        # non-truncatable scalar


def test_apply_lossy_truncates_marked_record_field():
    obj = {"result": [{"id": i, "body": "B" * 50} for i in range(3)]}
    out = lossy.apply_lossy(obj, _rule({"result[].body": {"lossy": "truncate", "max": 5}}))
    assert all(r["body"] == "BBBBB…⟨+45 chars⟩" for r in out["result"])
    assert all(r["id"] == i for i, r in enumerate(out["result"]))   # other fields intact
    assert obj["result"][0]["body"] == "B" * 50                     # original untouched


def test_critical_field_is_never_truncated():
    obj = {"result": [{"id": 1, "body": "B" * 50}]}
    # same field marked BOTH lossy and critical -> critical wins, nothing dropped
    rule = _rule({"result[].body": {"lossy": "truncate", "max": 5, "critical": True}})
    out = lossy.apply_lossy(obj, rule)
    assert out == obj
    assert lossy.critical_paths(rule) == {"result[].body"}


# --- acceptable-loss gate ---
def test_gate_accepts_a_valid_truncation():
    obj = {"result": [{"id": i, "body": "B" * 50} for i in range(3)]}
    rule = _rule({"result[].body": {"lossy": "truncate", "max": 5}})
    out = lossy.apply_lossy(obj, rule)
    assert lossy.acceptable_loss(obj, out, rule)


def test_gate_rejects_change_to_an_unmarked_field():
    obj = {"result": [{"id": 1, "body": "B" * 50, "url": "keep"}]}
    rule = _rule({"result[].body": {"lossy": "truncate", "max": 5}})
    out = lossy.apply_lossy(obj, rule)
    out["result"][0]["url"] = "TAMPERED"          # an unmarked field changed
    assert not lossy.acceptable_loss(obj, out, rule)


def test_gate_rejects_a_non_truncation_edit_of_a_marked_field():
    obj = {"result": [{"id": 1, "body": "B" * 50}]}
    rule = _rule({"result[].body": {"lossy": "truncate", "max": 5}})
    out = {"result": [{"id": 1, "body": "totally rewritten, not a prefix"}]}
    assert not lossy.acceptable_loss(obj, out, rule)


def test_gate_fails_closed_on_shape_mismatch():
    # path says result[].body but result isn't a list -> PathError -> unacceptable
    obj = {"result": {"body": "x" * 50}}
    rule = _rule({"result[].body": {"lossy": "truncate", "max": 5}})
    assert not lossy.acceptable_loss(obj, obj, rule)


# --- end to end through policy.apply ---
def test_apply_executes_truncate_and_warns_it_is_lossy():
    obj = {"result": [{"id": i, "body": "LONGBODY" * 20} for i in range(8)]}
    raw = json.dumps(obj)
    p = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"),
                           fields={"result[].body": {"lossy": "truncate", "max": 10}})])
    res = apply(raw, "gh.api.x", p)
    assert any("NOT lossless" in w for w in res.warnings)
    decoded = transforms.decompress(res.text)
    assert all(r["body"].endswith("chars⟩") for r in decoded["result"])   # truncated
    assert [r["id"] for r in decoded["result"]] == list(range(8))          # ids preserved
    assert len(res.text) < len(raw)


def test_apply_skips_lossy_when_shape_mismatches_and_stays_lossless():
    # body field marked truncate but result is an object, not a list -> fail closed
    raw = json.dumps({"result": {"body": "x" * 50}})
    p = Policy(rules=[Rule("gh.*", ("minify", "tabularize"),
                           fields={"result[].body": {"lossy": "truncate", "max": 5}})])
    res = apply(raw, "gh.api.x", p)
    assert transforms.decompress(res.text) == json.loads(raw)   # lossless fallback
    assert any("skipped" in w for w in res.warnings)
