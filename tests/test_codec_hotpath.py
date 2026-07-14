"""#79: codec hot-path — memoized canonicalization + the shared depth cap.

Two halves. (1) `_build_canon_memo` must produce byte-identical output to `minify`
for every container, since dict_encode's counting/replacement walks and
compress_with's base now read the memo instead of re-serializing per level.
(2) A payload nested past `MAX_DEPTH` must be screened out at every boundary
(policy.apply, the proxy, measure) instead of RecursionError-ing the recursive
transforms.
"""

from __future__ import annotations

import json

import pytest

from terse import transforms
from terse.measure import measure_payload
from terse.policy import Policy, Rule
from terse.policy import apply as policy_apply
from terse.proxy import Interceptor


def _deep(depth: int, leaf=1):
    obj = leaf
    for _ in range(depth):
        obj = [obj]
    return obj


CANON_CASES = [
    pytest.param({}, id="empty-dict"),
    pytest.param([], id="empty-list"),
    pytest.param(1, id="bare-int"),
    pytest.param("just a string", id="bare-str"),
    pytest.param(None, id="bare-null"),
    pytest.param(2.5, id="bare-float"),
    pytest.param(["x", "x"], id="repeated-strings"),
    pytest.param({"a": {"b": [1, 2.5, None, True, "héllo ☃"]}, "c": "héllo ☃"}, id="unicode-nested"),
    pytest.param({"records": [{"id": i, "tag": "same"} for i in range(5)]}, id="records"),
    pytest.param({"escape\n\"quote": [" ", "\\", ""]}, id="escapes"),
    pytest.param([[], {}, [[]], [{}]], id="empty-containers-nested"),
    pytest.param({1: "int-key", 2.5: "float-key"}, id="non-str-keys-numeric"),
    pytest.param({True: "bool-key"}, id="non-str-key-bool"),
    pytest.param({None: "none-key"}, id="non-str-key-none"),
]


@pytest.mark.parametrize("obj", CANON_CASES)
def test_canon_matches_minify_everywhere(obj):
    # The whole correctness contract of the memo pass: assembling a container's
    # minified form from its children's forms is byte-identical to json.dumps on it —
    # including the key-coercion fallback for non-str keys.
    root, memo = transforms._build_canon_memo(obj)
    assert root == transforms.minify(obj)

    def walk(node):
        if isinstance(node, (list, dict)):
            assert memo[id(node)] == transforms.minify(node)
            for child in (node if isinstance(node, list) else node.values()):
                walk(child)

    walk(obj)


def test_dict_encode_still_encodes_identically():
    # The memo is a performance change only: same candidates, same legend, same data.
    obj = {"outer": [{"id": 1, "u": "https://x.example/long/shared/url"},
                     {"id": 2, "u": "https://x.example/long/shared/url"}] * 3,
           "dup_tree": [{"a": [1, 2, 3]}, {"a": [1, 2, 3]}]}
    data, legend = transforms.dict_encode(obj)
    assert legend  # the repeated URL/subtrees are worth aliasing
    assert transforms.dict_decode(data, legend) == obj
    assert transforms.roundtrip_ok(obj)


def test_dict_encode_serializes_scalars_only_not_per_level(monkeypatch):
    # Pre-#79, _count_value_nodes and _replace_nodes each called minify() at every
    # container level (>= 2*depth full re-serializations on a deep chain). The memo
    # pass calls minify only for scalar leaves (and the non-str-key fallback).
    depth = 60
    obj: dict = {"pair": ["dup-value", "dup-value"]}
    for _ in range(depth):
        obj = {"wrap": obj}
    calls = []
    real_minify = transforms.minify
    monkeypatch.setattr(transforms, "minify",
                        lambda o: (calls.append(1), real_minify(o))[1])
    transforms.dict_encode(obj)
    assert len(calls) < depth


def test_exceeds_depth_boundary():
    assert not transforms.exceeds_depth(_deep(transforms.MAX_DEPTH))
    assert transforms.exceeds_depth(_deep(transforms.MAX_DEPTH + 1))
    # dict nesting counts the same as list nesting
    deep_dicts: dict = {}
    for _ in range(transforms.MAX_DEPTH + 1):
        deep_dicts = {"d": deep_dicts}
    assert transforms.exceeds_depth(deep_dicts)
    assert not transforms.exceeds_depth({"shallow": [1, 2, {"x": "y"}]})


def test_measure_payload_survives_pathological_nesting():
    # Regression for the issue's crash: this RecursionError'd measure_payload.
    raw = "[" * 500 + "1" + "]" * 500
    row = measure_payload(raw)
    assert row["applicable"] is False
    assert row["roundtrip_ok"] is True
    assert row["saved_cl100k"]["tier_total"] == 0


def test_policy_apply_passes_through_too_deep_payload():
    raw = json.dumps(_deep(transforms.MAX_DEPTH + 5))
    pol = Policy(rules=[Rule("t.*", ("minify", "tabularize", "dictionary"))])
    applied = policy_apply(raw, "t.deep", pol)
    assert applied.skipped
    assert applied.text == raw
    assert any("deeper than" in w for w in applied.warnings)


def test_depth_at_cap_still_compresses():
    # The guard must not fire early: a payload AT the cap goes through the tiers.
    obj = _deep(transforms.MAX_DEPTH - 1, leaf={"k": "v"})
    pol = Policy(rules=[Rule("t.*", ("minify", "tabularize", "dictionary"))])
    applied = policy_apply(json.dumps(obj, indent=1), "t.ok", pol)
    assert not applied.skipped
    assert transforms.decompress(applied.text) == obj


def test_proxy_forwards_too_deep_payload_unchanged():
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))])
    inter = Interceptor(pol)
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "gh.deep"}}))
    deep_text = json.dumps(_deep(transforms.MAX_DEPTH + 5))
    reply = json.dumps({"jsonrpc": "2.0", "id": 3,
                        "result": {"content": [{"type": "text", "text": deep_text}]}})
    out = inter.transform_response(reply)
    assert json.loads(out) == json.loads(reply)  # payload untouched
    assert inter.last == {}                       # and never stored as a diff base
