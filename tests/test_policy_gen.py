"""policy generate (#24): conservative, lossless auto-authoring from a corpus."""
from __future__ import annotations

import json

from terse.policy import load_policy
from terse.policy_gen import generate_policy, merge_policy


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


# A record list dominated by a huge, unique field (an embedding-like vector) — the drop-to-
# retrieve signature: lossless folding is powerless (nothing repeats) but the field is most
# of the payload.
def _blob_records(n=20):
    return {"result": [{"id": i, "status": "active",
                        "embedding": json.dumps([round((i * 100 + j) * 0.001, 3)
                                                 for j in range(200)])}
                       for i in range(n)]}


def test_drop_candidate_suggested_for_large_unique_field():
    doc, rows = generate_policy([_env("kb.nodes", _blob_records()) for _ in range(2)])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "kb.nodes")
    assert rule["_suggested_fields"] == {"result[].embedding": {"lossy": "drop-to-retrieve"}}
    # small / low-cardinality fields are NOT suggested
    assert "result[].status" not in rule["_suggested_fields"]   # repeated -> low cardinality
    assert "result[].id" not in rule["_suggested_fields"]       # tiny
    # the note flags it as lossy + opt-in
    assert "LOSSY" in rule["_suggested_fields_note"]


def test_suggestion_is_inactive_when_loaded():
    # `_suggested_fields` is NOT `fields`, so the loader enables no lossy op — stays lossless.
    doc, _ = generate_policy([_env("kb.nodes", _blob_records()) for _ in range(2)])
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "pol.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        pol = load_policy(p)
    assert pol.select("kb.nodes").fields == {}      # suggestion did not become active
    assert not pol.has_drop()                        # nothing enables drop-to-retrieve


def test_drop_candidate_appears_even_when_tier_decision_is_passthrough():
    # A tool whose lossless savings fall below threshold still gets the suggestion: the
    # highest-value drop case (kb embedding) is exactly a low-lossless-savings tool.
    doc, rows = generate_policy([_env("kb.nodes", _blob_records())], threshold=99.0)
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "kb.nodes")
    assert rule["tiers"] == []                                   # forced passthrough
    assert "result[].embedding" in rule["_suggested_fields"]     # suggestion survives


def test_top_level_record_list_yields_bracket_path():
    recs = [{"id": i, "embedding": json.dumps([float(i * 100 + j) for j in range(200)])}
            for i in range(20)]
    doc, _ = generate_policy([_env("x.list", recs)])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "x.list")
    assert "[].embedding" in rule.get("_suggested_fields", {})


def test_no_suggestion_when_no_field_qualifies():
    doc, _ = generate_policy([_env("gh.items", _records()) for _ in range(2)])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "gh.items")
    assert "_suggested_fields" not in rule                        # small, foldable fields only


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


def test_classify_field_role():
    from terse.policy_gen import classify_field_role
    for n in ["id", "result[].name", "path", "commandLine", "uuid", "status"]:
        assert classify_field_role(n) == "identity", n
    for n in ["evidence", "result[].bodyText", "notes", "description", "rationale"]:
        assert classify_field_role(n) == "prose", n
    for n in ["principle", "embedding", "result[].verdict", "foobar"]:
        assert classify_field_role(n) == "unknown", n


# A record with a large+unique IDENTITY field (name), a PROSE field, and an UNKNOWN field —
# all three clear the size/uniqueness/share thresholds, so only role distinguishes them.
def _mixed_records(n=20):
    return {"result": [{"id": i,
                        "name": "n" * 250 + str(i),          # identity, large -> must be EXCLUDED
                        "description": "d" * 250 + str(i),   # prose -> ranked first
                        "principle": "p" * 250 + str(i)}     # unknown -> after prose, flagged
                       for i in range(n)]}


def test_identity_field_excluded_and_prose_ranked_first():
    doc, rows = generate_policy([_env("kb.x", _mixed_records()) for _ in range(2)])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "kb.x")
    sug = rule["_suggested_fields"]
    assert "result[].description" in sug and "result[].principle" in sug
    assert "result[].name" not in sug        # identity excluded despite large+unique+high-share
    assert "result[].id" not in sug
    # prose ranks before unknown in both the suggestion and the report rows
    keys = list(sug.keys())
    assert keys.index("result[].description") < keys.index("result[].principle")
    dr = next(r for r in rows if r["tool"] == "kb.x")["drop_rows"]
    assert [d["role"] for d in dr] == ["prose", "unknown"]
    # the note carries the role tags, the dropeval gate, and the load-bearing caution
    note = rule["_suggested_fields_note"]
    assert "[prose]" in note and "[unknown]" in note
    assert "--drop-eval" in note and "LOAD-BEARING" in note


def test_activate_suggestions_promotes_inactive_to_fields():
    from terse.policy_gen import activate_suggestions
    doc = {"version": 1, "policies": [
        {"match": {"tool": "kb.x"}, "tiers": ["minify"],
         "_suggested_fields": {"result[].body": {"lossy": "drop-to-retrieve"}},
         "_suggested_fields_note": "n"},
        {"match": {"tool": "gh.y"}, "tiers": ["minify"]},
    ]}
    out = activate_suggestions(doc)
    p0 = out["policies"][0]
    assert p0["fields"] == {"result[].body": {"lossy": "drop-to-retrieve"}}   # promoted
    assert "_suggested_fields" not in p0 and "_suggested_fields_note" not in p0
    assert "fields" not in out["policies"][1]                                  # untouched
    assert "_suggested_fields" in doc["policies"][0]                           # original intact (deep copy)


# --- #136: merge_policy — re-tuning an EXISTING policy without destroying it ---

def _gen(*rules):
    return {"version": 1, "policies": [{"match": {"tool": t}, "tiers": list(ti),
                                        "_comment": "generated"} for t, ti in rules]}


def test_merge_preserves_every_key_the_corpus_cannot_decide():
    # capture / structured / active fields are safety decisions a payload cannot inform.
    # A regeneration path that reverses them would be the one hole in terse's fail-safe
    # posture (#85, #135).
    existing = {"version": 1, "never_lossy_servers": ["secret-broker"],
                "policies": [{"match": {"tool": "kb.*"}, "tiers": ["minify"],
                              "capture": False, "structured": "leave",
                              "fields": {"result[].id": {"critical": True}}}]}
    merged, changes = merge_policy(existing, _gen(("kb.*", ("minify", "tabularize"))))
    rule = merged["policies"][0]
    assert rule["capture"] is False
    assert rule["structured"] == "leave"
    assert rule["fields"] == {"result[].id": {"critical": True}}
    assert merged["never_lossy_servers"] == ["secret-broker"]
    assert rule["tiers"] == ["minify", "tabularize"]          # the corpus DID decide this
    assert changes[0]["kind"] == "tiers"


def test_merge_proposes_tier_removal():
    # The motivating case: a tier decision that went stale. Additive-only could never fix it.
    existing = {"version": 1, "policies": [
        {"match": {"tool": "kb.*"}, "tiers": ["minify", "tabularize", "dictionary"]}]}
    merged, changes = merge_policy(existing, _gen(("kb.*", ("minify", "tabularize"))))
    assert merged["policies"][0]["tiers"] == ["minify", "tabularize"]
    assert changes[0] == {"tool": "kb.*", "kind": "tiers",
                          "before": ["minify", "tabularize", "dictionary"],
                          "after": ["minify", "tabularize"], "preserved": []}


def test_merge_keeps_rules_absent_from_the_corpus_in_position():
    existing = {"version": 1, "policies": [
        {"match": {"tool": "gh.*"}, "tiers": ["minify"]},
        {"match": {"tool": "runecho.*"}, "tiers": []}]}
    merged, changes = merge_policy(existing, _gen(("runecho.*", ("minify", "tabularize"))))
    assert [p["match"]["tool"] for p in merged["policies"]] == ["gh.*", "runecho.*"]
    assert merged["policies"][0]["tiers"] == ["minify"]        # untouched
    assert {c["tool"]: c["kind"] for c in changes}["gh.*"] == "preserved"


def test_merge_inserts_a_new_rule_before_any_glob_that_would_shadow_it():
    # first-match-wins: appending `kb.read.search` after `kb.*` makes it DEAD, and the
    # policy would look re-tuned while changing nothing.
    existing = {"version": 1, "policies": [{"match": {"tool": "kb.*"}, "tiers": ["minify"]}]}
    merged, _ = merge_policy(existing, _gen(("kb.read.search", ("minify", "tabularize"))))
    order = [p["match"]["tool"] for p in merged["policies"]]
    assert order.index("kb.read.search") < order.index("kb.*")


def test_merge_appends_when_nothing_shadows_the_new_rule():
    existing = {"version": 1, "policies": [{"match": {"tool": "gh.*"}, "tiers": ["minify"]}]}
    merged, _ = merge_policy(existing, _gen(("runecho.structure", ("minify",))))
    assert [p["match"]["tool"] for p in merged["policies"]] == ["gh.*", "runecho.structure"]


def test_merge_leaves_an_unreachable_duplicate_alone():
    existing = {"version": 1, "policies": [
        {"match": {"tool": "kb.*"}, "tiers": ["minify"]},
        {"match": {"tool": "kb.*"}, "tiers": []}]}
    merged, changes = merge_policy(existing, _gen(("kb.*", ("minify", "tabularize"))))
    assert merged["policies"][0]["tiers"] == ["minify", "tabularize"]
    assert merged["policies"][1]["tiers"] == []               # already unreachable
    assert changes[1]["why"] == "unreachable duplicate"


def test_merge_does_not_mutate_its_inputs():
    existing = {"version": 1, "policies": [{"match": {"tool": "kb.*"}, "tiers": ["minify"],
                                            "capture": False}]}
    snapshot = json.dumps(existing, sort_keys=True)
    merge_policy(existing, _gen(("kb.*", ("minify", "tabularize"))))
    assert json.dumps(existing, sort_keys=True) == snapshot
