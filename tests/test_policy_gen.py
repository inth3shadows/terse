"""policy generate (#24): conservative, lossless auto-authoring from a corpus."""
from __future__ import annotations

import json

from terse.measure import measure_joined, measure_payload
from terse.policy import load_policy
from terse.policy_gen import (
    _tool_decision,
    generate_policy,
    group_results,
    merge_policy,
)


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


def test_non_json_payload_no_longer_disqualifies_the_tool():
    # REVERSED in #147, deliberately. This used to assert passthrough on the grounds that
    # the policy matches by tool name so we can't compress only "most" of its results. The
    # premise was wrong: `policy.apply` passes a non-JSON payload through untouched at
    # runtime, so the tier costs nothing on those results — and the old rule silently
    # zeroed real savings whenever a server returned an error string among its records
    # (measured: 4 of 436 `kb.read.search` payloads).
    doc, _ = generate_policy([_env("logs.tail", _records()),
                              _env("logs.tail", "2026-06-30 12:00:00 INFO started\n...")])
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "logs.tail")
    assert rule["tiers"]
    assert "non-JSON, passed through" in rule["_comment"]


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


def _gen_with_suggestion(tool, tiers, suggested_field):
    doc = _gen((tool, tiers))
    doc["policies"][0]["_suggested_fields"] = {suggested_field: {"lossy": "drop-to-retrieve"}}
    return doc


def test_an_added_rule_surfaces_its_drop_suggestion_in_the_diff():
    """A new passthrough rule whose only value is its `_suggested_fields` (the #139
    codegraph_explore case) must not render as a bare '(new rule)' — the suggestion is the
    whole reason it exists. The `added` change record carries `suggests` so the diff can
    name it; the `suggestions` change kind never fired for a NEW rule, only a changed one."""
    existing = {"version": 1, "policies": [{"match": {"tool": "gh.*"}, "tiers": ["minify"]}]}
    _, changes = merge_policy(existing,
                              _gen_with_suggestion("codegraph_explore", (), "$text.code_blocks"))
    added = next(c for c in changes if c["kind"] == "added"
                 and c["tool"] == "codegraph_explore")
    assert added["suggests"] == ["$text.code_blocks"]


def test_an_added_rule_without_a_suggestion_has_an_empty_suggests():
    existing = {"version": 1, "policies": [{"match": {"tool": "gh.*"}, "tiers": ["minify"]}]}
    _, changes = merge_policy(existing, _gen(("codegraph_search", ("minify", "tabularize"))))
    added = next(c for c in changes if c["kind"] == "added")
    assert added["suggests"] == []


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


# --- #147: score RESULTS the way the proxy compresses them, not block by block ---

def _blocks(n):
    return [json.dumps({"id": i, "status": "active", "city": "Berlin",
                        "url": "https://example/api/items"}) for i in range(n)]


def test_measure_joined_beats_per_block_on_one_record_per_block():
    # The whole reason #147 exists: a server returning one record per content block has
    # nothing to fold within a block, and everything to fold across them.
    raws = _blocks(20)
    per_block = sum(measure_payload(r)["saved_cl100k"]["tier_total"] for r in raws)
    joined = measure_joined(raws)
    assert joined["saved_cl100k"]["tier_total"] > per_block
    assert joined["roundtrip_ok"] and joined["blocks"] == 20


def test_measure_joined_refuses_exactly_where_apply_joined_would():
    assert measure_joined(_blocks(1)) is None                      # nothing to join
    assert measure_joined([*_blocks(2), "Error executing tool"]) is None   # non-JSON block
    assert measure_joined([*_blocks(2), json.dumps([1, 2])]) is None       # not a record
    assert measure_joined([*_blocks(2),
                           json.dumps({"__terse_table__": 1})]) is None    # reserved marker


def test_group_results_splits_on_a_gap_and_keeps_a_burst_together():
    def env(ts, i):
        return {"tool": "t", "raw": json.dumps({"id": i}), "captured_at": ts}
    # three blocks a millisecond apart, then a gap, then two more
    envs = [env(1_000_000_000, 0), env(1_001_000_000, 1), env(1_002_000_000, 2),
            env(9_000_000_000, 3), env(9_001_000_000, 4)]
    groups = group_results(envs)["t"]
    assert [len(g) for g in groups] == [3, 2]


def test_group_results_treats_an_untimed_envelope_as_its_own_result():
    # Envelopes predating `captured_at` must not be silently joined with anything.
    envs = [{"tool": "t", "raw": json.dumps({"id": 0})},
            {"tool": "t", "raw": json.dumps({"id": 1})}]
    assert [len(g) for g in group_results(envs)["t"]] == [1, 1]


def test_one_non_json_payload_no_longer_disqualifies_a_whole_tool():
    # The live case: kb.read.search measured 16.7% saved and was marked passthrough
    # because 4 of 436 payloads were the server's `Error executing tool …` text. The
    # runtime passes a non-JSON payload through untouched, so the tier costs nothing there.
    groups = [_blocks(20), ["Error executing tool t: boom"]]
    row = _tool_decision("t", groups, 5.0)
    assert row["tiers"], row["reason"]
    assert "non-JSON, passed through" in row["reason"]


def test_a_mostly_text_tool_is_still_passthrough_on_the_threshold():
    # Suppressed for the right reason: non-JSON contributes 0 saved while its raw tokens
    # stay in the denominator, so the percentage falls below the threshold by itself.
    groups = [["some long prose answer " * 200] for _ in range(10)] + [_blocks(3)]
    row = _tool_decision("t", groups, 5.0)
    assert row["tiers"] == []
    assert "threshold" in row["reason"]


def test_a_roundtrip_failure_still_disqualifies_the_tool():
    marker = json.dumps({"__terse_table__": 1, "n": 1, "cols": ["a"], "rows": [[1]]})
    row = _tool_decision("t", [_blocks(5), [marker]], 5.0)
    assert row["tiers"] == []
    assert "round-trip" in row["reason"]


# --- #146 review findings ---

def test_a_new_rule_inherits_the_operator_keys_of_the_rule_it_displaces():
    # Without this the anti-shadowing insertion is a SAFETY HOLE: the new rule sits ahead
    # of the operator's, so that tool would run with capture ON and structured "auto".
    existing = {"version": 1, "policies": [
        {"match": {"tool": "kb.*"}, "tiers": [], "capture": False, "structured": "leave"}]}
    merged, changes = merge_policy(existing, _gen(("kb.read.search", ("minify", "tabularize"))))
    new = merged["policies"][0]
    assert new["match"]["tool"] == "kb.read.search"       # inserted ahead, as intended
    assert new["capture"] is False                         # ...but not at the cost of these
    assert new["structured"] == "leave"
    assert any(c["kind"] == "inherited" and c["tool"] == "kb.read.search" for c in changes)


def test_autotune_will_not_activate_a_lossy_selector_by_turning_tiers_on():
    # `policy.apply` suppresses the text-drop path entirely under `tiers: []`, so the
    # selector is inert today. Flipping tiers on would put a LOSSY transform live — which a
    # merge documented as lossless and operator-preserving must never do.
    existing = {"version": 1, "policies": [
        {"match": {"tool": "docs.fetch"}, "tiers": [],
         "fields": {"$text.code_blocks": {"lossy": "drop-to-retrieve"}}}]}
    merged, _ = merge_policy(existing, _gen(("docs.fetch", ("minify", "tabularize"))))
    rule = merged["policies"][0]
    assert rule["tiers"] == []
    assert "ACTIVATE the lossy" in rule["_comment"]
    assert rule["fields"] == {"$text.code_blocks": {"lossy": "drop-to-retrieve"}}


def test_join_blocks_false_is_honoured_when_scoring():
    # apply_joined's first check is `if not policy.join_blocks`. Tuning a tool on
    # cross-block folding it will never perform proposes savings the operator can't observe.
    envs = [{"tool": "t", "captured_at": 1_000_000_000 + i,
             "raw": json.dumps({"id": i, "status": "active", "city": "Berlin"})}
            for i in range(20)]
    _, on = generate_policy(envs, join_blocks=True)
    _, off = generate_policy(envs, join_blocks=False)
    # Asserting on the measured saving, not the tier list: a single small object still
    # minifies past the threshold, so the tiers can agree while the NUMBER the operator is
    # shown — and the dictionary decision that hangs off it — is scored on folding that
    # will never happen.
    assert on[0]["saved_pct"] > off[0]["saved_pct"] * 1.5


def test_reason_ratios_count_results_not_blocks():
    # `n` counts BLOCKS and a joined result is one row, so a blocks denominator would
    # understate by the join factor and argue against the decision it justifies.
    groups = [_blocks(20), ["Error executing tool t: boom"]]
    row = _tool_decision("t", groups, 5.0)
    assert "1/2 non-JSON" in row["reason"], row["reason"]


# --------------------------------------------------------------------------- #
# text-payload drop candidates (#139): the shape `_drop_candidates` cannot see
# --------------------------------------------------------------------------- #
def _explore_doc(n_files=3, code_lines=60):
    """A `codegraph_explore`-shaped answer: markdown intelligence wrapping fenced source."""
    out = ["## Exploration: symbols", "", f"Found {n_files * 4} symbols.", "",
           "### Blast radius", ""]
    for f in range(n_files):
        out += [f"- `sym{f}` (src/f{f}.go:10) — 2 callers; no covering tests found"]
    out += ["", "### Source Code", ""]
    for f in range(n_files):
        body = "\n".join(f"{i}\t\tcall{f}_{i}(ctx, payload, opts) // widening the region"
                         for i in range(code_lines))
        out += [f"**`src/f{f}.go`**", "", "```go", body, "```", ""]
    return "\n".join(out) + "\n"


def test_a_long_text_tool_is_proposed_the_code_block_selector():
    """`_drop_candidates` opens with `json.loads` and skips on failure, so a 100%-text
    tool yielded ZERO candidates by construction — not a threshold missed, a shape unseen.
    That is why autotune never surfaced `codegraph_explore` (0.0% saved, 86.6% of its
    tokens inside droppable blocks)."""
    envs = [_env("codegraph_explore", _explore_doc()) for _ in range(3)]
    doc, rows = generate_policy(envs, threshold=5.0)
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "codegraph_explore")

    assert rule["_suggested_fields"] == {
        "$text.code_blocks": {"lossy": "drop-to-retrieve", "min": 400}}
    # Source is the load-bearing case the role split exists to flag — never tagged safe.
    row = next(r for r in rows[0]["drop_rows"] if r["path"] == "$text.code_blocks")
    assert row["role"] == "unknown"
    assert row["tok_share"] > 0.4
    # Measured, not asserted: three identical payloads means every block repeats, and
    # identical blocks are content-addressed to one handle. Printing a fabricated
    # "100% uniq" beside honestly-measured JSON rows would misreport the drop's cost.
    assert row["n"] == 9 and row["distinct"] == 3
    assert row["uniq_ratio"] == round(3 / 9, 4)

    # INACTIVE: the loader reads `fields`, so the generated policy is still fully lossless.
    assert "fields" not in rule
    assert not load_policy_from(doc).has_drop()


def test_a_text_tool_whose_bulk_is_not_fenced_is_not_proposed():
    """The guard against proposing it everywhere long text appears. `read_text_file`
    returns raw source: the only fences are docstring examples, ~1-5% of tokens, so the
    selector would evict almost nothing and still cost a retrieve round-trip."""
    prose = "\n".join(f"line {i} of a plain source file with no markdown fencing at all"
                      for i in range(400))
    doc, rows = generate_policy([_env("read_text_file", prose) for _ in range(3)],
                                threshold=5.0)
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "read_text_file")
    assert "_suggested_fields" not in rule


def test_one_text_payload_is_an_anecdote_not_a_shape():
    doc, _ = generate_policy([_env("codegraph_explore", _explore_doc())], threshold=5.0)
    rule = next(p for p in doc["policies"] if p["match"]["tool"] == "codegraph_explore")
    assert "_suggested_fields" not in rule


def load_policy_from(doc):
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        tf.write(json.dumps(doc))
        name = tf.name
    try:
        return load_policy(name)
    finally:
        Path(name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# #148 / #152 — the corpus records which RESULT and which SERVER a payload came from,
# so neither has to be inferred from capture timing or left unknowable.
# ---------------------------------------------------------------------------


def _idenv(tool, i, ts, result_id, server=None):
    env = {"tool": tool, "raw": json.dumps({"id": i, "status": "active"}),
           "captured_at": ts, "result_id": result_id}
    if server is not None:
        env["server"] = server
    return env


def test_result_ids_beat_the_timing_heuristic_on_the_parallel_call_burst():
    # The #148 repro: 200 INDEPENDENT single-block results, each a millisecond after the
    # last. Consecutive-window grouping chains all 200 into one result and scores the tool
    # as if that array had arrived together. With result ids there is nothing to infer.
    envs = [_idenv("t", i, 1_000_000_000 + i * 1_000_000, f"s:{i}") for i in range(200)]
    groups = group_results(envs)["t"]
    assert len(groups) == 200
    assert {len(g) for g in groups} == {1}


def test_result_ids_keep_a_genuine_multi_block_result_together():
    # Same tight timing, but the blocks really do share a result — the case the window was
    # right about. Exact ids must not over-split it either.
    envs = [_idenv("t", i, 1_000_000_000 + i * 1_000_000, "s:7") for i in range(4)]
    assert [len(g) for g in group_results(envs)["t"]] == [4]


def test_result_ids_group_across_a_gap_the_window_would_have_split():
    # Timing says "two results" (an 8-second gap); the ids say one. A slow server that
    # streams its blocks is exactly where the heuristic under-groups, and the ids win.
    envs = [_idenv("t", 0, 1_000_000_000, "s:1"), _idenv("t", 1, 9_000_000_000, "s:1")]
    assert [len(g) for g in group_results(envs)["t"]] == [2]


def test_a_mixed_corpus_groups_each_half_by_what_it_knows():
    # Re-capturing after an upgrade leaves both generations in one dir. Identified
    # envelopes group exactly; the rest still fall back to timing.
    envs = [_idenv("t", 0, 1_000_000_000, "s:1"), _idenv("t", 1, 1_000_500_000, "s:1"),
            {"tool": "t", "raw": json.dumps({"id": 2}), "captured_at": 5_000_000_000},
            {"tool": "t", "raw": json.dumps({"id": 3}), "captured_at": 5_001_000_000}]
    assert sorted(len(g) for g in group_results(envs)["t"]) == [2, 2]


def test_legacy_span_cap_bounds_an_unbroken_burst():
    # No ids to read, so the guess stands — but it is no longer unbounded. 600 envelopes a
    # millisecond apart stay inside the 50 ms consecutive window forever; the total-span cap
    # is what stops them chaining into one 600-block "result".
    envs = [{"tool": "t", "raw": json.dumps({"id": i}),
             "captured_at": 1_000_000_000 + i * 1_000_000} for i in range(600)]
    groups = group_results(envs)["t"]
    assert len(groups) > 1
    assert max(len(g) for g in groups) <= 260        # ~250 ms of 1 ms-apart writes


def test_heuristic_share_reports_what_was_guessed():
    from terse.policy_gen import heuristic_share
    envs = [_idenv("t", 0, 1, "s:1"), {"tool": "t", "raw": "{}"}]
    assert heuristic_share(envs) == (1, 2)
    assert heuristic_share([_idenv("t", 0, 1, "s:1")]) == (0, 1)


def test_a_rule_generated_from_a_server_tagged_corpus_is_reachable_by_select():
    # The property #152 says is silently false today. runecho does NOT self-prefix its tool
    # names, so the corpus holds a bare `structure`; `select` looks up `runecho.structure`
    # FIRST. A rule authored under the bare name is therefore unreachable — not merely
    # lower-priority — and the generated policy has to carry the qualified name.
    envs = [{"tool": "structure", "server": "runecho", "raw": json.dumps(_records()),
             "captured_at": 1_000_000_000 + i, "result_id": f"s:{i}"} for i in range(3)]
    doc, _rows = generate_policy(envs, threshold=5.0)
    names = [p["match"]["tool"] for p in doc["policies"]]
    assert names == ["runecho.structure"]
    rule = load_policy_from(doc).select("structure", server="runecho")
    assert rule.tool_glob == "runecho.structure"
    assert "tabularize" in rule.tiers


def test_a_self_prefixing_server_is_not_double_qualified():
    # kb names its own tools `kb.read.*`; qualifying again would author `kb.kb.read.search`
    # and miss the operator's `kb.*` rule. Mirrors Policy._match_candidates' own skip.
    envs = [{"tool": "kb.read.search", "server": "kb", "raw": json.dumps(_records()),
             "captured_at": 1_000_000_000 + i, "result_id": f"s:{i}"} for i in range(3)]
    doc, _rows = generate_policy(envs, threshold=5.0)
    assert [p["match"]["tool"] for p in doc["policies"]] == ["kb.read.search"]


def test_a_corpus_with_no_server_keeps_the_bare_name():
    envs = [{"tool": "structure", "raw": json.dumps(_records()),
             "captured_at": 1_000_000_000 + i} for i in range(3)]
    doc, _rows = generate_policy(envs, threshold=5.0)
    assert [p["match"]["tool"] for p in doc["policies"]] == ["structure"]


def test_a_deployed_server_glob_is_visible_to_the_shadow_check():
    # #148.2: with `runecho.* {tiers: []}` deployed and the corpus holding runecho's tool,
    # the merge must SEE that rule — inheriting its operator-owned keys and reporting the
    # tier change against it — rather than appending a rule the loader will never reach.
    existing = {"version": 1, "policies": [
        {"match": {"tool": "runecho.*"}, "tiers": [], "capture": False},
    ]}
    envs = [{"tool": "structure", "server": "runecho", "raw": json.dumps(_records()),
             "captured_at": 1_000_000_000 + i, "result_id": f"s:{i}"} for i in range(3)]
    generated, _ = generate_policy(envs, threshold=5.0)
    merged, changes = merge_policy(existing, generated)

    added = [c for c in changes if c["kind"] == "added"]
    assert [c["tool"] for c in added] == ["runecho.structure"]
    assert added[0]["before"] == []           # it displaces the deployed passthrough
    inherited = [c for c in changes if c["kind"] == "inherited"]
    assert inherited and inherited[0]["from"] == "runecho.*"
    assert "capture" in inherited[0]["keys"]  # the operator's #85 decision survives

    # And the merged doc actually re-decides the tool at runtime, which is the whole point.
    new_rule = load_policy_from(merged).select("structure", server="runecho")
    assert new_rule.tool_glob == "runecho.structure"
    assert new_rule.tiers and new_rule.capture is False
