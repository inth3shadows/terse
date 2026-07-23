"""Policy shell: selection, fail-closed defaults, lossless guarantee, round-trip."""

from __future__ import annotations

import json

import pytest

from terse import transforms
from terse.policy import Policy, Rule, apply, apply_joined, default_policy, load_policy

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


def test_select_server_scoped_rule_matches_a_server_that_does_not_self_prefix():
    # Regression (#83): a server-scoped rule like "runecho.*" only ever matched servers
    # that happen to self-prefix their own tool names. runecho calls its tool plain
    # "structure", so the rule silently missed and fell through to the defaults — and
    # nothing said so. With the server known, the qualified candidate makes it match.
    p = Policy(rules=[Rule(tool_glob="runecho.*", tiers=("minify",))])
    assert p.select("structure").tiers == ("minify", "tabularize", "dictionary")  # pre-#83
    assert p.select("structure", server="runecho").tiers == ("minify",)           # fixed
    # a server whose rule doesn't match still falls through to the defaults
    assert p.select("structure", server="codegraph").tiers == ("minify", "tabularize",
                                                               "dictionary")


def test_select_does_not_double_qualify_a_self_prefixed_tool():
    # kb names its OWN tools "kb.read.*", so qualifying by server must not synthesize
    # "kb.kb.read.search" and miss the "kb.*" rule the user actually wrote.
    p = _policy()
    assert p.select("kb.read.search", server="kb").tiers == ("minify", "tabularize")


def test_select_server_qualified_candidate_outranks_a_bare_rule():
    # A server-scoped rule is the more specific intent, so it wins over a bare-name rule
    # regardless of declaration order — mirroring how multiproxy's peer-qualified
    # candidate already outranks its bare fallback.
    p = Policy(rules=[Rule(tool_glob="structure", tiers=()),
                      Rule(tool_glob="runecho.*", tiers=("minify",))])
    assert p.select("structure", server="runecho").tiers == ("minify",)
    assert p.select("structure").tiers == ()          # no server: bare rule still wins


def test_select_server_none_is_byte_identical_to_pre_83_behavior():
    # The whole change is additive: every existing candidate is still tried in its
    # original order, so a policy that matched before matches the same rule now.
    p = _policy()
    for tool in ("gh.api.repos", "kb.read.search", "gh.api.rate_limit", "unknown.tool",
                 "gh__gh.api.repos", "gh__kb.read.search"):
        assert p.select(tool).tiers == p.select(tool, server=None).tiers


def test_select_server_scoped_rule_matches_a_multiproxy_peer_qualified_tool():
    # multiproxy passes the peer's config name as `server`, and its tool arrives
    # peer-qualified ("runecho__structure"). The qualified candidate is built from the
    # BARE part, so a "runecho.*" rule matches there too — the separator is "__", so
    # without this the peer-qualified name would miss the dot-globbed rule as well.
    p = Policy(rules=[Rule(tool_glob="runecho.*", tiers=("minify",))])
    assert p.select("runecho__structure", server="runecho").tiers == ("minify",)


def test_capture_defaults_true_and_parses_false(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({
        "version": 1,
        "policies": [
            {"match": {"tool": "secret-broker.*"}, "tiers": [], "capture": False},
            {"match": {"tool": "gh.*"}, "tiers": ["minify"]},
        ],
    }), encoding="utf-8")
    pol = load_policy(p)
    assert pol.select("secret-broker.reveal").capture is False
    assert pol.select("gh.items").capture is True        # omitted -> pre-#85 behavior
    assert Rule("x", ()).capture is True                 # dataclass default


@pytest.mark.parametrize("bad", ["false", "no", 0, 1, None, [], {}])
def test_capture_rejects_a_non_bool_rather_than_silently_enabling_itself(tmp_path, bad):
    # THE failure direction that matters (#85): every wrong-typed value in Python is
    # truthy (`bool("false") is True`), so a lax coercion would silently turn the guard
    # back ON — writing the very payloads it was written to keep off disk. Fail at load.
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"version": 1,
                             "policies": [{"match": {"tool": "x.*"}, "tiers": [],
                                           "capture": bad}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="'capture' must be true or false"):
        load_policy(p)


def test_capture_is_a_registered_rule_key(tmp_path):
    # Strict-key validation (#77) rejects unknown keys, so a `capture` that wasn't
    # registered would make every policy using it fail to load — the guard must be
    # spelled exactly this way to work at all.
    from terse.policy import _RULE_KEYS
    assert "capture" in _RULE_KEYS


def test_apply_passes_server_through_to_rule_selection():
    p = Policy(rules=[Rule(tool_glob="runecho.*", tiers=())])   # () = passthrough
    raw = json.dumps({"result": [{"id": 1, "k": "v"}, {"id": 2, "k": "v"}]})
    assert apply(raw, "structure", p).text != raw                        # defaults ran
    assert apply(raw, "structure", p, server="runecho").skipped is True  # rule matched


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


def test_never_lossy_server_suppresses_a_lossy_field():
    # On a never-lossy server (credential/personal store), a field marked lossy is kept
    # fully lossless — the structural floor from the security review. Enforced on the
    # verified server identity, not a tool-name match.
    raw = json.dumps({"result": [{"id": 1, "body": "x" * 200}]})
    rule = Rule(tool_glob="*", tiers=("minify", "tabularize"),
                fields={"result[].body": {"lossy": "truncate", "max": 5}})

    # (a) baked never_lossy list — identity-based, catches a store whose name looks innocuous
    pol = Policy(rules=[rule], never_lossy_servers=frozenset({"kb"}))
    kb = apply(raw, "kb.read.x", pol, server="kb")
    assert transforms.decompress(kb.text) == json.loads(raw)          # fully lossless
    assert any("never-lossy" in w for w in kb.warnings)

    # (b) non-overridable name floor — server not in the list, but its name screams secrets
    sec = apply(raw, "reveal", pol, server="secret-broker")
    assert transforms.decompress(sec.text) == json.loads(raw)
    assert any("never-lossy" in w for w in sec.warnings)

    # (c) an ordinary server: the same lossy field IS applied (truncated -> not lossless)
    ok = apply(raw, "x", pol, server="runecho")
    assert any("truncated" in w for w in ok.warnings)
    assert "x" * 200 not in ok.text                                   # the long body was cut


def test_server_never_lossy_predicate():
    pol = Policy(rules=[], never_lossy_servers=frozenset({"kb", "sb-run"}))
    assert pol.server_never_lossy("kb")                 # baked list
    assert pol.server_never_lossy("sb-run")             # baked list (launcher alias, no secret word)
    assert pol.server_never_lossy("secret-broker")      # floor: "secret"
    assert pol.server_never_lossy("acme-vault")         # floor: "vault"
    assert pol.server_never_lossy("my-authgw")          # floor: "auth"
    assert not pol.server_never_lossy("runecho")
    assert not pol.server_never_lossy(None)              # unknown identity is NOT auto-excluded
    assert not pol.server_never_lossy("")


def test_never_lossy_servers_loads_from_policy_json(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"version": 1, "never_lossy_servers": ["kb", "sb-run"],
                             "policies": []}), encoding="utf-8")
    pol = load_policy(p)
    assert pol.server_never_lossy("kb") and pol.server_never_lossy("sb-run")


# --- #116: multi-block join (apply_joined) ---

def _rec_blocks(n, change=None):
    rows = [{"id": i, "status": "active", "url": "https://x.example/api/items"}
            for i in range(n)]
    if change is not None:
        rows[change]["status"] = "closed"
    return [json.dumps(r) for r in rows]


def test_join_blocks_defaults_on_and_loader_parses_false(tmp_path):
    assert Policy(rules=[]).join_blocks is True            # ON by default (#116)
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"version": 1, "join_blocks": False, "policies": []}),
                 encoding="utf-8")
    assert load_policy(p).join_blocks is False


def test_apply_joined_folds_records_across_blocks_losslessly():
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))])
    raws = _rec_blocks(6)
    applied, curr, reason = apply_joined(raws, "gh.api.items", pol)
    assert reason == ""                                   # join applied
    assert transforms.TABLE_MARKER in applied.text        # records folded into one table
    assert transforms.decompress(applied.text) == [json.loads(r) for r in raws]  # lossless
    assert curr == [json.loads(r) for r in raws]          # raw parse is the diff base


def test_apply_joined_runs_lossy_per_block_not_over_the_array():
    # The §2 claim: a field path like `body` is authored against ONE record's shape, so it
    # must resolve per-block. If lossy ran AFTER the join, `body` would address the array
    # (which has no `body`) and truncate nothing.
    rule = Rule("gh.*", ("minify", "tabularize", "dictionary"),
                fields={"body": {"lossy": "truncate", "max": 5}})
    pol = Policy(rules=[rule])
    raws = [json.dumps({"id": 1, "body": "x" * 200}),
            json.dumps({"id": 2, "body": "y" * 200})]
    applied, curr, reason = apply_joined(raws, "gh.api.items", pol)
    assert reason == ""
    out = transforms.decompress(applied.text)
    assert [r["id"] for r in out] == [1, 2]
    assert all(len(r["body"]) < 200 for r in out)         # each record's body was cut
    assert "x" * 200 not in applied.text and "y" * 200 not in applied.text
    assert any("truncated" in w for w in applied.warnings)
    assert curr == [json.loads(r) for r in raws]          # base is the RAW (pre-lossy) parse


def test_apply_joined_never_lossy_server_keeps_the_join_fully_lossless():
    rule = Rule("*", ("minify", "tabularize"),
                fields={"body": {"lossy": "truncate", "max": 5}})
    pol = Policy(rules=[rule], never_lossy_servers=frozenset({"kb"}))
    raws = [json.dumps({"id": 1, "body": "x" * 200}),
            json.dumps({"id": 2, "body": "y" * 200})]
    applied, curr, reason = apply_joined(raws, "kb.read.x", pol, server="kb")
    assert reason == ""
    assert transforms.decompress(applied.text) == [json.loads(r) for r in raws]  # untouched
    assert any("never-lossy" in w for w in applied.warnings)


def test_apply_joined_refusal_reasons():
    pol = Policy(rules=[Rule("gh.*", ("minify", "tabularize", "dictionary"))])
    good = _rec_blocks(2)

    # join_blocks disabled
    off = Policy(rules=[Rule("gh.*", ("minify", "tabularize"))], join_blocks=False)
    assert apply_joined(good, "gh.api.items", off) == (None, None, "off")

    # explicit passthrough tier
    passthru = Policy(rules=[Rule("gh.*", ())])
    assert apply_joined(good, "gh.api.items", passthru) == (None, None, "passthrough")

    # a non-JSON block
    _, _, r = apply_joined([good[0], "not json {"], "gh.api.items", pol)
    assert r == "non_json"

    # a block that isn't a dict (a JSON array) — not a record sequence
    _, _, r = apply_joined([good[0], json.dumps([1, 2, 3])], "gh.api.items", pol)
    assert r == "heterogeneous"

    # a reserved terse marker key present
    _, _, r = apply_joined([json.dumps({transforms.TABLE_MARKER: 1}), good[0]],
                           "gh.api.items", pol)
    assert r == "marker"

    # nesting past the codec depth cap
    deep = {"leaf": 1}
    for _ in range(transforms.MAX_DEPTH + 5):
        deep = {"x": deep}
    _, _, r = apply_joined([json.dumps(deep), json.dumps(deep)], "gh.api.items", pol)
    assert r == "depth"


def test_apply_falls_back_to_lossless_when_codec_self_check_fails(monkeypatch):
    # Verify-before-emit: if the always-on Tier-0/0.5 codec ever produced output that did
    # not round-trip, apply() must fall back to the plain lossless minified form rather
    # than ship a corrupt payload. Simulate a latent codec bug by making decompress return
    # the wrong value, then assert the emitted text still reconstructs the original exactly.
    p = _policy()
    monkeypatch.setattr("terse.policy.transforms.decompress",
                        lambda _text: {"corrupted": "not the original"})
    result = apply(RECORDS, "gh.api.repos", p)
    assert json.loads(result.text) == json.loads(RECORDS)  # lossless despite the "bug"
    assert result.tiers == ()
    assert any("self-check failed" in w for w in result.warnings)


def test_structured_defaults_to_auto_and_accepts_only_known_literals(tmp_path):
    # Strict for the same reason `capture` is: this decides whether terse rewrites a field
    # carrying a declared outputSchema. A typo reverting to "leave" is a quiet no-op; a
    # typo enabling "compress" quietly rewrites a typed field. Both must fail at load.
    def write(rule):
        p = tmp_path / "p.json"
        p.write_text(json.dumps({"version": 1, "policies": [rule]}))
        return p
    assert load_policy(write({"match": {"tool": "*"}, "tiers": []})).rules[0].structured \
        == "auto"
    assert load_policy(write({"match": {"tool": "*"}, "tiers": [],
                              "structured": "compress"})).rules[0].structured == "compress"
    assert load_policy(write({"match": {"tool": "*"}, "tiers": [],
                              "structured": "replace"})).rules[0].structured == "replace"
    for bad in ("Compress", "Auto", "true", True, 1, None, "drop"):
        try:
            load_policy(write({"match": {"tool": "*"}, "tiers": [], "structured": bad}))
        except ValueError:
            continue
        raise AssertionError(f"load_policy accepted structured={bad!r}")


def test_structured_mode_resolves_against_the_declared_client(tmp_path):
    # "auto" is decided by the client's own `clientInfo.name`, and fails CLOSED: an
    # unknown client, or none at all, must never get the rewriting behavior (#128).
    from terse.policy import structured_mode_for_client as resolve
    assert resolve("auto", "claude-code") == "compress"      # measured not to validate
    assert resolve("auto", "some-other-client") == "leave"
    assert resolve("auto", None) == "leave"                  # no handshake seen
    assert resolve("auto", "Claude-Code") == "leave"          # exact match only
    # "auto" tops out at "compress" and never escalates to "replace": every mode up to
    # "compress" is invisible to a client that ignores the typed field, and "replace" is
    # the first one that removes information from the wire.
    assert resolve("auto", "claude-code") != "replace"
    # explicit settings always win over the resolution
    for client in ("claude-code", "some-other-client", None):
        assert resolve("leave", client) == "leave"
        assert resolve("compress", client) == "compress"
        assert resolve("replace", client) == "replace"
