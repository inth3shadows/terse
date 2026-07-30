"""Tier 0.6 `embedded`: fold a string leaf that is itself a JSON document.

The tier exists because `tabularize`/`dictionary` walk PARSED structure, so a payload a
server delivered double-encoded (`{"response_text": json.dumps(body)}`) is a leaf they
cannot reach — measured 41.9% saved as a real record array vs 0.0% inside a string.

The load-bearing property here is NOT the saving, it is that the fold is byte-exact.
`json.dumps(json.loads(s))` is not `s` in general, so the tier must DECLINE every string it
cannot regenerate exactly rather than round-trip to something the server never sent. Most of
this file pins the declining.
"""

from __future__ import annotations

import json

import pytest

from terse import policy as P
from terse import transforms as T


def _embedded(obj):
    return T.compress_with(obj, embedded=True)


def _records(n=12):
    return [{"id": i, "name": f"row {i}", "url": f"https://e.com/{i}"} for i in range(n)]


def _profitable(**extra) -> str:
    """A JSON string big enough that folding it CLEARLY pays, in Python's default form.

    Every declining test below must start from a payload the size guard would happily
    accept — otherwise the guard rejects it for its own reasons and the test passes while
    telling you nothing about the check it names. (Mutation-tested: with small payloads,
    inverting the byte-exact form check, the marker check, and the size guard itself all
    still passed.) Callers inject one defect into the returned string and assert the fold
    is declined *because of that defect*."""
    return json.dumps({"results": _records(20), **extra})


# --------------------------------------------------------------------------- #
# It reaches the payload the other tiers structurally cannot
# --------------------------------------------------------------------------- #
def test_records_inside_a_string_are_unreachable_without_the_tier():
    """The gap itself, pinned: identical data, 0% as a string vs a real saving as an array.

    If this ever fails because the OFF number improved, the tier's premise changed and its
    cost/benefit should be re-derived — not silently accepted."""
    payload = {"response_text": json.dumps({"results": _records(20)})}
    raw = T.minify(payload)
    off = T.count_cl100k(T.compress_with(payload))
    on = T.count_cl100k(_embedded(payload))
    assert off == T.count_cl100k(raw)      # no tier can fold a string leaf
    assert on < off                         # the embedded tier can
    assert T.decompress(_embedded(payload)) == payload


def test_the_inner_string_comes_back_byte_identical_not_merely_equal():
    """Equality of the whole object is the gate, but state the inner string explicitly:
    a re-serialization that differed by one space would still satisfy a sloppier check."""
    inner = json.dumps({"results": _records()})
    payload = {"response_text": inner}
    restored = T.decompress(_embedded(payload))
    assert restored["response_text"] == inner
    assert isinstance(restored["response_text"], str)   # still a STRING, not unwrapped


@pytest.mark.parametrize("dumps_kwargs, form", [
    ({}, "p"),                                              # json.dumps(body), the common one
    ({"ensure_ascii": False}, "P"),
    ({"separators": (",", ":")}, "c"),
    ({"separators": (",", ":"), "ensure_ascii": False}, "C"),
    ({"indent": 2}, "i2"),
    ({"indent": 2, "ensure_ascii": False}, "I2"),
    ({"indent": 4}, "i4"),
    ({"indent": 4, "ensure_ascii": False}, "I4"),
])
def test_every_registered_form_round_trips_and_is_tagged_as_itself(dumps_kwargs, form):
    """Each `_EMBED_FORMS` entry must both fire and be recorded, since "f" is the only
    instruction decode has for rebuilding the exact bytes.

    The records carry a non-ASCII value deliberately: on pure-ASCII data the
    `ensure_ascii=True/False` pair emits identical bytes, so the ids are indistinguishable
    and this test would assert nothing about half the registry."""
    recs = [dict(r, note="café") for r in _records()]
    inner = json.dumps({"results": recs}, **dumps_kwargs)
    payload = {"body": inner}
    wire = _embedded(payload)
    assert T.JSON_STR_MARKER in wire
    assert T.decompress(wire) == payload
    assert json.loads(wire)["body"]["f"] == form


def test_identical_forms_resolve_to_one_id_and_still_round_trip():
    """When two registry entries produce the same bytes (ASCII-only data under
    `ensure_ascii=True/False`) the first match wins. Either id is a correct recipe, so the
    only thing that matters is that decode reproduces the string."""
    inner = json.dumps({"results": _records()})          # no non-ASCII: "p" and "P" tie
    payload = {"body": inner}
    wire = _embedded(payload)
    assert json.loads(wire)["body"]["f"] == "p"          # dict order decides; documented
    assert T.decompress(wire) == payload


def test_a_nested_record_array_inside_the_string_is_itself_tabularized():
    """The point of folding is to let the OTHER tiers in — not merely to unwrap."""
    payload = {"body": json.dumps({"results": _records(20)})}
    assert T.TABLE_MARKER in _embedded(payload)


def test_it_reaches_a_string_inside_a_record_row_and_a_plain_list():
    """`_fold_records` and the plain-list branch both recurse; a param dropped in either
    silently disables the tier for those shapes while top-level cases still pass."""
    inner = json.dumps({"results": _records()})
    in_rows = [{"id": 1, "body": inner}, {"id": 2, "body": inner}]
    in_list = [inner, {"nested": [inner]}]          # non-uniform: the plain-list branch
    for payload in (in_rows, in_list):
        wire = _embedded(payload)
        assert T.JSON_STR_MARKER in wire
        assert T.decompress(wire) == payload


# --------------------------------------------------------------------------- #
# It DECLINES anything it cannot reproduce byte-exactly
# --------------------------------------------------------------------------- #
def _defect_cases():
    """Each case is a big, otherwise-perfectly-foldable document with ONE defect that makes
    it impossible to regenerate byte-exactly. `sanity` asserts the same document WITHOUT the
    defect does fold, proving the decline is caused by the defect and not by size."""
    clean = _profitable(n=1.5)
    return [
        # json.loads keeps only the last duplicate, so the key is simply gone on re-encode.
        pytest.param(_profitable()[:-1] + ', "a": 1, "a": 2}', id="duplicate-keys"),
        # 1.50 -> 1.5 and 1.5e0 -> 1.5: the VALUE survives, the spelling does not.
        pytest.param(clean.replace("1.5}", "1.50}"), id="trailing-zero"),
        pytest.param(clean.replace("1.5}", "1.5e0}"), id="exponent"),
        # Spacing that matches no single registered form.
        pytest.param(_profitable().replace('"results": ', '"results":', 1), id="mixed-spacing"),
        pytest.param("  " + _profitable(), id="leading-whitespace"),
        pytest.param(_profitable() + "\n", id="trailing-newline"),
    ]


@pytest.mark.parametrize("inner", _defect_cases())
def test_declines_a_string_no_form_can_regenerate(inner):
    """The corruption cases — the reason the bar is byte-equality and not "same data".

    Every one of these parses cleanly and carries the same information, so a tier that
    checked only `json.loads(a) == json.loads(b)` would fold them and then hand back bytes
    the server never emitted."""
    payload = {"body": inner}
    wire = _embedded(payload)
    assert T.JSON_STR_MARKER not in wire
    assert T.decompress(wire) == payload


def test_the_defect_cases_would_otherwise_fold():
    """Guards the guard: if `_profitable()` ever stops being worth folding, every case in
    `_defect_cases` would pass for the wrong reason and silently stop testing anything."""
    assert T.JSON_STR_MARKER in _embedded({"body": _profitable()})
    assert T.JSON_STR_MARKER in _embedded({"body": _profitable(n=1.5)})


@pytest.mark.parametrize("inner, why", [
    ("just some prose, not json at all", "not JSON"),
    ("42", "JSON scalar, not a container"),
    ('"hello"', "JSON string-scalar, not a container"),
    ("", "empty string"),
    ("{", "truncated JSON"),
    ('{"a": 1}extra', "trailing garbage"),
])
def test_declines_a_string_that_is_not_a_json_document(inner, why):
    payload = {"body": inner}
    assert T.JSON_STR_MARKER not in _embedded(payload), why
    assert T.decompress(_embedded(payload)) == payload


def test_declines_when_the_embedded_document_carries_a_terse_marker():
    """Same invariant `has_terse_marker` enforces outside: folding a doc that already
    contains an envelope key would make decode read the user's literal dict as terse's.

    Built on `_profitable()` so the size guard is NOT what declines it — with a small
    payload this test passed even with the marker check deleted."""
    inner = _profitable()[:-1] + f', "{T.TABLE_MARKER}": 1, "cols": ["a"], "rows": [[1]]}}'
    payload = {"body": inner}
    assert T.JSON_STR_MARKER in _embedded({"body": _profitable()})   # sanity: size is fine
    assert T.JSON_STR_MARKER not in _embedded(payload)
    assert T.decompress(_embedded(payload)) == payload


def test_declines_one_unprofitable_document_inside_a_payload_that_still_shrinks():
    """The per-occurrence size guard, isolated from `compress_with`'s whole-payload guard.

    A tiny `{}` alone is caught by the outer guard (the payload would grow), so testing it
    alone proves nothing — with the per-occurrence guard deleted, that version still passed.
    Here a big foldable sibling makes the payload shrink overall, so ONLY the per-occurrence
    guard can stop the tiny document from being wrapped in an envelope bigger than itself."""
    payload = {"big": _profitable(), "tiny": "{}"}
    wire = _embedded(payload)
    assert T.count_cl100k(wire) < T.count_cl100k(T.minify(payload))   # payload did shrink
    assert json.loads(wire)["tiny"] == "{}"                           # left as a string
    assert T.JSON_STR_MARKER in wire                                  # the big one folded
    assert T.decompress(wire) == payload


def test_declines_a_document_past_the_depth_cap():
    deep = json.dumps(json.loads("[" * (T.MAX_DEPTH + 5) + "]" * (T.MAX_DEPTH + 5)))
    payload = {"body": deep}
    assert T.JSON_STR_MARKER not in _embedded(payload)
    assert T.decompress(_embedded(payload)) == payload


# --------------------------------------------------------------------------- #
# Wiring: off by default, on only when the policy asks
# --------------------------------------------------------------------------- #
def test_the_tier_is_off_unless_requested():
    """Opt-in: `compress`/`compress_with` defaults must not start folding strings, or every
    existing measurement baseline and primer shifts under callers who never asked."""
    payload = {"body": json.dumps({"results": _records()})}
    assert T.JSON_STR_MARKER not in T.compress(payload)
    assert T.JSON_STR_MARKER not in T.compress_with(payload)


def test_default_tiers_stay_opt_in_even_though_embedded_is_valid():
    assert "embedded" in P.VALID_TIERS
    assert "embedded" not in P.DEFAULT_TIERS
    assert "embedded" not in P.default_policy().default_tiers


def test_policy_runs_the_tier_only_when_the_rule_lists_it():
    payload = {"body": json.dumps({"results": _records(20)})}
    raw = T.minify(payload)
    without = P.apply(raw, "t", P.Policy(rules=[P.Rule("*", ("minify", "tabularize"))]))
    with_it = P.apply(raw, "t", P.Policy(rules=[P.Rule("*", ("minify", "tabularize",
                                                             "embedded"))]))
    assert T.JSON_STR_MARKER not in without.text
    assert T.JSON_STR_MARKER in with_it.text
    assert T.count_cl100k(with_it.text) < T.count_cl100k(without.text)
    assert T.decompress(with_it.text) == json.loads(raw)


def test_the_primer_paragraph_is_charged_only_to_a_policy_that_can_emit_the_form():
    """#170's lesson: a primer paragraph is re-read every turn, so a form a server cannot
    produce must not be documented to it."""
    from terse.proxy import PRIMER_EMBEDDED, build_primer
    plain = P.Policy(rules=[P.Rule("*", ("minify", "tabularize", "dictionary"))])
    opted = P.Policy(rules=[P.Rule("*", ("minify", "tabularize", "embedded"))])
    assert PRIMER_EMBEDDED not in build_primer(plain)
    assert PRIMER_EMBEDDED in build_primer(opted)
    assert not P.default_policy().emits_embedded()
    assert opted.emits_embedded()


def test_an_unknown_form_id_raises_rather_than_returning_the_envelope():
    """A forward/corrupt payload must fail loudly: handing back the wrapper dict as if it
    were the data is the silent-corruption outcome. The proxy's self-check converts this
    into a fall-back to the plain minified form."""
    wire = json.dumps({"body": {T.JSON_STR_MARKER: 1, "f": "no-such-form", "v": {"a": 1}}})
    with pytest.raises(ValueError, match="unknown embedded-JSON form"):
        T.decompress(wire)


def test_marker_is_reserved_so_a_literal_payload_containing_it_is_left_alone():
    """The new key must join `_RESERVED_MARKERS`, or a real payload using it as a dict key
    would be mis-decoded as terse's own envelope."""
    assert T.has_terse_marker({"x": {T.JSON_STR_MARKER: 1}})
    payload = {T.JSON_STR_MARKER: 1, "f": "p", "v": "not really terse output"}
    assert P.apply(T.minify(payload), "t",
                   P.Policy(rules=[P.Rule("*", ("minify", "tabularize", "embedded"))])
                   ).text == T.minify(payload)


# --------------------------------------------------------------------------- #
# `policy generate` / autotune must be able to RECOMMEND the tier
# --------------------------------------------------------------------------- #
def _envs(tool, raw, n=4):
    return [{"tool": tool, "raw": raw} for _ in range(n)]


def _row(rows, tool):
    return next(r for r in rows if r["tool"] == tool)


def test_generate_recommends_the_tier_for_a_tool_that_double_encodes():
    """The whole point of the tier being opt-in: something has to turn it on, on evidence.
    Shipped without this, `embedded` is reachable only by hand-editing a policy — and #144
    is the standing proof that hand-edited tier decisions go stale and nothing re-derives
    them."""
    from terse.policy_gen import generate_policy
    raw = json.dumps({"response_text": json.dumps({"results": _records(20)})})
    _doc, rows = generate_policy(_envs("broker.exa_search", raw))
    row = _row(rows, "broker.exa_search")
    assert "embedded" in row["tiers"]
    assert row["emb_pct"] > 0
    assert "embedded +" in row["reason"]


def test_generate_withholds_the_tier_from_an_ordinary_record_tool():
    """It must not be added everywhere: a tool with no embedded JSON gains nothing and would
    pay a primer paragraph every turn for a form it can never emit."""
    from terse.policy_gen import generate_policy
    raw = json.dumps({"results": _records(20)})
    _doc, rows = generate_policy(_envs("gh.list_items", raw))
    row = _row(rows, "gh.list_items")
    assert "embedded" not in row["tiers"]
    assert row["emb_pct"] == 0
    assert "embedded" not in row["reason"]      # silent when it saved nothing, not noise


def test_a_tool_saved_ONLY_by_the_embedded_tier_is_not_marked_passthrough():
    """`tier_total` has to include the embedded step. A body delivered as one JSON string
    saves ~0% under the other tiers, so scoring it without `embedded` would put it below the
    threshold and hand back `tiers: []` — permanently hiding the tool the tier exists for."""
    from terse.policy_gen import generate_policy
    raw = json.dumps({"response_text": json.dumps({"results": _records(20)})})
    _doc, rows = generate_policy(_envs("broker.exa_search", raw))
    row = _row(rows, "broker.exa_search")
    assert row["tiers"] != []
    assert row["saved_pct"] > 5.0
    # ...and the same payload scores ~0 with the tier's contribution removed.
    from terse.measure import measure_payload
    m = measure_payload(raw)
    without = m["cl100k"]["raw"] - m["cl100k"]["compressed"]
    assert abs(without) < m["saved_cl100k"]["embedded"]


def test_measure_reports_the_embedded_step_as_its_own_marginal_saving():
    from terse.measure import measure_payload
    raw = json.dumps({"response_text": json.dumps({"results": _records(20)})})
    m = measure_payload(raw)
    assert m["saved_cl100k"]["embedded"] > 0
    assert m["cl100k"]["embedded"] < m["cl100k"]["compressed"]
    # tier_total is measured against the embedded form, not the dictionary one
    assert m["saved_cl100k"]["tier_total"] == m["cl100k"]["raw"] - m["cl100k"]["embedded"]


def test_measure_reports_zero_when_there_is_no_embedded_json():
    from terse.measure import measure_payload
    m = measure_payload(json.dumps({"results": _records(20)}))
    assert m["saved_cl100k"]["embedded"] == 0
    assert m["cl100k"]["embedded"] == m["cl100k"]["compressed"]


# --------------------------------------------------------------------------- #
# Adversarial-review follow-ups (#183)
# --------------------------------------------------------------------------- #
def test_embedded_honours_tabularize_off_and_emits_no_undocumented_marker():
    """Folding a string opens a NEW structural walk, and it must inherit the caller's
    `tabularize`. Defaulting it to True inside the fold emitted `__terse_table__` for a
    policy whose tiers were ["minify","embedded"] — while `emits_table()` was correctly
    False, so the primer never documented it and the model got an unexplained envelope.

    Every other test in this file enables both tiers together, which is exactly why this
    went unnoticed until an adversarial review pulled them apart."""
    payload = {"body": json.dumps({"results": _records(20)})}
    wire = T.compress_with(payload, tabularize=False, embedded=True)
    assert T.JSON_STR_MARKER in wire          # the tier still does its job
    assert T.TABLE_MARKER not in wire         # ...without a tier the caller disabled
    assert T.decompress(wire) == payload


def test_policy_with_embedded_but_not_tabularize_never_emits_a_table():
    """End-to-end through the runtime path, against the invariant the codebase states:
    `select(tool).tiers <= reachable_tiers()`, i.e. never emit a form the primer omits."""
    pol = P.Policy(rules=[], default_tiers=("minify", "embedded"))
    raw = T.minify({"response_text": json.dumps({"results": _records(20)})})
    out = P.apply(raw, "t", pol)
    assert not pol.emits_table()
    assert T.TABLE_MARKER not in out.text
    assert T.decompress(out.text) == json.loads(raw)


def _double_encoded(outer=6, inner=20) -> str:
    """A payload that compresses under the DEFAULT pipeline *and* carries a JSON document
    inside a string, so the two gates have visibly different consequences.

    Both halves are load-bearing. Without the plain record array the default pipeline saves
    exactly 0 here (`{"response_text": "..."}` has no repeated keys to fold), and a test
    asserting that the default savings survive an embedded-gate failure would be asserting
    `0 == 0` — green against the very over-reach it is meant to pin.

    The inner array is the larger of the two on purpose: `embedded` is scored as a MARGINAL
    tier, so it must clear `policy generate`'s threshold on its own to be offered at all.
    At `outer == inner` it measures 3.4% and is dropped as uneconomic — which would let a
    "the tier was dropped" assertion pass without the gate ever being consulted."""
    return json.dumps({"results": _records(outer),
                       "response_text": json.dumps({"results": _records(inner)})})


def _break_only_embedded(monkeypatch):
    """Corrupt the embedded pipeline and NOTHING else.

    Patching `decompress` instead would also break `roundtrip_ok`, so the DEFAULT gate
    would catch the mutation and a test built on it would pass with the fix reverted —
    mutation-tested, that is exactly what happened on the first attempt."""
    import terse.measure as M
    real = M.transforms.compress_with

    def only_embedded_is_broken(obj, *a, **kw):
        text = real(obj, *a, **kw)
        return T.minify({"corrupted": True}) if kw.get("embedded") else text

    monkeypatch.setattr(M.transforms, "compress_with", only_embedded_is_broken)


def test_measure_gate_covers_the_embedded_pipeline_it_scores(monkeypatch):
    """`measure` reports `embedded`/`tier_total` from `compress_with(embedded=True)`, so the
    round-trip gate has to validate THAT pipeline — not just the default one. Otherwise a
    tier that failed only with the flag on would keep its savings banked and feed them to
    `policy generate`."""
    from terse.measure import measure_payload
    m = measure_payload(_double_encoded())
    assert m["roundtrip_ok"] is True and m["embedded_ok"] is True
    assert m["saved_cl100k"]["embedded"] > 0

    _break_only_embedded(monkeypatch)
    bad = measure_payload(_double_encoded())
    # #188: the two gates are SEPARATE. The default pipeline round-tripped, so the row is
    # still valid and still banks its real savings — only the embedded tier is zeroed.
    # Folding these into one flag (as #186 did) demoted a working tool to passthrough.
    assert bad["roundtrip_ok"] is True
    assert bad["embedded_ok"] is False
    assert bad["saved_cl100k"]["embedded"] == 0
    cl = bad["cl100k"]
    assert bad["saved_cl100k"]["tier_total"] == cl["raw"] - cl["compressed"] > 0
    # ...and every tier below embedded keeps its saving — the specific over-reach #188 names.
    for tier in ("minify", "tabularize", "dictionary"):
        assert bad["saved_cl100k"][tier] == m["saved_cl100k"][tier]
    assert bad["saved_cl100k"]["tabularize"] > 0


def test_measure_joined_gates_the_embedded_pipeline_it_scores(monkeypatch):
    """The half #186 left unfixed (#188). `policy_gen._tool_decision` calls `measure_joined`
    FIRST for every result group and only falls back to `measure_payload` when the join
    refuses, so on a multi-block fleet this is the ONLY gate that ever runs."""
    from terse.measure import measure_joined
    raws = [_double_encoded(6) for _ in range(4)]
    good = measure_joined(raws)
    assert good is not None
    assert good["roundtrip_ok"] is True and good["embedded_ok"] is True
    assert good["saved_cl100k"]["embedded"] > 0

    _break_only_embedded(monkeypatch)
    bad = measure_joined(raws)
    assert bad is not None
    assert bad["roundtrip_ok"] is True        # default pipeline unaffected
    assert bad["embedded_ok"] is False        # ...and this is what #186 never checked here
    assert bad["saved_cl100k"]["embedded"] == 0
    cl = bad["cl100k"]
    assert bad["saved_cl100k"]["tier_total"] == cl["raw"] - cl["compressed"] > 0


def test_policy_gen_drops_only_the_embedded_tier_when_its_gate_fails(monkeypatch):
    """An embedded-only failure must cost the TIER, not the TOOL. #186's single flag sent
    `_tool_decision` down the `gate_fail` branch to `tiers: []`, so a tool whose default
    pipeline round-tripped fine lost its working compression entirely and the report claimed
    the codec was not lossless for it."""
    from terse.policy_gen import _tool_decision
    groups = [[_double_encoded(6) for _ in range(4)]]
    good = _tool_decision("srv.tool", groups, threshold=5.0)
    assert "embedded" in good["tiers"] and good["emb_fail"] == 0

    _break_only_embedded(monkeypatch)
    bad = _tool_decision("srv.tool", groups, threshold=5.0)
    assert bad["emb_fail"] == 1
    assert "embedded" not in bad["tiers"]
    assert bad["tiers"] == ["minify", "tabularize", "dictionary"]   # NOT [] — the tool works
    assert bad["saved_pct"] > 5.0
    # The reason must name the losslessness failure, not hide it behind "below threshold":
    # `embedded` sums to 0 for these rows, so the economic branch would fire misleadingly.
    assert "failed the embedded round-trip" in bad["reason"]


def test_diff_label_mirrors_the_runtime_coercion_rather_than_the_authors_intent(tmp_path):
    """DO NOT "fix" this to `is True`. `load_policy` builds the policy with
    `bool(doc.get("diff", False))`, so `"diff": "false"` genuinely diffs at runtime. The
    label reports EFFECTIVE behaviour; tightening it would print "off" while the proxy
    diffs — the label-vs-reality divergence #181 exists to kill. A review flagged the
    truthiness as a bug; this test is why it stays."""
    from terse.install_mcp import _default_diff_label
    from terse.policy import load_policy
    for i, value in enumerate((True, False, "false", "no", 1, 0)):
        path = tmp_path / f"policy{i}.json"
        path.write_text(json.dumps({"version": 1, "diff": value}), encoding="utf-8")
        label, runtime = _default_diff_label(str(path)), load_policy(str(path)).diff
        assert ("on" in label) is runtime, f"{value!r}: label {label} vs runtime {runtime}"


def test_diff_label_refuses_to_resolve_a_relative_policy_path(tmp_path, monkeypatch):
    """A relative `--policy` resolves against the MCP LAUNCHER's cwd, which a status scan
    cannot know — `install_mcp`'s own `policy_missing` check skips relative paths for
    exactly this reason. Reading one would resolve it against the SCANNER's cwd and report
    the diff setting of whatever file happens to sit there: a confidently wrong label, the
    same divergence #181 exists to kill."""
    from terse.install_mcp import _default_diff_label
    (tmp_path / "policy.json").write_text(json.dumps({"version": 1, "diff": True}),
                                          encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # The same file, named two ways. Absolute: read it and report what it says.
    assert _default_diff_label(str(tmp_path / "policy.json")) == "policy (on)"
    # Relative: fall through to the dataclass default rather than trust the scanner's cwd.
    assert _default_diff_label("policy.json").startswith("default (")
