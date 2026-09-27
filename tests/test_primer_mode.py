"""`terse proxy --primer always|never|auto` (#325).

#249 closed with: the primer's value tracks payload SHAPE, not model. It is load-bearing on
the stress corpus (every regression was `deref` over a hoisted `subcols` row or an object
alias with absent columns) and buys nothing on realistic payloads. So `auto` attaches only
on a result whose compressed form carries one of those reconstruction forms; `never` never
attaches; `always` is the lazy attach as it was, and stays the default.
"""
from __future__ import annotations

import json

import pytest

from terse import cli
from terse import install_mcp as im
from terse.policy import Policy, Rule
from terse.proxy import (
    PRIMER_HEAD,
    PRIMER_MODES,
    Interceptor,
    PrimerLatch,
    union_primer,
    wire_needs_primer,
)
from terse.stats import (
    PRIMER_CADENCE_ONCE,
    PRIMER_DECLINE_AUTO,
    PRIMER_DECLINE_NEVER,
    PRIMER_DECLINE_STRUCTURED,
    aggregate,
    build_primer_record,
    build_primer_section,
    primer_liability,
)
from terse.transforms import compress

TIERS = ("minify", "tabularize", "dictionary")
POL = Policy(rules=[Rule("gh.*", TIERS)])


def _flat_text():
    """A flat table with scalar aliases -- the form `auto` sends unexplained."""
    return json.dumps({"result": [{"id": i, "status": "awaiting-triage-from-maintainer",
                                   "url": "https://x.example/api/items"}
                                  for i in range(20)]}, indent=2)


def _nested_text():
    """A uniform nested-dict column -> a hoisted `subcols` header: stress.nested_records'
    shape, where #249's `deref` regression lived."""
    return json.dumps({"result": [{"id": i, "owner": {"name": f"user-{i:02d}",
                                                      "team": "platform-infrastructure"},
                                   "count": i * 3} for i in range(12)]})


def _call(inter, mid, name="gh.api.items"):
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": name}}))


def _result(mid, text, structured=None):
    res: dict = {"content": [{"type": "text", "text": text}]}
    if structured is not None:
        res["structuredContent"] = structured
    return json.dumps({"jsonrpc": "2.0", "id": mid, "result": res})


def _has_primer(line):
    return any(PRIMER_HEAD in b.get("text", "")
               for b in json.loads(line)["result"]["content"])


def _compressed(line):
    return any('"__terse_' in b.get("text", "")
               for b in json.loads(line)["result"]["content"])


def _standalone(mode, rows):
    return Interceptor(POL, primer_mode=mode,
                       stats_primer=lambda c, t, a=True, r=None: rows.append((c, a, r)))


def _drive(inter, mid, text, structured=None):
    _call(inter, mid)
    return inter.transform_response(_result(mid, text, structured))


# --- the shape signal ---

def test_the_signal_fixtures_are_the_shapes_they_claim():
    assert not wire_needs_primer(compress(json.loads(_flat_text())))
    assert wire_needs_primer(compress(json.loads(_nested_text())))


@pytest.mark.parametrize("obj, needs", [
    # Flat, wide (12 columns): read at raw parity without a primer in #249.
    ([{f"c{j}": i * 12 + j for j in range(12)} for i in range(8)], False),
    # Absent columns: near-chance on absent-vs-null without the table paragraph.
    ([{"id": i, "name": f"resource-{i:02d}", "size_kb": i * 13,
       **({"owner_team": f"team-{i}"} if i % 2 else {})} for i in range(20)], True),
    # An object-valued alias (stress.object_alias).
    ([{"id": i, "cfg": [{"r": "us-east-1", "t": "gold"},
                        {"z": "ap-south-1", "t": "bronze", "f": [1]}][i % 2]}
      for i in range(12)], True),
])
def test_the_signal_follows_249s_evidence(obj, needs):
    assert wire_needs_primer(compress(obj)) is needs


def test_unmeasured_forms_and_unparseable_text_attach():
    """Diff/dropped/embedded forms were never measured without a primer, and a parse
    failure says nothing -- both answer True, the `always` direction."""
    assert wire_needs_primer('{"__terse_diff__":1,"shape":"rows"}')
    assert wire_needs_primer('{"__terse_textdiff__":1,"ops":[]}')
    assert wire_needs_primer('{"__terse_dict__":1, broken')
    assert not wire_needs_primer('{"plain":"json"}')


# --- mode parsing ---

def test_mode_values_and_rejection():
    assert PRIMER_MODES == ("always", "never", "auto")
    with pytest.raises(ValueError):
        Interceptor(POL, primer_mode="sometimes")
    with pytest.raises(ValueError):
        PrimerLatch(mode="sometimes")


@pytest.mark.parametrize("argv, want", [([], "always"), (["--primer", "never"], "never"),
                                        (["--primer=auto"], "auto")])
def test_cli_passes_the_mode_to_run_proxy(monkeypatch, argv, want):
    seen = {}
    monkeypatch.setattr("terse.proxy.run_proxy",
                        lambda cmd, pol, **kw: seen.update(kw) or 0)
    assert cli.main(["proxy", "--no-stats", *argv, "--", "some-server"]) == 0
    assert seen["primer_mode"] == want


def test_cli_passes_the_mode_to_the_router(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("terse.multiproxy.run_multi_proxy",
                        lambda path, pol, **kw: seen.update(kw) or 0)
    cfg = tmp_path / "peers.json"
    cfg.write_text("{}")
    assert cli.main(["proxy", "--no-stats", "--primer", "never", "--config", str(cfg)]) == 0
    assert seen["primer_mode"] == "never"


def test_cli_rejects_an_unknown_mode(capsys):
    with pytest.raises(SystemExit):
        cli.main(["proxy", "--primer", "sometimes", "--", "x"])


# --- always: exactly the lazy attach as before ---

def test_always_is_the_default_and_attaches_on_a_flat_result():
    rows: list = []
    inter = Interceptor(POL, stats_primer=lambda c, t, a=True: rows.append((c, a)))
    out = _drive(inter, 1, _flat_text())
    assert _has_primer(out)
    assert rows == [(PRIMER_CADENCE_ONCE, True)]


# --- never ---

def test_never_attaches_nothing_on_any_result():
    rows: list = []
    inter = _standalone("never", rows)
    for mid, text in enumerate([_flat_text(), _nested_text(), _flat_text()], start=1):
        out = _drive(inter, mid, text)
        assert _compressed(out) and not _has_primer(out)
    # structuredContent too, and it is still compressed text-side.
    assert not _has_primer(_drive(inter, 9, _nested_text(), structured={"a": 1}))
    # ONE decline row per session, reason `never`.
    assert rows == [(PRIMER_CADENCE_ONCE, False, PRIMER_DECLINE_NEVER)]


def test_never_rearms_its_decline_row_on_a_new_session():
    rows: list = []
    inter = _standalone("never", rows)
    _drive(inter, 1, _flat_text())
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": 5, "method": "initialize",
                                   "params": {}}))
    _drive(inter, 2, _flat_text())
    assert [r[2] for r in rows] == [PRIMER_DECLINE_NEVER, PRIMER_DECLINE_NEVER]


def test_never_writes_nothing_for_a_session_with_no_wire_form():
    rows: list = []
    inter = _standalone("never", rows)
    _call(inter, 1, name="other.tool")
    inter.transform_response(_result(1, '{"a":1}'))
    assert rows == []


# --- auto ---

def test_auto_declines_a_flat_result_then_attaches_on_a_stress_one_and_latches():
    rows: list = []
    inter = _standalone("auto", rows)
    first = _drive(inter, 1, _flat_text())
    assert _compressed(first) and not _has_primer(first)       # compressed, unexplained
    assert not _has_primer(_drive(inter, 2, _flat_text()))
    assert _has_primer(_drive(inter, 3, _nested_text()))       # the shape that regressed
    assert not _has_primer(_drive(inter, 4, _nested_text()))   # latched, as `always`
    assert rows == [(PRIMER_CADENCE_ONCE, False, PRIMER_DECLINE_AUTO),
                    (PRIMER_CADENCE_ONCE, True, None)]


def test_auto_attaches_immediately_when_the_first_result_is_stress_shaped():
    rows: list = []
    inter = _standalone("auto", rows)
    assert _has_primer(_drive(inter, 1, _nested_text()))
    assert rows == [(PRIMER_CADENCE_ONCE, True, None)]


# --- router: one mode, shared by every peer ---

def _router(mode, rows, pol=POL, n=2):
    latch = PrimerLatch(mode=mode)
    peers = [Interceptor(pol, server_name=f"p{i}", lazy_primer=False, shared_primer=latch,
                         stats_primer=lambda c, t, a=True, r=None: rows.append((a, r)))
             for i in range(n)]
    latch.set_text(union_primer([(pol, f"p{i}") for i in range(n)], structured_wrap=True))
    return latch, peers


def test_router_never_attaches_nothing_and_holds_nothing():
    rows: list = []
    pol = Policy(rules=[Rule("gh.*", TIERS, structured="replace")])
    _latch, (a, b) = _router("never", rows, pol)
    assert not _has_primer(_drive(a, 1, _nested_text()))
    assert not _has_primer(_drive(b, 2, _flat_text()))
    # No primer is ever coming, so a typed result is no longer held raw for one.
    structured = {"rows": [{"id": i, "status": "active"} for i in range(12)]}
    b.client_name = "claude-code"
    out = json.loads(_drive(b, 3, _flat_text(), structured=structured))["result"]
    assert out.get("structuredContent") != structured
    assert rows == [(False, PRIMER_DECLINE_NEVER)]              # one row for the fleet


def test_router_auto_shares_the_decision_across_peers():
    rows: list = []
    latch, (a, b) = _router("auto", rows)
    assert a._primer_mode == b._primer_mode == "auto"
    assert not _has_primer(_drive(a, 1, _flat_text()))
    assert not _has_primer(_drive(b, 2, _flat_text()))
    assert _has_primer(_drive(b, 3, _nested_text()))
    assert not _has_primer(_drive(a, 4, _nested_text()))       # the shared latch is spent
    assert rows == [(False, PRIMER_DECLINE_AUTO), (True, None)]


def test_router_auto_sends_a_flat_typed_result_compressed_and_unwrapped():
    """#463's tentative wrap: a wire form `auto` sends unexplained is released compressed
    rather than reverted to the raw hold -- no primer is owed for it."""
    rows: list = []
    pol = Policy(rules=[Rule("gh.*", TIERS, structured="auto")])
    latch, (a, _b) = _router("auto", rows, pol)
    a.client_name = "claude-code"
    structured = {"rows": [{"id": i, "status": "awaiting-triage-from-maintainer"}
                           for i in range(12)]}
    out = json.loads(_drive(a, 1, _flat_text(), structured=structured))["result"]
    assert "__terse_primer__" not in out["structuredContent"]
    assert "__terse_" in json.dumps(out["structuredContent"])
    assert latch.pending()
    assert rows == [(False, PRIMER_DECLINE_AUTO)]


def test_router_always_is_unchanged():
    rows: list = []
    _latch, (a, b) = _router("always", rows)
    assert _has_primer(_drive(a, 1, _flat_text()))
    assert not _has_primer(_drive(b, 2, _nested_text()))
    assert rows == [(True, None)]


# --- the ledger ---

def test_a_decline_row_carries_its_reason_and_zero_cost():
    for reason in (PRIMER_DECLINE_NEVER, PRIMER_DECLINE_AUTO):
        rec = build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE, primer="P" * 400,
                                  attached=False, reason=reason)
        assert rec["reason"] == reason and rec["tokens"] == 0 and rec["bytes"] == 0
    # The #286 suppression, written without a reason, is the `structured` one.
    rec = build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE, primer="P",
                              attached=False)
    assert rec["reason"] == PRIMER_DECLINE_STRUCTURED
    assert "reason" not in build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE,
                                               primer="P")


def test_aggregate_keeps_attach_and_each_decline_apart():
    recs = [build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40),
            build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE, primer="P",
                                attached=False, reason=PRIMER_DECLINE_AUTO),
            build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE, primer="P",
                                attached=False, reason=PRIMER_DECLINE_NEVER),
            # A pre-#325 suppression row: no `reason` key at all.
            {**build_primer_record("gh", cadence=PRIMER_CADENCE_ONCE, primer="P",
                                   attached=False), "reason": None}]
    got = {(r["attached"], r["reason"]) for r in aggregate(recs)["primers"]}
    assert got == {(True, None), (False, "auto"), (False, "never"), (False, "structured")}


# --- primer_liability under each mode ---

def _scan_row(primer=None):
    return {"server": "gh", "state": "wrapped", "wraps": "gh-server", "scope": "user",
            "policy": None, "primer": primer}


def _agg(primer_rows=()):
    recs = [{"server": "gh-server", "tool": "gh.api.items", "raw_chars": 1000,
             "out_chars": 400, "raw_tokens": 250, "out_tokens": 100,
             "decision": "tabularize", "passthrough": False}]
    return aggregate(recs + list(primer_rows))


def _decline(reason):
    return build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 900,
                               attached=False, reason=reason)


def test_liability_always_bills_the_estimate_as_before():
    row = primer_liability([_scan_row()], _agg())["servers"][0]
    assert row["primer_mode"] == "always" and row["primer_source"] == "estimated"
    assert row["primer_tokens"] > 0


def test_liability_never_is_a_configured_zero_even_without_ledger_rows():
    liab = primer_liability([_scan_row("never")], _agg())
    row = liab["servers"][0]
    assert row["primer_tokens"] == 0 and row["primer_source"] == "configured"
    assert row["break_even_verdict"] == "no primer"
    assert liab["session_once_tokens"] == 0
    assert any("--primer never" in ln for ln in build_primer_section(liab))


def test_liability_never_with_its_decline_row_is_measured():
    row = primer_liability([_scan_row("never")],
                           _agg([_decline(PRIMER_DECLINE_NEVER)]))["servers"][0]
    assert row["primer_tokens"] == 0 and row["primer_source"] == "recorded"


def test_liability_never_still_bills_a_recorded_attach():
    """The window can predate the flag: an attach that went out is a fact."""
    att = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    row = primer_liability([_scan_row("never")], _agg([att]))["servers"][0]
    assert row["primer_source"] == "recorded" and row["primer_tokens"] > 0


def test_liability_auto_declined_everywhere_is_a_measured_zero_named_by_mode():
    liab = primer_liability([_scan_row("auto")], _agg([_decline(PRIMER_DECLINE_AUTO)]))
    row = liab["servers"][0]
    assert row["primer_tokens"] == 0 and row["primer_source"] == "recorded"
    text = "\n".join(build_primer_section(liab))
    assert "#325" in text and "structuredContent" not in text


def test_liability_auto_decline_then_attach_bills_the_attach():
    att = build_primer_record("gh-server", cadence=PRIMER_CADENCE_ONCE, primer="P" * 40)
    row = primer_liability([_scan_row("auto")],
                           _agg([_decline(PRIMER_DECLINE_AUTO), att]))["servers"][0]
    assert row["primer_source"] == "recorded" and row["primer_tokens"] > 0


def test_liability_auto_without_rows_is_an_estimate_flagged_as_an_upper_bound():
    liab = primer_liability([_scan_row("auto")], _agg())
    row = liab["servers"][0]
    assert row["primer_source"] == "estimated" and row["primer_tokens"] > 0
    assert any("UPPER bound" in ln for ln in build_primer_section(liab))


def test_liability_reads_an_unknown_mode_as_always():
    row = primer_liability([_scan_row("bogus")], _agg())["servers"][0]
    assert row["primer_mode"] == "always" and row["primer_tokens"] > 0


# --- the baked flag is terse's, never the downstream's ---

def test_parse_proxy_opts_recognizes_primer_both_spellings():
    for flag in (["--primer", "never"], ["--primer=never"]):
        entry = {"command": "/usr/bin/python",
                 "args": ["-m", "terse", "proxy", *flag, "--server-name", "kb", "--", "kb"]}
        assert im.parse_proxy_opts(entry) == {"primer": "never", "server_name": "kb"}


def test_a_downstream_primer_flag_is_not_terses():
    entry = {"command": "/usr/bin/python",
             "args": ["-m", "terse", "proxy", "--", "kb-server", "--primer", "never"]}
    assert "primer" not in (im.parse_proxy_opts(entry) or {})
