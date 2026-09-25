"""#212 — the multiproxy router primes LAZILY: one union primer per session, attached to the
first terse-marked result from ANY peer, instead of riding `initialize.instructions` into
every turn's system prompt.

Measured before building (program plan 2.2): 71% of router sessions never see a terse
marker, and the eager primer was 0.25% of all spend.
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import threading

import pytest

from terse.multiproxy import run_multi_proxy
from terse.policy import Policy, Rule
from terse.proxy import PRIMER_HEAD, Interceptor, PrimerLatch, union_primer

FAKE = pathlib.Path(__file__).parent / "fake_mcp_server.py"
TIERS = ("minify", "tabularize", "dictionary")
POL = Policy(rules=[Rule("plain.*", ("minify",)), Rule("gh.*", TIERS)])


def _records_text():
    return json.dumps({"result": [{"id": i, "status": "active",
                                   "url": "https://x.example/api/items"}
                                  for i in range(20)]}, indent=2)


def _call(inter, mid, name="gh.api.items"):
    inter.note_request(json.dumps({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                                   "params": {"name": name}}))


def _result(mid, text=None, structured=None):
    res: dict = {"content": [{"type": "text", "text": text or _records_text()}]}
    if structured is not None:
        res["structuredContent"] = structured
    return json.dumps({"jsonrpc": "2.0", "id": mid, "result": res})


def _peers(n=2, emitted=None):
    latch = PrimerLatch()
    peers = [Interceptor(POL, server_name=f"p{i}", lazy_primer=False, shared_primer=latch,
                         stats_primer=(lambda c, t, a=True: emitted.append((c, len(t), a)))
                         if emitted is not None else None)
             for i in range(n)]
    latch.set_text(union_primer([(POL, f"p{i}") for i in range(n)]))
    return latch, peers


def _has_primer(line):
    return any(PRIMER_HEAD in b.get("text", "")
               for b in json.loads(line)["result"]["content"])


def test_one_union_primer_per_session_across_all_peers():
    emitted: list = []
    latch, (a, b) = _peers(emitted=emitted)
    _call(a, 1)
    _call(b, 2)
    first, second = a.transform_response(_result(1)), b.transform_response(_result(2))
    assert _has_primer(first) and not _has_primer(second)
    assert json.loads(first)["result"]["content"][0]["text"] == latch.text
    # Recorded once, attached, at the union primer's real size.
    assert emitted == [("once/session", len(latch.text), True)]


def test_a_reinitialize_on_any_peer_rearms_the_shared_primer():
    latch, (a, b) = _peers()
    _call(a, 1)
    assert _has_primer(a.transform_response(_result(1)))
    # The router broadcasts initialize to every peer; each resets the ONE latch.
    for p in (a, b):
        p.note_request(json.dumps({"jsonrpc": "2.0", "id": 9, "method": "initialize",
                                   "params": {}}))
    _call(b, 2)
    assert _has_primer(b.transform_response(_result(2)))


def test_structured_content_is_held_raw_until_the_router_primer_has_attached():
    """#212 review: the primer cannot ride a result carrying `structuredContent` (the client
    discards the text block), and before #212 the eager `initialize` primer explained it.
    So while the shared primer is owed, the typed field goes out RAW — never an envelope the
    model has no legend for — and no suppression is recorded, since none was owed."""
    emitted: list = []
    latch, (a, b) = _peers(emitted=emitted)
    for p in (a, b):
        p.client_name = "claude-code"          # `auto` compresses the typed field here
    rows = {"rows": [{"id": i, "status": "active"} for i in range(12)]}
    _call(a, 1)
    held = json.loads(a.transform_response(_result(1, structured=rows)))["result"]
    assert held["structuredContent"] == rows
    _call(b, 2)
    assert _has_primer(b.transform_response(_result(2)))     # a text result primes
    _call(a, 3)
    after = json.loads(a.transform_response(_result(3, structured=rows)))["result"]
    assert after["structuredContent"] != rows                # ...and compression resumes
    assert "__terse_" in json.dumps(after["structuredContent"])
    assert emitted == [("once/session", len(latch.text), True)]


def test_an_unmarked_result_does_not_spend_the_primer():
    _latch, (a, _b) = _peers()
    # minify-only: the bytes change, but no terse marker reaches the model to explain.
    _call(a, 1, name="plain.tool")
    assert not _has_primer(a.transform_response(_result(1)))
    _call(a, 2)
    assert _has_primer(a.transform_response(_result(2)))


def test_concurrent_claims_attach_exactly_once():
    latch = PrimerLatch("x")
    wins: list[bool] = []
    barrier = threading.Barrier(16)

    def go():
        barrier.wait()
        wins.append(latch.claim())
    threads = [threading.Thread(target=go) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wins.count(True) == 1


def test_run_multi_proxy_is_lazy_end_to_end_and_bills_the_router_label(tmp_path):
    """Through the real router: no primer at initialize, one on the first compressed
    result, and the ledger row under `router:<peers file>` — never a peer label (#396)."""
    from terse.stats import load_stats
    cfg = tmp_path / "multi.json"
    cfg.write_text(json.dumps({"downstreams": [
        {"name": "gh", "command": [sys.executable, str(FAKE)]}]}), encoding="utf-8")
    log = tmp_path / "stats.jsonl"
    cin = io.StringIO("\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "gh__gh.api.items"}}]) + "\n")
    cout = io.StringIO()
    assert run_multi_proxy(str(cfg), POL, stdin=cin, stdout=cout, stats_log=str(log)) == 0
    msgs = {m["id"]: m for m in (json.loads(ln) for ln in cout.getvalue().splitlines()
                                 if ln.strip())}
    assert PRIMER_HEAD not in (msgs[1]["result"].get("instructions") or "")
    assert msgs[2]["result"]["content"][0]["text"].startswith(PRIMER_HEAD)
    from terse.stats import router_ledger_label
    label = router_ledger_label(cfg)
    rows = load_stats(log)
    primers = [r for r in rows if r.get("event") == "primer"]
    assert [(p["server"], p["attached"]) for p in primers] == [(label, True)]
    # ...and one session row per initialize, under the same label (#212 accounting).
    assert [r["server"] for r in rows if r.get("event") == "router_session"] == [label]


# --- #212 accounting: `primer_liability` reads a lazy router from its own rows ---------

def _pfile(tmp_path, name="peers.json"):
    f = tmp_path / name
    f.write_text("{}", encoding="utf-8")
    return str(f)


def _pol(tmp_path):
    p = tmp_path / "pol.json"
    p.write_text(json.dumps({"version": 1, "policies": [
        {"match": {"tool": "*"}, "tiers": ["minify", "tabularize"]}]}), encoding="utf-8")
    return str(p)


def _router_row(tmp_path, peers_file, wraps="gh, kb"):
    return {"scope": "user", "server": "terse", "state": "router", "wraps": wraps,
            "policy": _pol(tmp_path), "peers_file": peers_file}


def _agg(tools=(), sessions=None, attaches=0, tokens=500):
    from terse.stats import router_ledger_label
    rows = [{"server": s, "tool": "t", "blocks": b, "tokenized": b, "encoded": b,
             "raw_tokens": 10 * b * 100, "out_tokens": 10 * b * 20,
             "context_raw_tokens": 10 * b * 100, "context_out_tokens": 10 * b * 20,
             "raw_chars": 0, "out_chars": 0, "diffs": 0,
             # (label, blocks, stamped): stamped = written by THIS lazy router's peer (#212).
             "router_stamps": ({router_ledger_label(sessions[0]): b}
                               if stamped and sessions is not None else {})}
            for s, b, stamped in tools]
    keys = ("blocks", "raw_tokens", "out_tokens", "context_raw_tokens", "context_out_tokens")
    agg = {"total": {k: sum(r[k] for r in rows) for k in keys}
           | {"raw_chars": 0, "out_chars": 0, "untokenized": 0},
           "decisions": {}, "diff_reasons": {}, "tools": rows, "primers": [],
           "router_sessions": []}
    if sessions is not None:
        label = router_ledger_label(sessions[0])
        agg["router_sessions"] = [{"server": label, "sessions": sessions[1]}]
        if attaches:
            agg["primers"] = [{"server": label, "cadence": "once/session", "attached": True,
                               "emissions": attaches, "bytes": 0,
                               "tokens": tokens * attaches, "untokenized": 0}]
    return agg


def test_an_idle_lazy_router_is_free_not_unwrap(tmp_path):
    """Review of PR A: billed per turn, a never-called lazy router read UNWRAP / "pure
    cost" — the #211 inversion. It recorded sessions and never attached: provably free."""
    from terse.stats import primer_liability
    pf = _pfile(tmp_path)
    liab = primer_liability([_router_row(tmp_path, pf)], _agg(sessions=(pf, 3, 100)))
    srv = liab["servers"][0]
    assert srv["cadence"] != "per-turn" and srv["verdict"] != "UNWRAP"
    assert liab["idle"] == [] and liab["per_turn_tokens"] == 0
    assert "terse" in liab["free"]


def test_a_lazy_router_that_attached_is_billed_its_recorded_primer_once_per_session(
        tmp_path):
    from terse.stats import primer_liability
    pf = _pfile(tmp_path)
    liab = primer_liability([_router_row(tmp_path, pf)],
                            _agg(tools=[("gh", 5, True)], sessions=(pf, 4, 100),
                                 attaches=2, tokens=480))
    srv = liab["servers"][0]
    assert (srv["cadence"], srv["primer_tokens"], srv["primer_source"]) == (
        "once/session", 480, "recorded")
    assert liab["per_turn_tokens"] == 0 and liab["session_once_tokens"] == 480


def test_a_window_straddling_the_upgrade_stays_per_turn(tmp_path):
    """An UNSTAMPED peer row means part of the window ran an eager (pre-#212) router that
    recorded nothing — the per-turn bill stands."""
    from terse.stats import primer_liability
    pf = _pfile(tmp_path)
    liab = primer_liability([_router_row(tmp_path, pf)],
                            _agg(tools=[("gh", 5, False)], sessions=(pf, 4, 100)))
    assert liab["servers"][0]["cadence"] == "per-turn"


def test_no_session_rows_means_per_turn_as_before(tmp_path):
    from terse.stats import primer_liability
    pf = _pfile(tmp_path)
    liab = primer_liability([_router_row(tmp_path, pf)], _agg(tools=[("gh", 5, False)]))
    assert liab["servers"][0]["cadence"] == "per-turn"


def test_another_routers_sessions_do_not_make_this_one_lazy(tmp_path):
    """Two project-scope routers share a peers-file BASENAME across repos; the label hashes
    the resolved path so one repo's sessions cannot speak for another's router."""
    from terse.stats import primer_liability, router_ledger_label
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    mine = _pfile(tmp_path / "r1", ".terse-peers-project-x.json")
    other = _pfile(tmp_path / "r2", ".terse-peers-project-x.json")
    assert router_ledger_label(mine) != router_ledger_label(other)
    liab = primer_liability([_router_row(tmp_path, mine)], _agg(sessions=(other, 3, 100)))
    assert liab["servers"][0]["cadence"] == "per-turn"


def test_the_scan_and_the_router_agree_on_the_label(tmp_path):
    """Wiring: the label `primer_liability` derives from a REAL scan row's `peers_file` is
    the one a router launched from that install's `--config` writes under."""
    from terse import install_mcp as im
    from terse.stats import router_ledger_label
    cfg = tmp_path / "claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"kb": {"command": "kb-mcp"}}}),
                   encoding="utf-8")
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"version": 1, "defaults": {"tiers": ["minify"]}}),
                   encoding="utf-8")
    im.do_install(["kb"], str(pol), cfg=cfg, multiproxy=True)
    router = next(r for r in im.scan_scopes(cfg=cfg) if r["state"] == "router")
    args = json.loads(cfg.read_text())["mcpServers"]["terse"]["args"]
    launched_with = args[args.index("--config") + 1]
    assert router_ledger_label(router["peers_file"]) == router_ledger_label(launched_with)


def test_two_real_peers_share_ONE_primer_end_to_end(tmp_path):
    """Review of PR A: every end-to-end test had one peer, where a peer's own primer and
    the union primer are the same text — a per-peer latch passed them all. Two peers through
    the real router: exactly one of the two results carries the primer, one attach row."""
    from terse.stats import load_stats
    cfg = tmp_path / "multi.json"
    cfg.write_text(json.dumps({"downstreams": [
        {"name": "gh", "command": [sys.executable, str(FAKE)]},
        {"name": "gx", "command": [sys.executable, str(FAKE)]}]}), encoding="utf-8")
    log = tmp_path / "stats.jsonl"
    cin = io.StringIO("\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "gh__gh.api.items"}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "gx__gh.api.items"}}]) + "\n")
    cout = io.StringIO()
    assert run_multi_proxy(str(cfg), POL, stdin=cin, stdout=cout, stats_log=str(log)) == 0
    msgs = {m["id"]: m for m in (json.loads(ln) for ln in cout.getvalue().splitlines()
                                 if ln.strip())}
    primed = [i for i in (2, 3) if any(PRIMER_HEAD in b.get("text", "")
                                       for b in msgs[i]["result"]["content"])]
    assert len(primed) == 1, primed
    attaches = [r for r in load_stats(log) if r.get("event") == "primer"]
    assert len(attaches) == 1


class _RacingLatch(PrimerLatch):
    """Forces two peers to both see `pending()` before either claims — the window the GIL
    otherwise makes too small for a plain check-then-set to lose in a test."""

    def __init__(self, text):
        super().__init__(text)
        self.barrier = threading.Barrier(2, timeout=5)

    def pending(self):
        p = super().pending()
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return p


def test_two_peers_racing_through_the_attach_prime_exactly_once():
    latch = _RacingLatch("")
    peers = [Interceptor(POL, server_name=f"p{i}", lazy_primer=False, shared_primer=latch)
             for i in range(2)]
    latch.set_text(union_primer([(POL, "p0"), (POL, "p1")]))
    for i, p in enumerate(peers):
        _call(p, i + 1)
    outs: dict[int, str] = {}

    def go(i):
        outs[i] = peers[i].transform_response(_result(i + 1))
    threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(_has_primer(o) for o in outs.values()) == 1


def test_a_busy_lazy_router_with_no_attach_is_billed_the_estimate_not_measured_free(
        tmp_path):
    """Review round 2: one idle session row, then compressed peer rows from ANOTHER router
    session whose own attach row fell outside the window — all stamped, so the router IS
    lazy, but its attach cannot be seen. Declaring that "provably free" under-billed as a
    measurement. Absence is evidence of nothing: `encoded > 0` with no attach bills the
    estimate once per session."""
    from terse.stats import primer_liability
    pf = _pfile(tmp_path)
    liab = primer_liability([_router_row(tmp_path, pf)],
                            _agg(tools=[("gh", 6, True)], sessions=(pf, 1, 100)))
    srv = liab["servers"][0]
    assert srv["cadence"] == "once/session" and srv["primer_source"] == "estimated"
    assert srv["primer_tokens"] and "terse" not in liab["free"]


def test_a_contested_peer_does_not_make_a_lazy_routers_primer_unknowable(tmp_path):
    """The router's primer rows live under its OWN label, so a peer label contested by a
    live duplicate (#396) says nothing about them: recorded, not `1x?`."""
    from terse.stats import primer_liability
    pf = _pfile(tmp_path)
    pol = _pol(tmp_path)
    rows = [_router_row(tmp_path, pf),
            {"scope": "user", "server": "kb", "state": "folded-and-live",
             "wraps": "kb-server --stdio", "policy": pol, "ledger_identity": "kb",
             "ledger_identity_explicit": True}]
    liab = primer_liability(rows, _agg(tools=[("gh", 5, True), ("kb", 3, True)],
                                       sessions=(pf, 2, 100), attaches=1, tokens=480))
    router = next(s for s in liab["servers"] if s["server"] == "terse")
    assert router["contested_labels"] == ["kb"]
    assert (router["cadence"], router["primer_tokens"]) == ("once/session", 480)



def _stats_sink(rows):
    return lambda tool, raw, emitted, passthrough, reason, st, so: rows.append(
        {"raw": raw, "emitted": emitted, "passthrough": passthrough, "reason": reason,
         "structured": st, "structured_out": so})


def test_a_held_result_passes_through_WHOLE_and_is_ledgered_as_passthrough():
    """Review round 2: holding only the typed field left a text-reading client (Cursor:
    `auto` resolves to `leave`) reading a terse text block no primer explained, and the
    ledger credited a context saving the model never received. The whole result is held,
    and the row says so."""
    for client in ("claude-code", "cursor"):
        rows: list = []
        latch = PrimerLatch()
        a = Interceptor(POL, server_name="p0", lazy_primer=False, shared_primer=latch,
                        stats=_stats_sink(rows))
        latch.set_text(union_primer([(POL, "p0")]))
        a.client_name = client
        typed = {"rows": [{"id": i, "status": "active"} for i in range(12)]}
        raw = _result(1, structured=typed)
        _call(a, 1)
        out = json.loads(a.transform_response(raw))["result"]
        assert out == json.loads(raw)["result"], client        # byte-for-byte untouched
        assert rows and all(r["passthrough"] and r["raw"] == r["emitted"]
                            and r["reason"] == "primer_hold" for r in rows), client
        assert latch.pending(), client                         # still owed


def test_the_stamp_survives_a_since_cut_that_drops_the_session_row(tmp_path):
    """Round 4: `--since` drops a session's `router_session` row but keeps its later peer
    rows, so the round-2 timestamp rule read a fleet lazy ALL ALONG as per-turn in almost
    every window. The per-row stamp is positive proof and cannot be cut that way. Driven
    through the real `load_stats(since)` + `aggregate`."""
    from terse.stats import (
        aggregate,
        append_stats,
        load_stats,
        primer_liability,
        router_ledger_label,
    )
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    log = tmp_path / "stats.jsonl"
    for rec in [{"ts": 100, "server": label, "event": "router_session"},
                _rec(110, stamp=label), _rec(160, stamp=label),
                {"ts": 200, "server": label, "event": "router_session"},
                _rec(210, stamp=label)]:
        append_stats(rec, log)

    def cadence(since):
        agg = aggregate(load_stats(log, since_ts=since))
        return primer_liability([_router_row(tmp_path, pf)], agg)["servers"][0]["cadence"]

    for since in (None, 150, 205):
        assert cadence(since) != "per-turn", since


def test_the_label_resolves_the_path_so_relative_and_absolute_agree(tmp_path, monkeypatch):
    from terse.stats import router_ledger_label
    pf = _pfile(tmp_path, "peers.json")
    monkeypatch.chdir(tmp_path)
    assert router_ledger_label("peers.json") == router_ledger_label(pf)
    assert router_ledger_label("./peers.json") == router_ledger_label(pf)


# --- review round 3 ---------------------------------------------------------------

def test_after_the_attach_structured_results_compress_again_text_and_ledger_too():
    """`hold_all` must end with the primer: a 'hold forever' passed the older test, which
    only looked at the typed field."""
    rows: list = []
    latch = PrimerLatch()
    a = Interceptor(POL, server_name="p0", lazy_primer=False, shared_primer=latch,
                    stats=_stats_sink(rows))
    latch.set_text(union_primer([(POL, "p0")]))
    a.client_name = "claude-code"
    _call(a, 1)
    assert _has_primer(a.transform_response(_result(1)))     # text result primes
    rows.clear()
    typed = {"rows": [{"id": i, "status": "active"} for i in range(12)]}
    _call(a, 2)
    out = json.loads(a.transform_response(_result(2, structured=typed)))["result"]
    assert "__terse_" in out["content"][0]["text"]
    assert rows and not any(r["passthrough"] or r["reason"] == "primer_hold" for r in rows)


def test_a_held_result_drops_the_tools_diff_bases():
    """The client last received the RAW held text, so a later diff must not reference a
    base from before it (same discipline as the mirror drop)."""
    latch = PrimerLatch()
    a = Interceptor(POL, server_name="p0", lazy_primer=False, shared_primer=latch)
    latch.set_text(union_primer([(POL, "p0")]))
    tool = "gh.api.items"
    for m in (a.last, a.last_args, a.last_joined, a.since_keyframe, a.last_text,
              a.since_text_keyframe):
        m[tool] = object()
    a.diff = True
    _call(a, 1)
    a.transform_response(_result(1, structured={"rows": [{"id": 1}]}))
    for m in (a.last, a.last_args, a.last_joined, a.since_keyframe, a.last_text,
              a.since_text_keyframe):
        assert tool not in m


def test_an_empty_union_primer_is_never_pending():
    """A default-deny fleet assembles no primer: nothing is owed, so nothing is held."""
    latch = PrimerLatch("")
    assert not latch.pending()
    latch.set_text("")
    assert not latch.pending()
    latch.reset()
    assert not latch.pending()
    latch.set_text("x")
    assert latch.pending()


def test_only_initialize_writes_a_session_row(tmp_path):
    from terse.stats import load_stats
    cfg = tmp_path / "multi.json"
    cfg.write_text(json.dumps({"downstreams": [
        {"name": "gh", "command": [sys.executable, str(FAKE)]}]}), encoding="utf-8")
    log = tmp_path / "stats.jsonl"
    cin = io.StringIO("\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}]) + "\n")
    assert run_multi_proxy(str(cfg), POL, stdin=cin, stdout=io.StringIO(),
                           stats_log=str(log)) == 0
    assert len([r for r in load_stats(log) if r.get("event") == "router_session"]) == 1


def _rec(ts, server="gh", tool="gh.api.items", stamp=None):
    r = {"ts": ts, "version": "9", "server": server, "tool": tool,
         "decision": "compressed", "raw_chars": 400, "out_chars": 40,
         "raw_tokens": 100, "out_tokens": 10}
    if stamp is not None:
        r["router"] = stamp
    return r


def test_one_unstamped_row_under_any_tool_or_peer_keeps_the_router_per_turn(tmp_path):
    """No claimed row may be UNSTAMPED: one (an eager router's) under a second tool or a
    second peer keeps the router per-turn."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    base = [{"ts": 100, "server": label, "event": "router_session"},
            _rec(200, tool="gh.api.a", stamp=label), _rec(200, server="kb", stamp=label)]

    def cadence(extra):
        return primer_liability([_router_row(tmp_path, pf)],
                                aggregate(base + extra))["servers"][0]["cadence"]

    assert cadence([]) != "per-turn"
    assert cadence([_rec(300, tool="gh.api.b")]) == "per-turn"
    assert cadence([_rec(300, server="kb")]) == "per-turn"
    # Another LAZY router's stamp is no evidence of an eager period (round 5): still lazy.
    assert cadence([_rec(300, stamp="router:other.json:0000000000")]) != "per-turn"


def _dup_rows(tmp_path, pf):
    return [_router_row(tmp_path, pf),
            {"scope": "user", "server": "kb", "state": "folded-and-live",
             "wraps": "kb-server --stdio", "policy": _pol(tmp_path),
             "ledger_identity": "kb", "ledger_identity_explicit": True}]


def test_a_contested_peers_old_rows_still_keep_a_straddling_window_per_turn(tmp_path):
    """Review round 3: `labels` has contested labels removed, so kb's ts-50 row (possibly
    the pre-#212 router's own) never reached the straddle check and the window read lazy."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    recs = [{"ts": 100, "server": label, "event": "router_session"},
            _rec(50, server="kb"), _rec(200, stamp=label)]
    router = next(s for s in primer_liability(_dup_rows(tmp_path, pf),
                                              aggregate(recs))["servers"]
                  if s["server"] == "terse")
    assert router["cadence"] == "per-turn"


def test_a_name_only_contest_does_not_make_a_lazy_router_uncertain(tmp_path):
    """Review round 3: kb contested with no kb rows sent `blocks=None` to `_cadence`, so a
    router with 5 real compressed blocks read `1x?` and its primer left both totals."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    recs = [{"ts": 100, "server": label, "event": "router_session"}] + \
        [_rec(200 + i, stamp=label) for i in range(5)]
    liab = primer_liability(_dup_rows(tmp_path, pf), aggregate(recs))
    router = next(s for s in liab["servers"] if s["server"] == "terse")
    assert router["cadence"] == "once/session" and router["primer_tokens"]
    assert liab["session_once_tokens"] >= router["primer_tokens"]


def test_the_real_router_stamps_every_peer_row_with_its_label(tmp_path):
    """Wiring: the stamp is written by `_build_peers`' stats writer, not only read."""
    from terse.stats import load_stats, router_ledger_label
    cfg = tmp_path / "multi.json"
    cfg.write_text(json.dumps({"downstreams": [
        {"name": "gh", "command": [sys.executable, str(FAKE)]}]}), encoding="utf-8")
    log = tmp_path / "stats.jsonl"
    cin = io.StringIO("\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "gh__gh.api.items"}}]) + "\n")
    assert run_multi_proxy(str(cfg), POL, stdin=cin, stdout=io.StringIO(),
                           stats_log=str(log)) == 0
    rows = [r for r in load_stats(log) if not r.get("event")]
    assert rows and all(r.get("router") == router_ledger_label(cfg) for r in rows)


@pytest.mark.parametrize("reasons,diff_note,hold_note", [
    ({"diff_off": 3, "primer_hold": 2}, True, True),
    ({"primer_hold": 2}, False, True),
    ({"diff_off": 3}, True, False),
    ({"diff_off": 3, "emitted": 1, "primer_hold": 1}, False, True),
])
def test_primer_hold_neither_hides_the_diff_off_note_nor_goes_unexplained(
        reasons, diff_note, hold_note):
    """Round 4, L2: on a default router fleet `primer_hold` rides beside `diff_off`."""
    from terse.stats import build_stats_report
    agg = {"total": {"blocks": 1, "raw_chars": 4, "out_chars": 4, "raw_tokens": 1,
                     "out_tokens": 1, "context_raw_tokens": 1, "context_out_tokens": 1,
                     "untokenized": 0, "unversioned": 0},
           "decisions": {"compressed": 1}, "diff_reasons": reasons, "tools": [],
           "versions": {}, "retrieves": [], "primers": [], "router_sessions": []}
    text = build_stats_report(agg, log_path="x")
    assert ("diff_off = cross-call diffing is OFF" in text) is diff_note
    assert ("primer_hold = " in text) is hold_note


def test_a_router_that_writes_no_rows_is_never_proven_lazy(tmp_path):
    """Round 4, L1: a router now baked `--no-stats` claims no labels, so `all()` over an
    empty `claimed` passed vacuously on its old session rows and the primer left
    `per_turn_tokens` — even beside unstamped (eager) peer rows."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    row = {**_router_row(tmp_path, pf), "stats": False}
    recs = [{"ts": 100, "server": router_ledger_label(pf), "event": "router_session"},
            _rec(50), _rec(200)]
    liab = primer_liability([row], aggregate(recs))
    assert liab["servers"][0]["cadence"] == "per-turn" and liab["per_turn_tokens"]



def test_a_lazy_routers_contested_label_with_its_own_stamped_rows_is_still_billed(tmp_path):
    """Round 5, M1: the #396 blackout nulled `blocks`/`encoded` before `_cadence`, so a lazy
    router whose contested peer had rows (all carrying its stamp) and no attach read `1x?`
    and its primer left BOTH totals. The blackout still governs the savings; the primer
    question is answered from the router's own evidence."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    recs = [{"ts": 100, "server": label, "event": "router_session"}] + \
        [_rec(200 + i, stamp=label) for i in range(5)] + [_rec(300, server="kb", stamp=label)]
    liab = primer_liability(_dup_rows(tmp_path, pf), aggregate(recs))
    router = next(s for s in liab["servers"] if s["server"] == "terse")
    assert router["blocks"] is None                          # savings still blacked out
    assert router["cadence"] == "once/session" and router["primer_tokens"]
    assert liab["session_once_tokens"] >= router["primer_tokens"]


def test_two_stamped_tools_under_one_peer_are_summed(tmp_path):
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    recs = [{"ts": 100, "server": label, "event": "router_session"},
            _rec(200, tool="gh.api.a", stamp=label), _rec(201, tool="gh.api.b", stamp=label),
            _rec(202, tool="gh.api.b", stamp=label)]
    liab = primer_liability([_router_row(tmp_path, pf)], aggregate(recs))
    assert liab["servers"][0]["cadence"] != "per-turn"


def test_stamps_under_a_contested_label_count_as_evidence(tmp_path):
    """No session row in the window (cut by `--since`), and the only stamped rows sit under
    a contested label the router still WRITES: they prove it ran lazily."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    liab = primer_liability(_dup_rows(tmp_path, pf),
                            aggregate([_rec(300, server="kb", stamp=label)]))
    router = next(s for s in liab["servers"] if s["server"] == "terse")
    # Lazy AND billed: those stamped rows are also the proof it was called (round 6).
    assert router["cadence"] == "once/session" and "terse" not in liab["free"]


def test_only_another_routers_stamps_are_no_evidence_for_this_one(tmp_path):
    from terse.stats import aggregate, primer_liability
    pf = _pfile(tmp_path)
    liab = primer_liability([_router_row(tmp_path, pf)],
                            aggregate([_rec(300, stamp="router:other.json:0000000000")]))
    assert liab["servers"][0]["cadence"] == "per-turn"


def test_router_sessions_are_counted_not_flagged(tmp_path):
    from terse.stats import aggregate
    agg = aggregate([{"ts": t, "server": "router:x:1", "event": "router_session"}
                     for t in (1, 2, 3)])
    assert agg["router_sessions"] == [{"server": "router:x:1", "sessions": 3}]



def test_a_lazy_router_called_only_through_a_contested_peer_is_billed_not_free(tmp_path):
    """Round 6: `labels` omits a contested label with rows, so a router whose only calls
    went through it (all rows its own stamp, `gh` idle, no attach in the window) read
    "cost nothing at all" — a fabricated zero. Its own stamped rows prove it was called."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    recs = [{"ts": 100, "server": label, "event": "router_session"}] + \
        [_rec(200 + i, server="kb", stamp=label) for i in range(10)]
    liab = primer_liability(_dup_rows(tmp_path, pf), aggregate(recs))
    router = next(s for s in liab["servers"] if s["server"] == "terse")
    assert router["cadence"] == "once/session"
    assert "terse" not in liab["free"] and liab["session_once_tokens"] > 0


def test_an_untokenized_attach_under_the_routers_label_is_never_free(tmp_path):
    """An attach row is proof of payment even without tiktoken — never "never attached"."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    recs = [{"ts": 100, "server": label, "event": "router_session"},
            {"ts": 101, "server": label, "event": "primer", "cadence": "once/session",
             "attached": True, "bytes": 900, "tokens": None}]
    liab = primer_liability([_router_row(tmp_path, pf)], aggregate(recs))
    assert liab["servers"][0]["cadence"] == "once/session"
    assert "terse" not in liab["free"]


def test_a_passthrough_only_uncontested_peer_cannot_zero_the_contested_evidence(tmp_path):
    """`gh` ran but encoded nothing (`encoded == 0`): kept as-is, that 0 read the router
    `free` beside 10 of its own compressed rows under the contested `kb`."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    recs = [{"ts": 100, "server": label, "event": "router_session"},
            {**_rec(150, stamp=label), "decision": "passthrough", "out_chars": 400,
             "out_tokens": 100}] + \
        [_rec(200 + i, server="kb", stamp=label) for i in range(10)]
    liab = primer_liability(_dup_rows(tmp_path, pf), aggregate(recs))
    router = next(s for s in liab["servers"] if s["server"] == "terse")
    assert router["cadence"] == "once/session" and "terse" not in liab["free"]


def test_another_routers_stamps_under_a_shared_contested_peer_do_not_bill_this_one(
        tmp_path):
    """Round 7 (M3): `own_contested` must count only THIS router's stamps. `ra` and `rb`
    both front `kb`; every `kb` row is `rb`'s. Idle `ra` must not be billed for them."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pa, pb = _pfile(tmp_path / "a"), _pfile(tmp_path / "b")
    la, lb = router_ledger_label(pa), router_ledger_label(pb)
    rows = [{**_router_row(tmp_path, pa, wraps="gh, kb"), "server": "ra"},
            {**_router_row(tmp_path, pb, wraps="kb"), "server": "rb"}]
    recs = [{"ts": 100, "server": la, "event": "router_session"},
            {"ts": 100, "server": lb, "event": "router_session"}] + \
        [_rec(200 + i, server="kb", stamp=lb) for i in range(10)]
    by = {s["server"]: s for s in primer_liability(rows, aggregate(recs))["servers"]}
    assert by["ra"]["cadence"] != "once/session"


# --- #452 ---------------------------------------------------------------------------

def _two_routers(tmp_path, a_wraps):
    from terse.stats import router_ledger_label
    (tmp_path / "a").mkdir(parents=True)
    (tmp_path / "b").mkdir(parents=True)
    pa, pb = _pfile(tmp_path / "a"), _pfile(tmp_path / "b")
    rows = [{**_router_row(tmp_path, pa, wraps=a_wraps), "server": "ra"},
            {**_router_row(tmp_path, pb, wraps="kb"), "server": "rb"}]
    lb = router_ledger_label(pb)
    recs = [{"ts": 100, "server": router_ledger_label(pa), "event": "router_session"},
            {"ts": 100, "server": lb, "event": "router_session"}] + \
        [_rec(200 + i, server="kb", stamp=lb) for i in range(10)]
    return rows, recs


def test_a_router_whose_claimed_rows_are_all_another_routers_reads_free_either_way(
        tmp_path):
    """#452 item 2: `ra` wrote nothing (every `kb` row is `rb`'s). It read `1x?` when it
    fronted only `kb` and `free` when it also fronted an idle `gh` — one fact, two answers."""
    from terse.stats import aggregate, primer_liability
    for wraps in ("kb", "gh, kb"):
        rows, recs = _two_routers(tmp_path / wraps.replace(", ", "_"), wraps)
        liab = primer_liability(rows, aggregate(recs))
        ra = next(s for s in liab["servers"] if s["server"] == "ra")
        assert ra["cadence"] == "once/session (unpaid)", wraps
        assert "ra" in liab["free"] and "ra" not in liab["uncertain"], wraps


def test_a_lazy_router_whose_peers_all_ran_passthrough_is_free(tmp_path):
    """#452 item 4: every claimed label has rows, all passthrough — the primer never had a
    marker to ride. `encoded == 0` must reach `_cadence` (mutants that forced it to None
    billed this router, and no test could see them)."""
    from terse.stats import aggregate, primer_liability, router_ledger_label
    pf = _pfile(tmp_path)
    label = router_ledger_label(pf)
    passthrough = {"decision": "passthrough", "out_chars": 400, "out_tokens": 100}
    recs = [{"ts": 100, "server": label, "event": "router_session"},
            {**_rec(200, stamp=label), **passthrough},
            {**_rec(201, server="kb", stamp=label), **passthrough}]
    liab = primer_liability([_router_row(tmp_path, pf)], aggregate(recs))
    assert liab["servers"][0]["cadence"] == "once/session (unpaid)"
    assert "terse" in liab["free"]


def test_a_standalone_untokenized_attach_is_never_listed_free(tmp_path):
    """#452 item 3: the round-6 fix was gated to lazy routers; a standalone with an attach
    row recorded without tiktoken and `encoded == 0` was listed free beside that row."""
    from terse.stats import aggregate, primer_liability
    row = {"scope": "user", "server": "kb", "state": "wrapped", "wraps": "kb-server",
           "policy": _pol(tmp_path)}
    recs = [{**_rec(200, server="kb-server"), "decision": "passthrough", "out_chars": 400,
             "out_tokens": 100},
            {"ts": 201, "server": "kb-server", "event": "primer", "cadence": "once/session",
             "attached": True, "bytes": 900, "tokens": None}]
    liab = primer_liability([row], aggregate(recs))
    assert liab["servers"][0]["cadence"] == "once/session"
    assert "kb" not in liab["free"]
