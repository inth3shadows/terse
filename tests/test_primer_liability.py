"""#168: the primer is charged per wrapped server per TURN and the ledger never sees it.

The ledger charges terse for the payloads it compresses and never for the context it adds,
so `terse stats` could report a win in a session that was a net loss — measured from outside
terse as a 14.0% win at one wrapped server and a 2.1% LOSS at three.

What is deliberately NOT built here is a per-turn charge in the ledger: `turns` is not
observable from a stdio proxy, and inventing one would be the #144/#186/#188 defect family
again — a number describing something the code never measured. These tests pin the
break-even framing that replaces it, and the two ways sizing it can quietly lie: counting
from the ledger (which cannot see the install that is pure cost) and mistaking "unknown" for
"never called".
"""
from __future__ import annotations

import json

from terse.stats import build_primer_section, build_stats_report, primer_liability


def _policy(tmp_path, name="p.json", tool="kb.*", tiers=("minify", "tabularize")):
    path = tmp_path / name
    path.write_text(json.dumps({"version": 1,
                                "policies": [{"match": {"tool": tool},
                                              "tiers": list(tiers)}]}), encoding="utf-8")
    return str(path)


def _scan(server, state, wraps, policy, scope="user"):
    return {"scope": scope, "server": server, "state": state, "wraps": wraps,
            "policy": policy}


def _agg(*rows):
    """Minimal `aggregate()`-shaped result: (ledger_label, blocks, raw, out) tuples."""
    tools = [{"server": s, "tool": "t", "blocks": b, "raw_tokens": r, "out_tokens": o,
              "raw_chars": r, "out_chars": o, "diffs": 0} for s, b, r, o in rows]
    return {"total": {"blocks": sum(t["blocks"] for t in tools),
                      "raw_tokens": sum(t["raw_tokens"] for t in tools),
                      "out_tokens": sum(t["out_tokens"] for t in tools),
                      "raw_chars": 0, "out_chars": 0, "untokenized": 0},
            "decisions": {}, "diff_reasons": {}, "tools": tools}


def test_liability_is_sized_from_the_INSTALL_not_from_the_ledger(tmp_path):
    """The whole point of scanning the config: a wrapped server nobody called still ships
    its primer every turn and contributes zero ledger rows. Sizing this from the ledger
    would hide exactly the worst case — the install that is pure cost."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb-server --stdio", pol)],
                            _agg())          # empty ledger
    assert liab["per_turn_tokens"] > 0
    assert liab["idle"] == ["kb"]            # pays every turn, banked nothing
    assert liab["turns_covered"] == 0.0


def test_folded_peers_do_not_double_charge_behind_their_router(tmp_path):
    """A folded peer is stashed BEHIND a router: the router pays one union primer for the
    fleet and the peers pay nothing. Counting them would charge the same primer four times
    for a three-peer install."""
    pol = _policy(tmp_path)
    rows = [_scan("terse", "router", "codegraph, kb, runecho", pol),
            *[_scan(p, "folded", "", pol) for p in ("codegraph", "kb", "runecho")]]
    liab = primer_liability(rows, _agg())
    assert [s["server"] for s in liab["servers"]] == ["terse"]


def test_a_router_is_sized_by_the_union_over_PEER_names_not_its_own(tmp_path):
    """Each peer's policy is gated against its OWN name. Gating the union on the router's
    name tests a rule like `kb.*` against "terse", which matches nothing — so the default
    policy applies to every unmatched tool and the primer is sized LARGER than the one the
    router actually ships."""
    pol = _policy(tmp_path)
    router = primer_liability([_scan("terse", "router", "kb", pol)], _agg())
    direct = primer_liability([_scan("kb", "wrapped", "kb-server", pol)], _agg())
    assert router["per_turn_tokens"] == direct["per_turn_tokens"]
    # Pin the failure mode itself: sizing it as `build_primer(pol, "terse")` is bigger.
    from terse.policy import load_policy
    from terse.proxy import build_primer
    from terse.tokenize import count_cl100k
    assert count_cl100k(build_primer(load_policy(pol), "terse")) > router["per_turn_tokens"]


def test_a_busy_router_is_not_reported_as_never_called(tmp_path):
    """`wraps` means two different things by state — the downstream COMMAND for a wrapped
    entry, the comma-joined peer NAMES for a router. Reading a router's peer list as a
    command is how a router fronting three busy peers reported "pure cost, never called"
    against a ledger full of its own traffic."""
    pol = _policy(tmp_path)
    liab = primer_liability(
        [_scan("terse", "router", "codegraph, kb, runecho", pol)],
        _agg(("kb", 5, 1000, 400), ("codegraph", 2, 500, 500)))
    row = liab["servers"][0]
    assert row["ledger_labels"] == ["codegraph", "kb", "runecho"]
    assert row["blocks"] == 7          # summed across the peers that DID work
    assert liab["idle"] == []


def test_a_wrapped_entrys_ledger_label_comes_from_its_downstream_command(tmp_path):
    """The ledger keys on `server_label` of the wrapped command, not the MCP entry name."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("my-entry", "wrapped", "/opt/bin/kb-server --stdio", pol)],
                            _agg(("kb-server", 3, 900, 300)))
    assert liab["servers"][0]["ledger_labels"] == ["kb-server"]
    assert liab["servers"][0]["blocks"] == 3


def test_an_unknown_label_is_not_an_accusation_of_being_idle(tmp_path):
    """"Unknown" and "never called" are different claims, and only the second one accuses
    an install of being pure cost. A row with nothing to derive a label from reports
    `blocks: None` and stays out of `idle`."""
    liab = primer_liability([_scan("x", "wrapped", "", _policy(tmp_path))], _agg())
    assert liab["servers"][0]["blocks"] is None
    assert liab["idle"] == []


def test_an_unreadable_policy_is_excluded_and_the_total_labelled_a_lower_bound(tmp_path):
    """Substituting the built-in default would OVERSTATE (the default emits every form).
    Leave it out, count it, and say the number is a floor."""
    good = _policy(tmp_path)
    liab = primer_liability([_scan("a", "wrapped", "a", good),
                             _scan("b", "wrapped", "b", str(tmp_path / "nope.json"))],
                            _agg())
    assert liab["unresolved"] == 1
    assert liab["servers"][1]["primer_tokens"] is None
    assert liab["per_turn_tokens"] == liab["servers"][0]["primer_tokens"]
    assert "lower bound" in "\n".join(build_primer_section(liab))


def test_a_server_that_can_emit_no_compressed_form_pays_nothing(tmp_path):
    """`build_primer` returns "" for a policy that can emit no compressed form — a
    default-deny server explains nothing because it produces nothing. Sizing every server
    at one shared constant would invent a cost it does not pay."""
    deny = _policy(tmp_path, name="deny.json", tool="*", tiers=())
    liab = primer_liability([_scan("secret-broker", "wrapped", "sb", deny)], _agg())
    assert liab["servers"][0]["primer_tokens"] == 0
    assert liab["per_turn_tokens"] == 0
    # Nothing to break even on — not "infinite turns", which would read as a measurement.
    assert liab["turns_covered"] is None
    assert liab["idle"] == []


def test_duplicate_names_across_scopes_are_one_server(tmp_path):
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol, scope="project"),
                             _scan("kb", "wrapped", "kb", pol, scope="user")], _agg())
    assert len(liab["servers"]) == 1 and liab["servers"][0]["scope"] == "project"


def test_the_report_calls_out_a_net_negative_window(tmp_path):
    """The line the issue exists for: savings that do not cover even one turn of primer."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 1, 110, 100)))    # 10 tok saved, ~200 tok/turn
    assert liab["turns_covered"] < 1
    text = "\n".join(build_primer_section(liab))
    assert "NET NEGATIVE" in text


def test_the_liability_still_prints_when_the_ledger_is_empty(tmp_path):
    """An install with wrapped servers and NO recorded results is the purest net-negative
    case there is; suppressing the line here would hide it behind "nothing to report"."""
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))], _agg())
    out = build_stats_report(_agg(), log_path="/tmp/x.jsonl", liability=liab)
    assert "no results recorded" in out
    assert "primer liability" in out


def test_the_liability_is_absent_rather_than_zero_when_nothing_is_wrapped():
    assert build_primer_section(primer_liability([], _agg())) == []


# --- per-server break-even (#175) -----------------------------------------------------
#
# #175 stated the rule ("wrap a server when its typical payload saves more than
# `primer x turns-per-call`") and then computed the table proving it BY HAND. These pin the
# same arithmetic inside `terse stats`, and — more importantly — the four cases where the
# honest answer is not a number.


def test_break_even_divides_this_servers_savings_by_its_own_primer(tmp_path):
    """The headline number: 600 tok banked over 10 blocks is 60/call, so a primer of P
    tokens is repaid by P/60 calls in every turn it is charged."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 10, 1_000, 400)))
    srv = liab["servers"][0]
    assert srv["saved_per_call"] == 60.0
    assert srv["calls_to_break_even"] == srv["primer_tokens"] / 60.0
    assert "calls/turn to break even" in "\n".join(build_primer_section(liab))


def test_a_server_that_never_breaks_even_says_so_instead_of_printing_a_huge_number(tmp_path):
    """A non-positive rate does not break even at ANY call volume. Rendering that as
    `999,999.00 calls/turn` invites an operator to read it as merely expensive; it is the
    one verdict in this table that should stop them, so it is a word."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 4, 100, 100)))    # codec banked nothing
    srv = liab["servers"][0]
    assert srv["saved_per_call"] == 0.0
    assert srv["calls_to_break_even"] is None
    assert "never" in "\n".join(build_primer_section(liab))


def test_an_untokenized_ledger_reports_no_token_data_not_a_zero_rate(tmp_path):
    """Rows recorded without tiktoken carry char totals only. Savings in TOKENS are then
    unknown, not zero — and dividing a cl100k primer by a char-derived rate would silently
    mix units. The distinction matters: `0` accuses the server of being incompressible."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 7, 0, 0)))        # blocks, but no token counts
    srv = liab["servers"][0]
    assert srv["blocks"] == 7                              # it was called...
    assert srv["saved_per_call"] is None                   # ...but we cannot rate it
    assert srv["calls_to_break_even"] is None
    assert "no token data" in "\n".join(build_primer_section(liab))


def test_a_zero_primer_server_breaks_even_at_no_calls_at_all(tmp_path):
    """A default-deny server ships no primer, so there is nothing to earn back — distinct
    from "never", which is the opposite verdict."""
    deny = _policy(tmp_path, name="deny2.json", tool="*", tiers=())
    liab = primer_liability([_scan("sb", "wrapped", "sb", deny)], _agg(("sb", 2, 100, 90)))
    srv = liab["servers"][0]
    assert srv["primer_tokens"] == 0
    assert srv["calls_to_break_even"] == 0.0
    assert srv["break_even_verdict"] == "no primer"
    assert "no primer" in "\n".join(build_primer_section(liab))


def test_a_routers_rate_pools_every_peer_it_fronts(tmp_path):
    """A router pays ONE union primer for the fleet, so its break-even is against the
    pooled savings of all its peers — charging each peer separately would be the
    double-charge `_PAYS_PRIMER` already refuses upstream."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb, runecho", pol)],
                            _agg(("kb", 2, 500, 300), ("runecho", 2, 500, 100)))
    srv = liab["servers"][0]
    assert srv["blocks"] == 4
    assert srv["saved_per_call"] == 150.0          # (200 + 400) / 4


def test_the_table_is_suppressed_when_no_server_has_a_rate(tmp_path):
    """An install with nothing in the ledger renders four dashes and no information; the
    `idle` line above already made that point."""
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))], _agg())
    text = "\n".join(build_primer_section(liab))
    assert "primer liability" in text
    assert "calls/turn to break even" not in text


def test_an_unreadable_policy_is_primer_unknown_not_never(tmp_path):
    """Review of #197: `never` and "we could not read the policy" both produced
    `calls_to_break_even is None`, and the renderer read that single sentinel as the first.
    A server saving 600 tok/call was condemned as `never` — the one verdict here meant to
    stop an operator — on evidence about a missing FILE."""
    liab = primer_liability([_scan("b", "wrapped", "b", str(tmp_path / "gone.json"))],
                            _agg(("b", 5, 5_000, 2_000)))
    srv = liab["servers"][0]
    assert srv["primer_tokens"] is None                 # policy unreadable
    assert srv["saved_per_call"] == 600.0               # ...but the rate is real
    assert srv["break_even_verdict"] == "primer unknown"
    text = "\n".join(build_primer_section(liab))
    assert "primer unknown" in text and "never" not in text


def test_the_rate_divides_by_TOKENIZED_calls_not_by_every_block(tmp_path):
    """Review of #197: `aggregate` counts every record in `blocks` but only tokenized ones
    in the token sums, so a ledger spanning one offline session (`count_cl100k` -> None) and
    later online ones divided a partial numerator by a full denominator. The error is always
    pessimistic, i.e. it argues for unwrapping a server that is paying for itself."""
    from terse.stats import aggregate

    recs = [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4}
            for _ in range(90)]                                   # recorded without tiktoken
    recs += [{"server": "kb", "tool": "t", "raw_chars": 10, "out_chars": 4,
              "raw_tokens": 1_000, "out_tokens": 400} for _ in range(10)]
    agg = aggregate(recs)
    assert agg["tools"][0]["blocks"] == 100 and agg["tools"][0]["tokenized"] == 10

    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))], agg)
    srv = liab["servers"][0]
    assert srv["blocks"] == 100 and srv["tokenized_blocks"] == 10
    assert srv["saved_per_call"] == 600.0            # 6,000 tok over the 10 MEASURED calls
    assert srv["calls_to_break_even"] == srv["primer_tokens"] / 600.0
    # And the contaminated denominator is visible where it changed the number.
    assert "10/100" in "\n".join(build_primer_section(liab))


def test_an_unrecoverable_ledger_label_is_not_the_tiktoken_accusation(tmp_path):
    """Review of #197: `blocks is None` (no label could be derived) and `blocks == 0` (never
    called) both rendered `no token data`, which blames the tokenizer for a missing label."""
    liab = primer_liability([_scan("x", "wrapped", "", _policy(tmp_path))],
                            _agg(("kb", 4, 400, 100)))
    srv = liab["servers"][0]
    assert srv["blocks"] is None
    assert srv["break_even_verdict"] == "no ledger label"


def test_a_losing_server_sorts_below_one_that_merely_breaks_even(tmp_path):
    """Review of #197: the `or -1` sort key coerced a 0.0 rate (falsy) to -1 while leaving
    -0.5 (truthy) alone, so the server actively EXPANDING payloads printed above the one
    that merely broke even. The gap only opens for losses inside (-1, 0), which is why this
    fixture uses -0.5 rather than a large loss."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("losing", "wrapped", "losing", pol),
                             _scan("flat", "wrapped", "flat", pol)],
                            _agg(("losing", 4, 100, 102), ("flat", 4, 100, 100)))
    rows = [ln for ln in build_primer_section(liab) if "losing" in ln or "flat" in ln]
    assert len(rows) == 2 and "flat" in rows[0] and "losing" in rows[1]


def test_a_sub_unit_rate_does_not_print_as_zero(tmp_path):
    """Review of #197: `:,.0f` rendered every rate in (-1, 1) as `0`/`-0`, which sat beside
    a finite break-even and read as a contradiction."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 10, 1_004, 1_000)))     # 0.4 tok/call
    assert liab["servers"][0]["saved_per_call"] == 0.4
    assert "0.4" in "\n".join(build_primer_section(liab))
