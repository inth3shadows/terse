"""#168: the primer is a real cost the ledger never sees, and `terse stats` has to say so.

The ledger charges terse for the payloads it compresses and never for the context it adds,
so `terse stats` could report a win in a session that was a net loss — measured from outside
terse as a 14.0% win at one wrapped server and a 2.1% LOSS at three.

There are TWO cadences and they are never summed (#211 follow-up): a multiproxy router still
primes eagerly at `initialize`, which the client re-reads every turn, while a standalone
`terse proxy` attaches its primer once, to the first compressible result, and not at all if
that result never comes. Charging the second as if it were the first is what these tests
mostly pin now — it inflated the headline by a whole session's worth of turns and told the
operator that the servers #211 made FREE were "pure cost".

What is deliberately NOT built here is a per-turn charge in the ledger: `turns` is not
observable from a stdio proxy, and inventing one would be the #144/#186/#188 defect family
again — a number describing something the code never measured. These tests pin the
break-even framing that replaces it, and the ways sizing it can quietly lie: counting from
the ledger (which cannot see the install that is pure cost), mistaking "unknown" for "never
called", and now mistaking a one-time charge for a recurring one.
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
    """The whole point of scanning the config: a ROUTER nobody called still ships its union
    primer every turn and contributes zero ledger rows. Sizing this from the ledger would
    hide exactly the worst case — the install that is pure cost.

    Uses a router, not a `wrapped` entry, because after #211 only a router still pays for
    being installed; the standalone case is the test below."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol)],
                            _agg())          # empty ledger
    assert liab["per_turn_tokens"] > 0
    assert liab["idle"] == ["terse"]         # pays every turn, banked nothing
    assert liab["turns_covered"] == 0.0


def test_an_uncalled_standalone_entry_costs_nothing_rather_than_a_turn_of_primer(tmp_path):
    """The #211 follow-up. A lazily-primed `terse proxy` attaches its primer to the first
    compressible result, so an entry that never handles one pays NOTHING — not a primer
    every turn.

    The old model billed it into a `tok/turn` headline and then listed it under "pure cost
    until they handle a compressible result", which is precisely inverted: #211 is what made
    these free. An operator acting on that line would unwrap the servers costing them least."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb-server --stdio", pol)], _agg())
    assert liab["servers"][0]["primer_tokens"] > 0     # it still HAS a primer...
    assert liab["per_turn_tokens"] == 0                # ...and is charged for none of it
    assert liab["session_once_tokens"] == 0
    assert liab["idle"] == []                          # NOT "pure cost"
    assert liab["free"] == ["kb"]
    assert "costing nothing at all" in "\n".join(build_primer_section(liab))


def test_a_called_standalone_entry_is_billed_once_per_session_not_per_turn(tmp_path):
    """The other half: once it handles a compressible result the primer IS paid — once. It
    belongs to the session bucket, never to the per-turn one, and the two are not summed
    because `tok/turn` and `tok/session` are different units."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb-server", pol)],
                            _agg(("kb-server", 4, 4_000, 1_000)))
    srv = liab["servers"][0]
    assert srv["cadence"] == "once/session"
    assert liab["session_once_tokens"] == srv["primer_tokens"] > 0
    assert liab["per_turn_tokens"] == 0
    assert liab["free"] == [] and liab["idle"] == []
    text = "\n".join(build_primer_section(liab))
    assert "tok/session" in text and "once/session" in text


def test_the_two_cadences_are_reported_separately_and_never_added(tmp_path):
    """A mixed install is where summing them did the visible damage: the router's genuinely
    recurring cost and the standalone's one-time cost landed in one `tok/turn` total that was
    true of neither."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol),
                             _scan("rc", "wrapped", "runecho-mcp", pol)],
                            _agg(("kb", 2, 2_000, 500), ("runecho-mcp", 2, 2_000, 500)))
    by_name = {s["server"]: s for s in liab["servers"]}
    assert by_name["terse"]["cadence"] == "per-turn"
    assert by_name["rc"]["cadence"] == "once/session"
    assert liab["per_turn_tokens"] == by_name["terse"]["primer_tokens"]
    assert liab["session_once_tokens"] == by_name["rc"]["primer_tokens"]
    # The old bug in one line: the headline is no longer the sum of the two.
    assert liab["per_turn_tokens"] != (by_name["terse"]["primer_tokens"]
                                       + by_name["rc"]["primer_tokens"])


def test_an_unlabelled_standalone_entry_is_uncertain_not_free(tmp_path):
    """`blocks is None` means no ledger label was recoverable, so whether the lazy attach ever
    fired is unknown. Collapsing that to "never called" would move it into `free` and
    under-report the install — the same None-vs-0 distinction `_break_even` already keeps."""
    liab = primer_liability([_scan("x", "wrapped", "", _policy(tmp_path))], _agg())
    assert liab["servers"][0]["cadence"] == "once/session (?)"
    assert liab["free"] == [] and liab["uncertain"] == ["x"]
    assert liab["session_once_tokens"] == 0       # not billed either — it is unknown
    assert "unknown whether the lazy primer ever attached" in "\n".join(
        build_primer_section(liab))


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
    # Compared per-server, not via the totals: those now live in different cadence buckets
    # (#211 follow-up), and this test is about SIZING, not about who is billed.
    assert router["servers"][0]["primer_tokens"] == direct["servers"][0]["primer_tokens"]
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
    # Both entries are CALLED, so the exclusion shows up in a bucket that is actually billed
    # — against an empty ledger every standalone entry is free and the total would be 0
    # whether or not the unreadable policy was excluded, proving nothing.
    liab = primer_liability([_scan("a", "wrapped", "a", good),
                             _scan("b", "wrapped", "b", str(tmp_path / "nope.json"))],
                            _agg(("a", 2, 900, 300), ("b", 2, 900, 300)))
    assert liab["unresolved"] == 1
    assert liab["servers"][1]["primer_tokens"] is None
    assert liab["session_once_tokens"] == liab["servers"][0]["primer_tokens"]
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
    liab = primer_liability([_scan("terse", "router", "kb", pol)],
                            _agg(("kb", 1, 110, 100)))    # 10 tok saved, ~200 tok/turn
    assert liab["turns_covered"] < 1
    text = "\n".join(build_primer_section(liab))
    assert "NET NEGATIVE" in text


def test_a_standalone_only_install_still_gets_a_net_negative_verdict(tmp_path):
    """With no router there is no recurring charge, so `turns_covered` is None and the old
    renderer printed no bottom line at all — silence on exactly the install shape #211 made
    the common one. The verdict is per SESSION there, against the one-time charge."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 1, 110, 100)))    # 10 tok saved vs a ~200 tok primer
    assert liab["turns_covered"] is None
    assert liab["session_covered"] < 1
    text = "\n".join(build_primer_section(liab))
    # The shortfall in tokens, not a ratio: `~0x over` would round to a zero that reads as a
    # measurement (the print-as-zero hole review of #197 closed for `saved/block`).
    assert "NET NEGATIVE" in text
    assert f"does not cover the {liab['session_once_tokens']:,} tok one-time charge" in text
    assert "0x over" not in text


def test_the_one_time_charge_is_settled_before_savings_buy_turns(tmp_path):
    """A mixed install pays both out of the same savings pot, so the one-time charge comes
    off the top and only the remainder buys turns of the recurring one. Reporting the full
    savings against the router alone would overstate how far the window goes."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol),
                             _scan("rc", "wrapped", "runecho-mcp", pol)],
                            _agg(("kb", 2, 10_000, 0), ("runecho-mcp", 2, 10_000, 0)))
    expected = ((liab["saved_tokens"] - liab["session_once_tokens"])
                / liab["per_turn_tokens"])
    assert liab["turns_covered"] == expected
    assert liab["turns_covered"] < liab["saved_tokens"] / liab["per_turn_tokens"]


def test_a_router_only_install_is_not_told_about_a_charge_it_does_not_pay(tmp_path):
    """`once == 0` for an install with no lazy entry, so every sentence about a one-time
    charge is about a charge that does not exist. The stanza and the "settles the one-time
    charge" clause are both gated on there BEING one — the same reason the report says
    `no primer` rather than `0` elsewhere: absent and zero are different claims."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("terse", "router", "kb", pol)],
                            _agg(("kb", 20, 40_000, 10_000)))
    text = "\n".join(build_primer_section(liab))
    assert "tok/turn" in text
    assert "one-time" not in text and "tok/session" not in text
    assert "settles the one-time charge" not in text
    assert "not summed" not in text


def test_a_liability_blob_from_a_pre_cadence_terse_still_renders(tmp_path):
    """`build_primer_section` is public and `--json` emits this dict, so a blob round-tripped
    through a terse that predates the cadence split carries no `cadence`/`session_once_tokens`
    at all. It measured the eager model, so it renders as the recurring half — degrading, not
    raising, and never guessing a cadence it did not record."""
    liab = primer_liability([_scan("terse", "router", "kb", _policy(tmp_path))],
                            _agg(("kb", 5, 5_000, 1_000)))
    legacy = {k: v for k, v in liab.items()
              if k not in ("session_once_tokens", "session_covered", "free", "uncertain")}
    legacy["servers"] = [{k: v for k, v in s.items() if k != "cadence"}
                         for s in liab["servers"]]
    text = "\n".join(build_primer_section(legacy))
    assert "tok/turn" in text and "tok/session" not in text
    assert f"{'–':>21}" in text          # cadence cell blank rather than guessed


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
    assert srv["saved_per_block"] == 60.0
    assert srv["blocks_to_break_even"] == srv["primer_tokens"] / 60.0
    assert "blocks to break even" in "\n".join(build_primer_section(liab))


def test_a_server_that_never_breaks_even_says_so_instead_of_printing_a_huge_number(tmp_path):
    """A non-positive rate does not break even at ANY call volume. Rendering that as
    `999,999.00 calls/turn` invites an operator to read it as merely expensive; it is the
    one verdict in this table that should stop them, so it is a word."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 4, 100, 100)))    # codec banked nothing
    srv = liab["servers"][0]
    assert srv["saved_per_block"] == 0.0
    assert srv["blocks_to_break_even"] is None
    assert srv["break_even_verdict"] == "never"
    # not merely `"never" in text` — the `never called` verdict also matches that, and so
    # does the surrounding prose. Match the right-aligned CELL (width 21).
    assert f"{'never':>21}" in "\n".join(build_primer_section(liab))


def test_an_untokenized_ledger_reports_no_token_data_not_a_zero_rate(tmp_path):
    """Rows recorded without tiktoken carry char totals only. Savings in TOKENS are then
    unknown, not zero — and dividing a cl100k primer by a char-derived rate would silently
    mix units. The distinction matters: `0` accuses the server of being incompressible."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 7, 0, 0)))        # blocks, but no token counts
    srv = liab["servers"][0]
    assert srv["blocks"] == 7                              # it was called...
    assert srv["saved_per_block"] is None                   # ...but we cannot rate it
    assert srv["blocks_to_break_even"] is None
    assert "no token data" in "\n".join(build_primer_section(liab))


def test_a_zero_primer_server_breaks_even_at_no_calls_at_all(tmp_path):
    """A default-deny server ships no primer, so there is nothing to earn back — distinct
    from "never", which is the opposite verdict."""
    deny = _policy(tmp_path, name="deny2.json", tool="*", tiers=())
    liab = primer_liability([_scan("sb", "wrapped", "sb", deny)], _agg(("sb", 2, 100, 90)))
    srv = liab["servers"][0]
    assert srv["primer_tokens"] == 0
    assert srv["blocks_to_break_even"] == 0.0
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
    assert srv["saved_per_block"] == 150.0          # (200 + 400) / 4


def test_the_table_is_suppressed_when_no_server_has_a_rate(tmp_path):
    """An install with nothing in the ledger renders four dashes and no information; the
    `idle` line above already made that point."""
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))], _agg())
    text = "\n".join(build_primer_section(liab))
    assert "primer liability" in text
    assert "blocks to break even" not in text


def test_an_unreadable_policy_is_primer_unknown_not_never(tmp_path):
    """Review of #197: `never` and "we could not read the policy" both produced
    `calls_to_break_even is None`, and the renderer read that single sentinel as the first.
    A server saving 600 tok/call was condemned as `never` — the one verdict here meant to
    stop an operator — on evidence about a missing FILE."""
    liab = primer_liability([_scan("b", "wrapped", "b", str(tmp_path / "gone.json"))],
                            _agg(("b", 5, 5_000, 2_000)))
    srv = liab["servers"][0]
    assert srv["primer_tokens"] is None                 # policy unreadable
    assert srv["saved_per_block"] == 600.0               # ...but the rate is real
    assert srv["break_even_verdict"] == "primer unknown"
    text = "\n".join(build_primer_section(liab))
    # The `never` that must be absent is the VERDICT CELL, not the word anywhere in the
    # section — the surrounding prose legitimately uses it.
    assert "primer unknown" in text and f"{'never':>21}" not in text


def test_the_rate_divides_by_TOKENIZED_blocks_not_by_every_block(tmp_path):
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
    assert srv["saved_per_block"] == 600.0            # 6,000 tok over the 10 MEASURED calls
    assert srv["blocks_to_break_even"] == srv["primer_tokens"] / 600.0
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
    assert liab["servers"][0]["saved_per_block"] == 0.4
    assert "0.4" in "\n".join(build_primer_section(liab))


def test_the_denominator_column_is_labelled_blocks_because_that_is_what_it_counts(tmp_path):
    """Round-2 review of #197: the header was renamed `blocks` -> `calls` over a counter
    `aggregate` increments once per emitted tool-result BLOCK (>=1 per call, #141). A server
    emitting three blocks per call would have had its break-even overstated 3x with nothing
    in the report disclosing the unit."""
    liab = primer_liability([_scan("kb", "wrapped", "kb", _policy(tmp_path))],
                            _agg(("kb", 30, 10_000, 4_000)))
    text = "\n".join(build_primer_section(liab))
    assert "blocks to break even" in text and "calls/turn" not in text
    assert "saved/block" in text and "saved/call" not in text
    assert "a BLOCK is one emitted tool-result text block" in text


def test_a_rate_too_small_to_render_is_a_bound_not_a_zero(tmp_path):
    """Round-2 review of #197: the first fix moved the print-as-zero hole from (-1, 1) to
    (-0.05, 0.05) rather than closing it, so `0.0` still sat beside a finite break-even."""
    pol = _policy(tmp_path)
    liab = primer_liability([_scan("kb", "wrapped", "kb", pol)],
                            _agg(("kb", 1_000, 100_004, 100_000)))    # 0.004 tok/block
    assert liab["servers"][0]["saved_per_block"] == 0.004
    text = "\n".join(build_primer_section(liab))
    assert "<0.01" in text


def test_an_unrecoverable_label_leaves_the_tokenized_count_unknown_too(tmp_path):
    """Round-2 review of #197: `blocks` got the `if labels else None` guard and `tokenized`
    did not, so `--json` carried `{"blocks": null, "tokenized_blocks": 0}` — a 0 claiming
    "nothing was tokenized" where the truth is that no rows were ever found to ask."""
    liab = primer_liability([_scan("x", "wrapped", "", _policy(tmp_path))],
                            _agg(("kb", 4, 400, 100)))
    srv = liab["servers"][0]
    assert srv["blocks"] is None and srv["tokenized_blocks"] is None
    # And it still renders, rather than being suppressed alongside a genuinely idle install.
    rendered = build_primer_section(primer_liability(
        [_scan("x", "wrapped", "", _policy(tmp_path)),
         _scan("kb", "wrapped", "kb", _policy(tmp_path))], _agg(("kb", 4, 400, 100))))
    assert "no ledger label" in "\n".join(rendered)


def test_a_liability_blob_without_a_verdict_degrades_instead_of_raising(tmp_path):
    """Round-2 review of #197: `build_primer_section` is public and `--json` emits this exact
    dict shape, so a blob round-tripped through an older terse carries no `break_even_verdict`
    and took the numeric branch with a None. A report is never load-bearing — it degrades."""
    from terse.stats import _fmt_break_even

    assert _fmt_break_even({"saved_per_block": 0.0, "blocks_to_break_even": None}) == ("0", "–")
