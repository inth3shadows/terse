"""Deterministic verdicts for LIVE drop-to-retrieve rules, from the ledger (#252, phase B).

`terse tune` (corpus-based) decides which fields COULD be dropped. This answers the other
question: for a rule already firing, does the evidence say keep it, tune it, or take it
back? Inputs are ledger rows only — each rule's firings (`drops`, #252 phase A) and the
retrieves billed to it (#251) — so the verdict is reproducible from the file, with no model
in the loop.

The honest limit is printed with every report rather than buried here: a retrieve rate
measures whether the model ASKED for a value, not whether it needed it. A model that
re-reads the source instead of calling `terse.retrieve` leaves a drop with no retrieve, so
a low rate is a best case. That is why nothing below ever recommends widening a rule as
safe — only as a candidate to verify with `terse tune --drop-eval` first.
"""

from __future__ import annotations

from typing import Any

from . import lossy
from .stats import RETRIEVE_EVENT, aggregate, canonical_tool

# Below this many dropped values a rate is noise: 1 retrieve in 10 is "10%" and could as
# easily be 0% or 30% next week. #252's own reopen bar, for the same reason.
MIN_SPANS = 30
# Retract above this: four retrieves in ten means the drop costs a round-trip on most of the
# results that needed the value at all, and keeping it inline is cheaper than the calls.
RETRACT_RATE = 0.40
# At or below this the drop is close to free (on the evidence available — see the module
# docstring). Between the two bands the rule is left alone and watched.
KEEP_RATE = 0.05
# `keep_first`: when at least this share of retrieves fetched blocks earlier than k, keeping
# those k inline would have avoided them. Needs a few indexed retrieves to say anything.
KEEP_FIRST_SHARE = 0.80
MIN_INDEXED_RETRIEVES = 5

THRESHOLDS = {"min_spans": MIN_SPANS, "retract_rate": RETRACT_RATE, "keep_rate": KEEP_RATE,
              "keep_first_share": KEEP_FIRST_SHARE,
              "min_indexed_retrieves": MIN_INDEXED_RETRIEVES}


def _retrieve_indexes(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[int]]:
    """Block indexes of the HIT retrieves billed to each rule (#473). JSON-field drops have
    no index and contribute nothing; a miss fetched nothing and has no rule."""
    out: dict[tuple[str, str, str], list[int]] = {}
    for rec in records:
        if rec.get("event") != RETRIEVE_EVENT or not rec.get("hit"):
            continue
        idx = rec.get("index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            continue
        srv = str(rec.get("server", "unknown"))
        key = (srv, canonical_tool(srv, str(rec.get("tool", "unknown"))),
               str(rec.get("path", "")))
        out.setdefault(key, []).append(idx)
    return out


def _smallest_covering_k(indexes: list[int], share: float) -> int:
    """Smallest k such that at least `share` of `indexes` are < k."""
    ordered = sorted(indexes)
    need = share * len(ordered)
    for k in range(1, ordered[-1] + 2):
        if sum(1 for i in ordered if i < k) >= need:
            return k
    return ordered[-1] + 1


def verdicts(records: list[dict[str, Any]], policy: Any = None) -> list[dict[str, Any]]:
    """One verdict per drop rule seen in `records`.

    `policy` (a loaded `Policy`, optional) supplies each rule's current spec, so a
    `keep_first` suggestion is only made when it would RAISE the value in force and the
    report can name what it changes from. Without it, a verdict is still reached and the
    current value is reported as unknown."""
    rules = aggregate(records)["drop_rules"]
    indexes = _retrieve_indexes(records)
    out = []
    for rule in rules:
        key = (rule["server"], rule["tool"], rule["path"])
        spec = None
        if policy is not None:
            spec = policy.select(rule["tool"], rule["server"]).fields.get(rule["path"])
        current_kf = lossy.keep_first(spec) if spec is not None else None
        row = {**rule, "verdict": "", "change": None, "candidate": None, "reason": "",
               "keep_first": current_kf}
        rate = rule["rate"]
        idx = indexes.get(key, [])
        if rule["spans"] < MIN_SPANS or rate is None:
            row["verdict"] = "insufficient"
            row["reason"] = (f"{rule['spans']} dropped value(s) in the window; "
                             f"{MIN_SPANS} needed before a rate means anything")
        elif rate >= RETRACT_RATE:
            row["verdict"] = "retract"
            row["change"] = {"remove": "drop-to-retrieve"}
            row["reason"] = (f"{rate:.0%} of dropped values were fetched back "
                             f"(>= {RETRACT_RATE:.0%}): the round-trips cost more than "
                             "keeping the value inline")
        elif len(idx) >= MIN_INDEXED_RETRIEVES and lossy.is_text_path(rule["path"]) \
                and (k := _smallest_covering_k(idx, KEEP_FIRST_SHARE)) > (current_kf or 0):
            row["verdict"] = "raise_keep_first"
            row["change"] = {"keep_first": k}
            row["reason"] = (f"{KEEP_FIRST_SHARE:.0%} of {len(idx)} retrieves fetched a block "
                             f"before index {k}; keeping {k} inline would have avoided them")
        elif rate <= KEEP_RATE:
            row["verdict"] = "keep"
            row["reason"] = (f"{rate:.0%} fetched back (<= {KEEP_RATE:.0%}): the drop is close "
                             "to free on this evidence")
            row["candidate"] = ("widen (a lower `min`, or a lower `keep_first`) — only after "
                                "`terse tune --drop-eval` passes: a low rate cannot show "
                                "the model did not need the value")
        else:
            row["verdict"] = "watch"
            row["reason"] = (f"{rate:.0%} fetched back, between keep ({KEEP_RATE:.0%}) and "
                             f"retract ({RETRACT_RATE:.0%}); no change")
        if rule["estimated"]:
            row["reason"] += (f" [{rule['estimated']} of {rule['blocks']} block(s) predate "
                              "drop recording and are counted as one value each]")
        out.append(row)
    return out


def build_report(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ("tune --from-ledger: no drop-to-retrieve rule dropped or was retrieved in "
                "this window — nothing to judge.\n")
    lines = ["drop-rule verdicts from the ledger (deterministic; thresholds below):", ""]
    for r in rows:
        rate = "–" if r["rate"] is None else f"{r['rate']:.0%}"
        lines.append(f"{r['verdict'].upper():<17} {r['server']} {r['tool']} {r['path']}  "
                     f"dropped={r['spans']} retrieved={r['retrieves']} rate={rate}")
        lines.append(f"  why: {r['reason']}")
        if r["change"]:
            lines.append(f"  suggest: {r['change']}"
                         + (f" (keep_first now {r['keep_first']})"
                            if "keep_first" in r["change"] else ""))
        if r["candidate"]:
            lines.append(f"  candidate: {r['candidate']}")
    lines += ["", "thresholds: " + ", ".join(f"{k}={v}" for k, v in THRESHOLDS.items()),
              "nothing is applied: edit the policy by hand. A rate counts retrieves only — a "
              "model that re-reads the source instead shows no retrieve, so a low rate is a "
              "best case."]
    return "\n".join(lines) + "\n"
