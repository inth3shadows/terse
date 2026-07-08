"""Per-tool / per-type policy — the selective shell in front of arbitrary tool output.

The spike showed terse's value is strongly per-tool (0-30%): big on record/symbol-
shaped verbose output (gh APIs, runecho), ~0 on already-projected high-cardinality
data (kb) and single compact objects. So compression must be SELECTIVE, not blanket.

A policy is an ordered list of rules. The first rule whose `match.tool` glob matches
the tool name wins; if none match, `defaults` applies. Each rule names the lossless
TIERS to run (subset of minify/tabularize/dictionary; empty = passthrough) and an
optional per-field map.

Fail-closed (principle #37): an unmatched tool gets the lossless default and NEVER a
lossy op. Lossy field modes are parsed and surfaced but NOT executed in v1 — those
tiers aren't built — so this layer is 100% lossless regardless of policy. `critical`
fields are honored trivially today (lossless preserves everything); they become the
denylist once lossy tiers exist.

Format is JSON (stdlib, zero deps). YAML/TOML can be added behind the same loader.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import lossy as lossy_mod
from . import transforms

VALID_TIERS = ("minify", "tabularize", "dictionary")
LOSSY_MODES = ("truncate", "drop-to-retrieve")  # implemented; summarize is still deferred

# multiproxy's peer-qualifier separator (e.g. "gh__search" for peer "gh"'s "search"
# tool). Defined here, not in multiproxy.py, so Policy.select can recognize and strip
# it: a corpus captured through multiproxy stores each payload under its peer-qualified
# name (to keep same-named tools on different peers from colliding in the corpus), but
# a policy rule is authored against the downstream tool's own bare name — without this,
# every rule silently misses for a multiproxy-captured corpus (fnmatch("gh__search",
# "search") is False), and drop-eval/measure would score it as having nothing to test.
PREFIX_SEP = "__"


@dataclass
class Rule:
    tool_glob: str
    tiers: tuple[str, ...]
    # Per-field map: {path: {"lossy": mode, "max": N} | {"critical": true}}. `critical`
    # is a field flag (never made lossy); see lossy.critical_paths.
    fields: dict[str, dict] = field(default_factory=dict)

    def lossy_fields(self) -> list[str]:
        return [k for k, v in self.fields.items() if v.get("lossy")]


@dataclass
class Policy:
    rules: list[Rule]
    default_tiers: tuple[str, ...] = ("minify", "tabularize", "dictionary")
    # Cross-call diffing is stateful and its model-fluency is unproven, so it is OFF by
    # default: enable per-policy (`"diff": true`) or with `proxy --diff`. The proxy still
    # falls back to the full compressed form whenever a diff doesn't apply or win.
    diff: bool = False
    # Bound dangling-reference drift (#8): force a self-contained full result (a
    # keyframe, like video I-frames) after this many consecutive diffs per tool, so a
    # chained diff can never drift more than K turns from an anchor the model can
    # reconstruct from scratch. 0 disables keyframing (diff whenever it wins).
    diff_keyframe_interval: int = 5
    # Cross-peer session dictionary (#64 Phase 1): a shared legend spanning every peer, so a
    # value defined by one payload is referenced (definition elided) by later ones. Like
    # `diff`, it is a stateful cross-call scheme with an unproven-in-practice desync risk, so
    # OFF by default: enable per-policy (`"session_dict": true`) or with `proxy --session-dict`.
    # Mutually exclusive with `diff` (both are cross-call wires competing for the same slot);
    # the CLI rejects setting both. Reuses `diff_keyframe_interval` as its per-entry re-emit
    # bound. The proxy always falls back to the full compressed form when it doesn't apply.
    session_dict: bool = False

    def select(self, tool: str) -> Rule:
        """First rule whose glob matches the tool name, else the lossless default.

        Tries `tool` as given first, then — if it's a multiproxy peer-qualified name
        (contains PREFIX_SEP) — the bare part after the separator, so a rule authored
        for a downstream tool's own name still matches a multiproxy-captured corpus
        entry for it. A single-proxy `tool` (no PREFIX_SEP) is unaffected."""
        for candidate in self._match_candidates(tool):
            for rule in self.rules:
                if fnmatch.fnmatch(candidate, rule.tool_glob):
                    return rule
        return Rule(tool_glob="*", tiers=self.default_tiers)

    @staticmethod
    def _match_candidates(tool: str) -> list[str]:
        if PREFIX_SEP in tool:
            return [tool, tool.partition(PREFIX_SEP)[2]]
        return [tool]

    def has_drop(self) -> bool:
        """True if any rule marks a field drop-to-retrieve. Gates whether the proxy injects
        the synthetic terse.retrieve tool into tools/list (#10)."""
        return any(isinstance(s, dict) and s.get("lossy") == "drop-to-retrieve"
                   for r in self.rules for s in r.fields.values())


def default_policy() -> Policy:
    """Lossless-everywhere default: full Tier-0/0.5 on every tool, no lossy."""
    return Policy(rules=[], default_tiers=("minify", "tabularize", "dictionary"))


def _coerce_tiers(raw: Any, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{where}: 'tiers' must be a list, got {type(raw).__name__}")
    bad = [t for t in raw if t not in VALID_TIERS]
    if bad:
        raise ValueError(f"{where}: unknown tier(s) {bad}; valid: {list(VALID_TIERS)}")
    return tuple(raw)


def load_policy(path: str | Path) -> Policy:
    """Parse + validate a JSON policy file. Raises ValueError on a malformed policy."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("version") != 1:
        raise ValueError(f"unsupported policy version: {doc.get('version')!r} (expected 1)")
    default_tiers = _coerce_tiers(
        doc.get("defaults", {}).get("tiers", list(VALID_TIERS)), "defaults"
    )
    rules: list[Rule] = []
    for i, r in enumerate(doc.get("policies", [])):
        match = r.get("match", {})
        glob = match.get("tool", "*")
        rules.append(Rule(tool_glob=glob, tiers=_coerce_tiers(r.get("tiers", []), f"policies[{i}]"),
                          fields=r.get("fields", {})))
    return Policy(rules=rules, default_tiers=default_tiers, diff=bool(doc.get("diff", False)),
                  diff_keyframe_interval=int(doc.get("diff_keyframe_interval", 5)),
                  session_dict=bool(doc.get("session_dict", False)))


@dataclass
class Applied:
    text: str
    tool: str
    tiers: tuple[str, ...]
    skipped: bool
    warnings: list[str]


def _lossy_warnings(rule: Rule) -> list[str]:
    """Warn about field lossy requests that won't be executed as asked: deferred modes,
    unknown modes, and the truncate-vs-critical contradiction."""
    critical = lossy_mod.critical_paths(rule)
    out: list[str] = []
    for path, spec in rule.fields.items():
        mode = spec.get("lossy") if isinstance(spec, dict) else None
        if not mode:
            continue
        if mode == "summarize":
            out.append(f"field '{path}': lossy mode '{mode}' not implemented yet (left lossless)")
        elif mode not in LOSSY_MODES:
            out.append(f"field '{path}': unknown lossy mode '{mode}' (ignored)")
        elif path in critical:
            out.append(f"field '{path}': marked '{mode}' AND critical — kept lossless")
    return out


def apply(raw: str, tool: str, policy: Policy,
          drop_sink: Any = None) -> Applied:
    """Compress one raw payload per policy. Lossless by default; a field marked
    `truncate` (and not `critical`) is reduced, gated by the acceptable-loss invariant.
    Non-JSON passes through.

    `drop_sink` — the per-session drop-to-retrieve store, `handle -> value` — exists only
    in the running proxy, per issue #10. When it is None a drop-marked field can't be made
    recoverable, so it is left lossless with a warning instead of silently vanishing.

    Returns the (possibly unchanged) text plus what was applied — so a caller/proxy
    can log why a payload was or wasn't compressed, and whether anything was dropped.
    """
    rule = policy.select(tool)
    warnings = _lossy_warnings(rule)

    if not rule.tiers:
        return Applied(text=raw, tool=tool, tiers=(), skipped=True, warnings=warnings)
    if "minify" not in rule.tiers:
        warnings.append("'minify' implied by serialization; added")

    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return Applied(text=raw, tool=tool, tiers=(), skipped=True,
                       warnings=warnings + ["payload is not JSON; passed through"])

    # Marker-collision guard: a payload that already carries a reserved terse marker key
    # can't be compressed without the consumer mis-reading its own data as a terse
    # envelope (the codec has no escape convention). Pass it through untouched (#6).
    if transforms.has_terse_marker(obj):
        return Applied(text=raw, tool=tool, tiers=(), skipped=True,
                       warnings=warnings + ["payload contains a reserved terse marker key; "
                                            "passed through uncompressed to stay lossless"])

    # Tier-1 lossy (truncate) runs BEFORE the lossless tiers and is fail-closed: any
    # path that doesn't resolve, or a gate failure, keeps the fully-lossless object.
    data = obj
    if lossy_mod._truncate_specs(rule):
        try:
            cand = lossy_mod.apply_lossy(obj, rule)
            if cand is not obj and cand != obj and lossy_mod.acceptable_loss(obj, cand, rule):
                data = cand
                warnings.append("lossy: truncated marked field(s) — output is NOT lossless")
            elif cand != obj:
                warnings.append("lossy step skipped: acceptable-loss gate failed (kept lossless)")
        except lossy_mod.PathError as exc:
            warnings.append(f"lossy step skipped: {exc} (kept lossless)")

    # Tier-1 lossy (drop-to-retrieve, #10): replace a marked field with a handle marker and
    # persist the original to the session store, so the model can fetch it back on demand.
    # Same fail-closed contract as truncate, plus: writes are STAGED and committed to the
    # real store only after the gate passes, so a gate failure leaves no orphan handles.
    if lossy_mod._drop_specs(rule):
        if drop_sink is None:
            warnings.append("lossy: drop-to-retrieve needs the proxy store; left lossless")
        else:
            staging: dict[str, Any] = {}
            try:
                cand = lossy_mod.apply_drops(data, rule, tool, staging.__setitem__)
                if cand != data and lossy_mod.droppable_loss(data, cand, rule, staging.__getitem__):
                    for handle, value in staging.items():
                        drop_sink(handle, value)  # commit only once recoverability is proven
                    data = cand
                    warnings.append("lossy: dropped marked field(s) to retrieve handle(s) — "
                                    "output is NOT lossless")
                elif cand != data:
                    warnings.append("lossy step skipped: droppable-loss gate failed (kept lossless)")
            except lossy_mod.PathError as exc:
                warnings.append(f"lossy step skipped: {exc} (kept lossless)")

    text = transforms.compress_with(
        data,
        tabularize="tabularize" in rule.tiers,
        dictionary="dictionary" in rule.tiers,
    )
    return Applied(text=text, tool=tool, tiers=rule.tiers, skipped=False, warnings=warnings)
