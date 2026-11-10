"""The risk-ranked review queue (design section 4).

``score = prominence * sum(weight * count)``; unchecked (U) numbers count at most 3 per sentence (a
payload pseudo-sentence at most 1); numbers checked only by ``recheck`` (R) count once per sentence:
their residual risk is a referent or rounding slip, about the same for one number or five. Tiers: ``G`` (any gate), ``A`` (score >= 5; the top 25 by score),
``A'`` (tier-A overflow beyond the cap, sampled), ``B`` (2 <= score < 5), ``C`` (below 2), ``-``
(nothing checkable: no number and no number word; not queued or sampled).
Order: G (failed, verifier-refuted, changed, open finding, scope mismatch), then A by score, then
A', B, C; ties in reading order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

WEIGHTS = {
    "u_atom": 5,                  # unchecked hand-typed number (per atom, at most 3)
    "r_atom": 2,                  # retyped number(s) checked only by recheck (once per sentence)
    "citation_gap": 3,
    "scope_word": 4,              # "only", "all", "zero", ... not proved by a reference
    "formula_eq_unchecked": 4,    # "=" or "reduces to" without an exact check (approx counts too)
    "approx_no_tol": 2,
    "small_n": 2,
    "direct_whole_ambiguity": 4,
    "exception_group": 2,
    "rounding_boundary": 1,
    "other_config": 2,
    "verifier_weakened": 2,
}
PROMINENCE = 1.5
TIER_A = 5.0
TIER_B = 2.0
A_CAP = 25
GATE_ORDER = ["failed", "verifier_refuted", "changed", "open_finding", "scope_mismatch"]


@dataclass
class Item:
    sid: str
    ref: str
    text: str
    badge: str
    score: float
    tier: str
    factors: List[str] = field(default_factory=list)
    gates: List[str] = field(default_factory=list)
    verifier: List[Dict[str, Any]] = field(default_factory=list)
    order: int = 0

    def to_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def score_features(f: Dict[str, Any], verifier_weakened: bool = False):
    """(score, factor list) from a :class:`~explaneat.review.status.Status`'s features."""
    cap = 1 if f.get("payload") else 3
    parts: List[tuple] = []
    if f.get("u_atoms"):
        parts.append((f"unchecked number x{min(f['u_atoms'], cap)}", WEIGHTS["u_atom"] * min(f["u_atoms"], cap)))
    if f.get("r_atoms"):
        parts.append(("number(s) checked only by recheck", WEIGHTS["r_atom"]))
    if f.get("citation_gap"):
        parts.append(("citation gap", WEIGHTS["citation_gap"]))
    if f.get("scope_words"):
        parts.append((f"scope word ({', '.join(sorted(set(w.lower() for w in f['scope_words'])))})",
                      WEIGHTS["scope_word"]))
    if f.get("formula") in ("unchecked", "approx"):
        parts.append(("formula '=' not exact-checked", WEIGHTS["formula_eq_unchecked"]))
    if f.get("approx_no_tol"):
        parts.append(("'approx' formula without a tolerance", WEIGHTS["approx_no_tol"]))
    if f.get("small_n"):
        parts.append((f"small n ({min(f['small_n'])})", WEIGHTS["small_n"]))
    if f.get("direct_whole_ambiguity"):
        parts.append(("direct or whole-model? no scope word", WEIGHTS["direct_whole_ambiguity"]))
    if f.get("exception_group"):
        parts.append(("named exception group", WEIGHTS["exception_group"]))
    if f.get("rounding_boundary"):
        parts.append(("near a rounding boundary", WEIGHTS["rounding_boundary"]))
    if f.get("other_config"):
        parts.append(("AUC/accuracy without split or configuration", WEIGHTS["other_config"]))
    if verifier_weakened:
        parts.append(("verifier: weakened", WEIGHTS["verifier_weakened"]))
    total = float(sum(w for _, w in parts))
    if f.get("prominent") and total:
        total *= PROMINENCE
        parts.append(("prominent (x1.5)", 0))
    return total, [p for p, _ in parts]


def build_queue(sentences: Sequence, statuses: Dict[str, Any], overlay: Optional[Dict[str, List[Dict]]] = None,
                changed: Optional[Dict[str, str]] = None, cap: int = A_CAP) -> List[Item]:
    """Every sentence as a queue item, in queue order. ``overlay``: sid -> attached verifier findings
    (each with ``grade`` and ``stale``); ``changed``: sid -> change state from :mod:`.diff`."""
    overlay = overlay or {}
    changed = changed or {}
    items: List[Item] = []
    for k, s in enumerate(sentences):
        st = statuses[s.sid]
        found = overlay.get(s.sid) or []
        live = [f for f in found if not f.get("stale")]
        weakened = any(f.get("grade") == "weakened" for f in live)
        score, factors = score_features(st.features, weakened)
        gates = []
        if st.badge == "F":
            gates.append("failed")
        if any(f.get("grade") == "refuted" for f in live):
            gates.append("verifier_refuted")
        if changed.get(s.sid) in ("changed", "edited"):
            gates.append("changed")
        if any(r.startswith("open write-time finding") for r in st.reasons):
            gates.append("open_finding")
        if any(r.startswith("scope word differs") for r in st.reasons):
            gates.append("scope_mismatch")
        tier = "G" if gates else ("A" if score >= TIER_A else ("B" if score >= TIER_B else "C"))
        if tier == "C" and st.badge == "n/a" and not score:
            tier = "-"          # nothing checkable (no number, no number word): not queued, not sampled
        items.append(Item(s.sid, s.ref, s.text, st.badge, round(score, 2), tier, factors, gates, found, k))
    a = sorted([i for i in items if i.tier == "A"], key=lambda i: (-i.score, i.order))
    for i in a[cap:]:
        i.tier = "A'"

    def key(i: Item):
        t = {"G": 0, "A": 1, "A'": 2, "B": 3, "C": 4, "-": 5}[i.tier]
        g = min((GATE_ORDER.index(x) for x in i.gates), default=9)
        return (t, g, -i.score if i.tier in ("A", "A'") else 0, i.order)
    return sorted(items, key=key)


def tier_counts(items: Sequence[Item]) -> Dict[str, int]:
    out = {"G": 0, "A": 0, "A'": 0, "B": 0, "C": 0, "-": 0}
    for i in items:
        out[i.tier] += 1
    return out
