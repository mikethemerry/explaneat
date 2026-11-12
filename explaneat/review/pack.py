"""The review data the verification pack prints for one genome (pure data; LaTeX in
:mod:`explaneat.export.verification`): the risk-ranked queue (gates, tier A), the signoff-v1
random sample, verifier findings attached to sentences, what changed since the last verdicts, and
the leaves with their recorded leaf verdicts (understood / partial / not).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .signoff import SAMPLE, sample_order

SECONDS_QUEUE = 45      # per gate or tier-A item (legacy technical text: often a formula or a table row)
SECONDS_SAMPLE = 30     # per sampled item
MINUTES_FIXED = 5       # the recorded-vs-live table, the diff and the sign-off command


def _item(i, r, last) -> Dict[str, Any]:
    v = last.get(i.sid)
    cur = r.sentence(i.sid)
    verdict = None
    if v:
        verdict = v["verdict"] if v["target"].get("content_hash") == cur.content_hash else "stale"
    return {"sid": i.sid, "tier": i.tier, "badge": i.badge, "score": i.score, "ref": i.ref, "text": i.text,
            "why": i.gates + i.factors, "verdict": verdict,
            "verifier": [f for f in i.verifier]}


def review_summary(r) -> Dict[str, Any]:
    from .queue import tier_counts
    from .status import badge_counts
    last = {}
    for ln in r.verdicts:
        last[ln["target"]["sid"]] = ln
    queue = [_item(i, r, last) for i in r.queue if i.tier in ("G", "A")]
    sample: Dict[str, List[Dict[str, Any]]] = {}
    for name, tiers in (("BC", ("B", "C")), ("A_prime", ("A'",))):
        its = {i.sid: i for i in r.queue if i.tier in tiers}
        order = sample_order(r.label, list(its))[:SAMPLE]
        sample[name] = [_item(its[s], r, last) for s in order]
    n_q = len(queue)
    n_s = sum(len(v) for v in sample.values())
    from .verdicts import latest_leaf
    last_leaf = latest_leaf(r.leaf_verdicts)
    leaves = [dict(lf, verdict=(last_leaf.get(lf["id"]) or {}).get("verdict"),
                   verdict_note=(last_leaf.get(lf["id"]) or {}).get("note"))
              for lf in r.leaves]
    return {
        "leaves": leaves,
        "label": r.label, "target_hash": r.target_hash, "badges": badge_counts(r.statuses),
        "tiers": tier_counts(r.queue), "queue": queue, "sample": sample,
        "changes": r.diff["changes"], "n_verdicts": len(r.verdicts),
        "overlay": {k: r.overlay_summary[k] for k in ("attached", "stale", "dropped")}
        | {"files": len(r.overlay_summary["files"])},
        "unattached_verifier": [f for f in r.overlay_summary["unattached"] if not f.get("stale")],
        "recheck": (r.recheck or {}).get("summary"),
        "estimate_minutes": round(MINUTES_FIXED + (SECONDS_QUEUE * n_q + SECONDS_SAMPLE * n_s) / 60),
        "n_queue": n_q, "n_sample": n_s,
    }
