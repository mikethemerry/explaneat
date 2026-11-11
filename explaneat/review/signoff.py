"""``signoff-v1``: the definition of "Mike-verified" (design section 7; PREREG amendment 2026-11-11).

A genome passes when, on the current text:

1. every gate item (G) has a current verdict, none is an unfixed ``wrong``;
2. every tier-A item has a current verdict, none an unfixed ``wrong``;
3. a seeded random sample of 10 items from B/C (and 10 from A', if any) has current verdicts. The
   draw ranks each item by sha256(label | signoff-v1 | sid): no one chooses it, and it is stable
   when other sentences change. If any item of the first 10 was ever marked ``wrong``, the next 10
   of the same stratum are drawn too; a ``wrong`` among those fails the sign-off, unless every item
   of the stratum has a current verdict (the whole stratum was reviewed);
4. a ``recheck`` run of the current text (same target hash) has no mismatch outside sentences, and
   every mismatch on a sentence is a gate that was answered (1). Once references exist, a reference
   that does not reproduce makes its sentence F, so the same rule covers the reproduce run;
5. every ``unsure`` verdict on a G, A or sampled item is disclosed (``--disclose``: listed in the
   report's "Statements not confirmed");
6. the result gives the one-sided 95% Clopper-Pearson upper bound on the share of wrong sentences in
   the sampled strata (x ever-wrong in n sampled).

A verdict is **current** when its content hash equals the sentence's current content hash. A
``wrong`` verdict is fixed by changing the text (the sentence then needs a new verdict).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import PROTOCOL, review_dir

SAMPLE = 10


def clopper_pearson_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) upper bound on a binomial proportion."""
    if n <= 0:
        return 1.0
    if x >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1 - alpha, x + 1, n - x))


def clopper_pearson_lower(x: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0 or x <= 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(alpha, x, n - x + 1))


def sample_order(label: str, sids: Sequence[str]) -> List[str]:
    return sorted(sids, key=lambda s: hashlib.sha256(f"{label}|{PROTOCOL}|{s}".encode()).hexdigest())


def check(label: str, items: Sequence, sentences: Sequence, verdict_lines: List[Dict[str, Any]],
          recheck_report: Optional[Dict[str, Any]], target_hash: str) -> Dict[str, Any]:
    from .verdicts import latest
    cur = {s.sid: s for s in sentences}
    last = latest(verdict_lines)
    ever_wrong = {ln["target"]["sid"] for ln in verdict_lines if ln["verdict"] == "wrong"}
    missing: List[Dict[str, Any]] = []
    problems: List[str] = []

    def current(sid):
        v = last.get(sid)
        if v and cur.get(sid) is not None and v["target"].get("content_hash") == cur[sid].content_hash:
            return v
        return None

    def need(it, why):
        v = current(it.sid)
        if v is None:
            missing.append({"sid": it.sid, "ref": it.ref, "tier": it.tier, "why": why,
                            "stale": it.sid in last})
            return
        if v["verdict"] == "wrong":
            problems.append(f"{it.sid} ({it.ref}): marked wrong and not fixed yet")
        if v["verdict"] == "unsure" and not v.get("disclosed"):
            problems.append(f"{it.sid} ({it.ref}): 'unsure' must be disclosed (--disclose)")

    by_tier: Dict[str, List[Any]] = {}
    for it in items:
        by_tier.setdefault(it.tier, []).append(it)
    for it in by_tier.get("G", []):
        need(it, "gate: " + ", ".join(it.gates))
    for it in by_tier.get("A", []):
        need(it, "tier A")

    strata = {"BC": by_tier.get("B", []) + by_tier.get("C", []), "A_prime": by_tier.get("A'", [])}
    sample_info: Dict[str, Any] = {}
    n_sampled, x_wrong = 0, 0
    for name, its in strata.items():
        if not its:
            sample_info[name] = {"items": 0, "drawn": [], "extended": False}
            continue
        idx = {i.sid: i for i in its}
        order = sample_order(label, list(idx))
        first = order[:SAMPLE]
        drawn = list(first)
        extended = any(s in ever_wrong for s in first)
        failed = False
        if extended:
            second = order[SAMPLE:2 * SAMPLE]
            drawn += second
            if any(s in ever_wrong for s in second):
                full = all(current(s) is not None for s in order)
                if not full:
                    failed = True
                    problems.append(f"{name}: a 'wrong' in the extension sample: review every {name} item "
                                    f"({len(order)}) or regenerate the text")
        for s in drawn:
            need(idx[s], f"sample ({name})")
        reviewed = [s for s in drawn if s in last]          # the bound uses only reviewed sample items
        n_sampled += len(reviewed)
        x_wrong += sum(1 for s in reviewed if s in ever_wrong)
        sample_info[name] = {"items": len(its), "drawn": drawn, "extended": extended, "failed": failed}

    # recheck at the current text
    if recheck_report is None:
        problems.append("no recheck report: run `explaneat review status <label>` (it runs recheck)")
    else:
        if recheck_report.get("target_hash") and recheck_report["target_hash"] != target_hash:
            problems.append("the recheck report is for another version of the text: re-run it")
        sent_locs = {s.location for s in sentences}
        for f in recheck_report.get("findings") or []:
            if f.get("severity") != "mismatch":
                continue
            if f.get("location") in sent_locs:
                continue         # a sentence-level mismatch is an F gate, answered by a verdict (1)
            problems.append(f"recheck mismatch outside the text: {f.get('check')} at {f.get('location')}")

    valid = {sid: current(sid) for sid in cur}
    counts = {"confirmed": 0, "wrong": 0, "unsure": 0}
    for v in valid.values():
        if v:
            counts[v["verdict"]] += 1
    bound = clopper_pearson_upper(x_wrong, n_sampled)
    tiers = {t: len(v) for t, v in by_tier.items()}
    passed = not missing and not problems
    return {
        "label": label, "protocol": PROTOCOL, "passed": passed,
        "checked_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "content_hash": target_hash, "tiers": tiers, "missing": missing, "problems": problems,
        "sample": sample_info, "sample_n": n_sampled, "sample_wrong": x_wrong,
        "error_bound_95": round(bound, 4),
        "counts": counts, "wrong_fixed": len({ln["target"]["sid"] for ln in verdict_lines
                                              if ln["verdict"] == "wrong"} - {s for s, v in valid.items()
                                                                              if v and v["verdict"] == "wrong"}),
        "unsure_disclosed": sum(1 for v in valid.values() if v and v["verdict"] == "unsure" and v.get("disclosed")),
        "minutes": round(sum((ln.get("seconds") or 0) for ln in verdict_lines) / 60, 1),
    }


def write_result(result: Dict[str, Any], directory: Optional[Path] = None) -> Path:
    p = (directory or review_dir()) / f"{result['label']}.signoff.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return p


def signoff_block(result: Dict[str, Any]) -> Dict[str, Any]:
    """The ``mike_verification.signoff`` block written into the record when the sign-off passes."""
    keep = ("protocol", "checked_at", "content_hash", "tiers", "sample_n", "sample_wrong", "error_bound_95",
            "counts", "wrong_fixed", "unsure_disclosed")
    return {k: result[k] for k in keep}
