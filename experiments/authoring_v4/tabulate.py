"""Unblind and tabulate the authoring-v4 evaluation (design section 5.5, 5.6).

    uv run python experiments/authoring_v4/tabulate.py [adjudication dir] > experiments/authoring_v4/table.json

Reads the arm key (~/.explaneat/authoring_v4/arms.json), the adjudications, the verdicts, the
statement counts (statements.json) and the usage log; prints per-arm counts, densities with exact
Poisson intervals, verifier agreement, the conditional binomial test and cost.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
KEY = Path.home() / ".explaneat/authoring_v4"
TARGETED = {"b", "d", "e", "f"}


def poisson_ci(k, alpha=0.05):
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo, hi


def mech(m):
    return str(m or "").strip().split(":")[0].split()[0].lower() if m else "?"


def main(adj_dir):
    arms = {e["item_id"]: e for e in json.loads((KEY / "arms.json").read_text())["items"]}
    st = json.loads((HERE / "statements.json").read_text())
    usage = list(csv.DictReader((KEY / "usage.csv").open()))
    rows, agg = [], defaultdict(lambda: defaultdict(float))
    for iid, a in sorted(arms.items(), key=lambda kv: (kv[1]["label"], kv[1]["arm"])):
        adj = json.loads((adj_dir / f"{iid}.json").read_text())
        errs = adj["errors"]
        r = {"item": iid, "label": a["label"], "arm": a["arm"], "checkable": st[iid]["checkable"],
             "checkable_db": st[iid]["db"]["checkable"], "checkable_draft": st[iid]["draft"]["checkable"]}
        for lvl in ("refuted", "weakened"):
            for layer in ("database", "draft"):
                r[f"{lvl}_{layer}"] = sum(1 for e in errs if e.get("genuine_level") == lvl and e.get("layer") == layer)
        r["refuted_targeted_db"] = sum(1 for e in errs if e.get("genuine_level") == "refuted"
                                       and e.get("layer") == "database" and mech(e.get("mechanism")) in TARGETED)
        r["refuted_mechanisms"] = [f"{e.get('layer')}:{mech(e.get('mechanism'))}" for e in errs
                                   if e.get("genuine_level") == "refuted"]
        r["verifier_wrong"] = sum(1 for e in errs if e.get("genuine_level") == "none")
        gen = [e for e in errs if e.get("genuine_level") in ("refuted", "weakened")]
        fb = [set(e.get("found_by") or []) for e in gen]
        r["found_by_both"] = sum(1 for s in fb if {"A", "B"} <= s)
        r["found_by_one"] = sum(1 for s in fb if len(s) == 1)
        gref = [set(e.get("found_by") or []) for e in errs if e.get("genuine_level") == "refuted"]
        r["refuted_found_by_both"] = sum(1 for s in gref if {"A", "B"} <= s)
        r["verifier_minutes"] = round(sum(float(u["seconds"]) for u in usage if u["role"] == "verifier"
                                          and u["label_or_item"] == iid) / 60, 1)
        r["verifier_tokens"] = sum(int(u["tokens"]) for u in usage if u["role"] == "verifier" and u["label_or_item"] == iid)
        rows.append(r)
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                agg[a["arm"]][k] += v
    out = {"items": rows, "arms": {}}
    for arm, d in agg.items():
        ref = int(d["refuted_database"] + d["refuted_draft"])
        lo, hi = poisson_ci(ref)
        n = d["checkable"]
        gen = d["found_by_both"] + d["found_by_one"]
        out["arms"][arm] = {**{k: (int(v) if float(v).is_integer() else v) for k, v in d.items()},
                            "refuted_total": ref,
                            "refuted_per_100": round(100 * ref / n, 2),
                            "refuted_per_100_ci95": [round(100 * lo / n, 2), round(100 * hi / n, 2)],
                            "refuted_db_per_100_db": round(100 * d["refuted_database"] / d["checkable_db"], 2),
                            "weakened_per_100": round(100 * (d["weakened_database"] + d["weakened_draft"]) / n, 2),
                            "agreement_both_of_genuine": round(d["found_by_both"] / gen, 2) if gen else None}
    o, v = out["arms"]["original"]["refuted_total"], out["arms"]["v4"]["refuted_total"]
    out["binomial_v4_given_total"] = {"v4": v, "total": o + v,
                                      "p_one_sided_less": stats.binomtest(v, o + v, 0.5, alternative="less").pvalue
                                      if o + v else None}
    cost = defaultdict(lambda: defaultdict(list))
    for u in usage:
        cost[u["role"]]["tokens"].append(int(u["tokens"]))
        cost[u["role"]]["minutes"].append(float(u["seconds"]) / 60)
    out["cost"] = {role: {"n": len(d["tokens"]), "median_minutes": round(sorted(d["minutes"])[len(d["minutes"]) // 2], 1),
                          "total_tokens": sum(d["tokens"]), "median_tokens": sorted(d["tokens"])[len(d["tokens"]) // 2]}
                   for role, d in cost.items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "adjudication")
