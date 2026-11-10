"""Categorise the 47 'not an error (tool reading)' rows of the 2026-11-11 sweep triage by why
`recheck` v2 misread the text, using each row's settled explanation (how_confirmed).

Output: fp_categories.csv (row + category) and a count table on stdout.
"""
import csv
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "experiments/errata/2026-11-11/sweep_triage.csv"

CATS = {
    "A": "scope: a share/count within a stated subgroup compared with the whole training set",
    "B": "wrong referent: nearest-node heuristic picked another node (relay, upstream or downstream unit)",
    "C": "wrong quantity: a number of another kind caught by a keyword (AUC, fitted slope, mean readout x activation)",
    "D": "another configuration: counterfactual or linearised/core-only model, not 'this part off'",
    "E": "notation: aliases or raw units the formula parser does not resolve",
    "F": "prose direction: the direction word belongs to another clause, variable or idiom",
    "G": "prose number: value in a claim field the matcher skips, or a unit conversion (probability vs points)",
}


def category(r):
    lab, chk, how = r["label"], r["check"], r["how_confirmed"].lower()
    if chk == "freetext_active_share":
        if "tool compared the gate" in how or "1801 active" in how:
            return "B"
        return "A"
    if chk == "freetext_n_train":
        return "A"
    if chk == "freetext_active_count":
        if "tool took" in how or "state:9 gate" in how:
            return "B"
        if "counterfactual" in how:
            return "D"
        return "A"            # csc_tail acts on 3 rows: contribution non-zero vs node active
    if chk == "freetext_readout":
        if "auc" in how or "fitted slope" in how or "mean readout" in how:
            return "C"
        return "B"
    if chk == "freetext_auc":
        return "D"
    if chk == "freetext_formula":
        return "E"
    if chk == "prose_direction":
        return "F"
    if chk in ("prose_number", "prose_uncited"):
        return "G"
    return "?"


rows = [r for r in csv.DictReader(open(SRC)) if r["verdict"].startswith("not an error (tool reading)")]
out = []
for r in rows:
    r = dict(r)
    r["category"] = category(r)
    out.append(r)
with (HERE / "fp_categories.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
cnt = Counter(r["category"] for r in out)
by_check = Counter((r["category"], r["check"]) for r in out)
print(f"{len(out)} false-positive rows")
for k in sorted(cnt):
    print(f"  {k} {cnt[k]:2d}  {CATS.get(k, '?')}")
    for (c, chk), n in sorted(by_check.items()):
        if c == k:
            print(f"        {chk}: {n}")
