"""Summaries of the Frankenstein study (PLAN.md), read only from the run CSVs in results/.

Writes:
- summary_auc.csv: every comparator and standalone condition, plus the per-condition median
  over the ten unions, with test AUC, CI, the paired deltas and size;
- union_summary.csv: per condition over the ten unions: test AUC median / range; for each
  reference (best seed, ensemble, sham, base recipient) the median paired delta, the sign
  counts, a two-sided sign test and how many CIs lie above / below 0;
- families_summary.csv: per family and condition, the standalone value and the median over the
  ten unions of eff_n and net / gross.

    uv run python experiments/frankenstein/frankenstein.py summary
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REFS = ["best_seed", "ensemble", "sham", "base"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(v):
    return float(v) if v not in ("", None) else float("nan")


def write(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"wrote {path} ({len(rows)} rows)")


def union_summary(rows):
    out = []
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rr = [r for r in rows if r["condition"] == cond]
        auc = np.array([num(r["test_auc"]) for r in rr])
        o = {"condition": cond, "n_unions": len(rr), "test_auc_median": float(np.median(auc)),
             "test_auc_min": float(auc.min()), "test_auc_max": float(auc.max()),
             "C_median": float(np.median([num(r["C"]) for r in rr])),
             "hidden_nodes_median": float(np.median([num(r["hidden_nodes"]) for r in rr])),
             "connections_median": float(np.median(
                 [num(r["phenotype_connections_enabled"]) for r in rr])),
             "grafted_sd_median": float(np.median([num(r["grafted_contrib_sd_train"])
                                                   for r in rr])),
             "own_sd_median": float(np.median([num(r["other_contrib_sd_train"]) for r in rr]))}
        ep = [num(r["ft_epochs"]) for r in rr if r["ft_epochs"] != ""]
        o["ft_epochs_median"] = float(np.median(ep)) if ep else ""
        for ref in REFS:
            d = np.array([num(r[f"d_vs_{ref}"]) for r in rr])
            if np.all(np.isnan(d)):
                continue
            lo = np.array([num(r[f"d_vs_{ref}_lo"]) for r in rr])
            hi = np.array([num(r[f"d_vs_{ref}_hi"]) for r in rr])
            npos, nneg = int((d > 0).sum()), int((d < 0).sum())
            o.update({f"d_{ref}_median": float(np.median(d)), f"d_{ref}_pos": npos,
                      f"d_{ref}_neg": nneg,
                      f"d_{ref}_sign_p": binomtest(npos, npos + nneg, 0.5).pvalue
                      if npos + nneg else "",
                      f"d_{ref}_ci_above0": int((lo > 0).sum()),
                      f"d_{ref}_ci_below0": int((hi < 0).sum())})
        out.append(o)
    return out


def main(out: Path):
    comp = read(out / "comparators.csv")
    sa = read(out / "conditions_standalone.csv")
    un = read(out / "conditions_union.csv")
    us = union_summary(un)
    rows = []
    for r in comp:
        rows.append({"model": r["model"], "group": "comparator", **{k: r[k] for k in (
            "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "hidden_nodes",
            "phenotype_connections_enabled", "source")}, "best_by_train": r["best_by_train"]})
    for r in sa:
        rows.append({"model": f"{r['build']} {r['condition']}", "group": "standalone", **r})
    for u in us:
        rows.append({"model": f"union (median of 10) {u['condition']}", "group": "union median",
                     "test_auc": u["test_auc_median"], "hidden_nodes": u["hidden_nodes_median"],
                     "phenotype_connections_enabled": u["connections_median"],
                     "d_vs_best_seed": u["d_best_seed_median"],
                     "d_vs_ensemble": u["d_ensemble_median"],
                     "d_vs_sham": u.get("d_sham_median", ""),
                     "d_vs_base": u.get("d_base_median", "")})
    cols = ["model", "group", "best_by_train", "C", "n_parts", "train_auc", "test_auc",
            "test_auc_ci_lo", "test_auc_ci_hi", "d_vs_best_seed", "d_vs_best_seed_lo",
            "d_vs_best_seed_hi", "d_vs_ensemble", "d_vs_ensemble_lo", "d_vs_ensemble_hi",
            "d_vs_sham", "d_vs_sham_lo", "d_vs_sham_hi", "d_vs_base", "hidden_nodes",
            "phenotype_connections_enabled", "ft_epochs", "ft_best_epoch", "source"]
    write(out / "summary_auc.csv", rows, cols)
    ucols = list(us[0])
    for u in us[1:]:
        ucols += [k for k in u if k not in ucols]
    write(out / "union_summary.csv", us, ucols)

    fs = read(out / "families_standalone.csv") + read(out / "families_union.csv")
    fam_rows = []
    keys = dict.fromkeys((r["condition"], r["family"]) for r in fs)
    for cond, fam in keys:
        s = [r for r in fs if r["build"] == "standalone" and r["condition"] == cond
             and r["family"] == fam]
        d = [r for r in fs if r["build"] == "standalone-dedup" and r["condition"] == cond
             and r["family"] == fam]
        u = [r for r in fs if r["build"].startswith("union") and r["condition"] == cond
             and r["family"] == fam]
        src = (s or d)
        o = {"condition": cond, "family": fam}
        if src:
            r = src[0]
            o.update({"standalone_build": r["build"], "standalone_n_parts": r["n_parts"],
                      "standalone_eff_n": r["eff_n"],
                      "standalone_net_over_gross": r["net_over_gross"],
                      "standalone_top_feature": r["top_feature"],
                      "standalone_top_share": r["top_share_of_gross"]})
        if u:
            o.update({"unions": len(u),
                      "union_n_parts_median": float(np.median([num(r["n_parts"]) for r in u])),
                      "union_eff_n_median": float(np.nanmedian([num(r["eff_n"]) for r in u])),
                      "union_net_over_gross_median": float(np.nanmedian(
                          [num(r["net_over_gross"]) for r in u]))})
        fam_rows.append(o)
    write(out / "families_summary.csv", fam_rows,
          ["condition", "family", "standalone_build", "standalone_n_parts", "standalone_eff_n",
           "standalone_net_over_gross", "standalone_top_feature", "standalone_top_share",
           "unions", "union_n_parts_median", "union_eff_n_median",
           "union_net_over_gross_median"])


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results")
