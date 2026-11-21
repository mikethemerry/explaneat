"""Experiment 1 sidebar: PropNEAT hyperparameter sensitivity (Churn and the three synthetic sets).

Reads experiments/propneat_sensitivity/summary.json and cv.csv (written by that study's summarise.py
and choose.py) and experiments/benchmarks/benchmarks.csv (the comparator suite, read as it is now);
only formats them. Writes tables/exp1-sensitivity.tex with two floats:
  tab:exp1-sensitivity     per dataset: fixed configuration and its test AUC, the CV-selected
                           configuration and its test AUC, network size, best comparator, gradient boosting
  tab:exp1-sensitivity-cv  the CV surface: mean (SD) held-out-fold AUC of the 12 configurations

    uv run python thesis/ch7/exp1/sensitivity_table.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark_tables import COMPARATORS, rows as bench_rows

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STUDY = REPO / "experiments" / "propneat_sensitivity"
OUT = HERE / "tables" / "exp1-sensitivity.tex"
ORDER = ["churn", "both"]   # Mike, 22 Nov 2026: of the synthetic sets only U + saddle is presented
LABEL = {"churn": "Churn", "u": "Synth.\\ U", "saddle": "Synth.\\ saddle", "both": "Synth.\\ U $+$ saddle"}
FAMILY = {"heart": "Heart", "adult": "Adult/Churn"}
COMP_LABEL = dict(COMPARATORS)
SHORT = {"lr_l2": ("LR-$L_2$", "logistic regression, $L_2$"), "lr_l1": ("LR-$L_1$", "logistic regression, $L_1$"),
         "tree": ("Tree", "decision tree"), "rf": ("RF", "random forest"), "gb": ("GB", "gradient boosting"),
         "ebm": ("EBM", "explainable boosting"), "mlp": ("MLP", "one-hidden-layer MLP")}


def f3(x):
    return f"{float(x):.3f}"


def decay(w):
    w = float(w)
    return "0" if w == 0 else f"$10^{{{int(round(__import__('math').log10(w)))}}}$"


def conf(cid):
    e, fam, wd = cid.split("_")
    return f"{e[1:]}, {FAMILY[fam]}, {decay(wd[2:])}"


def mr(m):
    return f"\\makecell{{{f3(m['median'])}\\\\{{\\scriptsize {f3(m['min'])}--{f3(m['max'])}}}}}"


def size(h, c):
    return f"{h['median']:g} / {c['median']:g}"


def best_comparators(bench):
    K = {(r["dataset"], r["model_key"]): r for r in bench}
    bench_key = {"churn": "churn", "u": "shape_u", "saddle": "shape_saddle", "both": "shape_both"}
    out = {}
    for d in ORDER:
        comps = {k: K[(bench_key[d], k)] for k, _ in COMPARATORS}
        best = max(comps, key=lambda k: float(comps[k]["test_value"]))
        out[d] = (best, comps[best]["test_value"], comps["gb"]["test_value"])
    return out


def main_table(summary, bench):
    B = best_comparators(bench)
    out = ["  \\setlength{\\tabcolsep}{4pt}", "  \\renewcommand{\\arraystretch}{1.15}",
           "  \\begin{adjustbox}{max width=\\textwidth}",
           "  \\begin{tabular}{@{}lllcccccc@{}}", "    \\toprule",
           "    & \\multicolumn{2}{c}{Configuration} & \\multicolumn{2}{c}{Test AUC} & "
           "\\multicolumn{2}{c}{Size} & \\multicolumn{2}{c}{Comparators} \\\\",
           "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
           "    Dataset & Fixed & Selected & Fixed & Tuned & Fixed & Tuned & Best & GB \\\\", "    \\midrule"]
    for d in ORDER:
        s = summary[d]
        best, best_auc, gb_auc = B[d]
        sel = conf(s["selected_config"]) + ("$^\\ast$" if s["selected_is_fixed"] else "")
        out.append(f"    {LABEL[d]} & {conf(s['fixed_config'])} & {sel} & {mr(s['fixed_test_auc'])} & "
                   f"{mr(s['tuned_test_auc'])} & {size(s['fixed_hidden'], s['fixed_connections'])} & "
                   f"{size(s['tuned_hidden'], s['tuned_connections'])} & "
                   f"\\makecell{{{f3(best_auc)}\\\\{{\\scriptsize {SHORT[best][0]}}}}} & {f3(gb_auc)} \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "  \\end{adjustbox}"]
    return out


def cv_table(cv):
    configs = [r["config"] for r in cv if r["dataset"] == ORDER[0]]
    cell = {(r["dataset"], r["config"]): r for r in cv}
    out = ["  \\setlength{\\tabcolsep}{4pt}", "  \\renewcommand{\\arraystretch}{1.1}",
           "  \\begin{tabular}{@{}lll" + "c" * len(ORDER) + "@{}}", "    \\toprule",
           "    Epochs & Family & Decay & " + " & ".join(LABEL[d] for d in ORDER) + " \\\\", "    \\midrule"]
    for c in configs:
        e, fam, wd = c.split("_")
        cells = []
        for d in ORDER:
            r = cell[(d, c)]
            s = f"{f3(r['mean_auc'])} $\\pm$ {f3(r['sd_auc'])}"
            if r["selected"] == "True":
                s = f"\\textbf{{{f3(r['mean_auc'])}}} $\\pm$ {f3(r['sd_auc'])}"
            if r["is_fixed"] == "True":
                s += "$^\\dagger$"
            cells.append(s)
        out.append(f"    {e[1:]} & {FAMILY[fam]} & {decay(wd[2:])} & " + " & ".join(cells) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}"]
    return out


def main():
    summary = json.loads((STUDY / "summary.json").read_text())
    with open(STUDY / "cv.csv") as f:
        cv = list(csv.DictReader(f))
    bench = bench_rows()
    body = main_table(summary, bench)
    used = sorted({b for b, _, _ in best_comparators(bench).values()} | {"gb"}, key=lambda k: SHORT[k][0])
    comp_key = "; ".join(f"{SHORT[k][0]}: {SHORT[k][1]}" for k in used)
    cap = ("Experiment 1, sensitivity check: PropNEAT with a fixed configuration and with the configuration "
           "selected by cross-validation. Configuration: backprop epochs per generation, structural-mutation "
           "family, weight decay. Test AUC: median, below it the range over five seeds. Size: hidden nodes / "
           "enabled connections, median. Best: the comparators' highest test AUC (model below it). "
           + comp_key + "."
           + (" $^\\ast$The fixed configuration was selected." if any(summary[d]["selected_is_fixed"] for d in ORDER) else ""))
    cap_cv = ("Experiment 1, sensitivity check: cross-validated AUC of the twelve PropNEAT configurations. "
              "Each cell: mean $\\pm$ SD of the held-out-fold AUC over five folds of the training rows. "
              "Bold: selected; $^\\dagger$: the fixed configuration.")
    src = ("% Generated by thesis/ch7/exp1/sensitivity_table.py -- do not edit by hand.\n"
           "\\begin{table}[tbp]\n  \\thesistablestyle\n  \\centering\n  \\footnotesize\n" + "\n".join(body)
           + f"\n  \\caption{{{cap}}}\n  \\label{{tab:exp1-sensitivity}}\n\\end{{table}}\n\n"
           "\\begin{table}[tbp]\n  \\thesistablestyle\n  \\centering\n  \\footnotesize\n" + "\n".join(cv_table(cv))
           + f"\n  \\caption{{{cap_cv}}}\n  \\label{{tab:exp1-sensitivity-cv}}\n\\end{{table}}\n")
    OUT.write_text(src)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
