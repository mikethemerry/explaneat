"""One comparator suite for Experiments 1 and 2: the shared table builder (read-only).

Both benchmark tables (exp1-benchmarks.tex here, exp2-benchmarks.tex from thesis/ch7/exp2/
make_assets.py) are built by `classification_table` from experiments/benchmarks/benchmarks.csv, so
their rows (the seven comparators, in one order), cell format and caption key are identical:
each cell is the test AUC, its 95% bootstrap interval and the test accuracy; the column's best
AUC at the displayed precision is bold. Only the PropNEAT rows differ (Experiment 1: the explained
network; Experiment 2: the best seed by training AUC and the median of ten seeds).
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "experiments" / "benchmarks"
COMPARATORS = [("lr_l2", "Logistic regr., $L_2$"), ("lr_l1", "Logistic regr., $L_1$ (sparse)"),
               ("tree", "Decision tree"), ("rf", "Random forest"), ("gb", "Gradient boosting"),
               ("ebm", "Explainable boosting"), ("mlp", "MLP")]
SURVIVAL = [("cox_ph", "Cox PH, unpenalised"), ("cox_l2", "Cox PH, $L_2$ (ridge)"),
            ("cox_enet", "Cox PH, elastic net"), ("rsf", "Random survival forest"),
            ("gbs", "Gradient-boosted survival")]
# Experiment 1 benchmark columns: the original networks first (Heart, MONK-2, Adult), then the rest
EXP1_DATASETS = [("heart", "Heart"), ("monk2", "MONK-2"), ("adult", "Adult"), ("churn", "\\makecell{Churn\\\\(case)}"), ("c4_cells", "\\makecell{Connect-4\\\\cells}"),
                 ("c4_lines", "\\makecell{Connect-4\\\\lines}"), ("shape_u", "\\makecell{Synth.\\\\U}"),
                 ("shape_saddle", "\\makecell{Synth.\\\\saddle}"), ("shape_both", "\\makecell{Synth.\\\\U $+$ saddle}")]
# Caption key shared by both benchmark tables: only how to read a cell (Mike, 20 Nov: captions hold a title
# and the table's keys; the method is in the body text, see thesis/ch7/CAPTION_MOVED_TEXT.md).
SUITE_KEY = ("Each cell: test AUC; below it, its 95\\% interval and the test accuracy. "
             "Bold: the column's best AUC at this precision.")


def rows():
    with open(BENCH / "benchmarks.csv") as f:
        return list(csv.DictReader(f))


def by_key(R):
    return {(r["dataset"], r["model_key"]): r for r in R}


def f3(x):
    return f"{float(x):.3f}"


def f2(x):
    return f"{float(x):.2f}"


def cell(r, best, rng=None):
    auc = f3(r["test_value"])
    top = f"\\textbf{{{auc}}}" if auc == best else auc
    sub = (f"[{f2(r['ci_lo'])}, {f2(r['ci_hi'])}]" if r["ci_lo"] != "" else
           f"{f3(rng[0])}--{f3(rng[1])}" if rng else "--")
    acc = f"acc.\\ {f3(r['test_accuracy'])}" if r["test_accuracy"] != "" else ""
    return f"\\makecell{{{top}\\\\{{\\scriptsize {sub}}}\\\\{{\\scriptsize {acc}}}}}"


def classification_table(datasets, propneat, R=None, model_width="3.3cm"):
    """datasets: [(dataset key, column header)]; propneat: [(model_key, row label, {dataset: (lo, hi) range})].
    Returns the tabular body (lines) for a table float."""
    K = by_key(R or rows())
    keys = [p[0] for p in propneat] + [k for k, _ in COMPARATORS]
    best = {d: max(f3(K[(d, k)]["test_value"]) for k in keys if (d, k) in K) for d, _ in datasets}
    out = ["  \\setlength{\\tabcolsep}{3pt}", "  \\renewcommand{\\arraystretch}{1.1}",
           "  \\begin{adjustbox}{max width=\\textwidth}",
           f"  \\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{{model_width}}}" + "c" * len(datasets) + "@{}}",
           "    \\toprule", "    Model & " + " & ".join(h for _, h in datasets) + " \\\\", "    \\midrule",
           f"    \\multicolumn{{{len(datasets) + 1}}}{{@{{}}l}}{{\\textit{{PropNEAT}}}} \\\\"]
    for k, label, rngs in propneat:
        cells = [cell(K[(d, k)], best[d], (rngs or {}).get(d)) if (d, k) in K else "--" for d, _ in datasets]
        out.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    out += ["    \\midrule", f"    \\multicolumn{{{len(datasets) + 1}}}{{@{{}}l}}{{\\textit{{Comparators, tuned by 5-fold CV on the training rows}}}} \\\\"]
    for k, label in COMPARATORS:
        out.append(f"    {label} & " + " & ".join(cell(K[(d, k)], best[d]) for d, _ in datasets) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "  \\end{adjustbox}"]
    return out


def survival_table(R=None):
    """WHAS500 (the Experiment 1 survival case) with GBSG2 beside it as a side note."""
    K = by_key(R or rows())
    cols = [("whas500", "propneat_explained"), ("gbsg2", "propneat_selected")]
    keys = {"whas500": ["propneat_explained"] + [k for k, _ in SURVIVAL],
            "gbsg2": ["propneat_selected"] + [k for k, _ in SURVIVAL]}
    best = {d: max(f3(K[(d, k)]["test_value"]) for k in keys[d]) for d in keys}
    bold = lambda s, d: f"\\textbf{{{s}}}" if s == best[d] else s  # noqa: E731
    out = ["  \\setlength{\\tabcolsep}{3pt}", "  \\renewcommand{\\arraystretch}{1.1}",
           "  \\begin{adjustbox}{max width=\\textwidth}",
           "  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.6cm}ccccc@{}}", "    \\toprule",
           "    & \\multicolumn{4}{c}{WHAS500 (Experiment 1)} & GBSG2 (side note) \\\\",
           "    \\cmidrule(lr){2-5}\\cmidrule(l){6-6}",
           "    Model & \\makecell{Harrell's $C$\\\\(95\\% interval)} & Uno's $C$ & \\makecell{td-AUC\\\\(1, 3, 5 y)} & IBS & "
           "\\makecell{Harrell's $C$\\\\(95\\% interval)} \\\\", "    \\midrule"]

    def line(label, kw, kg):
        w, g = K[("whas500", kw)], K[("gbsg2", kg)]
        c = lambda r, d: f"{bold(f3(r['test_value']), d)} {{\\scriptsize [{f2(r['ci_lo'])}, {f2(r['ci_hi'])}]}}"  # noqa: E731
        return (f"    {label} & {c(w, 'whas500')} & {f3(w['uno_c'])} & {f3(w['td_auc_mean'])} & {f3(w['ibs'])} & "
                f"{c(g, 'gbsg2')} \\\\")
    out.append(line(f"PropNEAT Cox, selected network", "propneat_explained", "propneat_selected"))
    out.append("    \\midrule")
    for k, label in SURVIVAL:
        out.append(line(label, k, k))
    out += ["    \\bottomrule", "  \\end{tabular}", "  \\end{adjustbox}"]
    return out
