"""Tables of the three levels from levels/<label>.json (compute_levels.py).

    uv run python experiments/explanation_levels/make_tables.py

Writes ``tables.md`` (the per-study table, the ladder table, every Connect-4 / KRK leaf with its
best concept) and ``thesis_exports/ch7/ch7_levels_table.tex`` (the per-study table; a new file,
``ch7_table.tex`` is not touched). Cells are median [min-max] over the explanations of a study.
"""

from __future__ import annotations

import json
import statistics as st
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEVELS = HERE / "levels"
TEX = ROOT / "thesis_exports" / "ch7" / "ch7_levels_table.tex"

STUDIES = OrderedDict([("heart", "Heart"), ("monk2", "MONK-2"), ("adult", "Adult"), ("heart100", "Heart-100"),
                       ("churn100", "churn100"), ("churn200", "churn200")])
PREREG = {"heart", "monk2", "adult"}


def load():
    rows = []
    for p in sorted(LEVELS.glob("*.json")):
        d = json.loads(p.read_text())
        lv = d["levels"]
        if not lv["decomposed"]["n_leaves"]:      # an explanation with no annotations (nothing to grade)
            continue
        gt = (lv["understood"].get("ground_truth") or {})
        rows.append({
            "label": d["label"], "set": d["set"], "study": d["study"], "dataset": d["dataset"],
            "family": gt.get("family") or ("c4" if d["set"] == "c4cap" else None), "task": gt.get("task"),
            "arch": gt.get("arch") or ("evolved" if d["set"] == "c4cap" else None),
            "faithful": lv["faithful"]["function_preservation_max_abs_err"],
            "closed_form": lv["faithful"]["closed_form_max_abs_err"],
            "n_leaves": lv["decomposed"]["n_leaves"], "hidden": lv["decomposed"]["hidden_units"],
            "granularity": lv["decomposed"]["granularity"],
            "max_width": lv["decomposed"]["max_width"], "max_depth": lv["decomposed"]["max_depth"],
            "coarse": any(w.startswith("coarse") for w in lv["decomposed"]["warnings"]),
            "understood": lv["understood"]["understood_fraction"], "partial": lv["understood"]["partial_fraction"],
            "cfo": lv["understood"]["closed_form_only_fraction"],
            "domain": (lv["understood"].get("domain_only") or {}).get("understood_fraction"),
            "gt": gt.get("gt_recovery"), "gt_terms": (gt.get("gt_terms_recovered"), gt.get("gt_terms_total")),
            "leaves": d["leaves"],
        })
    return rows


def mmm(vals, d=2, pct=False):
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.{d}f}")
    if len(v) == 1:
        return f(v[0])
    return f"{f(st.median(v))} [{f(min(v))}-{f(max(v))}]"


def mmm_int(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    return f"{st.median(v):g} [{min(v)}-{max(v)}]" if len(v) > 1 else str(v[0])


def sci(x):
    return "n/a" if x is None else f"{x:.1e}"


def study_rows(rows):
    out = []
    for key, name in STUDIES.items():
        g = [r for r in rows if r["set"] == "v4" and r["study"] == key]
        if g:
            out.append((name, g))
    lad_syn = [r for r in rows if r["set"] == "ladder" and r["family"] == "synthetic"]
    lad_c4 = [r for r in rows if r["set"] == "ladder" and r["family"] == "c4"]
    c4cap = [r for r in rows if r["set"] == "c4cap"]
    if lad_syn:
        out.append(("Ladder, synthetic (MLP)", lad_syn))
    if lad_c4:
        out.append(("Ladder, Connect-4 (MLP)", lad_c4))
    if c4cap:
        out.append(("Connect-4 capacity (evolved)", c4cap))
    return out


def md_table(rows):
    L = ["| Study | n | Faithful: max abs err (worst) | Leaves | Granularity | Coarse (n) | Understood: meaningful share | Partial | Closed-form only | Domain-concept share |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, g in study_rows(rows):
        L.append(f"| {name} | {len(g)} | {sci(max(r['faithful'] for r in g))} | {mmm_int([r['n_leaves'] for r in g])} | "
                 f"{mmm([r['granularity'] for r in g])} | {sum(r['coarse'] for r in g)} | "
                 f"{mmm([r['understood'] for r in g], pct=True)} | {mmm([r['partial'] for r in g], pct=True)} | "
                 f"{mmm([r['cfo'] for r in g], pct=True)} | "
                 f"{mmm([r['domain'] for r in g], pct=True) if any(r['domain'] is not None for r in g) else 'n/a'} |")
    return "\n".join(L)


def ladder_table(rows):
    L = ["| Label | Family | Task | Arch | Hidden | Faithful | Leaves | Granularity | Max width | Max depth | Understood | Partial | CFO | Domain share | Ground truth |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in rows if r["set"] in ("ladder", "c4cap")], key=lambda r: (r["family"] or "", r["hidden"] or 0, r["label"])):
        gt = r["gt"] or ""
        if r["gt_terms"][0]:
            gt += f" ({r['gt_terms'][0]}/{r['gt_terms'][1]})"
        L.append(f"| {r['label']} | {r['family']} | {r['task'] or ''} | {r['arch']} | {r['hidden']} | {sci(r['faithful'])} | "
                 f"{r['n_leaves']} | {r['granularity']} | {r['max_width']} | {r['max_depth']} | "
                 f"{mmm([r['understood']], pct=True)} | {mmm([r['partial']], pct=True)} | {mmm([r['cfo']], pct=True)} | "
                 f"{mmm([r['domain']], pct=True) if r['domain'] is not None else 'n/a'} | {gt} |")
    return "\n".join(L)


def c4_leaf_table(rows):
    L = ["| Label | Arch | Leaf | Exits | Best concept (any) | Score | Grade | Best domain concept | Domain score | Domain grade |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in rows if r["family"] == "c4" or r["dataset"].startswith("krk")],
                    key=lambda r: (r["hidden"] or 0, r["label"])):
        for name, lf in r["leaves"].items():
            if "error" in lf:
                L.append(f"| {r['label']} | {r['arch']} | {name} | | error: {lf['error'][:60]} | | | | | |")
                continue
            L.append(f"| {r['label']} | {r['arch']} | {name} | {lf['n_exits']} | {', '.join(lf['best_concepts'][:3])} | "
                     f"{lf['score']:.2f} | {lf['grade']} | {', '.join((lf.get('best_domain_concepts') or [])[:3])} | "
                     f"{'n/a' if lf.get('domain_score') is None else f'{lf['domain_score']:.2f}'} | {lf.get('domain_grade') or ''} |")
    return "\n".join(L)


def tex_table(rows):
    def t(s):
        return s.replace("%", r"\%").replace("[", "[").replace("-", "--") if "[" in s else s.replace("%", r"\%")
    L = [r"% Generated by experiments/explanation_levels/make_tables.py from",
         r"% experiments/explanation_levels/levels/*.json (compute_levels.py, read only); do not edit by hand.",
         r"% Requires \usepackage{booktabs}.",
         r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{The three levels of explanation per study. Faithful: the largest function-preservation error "
         r"(explained model state against the phenotype, training rows) over the study's explanations. Decomposed: "
         r"leaves per explanation and granularity, the share of hidden units whose part is at most 6 units wide "
         r"(depth is not penalised); coarse counts explanations where one part wider than that holds half the "
         r"hidden units. Understood: the share of leaves whose best library concept grades meaningful (probe score "
         r"$\geq 0.8$), partial ($\geq 0.5$) or closed-form only; for Connect-4 the domain column counts only "
         r"domain concepts (threats, open lines, centre, rows, parity), not raw cells. Cells are median [min--max] "
         r"over the explanations. The ladder rows are plain backprop MLPs of the explanation complexity ladder; the "
         r"capacity row is the evolved Connect-4 network.}",
         r"\label{tab:ch7-levels}",
         r"\begin{tabular}{@{}lrlllrlllr@{}}", r"\toprule",
         r" & & \multicolumn{1}{c}{Faithful} & \multicolumn{3}{c}{Decomposed} & \multicolumn{4}{c}{Understood} \\",
         r"\cmidrule(lr){3-3}\cmidrule(lr){4-6}\cmidrule(lr){7-10}",
         r"Study & $n$ & Max err. & Leaves & Granularity & Coarse & Meaningful & Partial & Closed-form only & Domain \\",
         r"\midrule"]
    groups = study_rows(rows)
    did_rule = False
    for name, g in groups:
        if not did_rule and name not in [STUDIES[k] for k in PREREG]:
            L.append(r"\midrule")
            did_rule = True
        if name == "Heart":
            L.append(r"\multicolumn{10}{@{}l}{\emph{Pre-registered (ten seeds)}} \\")
        if did_rule and name == next(n for n, _ in groups if n not in [STUDIES[k] for k in PREREG]):
            L.append(r"\multicolumn{10}{@{}l}{\emph{Additional analyses and the ladder (not pre-registered)}} \\")
        dom = mmm([r["domain"] for r in g], pct=True) if any(r["domain"] is not None for r in g) else "--"
        worst = max(r["faithful"] for r in g)
        m, e = f"{worst:.1e}".split("e")
        worst_tex = "0" if worst == 0 else f"${m}\\times10^{{{int(e)}}}$"
        L.append(f"{name} & {len(g)} & {worst_tex} & "
                 f"{t(mmm_int([r['n_leaves'] for r in g]))} & {t(mmm([r['granularity'] for r in g]))} & "
                 f"{sum(r['coarse'] for r in g)} & {t(mmm([r['understood'] for r in g], pct=True))} & "
                 f"{t(mmm([r['partial'] for r in g], pct=True))} & {t(mmm([r['cfo'] for r in g], pct=True))} & {t(dom)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    rows = load()
    md = ["# Three levels per study", "", md_table(rows), "", "# The ladder and the Connect-4 capacity network", "",
          ladder_table(rows), "", "# Connect-4 and KRK leaves against the concept library", "", c4_leaf_table(rows), ""]
    (HERE / "tables.md").write_text("\n".join(md))
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(tex_table(rows))
    print(f"wrote {HERE / 'tables.md'} and {TEX} from {len(rows)} explanations")


if __name__ == "__main__":
    main()
