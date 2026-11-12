"""Thesis experiment tables (added 2026-11-13; `ch7_table.tex` is not touched).

    uv run python scripts/thesis/thesis_experiment_tables.py

Reads only committed results files (no database, no recomputation) and writes three LaTeX
tables to ``thesis_exports/ch7/``:

- ``ch7_experiment1_cases.tex``: Experiment 1, one row per case (dataset, n, model size, the
  test metric against the references, the three levels, verification);
- ``ch7_experiment2_consistency.tex``: Experiment 2, the frozen-method v4 run per set
  (coverage, recheck, levels, sign-off), with the reported test AUC and the Rashomon count;
- ``ch7_experiment3_complexity.tex``: Experiment 3, one line per study (the ladder, the
  three Connect-4 bases, MNIST) with its finding.

Every number is read from the file named in the column note; the finding sentences are
copied from the results files cited in ``THESIS_EXPERIMENTS.md``. Prints each table as
Markdown as well, for checking against the sources.
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis_exports" / "ch7"
E = ROOT / "experiments"
R = ROOT / "results" / "thesis"


def jload(p):
    return json.load(open(p))


def rows(p):
    return list(csv.DictReader(open(p)))


def tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")


def tex_table(caption, label, header, body, note=None, colspec=None):
    colspec = colspec or ("l" * len(header))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in body:
        if r == "midrule":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["\\begin{flushleft}\\footnotesize " + note + "\\end{flushleft}"]
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def md_table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in body:
        if r != "midrule":
            out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def levels_counts(p):
    L = jload(p)["levels"]
    u = L["understood"]["counts"]
    return dict(faithful=L["faithful"]["function_preservation_max_abs_err"],
                leaves=L["decomposed"]["n_leaves"], granularity=L["decomposed"]["granularity"],
                m=u["meaningful"], p=u["partial"], c=u["closed_form_only"],
                understood=L["understood"]["understood_fraction"],
                # the measured level: the author's leaf verdicts (None until a leaf has one)
                mike=L["understood"].get("understood_mike"), mike_n=L["understood"].get("mike_graded") or 0)


def verdict_counts(p):
    v = jload(p)
    c = v.get("counts") or Counter(f.get("grade") for f in v["findings"])
    return v["overall"]["grade"], int(c.get("upheld", 0)), int(c.get("weakened", 0)), int(c.get("refuted", 0)), int(c.get("flag", 0))


def verified_cell(lv):
    """Understood (verified): median of the author's understood share over the explanations with
    leaf verdicts, with the count; 'none yet' until one has them."""
    v = [x["mike"] for x in lv if x.get("mike") is not None]
    if not v:
        return "none yet"
    return f"{100 * st.median(v):.0f}%" + (f" ({len(v)} of {len(lv)})" if len(v) < len(lv) else "")


def f3(x):
    return f"{float(x):.3f}"


def summary_md_row(metric, columns):
    """Reported numbers from experiments/10seed/summary.md (do not recompute)."""
    text = (E / "10seed" / "summary.md").read_text().splitlines()
    out = {}
    current = None
    for line in text:
        m = re.match(r"\| Metric \| (.*) \|$", line)
        if m:
            current = [c.strip() for c in m.group(1).split("|")]
            continue
        if current and line.startswith(f"| {metric} |"):
            cells = [c.strip() for c in line.strip("|").split("|")][1:]
            for name, cell in zip(current, cells):
                if name in columns:
                    out[name] = cell.split(" ")[0]  # the median
    assert set(out) == set(columns), (out, columns)
    return out


# ----------------------------------------------------------------------------- Experiment 1
def experiment1():
    base = jload(R / "baselines.json")
    paper = {r["dataset"]: r for r in rows(E / "10seed" / "runs_paper_reference.csv")}
    body = []
    for key, name in (("heart", "Heart"), ("monk2", "MONK-2"), ("adult", "Adult")):
        p, b = paper[key], base[key]
        # the three levels of the paper explanation (compute_levels.py --set paper; probe v1 until the
        # probe v2 recompute replaces the file)
        lvp = E / "explanation_levels" / "levels" / f"paper-{key}.json"
        if lvp.exists():
            L = levels_counts(lvp)
            lv_cell = f"{L['faithful']:.1e}; {L['leaves']} leaves; {L['m']} / {L['p']} / {L['c']} (probe v1)"
        else:
            lv_cell = "not computed (paper explanation; levels script not run on it)"
        body.append([f"{name} (paper genome)", f"{int(p['n_train']):,} / {int(p['n_test']):,}",
                     f"{p['hidden_nodes']} / {p['phenotype_connections_enabled']}",
                     f"AUC {float(p['test_auc']):.3f} vs LR {b['logistic']['test_auc']:.3f}, GB {b['gradient_boosting']['test_auc']:.3f}",
                     lv_cell, "paper explanation; no blind verifier; not signed off"])
    # churn case (Experiment 1.2): one run, seed 10, 200 generations, best genome by training fitness
    cc = E / "churn_case"
    crun = rows(cc / "runs.csv")[0]
    cref = jload(cc / "references.json")
    clv = levels_counts(cc / "levels" / "churn-case-s10.json")
    g, u, w, rf, fl = verdict_counts(cc / "verdicts" / "churn-case-s10" / "verdict.json")
    body.append(["Churn case (churn-case-s10)", f"{int(crun['n_train']):,} / {int(crun['n_test']):,}",
                 f"{crun['hidden_nodes']} / {crun['phenotype_connections_enabled']}",
                 f"AUC {float(crun['test_auc']):.3f} vs LR {cref['l2_logistic']['test_auc']:.3f}, GB {cref['gradient_boosting']['test_auc']:.3f}",
                 f"{clv['faithful']:.1e}; {clv['leaves']} leaves; {clv['m']} / {clv['p']} / {clv['c']} (probe v2)",
                 f"blind verifier {g} ({u} upheld, {w} weakened, {rf} refuted, {fl} flags); not signed off"])
    body.append("midrule")
    # Cox
    refs = {(r["dataset"], r["model"]): r for r in rows(E / "survival" / "cox_case" / "references.csv")}
    runs = {(r["dataset"], r["seed"]): r for r in rows(E / "survival" / "cox_case" / "runs.csv")}
    for ds, tag, label, name in (("gbsg2", "gbsg2-s1", "cox-gbsg2-s1", "GBSG2 (Cox)"),
                                 ("whas500", "whas500-s0", "cox-whas500-s0", "WHAS500 (Cox)")):
        rj = jload(E / "survival" / "cox_case" / "runs" / f"{tag}.json")
        seed = tag.split("-s")[1]
        run = runs[(ds, seed)]
        lv1 = levels_counts(E / "survival" / "cox_case" / "levels" / f"{label}.json")
        g, u, w, rf, fl = verdict_counts(E / "survival" / "cox_case" / "verdicts" / label / "verdict.json")
        body.append([name, f"{rj['n_train']} / {rj['n_val']} / {rj['n_test']}",
                     f"{run['hidden']} / {run['k']}",
                     f"Harrell C {f3(refs[(ds, 'propneat_cox')]['harrell_c'])} vs ridge {f3(refs[(ds, 'ridge_cox')]['harrell_c'])}, "
                     f"Cox PH {f3(refs[(ds, 'cox_ph')]['harrell_c'])}, RSF {f3(refs[(ds, 'rsf')]['harrell_c'])}",
                     f"{lv1['faithful']:.1e}; {lv1['leaves']} leaves; {lv1['m']} / {lv1['p']} / {lv1['c']}",
                     f"blind verifier {g} ({u} upheld, {w} weakened, {rf} refuted, {fl} flags); not signed off"])
    # XOR
    x = jload(E / "xor" / "xor_results.json")["summary"]
    hc = {int(k): int(v) for k, v in x["raw_hidden_count"].items()}  # hidden count -> seeds
    body.append(["XOR (worked example)", "4 rows (the truth table)",
                 f"{x['n_seeds']} seeds; raw {min(hc)}--{max(hc)} hidden",
                 f"solved {x['n_solved']} / {x['n_seeds']} (median generation {x['solved_generation_median']:g}); reference: LP enumeration",
                 "not applicable (hand analysis: two families of realisations)",
                 "verify_xor.py claims (a)--(f) pass; not signed off"])
    body.append("midrule")
    # KRK
    cr = {r["name"]: r for r in rows(E / "complex_real" / "runs.csv")}
    cbase = jload(E / "complex_real" / "baselines.json")
    r, b = cr["krk-default-s2"], cbase["krk"]
    rec = jload(E / "complex_real" / "explanations" / "krk-default-s2.json")
    la = rec["leaf_assessments"]
    grades = Counter((a.get("assessment") or a.get("grade")) for a in (la.values() if isinstance(la, dict) else la))
    krk_vd = E / "complex_real" / "verification" / "krk" / "verdict.json"
    if krk_vd.exists():
        g, u, w, rf, fl = verdict_counts(krk_vd)
        krk_ver = f"blind verifier {g} ({u} upheld, {w} weakened, {rf} refuted, {fl} flags); not signed off"
    else:
        krk_ver = "orchestrator checks only; no blind verifier; not signed off"
    body.append(["KRK, boundary case (krk-default-s2)", f"{b['n_train']:,} / {b['n_test']:,}",
                 f"{r['hidden']} / {r['conns']}",
                 f"AUC {float(r['test_auc']):.4f} vs LR {b['logistic_regression']['test']:.3f}, GB {b['gradient_boosting']['test']:.3f}, rule {b['ground_truth_rule']['test']:.3f}",
                 f"{rec['function_preservation_max_abs_err']:.1e}; {rec['n_leaf_annotations']} leaves; "
                 f"{grades.get('meaningful', 0)} / {grades.get('partial', 0)} / {grades.get('closed-form-only', 0)} (author's grades)",
                 krk_ver])
    # Connect-4 capacity
    c4 = {r["name"]: r for r in rows(E / "c4_capacity" / "runs.csv")}["main-h10-e50-s1"]
    c4b = jload(E / "c4_capacity" / "baselines.json")
    c4rec = jload(E / "c4_capacity" / "explanations" / "record.json")
    c4lv = levels_counts(E / "explanation_levels" / "levels" / "c4cap-main-h10-e50-s1.json")
    g, u, w, rf, fl = verdict_counts(E / "c4_capacity" / "verification" / "independent" / "verdict.json")
    sa = c4rec["semantic_assessment"]
    lr, gb = c4b["lr_cells"], c4b["gradient_boosting"]
    body.append(["Connect-4, boundary case (capacity champion, seed 1)", f"{int(c4['fit_n']):,} / {cbase['c4']['n_test']:,}",
                 f"{c4['hidden']} / {c4['connections']}",
                 f"AUC {float(c4['test_auc']):.4f} vs LR {lr['test']:.4f}, GB {gb['test']:.4f}",
                 f"{c4lv['faithful']:.1e}; {c4lv['leaves']} leaves; probe {c4lv['m']} / {c4lv['p']} / {c4lv['c']}, "
                 f"author {sa['meaningful_leaves']} / {sa['partial_leaves']} / {sa['described_leaves']} described",
                 f"blind verifier {g} ({u} upheld, {w} weakened, {rf} refuted); not signed off"])
    header = ["Case", "n (train / test)", "Hidden / conns", "Test metric vs references",
              "Levels: faithful; decomposed; understood (m / p / c)", "Verification"]
    note = ("Sources: results/thesis/baselines.json and experiments/10seed/runs\\_paper\\_reference.csv (paper genomes); "
            "experiments/churn\\_case/ (runs.csv, references.json, levels/, verdicts/; the churn case); "
            "experiments/survival/cox\\_case/ (runs, references.csv, levels/, verdicts/; n is train / validation / test); "
            "experiments/xor/xor\\_results.json; experiments/complex\\_real/ (runs.csv, baselines.json, explanations/, verification/krk/); "
            "experiments/c4\\_capacity/ (runs.csv, baselines.json, explanations/record.json, verification/independent/) and "
            "experiments/explanation\\_levels/levels/ (the paper rows: levels/paper-*.json, concept probe v1). "
            "Understood counts are leaves graded meaningful / partial / closed-form only. "
            "Signed off means verified by the author under the signoff-v1 protocol; none is.")
    return header, body, note


# ----------------------------------------------------------------------------- Experiment 2
SETS = [("heart", "Heart", E / "10seed" / "explanations_v4.csv"),
        ("monk2", "MONK-2", E / "10seed" / "explanations_v4.csv"),
        ("adult", "Adult", E / "10seed" / "explanations_v4.csv"),
        ("heart100", "Heart-100 (supp.)", E / "10seed" / "explanations_heart100_v4.csv"),
        ("churn100", "churn100 (supp.)", E / "churn" / "explanations_churn100_v4.csv"),
        ("churn200", "churn200 (supp.)", E / "churn" / "explanations_churn200_v4.csv")]
SUMMARY_COL = {"heart": "Heart", "monk2": "MONK-2", "adult": "Adult", "heart100": "Heart-100",
               "churn100": "churn100", "churn200": "churn200"}


def experiment2():
    auc = summary_md_row("Test AUC", list(SUMMARY_COL.values()))
    rash = {(r["run_set"], r["reference"], r["eps"]): r for r in rows(R / "rashomon_counts.csv")}
    body = []
    for key, name, csvp in SETS:
        recs = [jload(p) for p in sorted((E / "10seed" / "explanations_v4").glob(f"{key}-s*.json"))]
        assert len(recs) == 10, key
        full = sum(1 for r in recs if r["full_coverage"] and r["hierarchy_complete"])
        mism = sum(r["recheck_summary"]["mismatch"] for r in recs)
        nr = sum(r["recheck_summary"]["needs_review"] for r in recs)
        csvr = [r for r in rows(csvp) if r["dataset"] == key or r["label"].startswith(key + "-s")]
        assert len(csvr) == 10, (key, len(csvr))
        # half-up to one decimal, as V4_RUN.md reports (21.55 -> 21.6)
        mins = Decimal(str(round(st.median(float(r["claude_minutes"]) for r in csvr), 2))).quantize(Decimal("0.1"), ROUND_HALF_UP)
        signed = sum(1 for r in csvr if r["verified_by_mike"].strip().lower() == "yes")
        lv = [levels_counts(p) for p in sorted((E / "explanation_levels" / "levels").glob(f"{key}-s*.json"))]
        assert len(lv) == 10, key
        rz = rash[(key, "best_observed", "0.01")]
        body.append([name, auc[SUMMARY_COL[key]], f"{rz['seeds_in_set']} / {rz['n_seeds']}",
                     f"{full} / 10", f"{mism} / {nr}", f"{mins}",
                     f"{st.median(x['leaves'] for x in lv):g}; {st.median(x['granularity'] for x in lv):.2f}; "
                     f"{100 * st.median(x['understood'] for x in lv):.0f}%",
                     verified_cell(lv), f"{signed} / 10"])
        if key == "adult":
            body.append("midrule")
    header = ["Set", "Test AUC (median)", "Rashomon $\\varepsilon=0.01$", "Full coverage (v4)",
              "Recheck: mismatch / needs review", "Minutes (median)", "Levels: leaves; granularity; probe meaningful (screen)",
              "Understood (verified)", "Signed off"]
    note = ("Ten seeds per set. Test AUC is the reported median (experiments/10seed/summary.md, unchanged). "
            "Rashomon: seeds within 0.01 test AUC of the best observed seed (results/thesis/rashomon\\_counts.csv). "
            "Coverage, recheck and minutes are the frozen-method v4 run (experiments/10seed/explanations\\_v4/ and the *\\_v4.csv files; "
            "recheck counts are the records' stored recheck\\_summary, before the amend\\_evidence fix that cleared the needs-review items). "
            "Levels are medians over the ten v4 records (experiments/explanation\\_levels/levels/); the probe's meaningful share is an "
            "automated screen. Understood (verified): the share of leaves the author graded understood in his leaf-by-leaf verification "
            "(review leaf verdicts), the measured level; none yet until a set has leaf verdicts. "
            "Signed off: verified by the author (signoff-v1); the three sets above the rule are pre-registered, the three below are supplementary.")
    return header, body, note


# ----------------------------------------------------------------------------- Experiment 3
def experiment3():
    lad = [r for r in rows(E / "explanation_barrier" / "ladder.csv") if r["explained"] == "True"]
    outcomes = Counter(r["outcome"] for r in lad)
    refuted = sum(int(r["refuted"]) for r in lad)
    stmts = sum(int(r["statements_db"]) for r in lad)  # "checkable statements" in RESULTS.md (2,274)
    weakened = sum(int(r["weakened"]) for r in lad)
    c4 = [r for r in lad if r["family"] == "c4"]
    widths = ", ".join(f"{r['arch']}: {float(r['test_auc']):.3f}" for r in sorted(c4, key=lambda r: int(r["n_params"])))
    body = [["Explanation complexity ladder (plain MLPs, known functions; exploratory)",
             f"{len(lad)} networks explained: {outcomes.get('explained', 0)} explained, "
             f"{outcomes.get('covered, not decomposed', 0)} covered but not decomposed; "
             f"verifiers refuted {refuted} of {stmts:,} statements ({100 * refuted / stmts:.2f} per 100), weakened {weakened}; "
             f"Connect-4 MLP test AUC by width {widths}",
             "The function's complexity did not break the method; it broke on width (a dense first layer wider than 26 units, "
             "the split-node letter limit), and meaning degrades smoothly with size and depth, not with interaction order."]]
    runs = rows(E / "c4_line_features" / "runs.csv")
    sel = {}
    for base_ in ("lines", "lines_cells", "perline"):
        br = [r for r in runs if r["name"].startswith(f"{base_}-") and r["name"].split("-")[0] == base_]
        sel[base_] = max(br, key=lambda r: float(r["train_auc"]))
    lv = {p.stem: levels_counts(p) for p in (E / "c4_line_features" / "levels").glob("c4lf-*.json")}
    vd = {b: verdict_counts(E / "c4_line_features" / "verification" / b / "verdict.json") for b in sel}
    parts = []
    for base_, label, inputs in (("lines", "lines", 16), ("lines_cells", "lines+cells", 58), ("perline", "per-line", 154)):
        r = sel[base_]
        key = f"c4lf-{r['name']}"
        L = lv[key]
        g, u, w, rf, fl = vd[base_]
        parts.append(f"{label} ({inputs} inputs): test AUC {float(r['test_auc']):.3f}, {L['leaves']} leaves, probe "
                     f"{L['m']} / {L['p']} / {L['c']}, verifier {g} ({rf} refuted, {w} weakened)")
    body.append(["Three Connect-4 bases (same rows, split and target; exploratory)", "; ".join(parts),
                 "Partly, and not at comparable AUC: the change of base moves what is understood from positional cell gates "
                 "to positional line terms; it does not reach tactical concepts at the cells' AUC."])
    mn = rows(E / "mnist" / "logs" / "linear_start" / "table.csv")
    by = {}
    for r in mn:
        by.setdefault(r["arm"], []).append(r)
    assert all(r["status"] == "complete" for r in mn), [r["name"] for r in mn if r["status"] != "complete"]
    mean = lambda arm, col: st.mean(float(r[col]) for r in by[arm])
    hidden = ", ".join(r["hidden"] for r in by["struct"])
    lc = jload(E / "mnist" / "baselines.json")
    lr_test = lc["mnist10"]["logistic_regression"]["test"]["acc"]
    body.append(["MNIST-10 linear check and linear start (exploratory)",
                 f"linear check: PropNEAT's backprop step equals nn.Linear per epoch; linear start, {len(mn)} of 9 runs complete, "
                 f"mean test accuracy full-start {mean('full', 'test_acc'):.4f}, LR-start {mean('lr', 'test_acc'):.4f}, "
                 f"structural {mean('struct', 'test_acc'):.4f} (hidden nodes kept: {hidden})"
                 + f"; sklearn LR {lr_test:.4f}",
                 "The training path is not broken, and a dense linear start closes most of the earlier sparse-network gap; "
                 "the structural arm did not improve on LR, but its species controller fragmented and reproduction stopped after "
                 "generations 6--7, so this does not establish a limit of structural evolution."])
    header = ["Study", "Networks and headline numbers", "Finding"]
    note = ("Sources: experiments/explanation\\_barrier/ladder.csv and RESULTS.md; experiments/c4\\_line\\_features/ "
            "(runs.csv, levels/, verification/, RESULTS.md); experiments/mnist/logs/linear\\_start/table.csv, "
            "LINEAR\\_CHECK.md and LINEAR\\_START.md. Probe counts are leaves graded meaningful / partial / closed-form only.")
    return header, body, note


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("ch7_experiment1_cases.tex", "tab:ch7-exp1-cases",
         "Experiment 1: one explained model per case against its references. The paper genomes are the AAAI paper's; "
         "every other model was selected by its study's stated rule.", experiment1, "p{3.2cm}p{1.6cm}p{1.4cm}p{3.6cm}p{3.6cm}p{3.2cm}"),
        ("ch7_experiment2_consistency.tex", "tab:ch7-exp2-consistency",
         "Experiment 2: the ten-seed study under the frozen v4 explanation method, per set.", experiment2,
         "lrrrrrp{3.0cm}p{1.6cm}r"),
        ("ch7_experiment3_complexity.tex", "tab:ch7-exp3-complexity",
         "Experiment 3: where the complexity lives. One line per study with its finding.", experiment3,
         "p{3.6cm}p{6.2cm}p{5.4cm}"),
    ]
    for fname, label, caption, fn, colspec in specs:
        header, body, note = fn()
        tex_body = [r if r == "midrule" else [tex_escape(c) for c in r] for r in body]
        (OUT / fname).write_text(tex_table(caption, label, [tex_escape(h) if "$" not in h else h for h in header],
                                           tex_body, note, colspec))
        print(f"## {fname}\n")
        print(md_table(header, body))
        print()
    print("wrote", ", ".join(s[0] for s in specs), "to", OUT)


if __name__ == "__main__":
    main()
