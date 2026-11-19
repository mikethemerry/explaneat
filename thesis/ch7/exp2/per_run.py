"""Experiment 2: one table per dataset with all ten runs (called by make_assets.main()).

Sources (read-only):
    runs CSVs          experiments/10seed/runs.csv (adult rows), experiments/churn/runs_churn100.csv,
                       experiments/10seed/runs_heart100.csv: AUCs and intervals, inputs, hidden nodes,
                       enabled connections, depth, skip connections
    explanations CSVs  experiments/10seed/explanations_v4.csv (adult rows),
                       experiments/churn/explanations_churn100_v4.csv,
                       experiments/10seed/explanations_heart100_v4.csv: annotations, splits, identity
                       nodes, coverage, function-preservation error, agent minutes (agent_minutes_db: computed
                       from the database's timestamped writes, experiments/10seed/explaining_time.py;
                       claude_minutes, the record's own start/end, is kept there for audit)
    v4 records         experiments/10seed/explanations_v4/<label>.json: mike_verification.minutes
    data/structure_metrics.json  ExplaNEAT.density / skippines / depth (extract_db.py, read-only DB)

Outputs: tables/exp2/exp2-runs-{adult,churn,heart100}.tex (one sidewaystable each, all ten seeds
and a median (range) line).
"""
from __future__ import annotations

import json
import statistics as st

from make_assets import DATA, EXPL, RUNS, STUDIES, TAB, jload, num, rows, thousands

SHORT = {"adult": "adult", "churn100": "churn", "heart100": "heart100"}
CAP_DATA = {
    "adult": "Adult (income; 10 seeds $\\times$ 100 generations, pre-registered)",
    "churn100": "Churn (10 seeds $\\times$ 100 generations)",
    "heart100": "Heart-100 (10 seeds $\\times$ 100 generations)",
}


def sci(x) -> str:
    """A small error as $a.b\\times10^{-k}$."""
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def med_rng(vals, fmt):
    v = [x for x in vals if x is not None]
    return f"{fmt(st.median(v))} ({fmt(min(v))}--{fmt(max(v))})"


def study_rows(k):
    p, f = RUNS[k]
    R = {int(r["seed"]): r for r in rows(p, **f)}
    p, f = EXPL[k]
    E = {int(r["seed"]): r for r in rows(p, **f)}
    M = json.loads((DATA / "structure_metrics.json").read_text())[k]
    assert sorted(R) == sorted(E) == list(range(10)), k
    out = []
    for s in range(10):
        r, e, m = R[s], E[s], M[str(s)]
        assert r["genome_id"] == e["genome_id"] == m["genome_id"], (k, s)
        assert m["layers"] == int(r["depth"]) + 1, (k, s, m["layers"], r["depth"])
        assert int(r["n_boot_used"]) == int(R[0]["n_boot_used"]), (k, s)
        rec = jload(f"experiments/10seed/explanations_v4/{e['label']}.json")
        mv = (rec.get("mike_verification") or {}).get("minutes")
        out.append({
            "seed": s, "train_auc": float(r["train_auc"]), "test_auc": float(r["test_auc"]),
            "lo": float(r["test_auc_ci_lo"]), "hi": float(r["test_auc_ci_hi"]),
            "inputs": int(r["inputs_connected"]), "n_inputs": int(r["n_inputs"]),
            "hidden": int(r["hidden_nodes"]), "conns": int(r["phenotype_connections_enabled"]),
            "skips": int(r["skip_connections"]), "skippiness": m["skippiness"], "density": m["density"],
            "depth": int(r["depth"]), "leaf": int(e["n_leaf"]), "comp": int(e["n_composition"]),
            "splits": int(e["n_splits"]), "identity": int(e["n_identity"]),
            "cov_s": float(e["structural_coverage"]), "cov_c": float(e["compositional_coverage_tool"]),
            "fp": float(e["function_preservation_max_abs_err"]), "agent": float(e["agent_minutes_db"]),
            "mike": None if mv is None else float(mv),
        })
    return out


HEAD = (r"Seed & \makecell{Train\\AUC} & \makecell{Test AUC\\(95\% CI)} & \makecell{Inputs\\used} & "
        r"\makecell{Hidden\\nodes} & \makecell{Enabled\\conn.} & \makecell{Skippiness\\(skip conn.)} & "
        r"Density & Depth & \makecell{Leaf $+$\\comp.} & Splits & \makecell{Id.\\nodes} & "
        r"\makecell{Coverage\\struct./comp.} & \makecell{Function\\pres.\ error} & \makecell{Agent\\min} & "
        r"\makecell{Mike\\min}")


def table(k, name):
    R = study_rows(k)
    n_in = {r["n_inputs"] for r in R}
    assert len(n_in) == 1
    n_in = n_in.pop()
    p, f = RUNS[k]
    nb = {int(r["n_boot_used"]) for r in rows(p, **f)}
    assert len(nb) == 1
    nb = nb.pop()
    body = ["  \\setlength{\\tabcolsep}{2.5pt}", "  \\renewcommand{\\arraystretch}{1.15}", "  \\begin{adjustbox}{max width=\\textheight}",
            "  \\begin{tabular}{@{}r" + "c" * 15 + "@{}}", "    \\toprule", f"    {HEAD} \\\\", "    \\midrule"]
    best_tr = max(num(r["train_auc"]) for r in R)  # best at the displayed precision; ties all bold
    best_te = max(num(r["test_auc"]) for r in R)
    bold = lambda x, b: f"\\textbf{{{x}}}" if x == b else x
    for r in R:
        cells = [f"s{r['seed']}", bold(num(r["train_auc"]), best_tr),
                 f"{bold(num(r['test_auc']), best_te)} {{\\scriptsize [{num(r['lo'])}, {num(r['hi'])}]}}",
                 f"{r['inputs']}/{n_in}", str(r["hidden"]), str(r["conns"]),
                 f"{num(r['skippiness'], 2)} ({r['skips']})", num(r["density"], 3), str(r["depth"]),
                 f"{r['leaf']} $+$ {r['comp']}", str(r["splits"]), str(r["identity"]),
                 f"{num(r['cov_s'], 2)}/{num(r['cov_c'], 2)}", sci(r["fp"]), num(r["agent"], 1),
                 "--" if r["mike"] is None else num(r["mike"], 0)]
        body.append("    " + " & ".join(cells) + " \\\\")
    i = lambda x: f"{x:g}"
    med = ["\\makecell[l]{Median\\\\(range)}",
           "\\makecell{" + num(st.median(r["train_auc"] for r in R)) + "\\\\{\\scriptsize "
           + f"{num(min(r['train_auc'] for r in R))}--{num(max(r['train_auc'] for r in R))}" + "}}",
           "\\makecell{" + num(st.median(r["test_auc"] for r in R)) + "\\\\{\\scriptsize "
           + f"{num(min(r['test_auc'] for r in R))}--{num(max(r['test_auc'] for r in R))}" + "}}"]
    for key, fmt in (("inputs", i), ("hidden", i), ("conns", i)):
        v = [r[key] for r in R]
        med.append(f"\\makecell{{{fmt(st.median(v))}\\\\{{\\scriptsize {fmt(min(v))}--{fmt(max(v))}}}}}")
    v = [r["skippiness"] for r in R]
    med.append(f"\\makecell{{{num(st.median(v), 2)}\\\\{{\\scriptsize {num(min(v), 2)}--{num(max(v), 2)}}}}}")
    for key, fmt in (("density", lambda x: num(x, 3)), ("depth", i)):
        v = [r[key] for r in R]
        med.append(f"\\makecell{{{fmt(st.median(v))}\\\\{{\\scriptsize {fmt(min(v))}--{fmt(max(v))}}}}}")
    v = [r["leaf"] + r["comp"] for r in R]
    med.append(f"\\makecell{{{i(st.median(v))}\\\\{{\\scriptsize {i(min(v))}--{i(max(v))}}}}}")
    for key in ("splits", "identity"):
        v = [r[key] for r in R]
        med.append(f"\\makecell{{{i(st.median(v))}\\\\{{\\scriptsize {i(min(v))}--{i(max(v))}}}}}")
    full = sum(r["cov_s"] == 1 and r["cov_c"] == 1 for r in R)
    med.append(f"{full}/10 full")
    v = [r["fp"] for r in R]
    med.append(f"$\\le$ {sci(max(v))}")
    for key, d in (("agent", 1), ("mike", 0)):
        v = [r[key] for r in R if r[key] is not None]
        med.append(f"\\makecell{{{num(st.median(v), d)}\\\\{{\\scriptsize {num(min(v), d)}--{num(max(v), d)}}}}}")
    body += ["    \\midrule", "    " + " & ".join(med) + " \\\\", "    \\bottomrule", "  \\end{tabular}", "  \\end{adjustbox}"]
    missing = [r["seed"] for r in R if r["mike"] is None]
    miss = (f" Seed{'s' if len(missing) > 1 else ''} {', '.join(f's{m}' for m in missing)} "
            f"ha{'ve' if len(missing) > 1 else 's'} no verification minutes in {'their records' if len(missing) > 1 else 'its record'}; "
            f"the median is over the other {10 - len(missing)}." if missing else "")
    cap = (f"{CAP_DATA[k]}: every run. AUCs are of the best genome of the run; the test interval is a "
           f"percentile bootstrap ({thousands(nb)} resamples). \\emph{{Inputs used}}: inputs with an enabled "
           f"path to the output, of the {n_in} model inputs (one-hot columns counted separately). "
           f"\\emph{{Hidden nodes}}, \\emph{{enabled connections}}, \\emph{{depth}} (edges on the longest "
           f"input-to-output path; layers $=$ depth $+\\,1$) and \\emph{{skip connections}} (edges that "
           f"jump a layer, direct input-to-output edges excluded) are of the phenotype. "
           f"\\emph{{Skippiness}} (mean number of layers a connection gene skips) and \\emph{{density}} "
           f"(node and connection genes over the parameters of a dense network with the same layer "
           f"widths) are the library's \\texttt{{ExplaNEAT}} measures on the genome. Explanation: "
           f"leaf $+$ composition annotations, node splits and identity nodes of the frozen-method "
           f"(v4) explanation; structural / compositional coverage; \\emph{{function-preservation "
           f"error}}, the largest absolute difference between the explained model and the network on "
           f"the training rows. \\emph{{Agent min}}: the explaining agent's minutes, from the explanation's "
           f"creation to its last timestamped write in the database (operations, evidence records, "
           f"logged queries), a lower bound on its wall-clock time; "
           f"\\emph{{Mike min}}: the author's verification minutes.{miss}")
    src = ("% Generated by thesis/ch7/exp2/per_run.py (via make_assets.py) -- do not edit by hand.\n"
           "\\begin{sidewaystable}\n  \\centering\n  \\footnotesize\n"
           + "\n".join(body) + f"\n  \\caption{{{cap}}}\n  \\label{{tab:exp2-runs-{SHORT[k]}}}\n"
           "\\end{sidewaystable}\n")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(src)


def build():
    for k, _ in STUDIES:
        table(k, f"exp2-runs-{SHORT[k]}")


if __name__ == "__main__":
    build()
