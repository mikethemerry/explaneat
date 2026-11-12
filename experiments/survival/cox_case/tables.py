"""Tables for the Cox survival case: ``tables.md`` here and the Experiment 1 row file
``thesis_exports/ch7/ch7_cox_case_table.tex`` (a new file; nothing else in that directory is
touched).

    uv run python experiments/survival/cox_case/tables.py

Inputs: runs.csv, selected.json, references.json (run_case.py, references.py),
explanations/<label>.json (verify_record.py), levels/<label>.json (levels.py),
verdicts/<label>/verdict.json (the verifier agents).
"""

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEX = ROOT / "thesis_exports" / "ch7" / "ch7_cox_case_table.tex"

NAMES = {"gbsg2": "GBSG2", "whas500": "WHAS500"}
MODELS = [("propneat_cox", "PropNEAT-Cox (selected seed)"), ("ridge_cox", "Ridge Cox (alpha on validation)"),
          ("cox_ph", "Cox PH, unpenalised"), ("rsf", "Random survival forest")]
LABELS = {"gbsg2": "cox-gbsg2-s1", "whas500": "cox-whas500-s0"}


def f(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def sci(x):
    if x is None:
        return "--"
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    runs = list(csv.DictReader((HERE / "runs.csv").open()))
    selected = json.loads((HERE / "selected.json").read_text())
    refs = json.loads((HERE / "references.json").read_text())
    md = []

    # ---- per-seed table
    md.append("### Per seed (test fold; selection used the validation Cox loss only)\n")
    md.append("| Dataset | Seed | Val Cox loss | Train Cox loss (null) | Harrell C | Uno C | td-AUC 1y / 3y / 5y | IBS | Calibration E/O (horizon) | Hidden | k | Inputs used | Events / param | Wall s |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        ds = r["dataset"]
        sel = "**" if int(r["seed"]) == selected[ds]["seed"] else ""
        h = int(float(r["cal_horizon"]))
        md.append(f"| {NAMES[ds]} | {sel}{r['seed']}{sel} | {float(r['val_cox_loss']):.4f} | {float(r['train_cox_loss']):.3f} ({float(r['null_cox_loss_train']):.3f}) "
                  f"| {float(r['harrell_c']):.3f} | {float(r['uno_c']):.3f} | {float(r['td_auc_1y']):.3f} / {float(r['td_auc_3y']):.3f} / {float(r['td_auc_5y']):.3f} "
                  f"| {float(r['ibs']):.3f} | {float(r['cal_eo']):.3f} ({h} d) | {r['hidden']} | {r['k']} | {r['inputs_used']} | {r['events_per_parameter']} | {r['wall_s']} |")
    md.append("\nThe selected seed is in bold. Val / train Cox loss: Efron partial likelihood per event on that fold; null = the constant model on the training fold.\n")

    # ---- references table
    md.append("### References on the same split (test fold)\n")
    md.append("| Dataset | Model | Params | Harrell C | Uno C | td-AUC 1y / 3y / 5y | IBS | Brier at horizon | Calibration E/O (horizon) | Max group gap | Test Cox loss |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, dd in refs.items():
        cal_h = int(dd["calibration_horizon"])
        for key, name in MODELS:
            m = dd["models"][key]
            auc = dict(zip([int(t) for t in m["td_auc"]["times"]], m["td_auc"]["auc"]))
            br = dict(zip([int(t) for t in m.get("brier", {}).get("times", [])], m.get("brier", {}).get("brier", [])))
            cal = m.get("calibration", {})
            k = m["genome"]["complexity_k"] if key == "propneat_cox" else (m.get("n_params") or (f"{m['params']['n_estimators']} trees" if key == "rsf" else "--"))
            md.append(f"| {NAMES[ds]} | {name} | {k} | {f(m['harrell_c'])} | {f(m['uno_c'])} | {f(auc.get(365))} / {f(auc.get(1095))} / {f(auc.get(1825))} "
                      f"| {f(m['ibs'])} | {f(br.get(cal_h))} | {f(cal.get('expected_over_observed'))} ({cal_h} d) | {f(cal.get('max_abs_group_diff'))} | {f(m.get('test_cox_loss'))} |")
    md.append("")

    # ---- explanation / levels / verifier
    md.append("### Explanation, the three levels and the verifier\n")
    md.append("| Dataset | Label | Leaves | Compositions | Identity nodes | Structural / compositional coverage | Faithful: max abs err | Closed-form err | Granularity | Understood: meaningful / partial / closed-form only | Recheck mismatch / needs review | Verifier |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows_tex = []
    for ds in refs:
        label = LABELS[ds]
        rec = load_json(HERE / "explanations" / f"{label}.json") or {}
        lv = (load_json(HERE / "levels" / f"{label}.json") or {}).get("levels") or {}
        vd = load_json(HERE / "verdicts" / label / "verdict.json") or {}
        v = rec.get("verified_by_orchestrator") or {}
        rc = v.get("recheck_summary") or rec.get("recheck_summary") or {}
        und = lv.get("understood") or {}
        c = und.get("counts") or {}
        dec = lv.get("decomposed") or {}
        fa = lv.get("faithful") or {}
        nf = len(vd.get("findings") or [])
        nref = sum(1 for x in vd.get("findings") or [] if x.get("grade") == "refuted")
        nweak = sum(1 for x in vd.get("findings") or [] if x.get("grade") == "weakened")
        verdict = f"{(vd.get('overall') or {}).get('grade', '--')} ({nf} findings: {nref} refuted, {nweak} weakened)" if vd else "--"
        md.append(f"| {NAMES[ds]} | {label} | {rec.get('n_leaf_annotations', '--')} | {rec.get('n_composition_annotations', '--')} | {rec.get('n_identity_nodes', '--')} "
                  f"| {f(v.get('structural_coverage'), 2)} / {f(v.get('compositional_coverage'), 2)} | {sci(fa.get('function_preservation_max_abs_err'))} | {sci(v.get('closed_form_max_abs_err'))} "
                  f"| {f(dec.get('granularity'), 2)} | {c.get('meaningful', '--')} / {c.get('partial', '--')} / {c.get('closed_form_only', '--')} "
                  f"| {rc.get('mismatch', '--')} / {rc.get('needs_review', '--')} | {verdict} |")
        rows_tex.append((ds, rec, lv, vd, v, c, dec, fa, verdict))
    md.append("")
    (HERE / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- the Experiment 1 row file
    lines = [
        "% Generated by experiments/survival/cox_case/tables.py from experiments/survival/cox_case/",
        "% (runs.csv, references.json, explanations/*.json, levels/*.json, verdicts/*/verdict.json); do not edit by hand.",
        "% Requires \\usepackage{booktabs}. Rows for the Experiment 1 case table (one model per case,",
        "% explained to three levels against a reference): the Cox survival case.",
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{The Cox survival case. One PropNEAT-Cox network per dataset (the seed with the lowest validation Cox loss of five), against ridge Cox, unpenalised Cox PH and a random survival forest fitted on the same event-stratified 80/20 split (seed 42); Harrell's and Uno's C, integrated Brier score and the expected/observed risk ratio at the calibration horizon (5 years for GBSG2, 3 years for WHAS500) on the test fold. Explanation: leaves, function-preservation error (faithful), granularity (decomposed), leaves whose best library concept grades meaningful (understood), and the blind verifier's overall verdict.}",
        r"\label{tab:ch7-cox-case}",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
        r"Dataset & Model & $k$ & Harrell C & Uno C & IBS & E/O \\", r"\midrule",
    ]
    for ds, dd in refs.items():
        cal_h = int(dd["calibration_horizon"])
        for i, (key, name) in enumerate(MODELS):
            m = dd["models"][key]
            k = m["genome"]["complexity_k"] if key == "propneat_cox" else (m.get("n_params") if key != "rsf" else "--")
            cal = m.get("calibration", {})
            first = NAMES[ds] + f" ($n$ = {dd['n_train'] + dd['n_val'] + dd['n_test']})" if i == 0 else ""
            lines.append(f"{first} & {name.replace('(alpha on validation)', '').strip()} & {k} & {f(m['harrell_c'])} & {f(m['uno_c'])} & {f(m['ibs'])} & {f(cal.get('expected_over_observed'), 2)} \\\\")
        if ds != list(refs)[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\medskip", "",
              r"\begin{tabular}{@{}llrlrrrl@{}}", r"\toprule",
              r"Dataset & Explanation & Leaves & Faithful (max err.) & Granularity & Understood & Recheck mismatches & Verifier \\", r"\midrule"]
    for ds, rec, lv, vd, v, c, dec, fa, verdict in rows_tex:
        und = lv.get("understood") or {}
        uf = und.get("understood_fraction")
        rc = v.get("recheck_summary") or {}
        lines.append(f"{NAMES[ds]} & {LABELS[ds]} & {rec.get('n_leaf_annotations', '--')} & {sci(fa.get('function_preservation_max_abs_err'))} & {f(dec.get('granularity'), 2)} "
                     f"& {('--' if uf is None else f'{100 * uf:.0f}\\%')} & {rc.get('mismatch', '--')} & {(vd.get('overall') or {}).get('grade', '--')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    TEX.write_text("\n".join(lines))
    print("wrote", TEX)


if __name__ == "__main__":
    main()
