"""Write a feature dossier: ``dossier.json``, ``report.md`` and ``figures/*.png``.

See :mod:`explaneat.analysis.feature_dossier` for what is computed. Everything here only
formats and draws what the dossier holds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# validated categorical order (dataviz reference palette, light mode), never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#1f1f1e", "#5c5b55", "#8a8980", "#e6e5df"


def _color(i: int) -> str:
    return SERIES[i] if i < len(SERIES) else MUTED


def _pct(p) -> str:
    if p is None:
        return "–"
    p = float(p)
    if p < 0.01:
        return f"{p * 100:.1f}%" if p >= 0.001 else "<0.1%"
    if p > 0.99:
        return f"{p * 100:.1f}%" if p <= 0.999 else ">99.9%"
    return f"{p * 100:.0f}%"


def _rate(r) -> str:
    """An observed proportion (k/n): 0% and 100% are exact, not '<0.1%'."""
    if r is None:
        return "–"
    return f"{float(r) * 100:.0f}%"


def _num(x, d: int = 0) -> str:
    return "–" if x is None else f"{float(x):.{d}f}"


# ============================================================================ figures

def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150,
                         "savefig.bbox": "tight", "savefig.facecolor": "white"})
    return plt


def _unit_name(d: Dict[str, Any], unit: str) -> str:
    u = next((x for x in d["units"] if x["unit"] == unit), None)
    part = u.get("part") if u else None
    return f"{unit} ({part})" if part else unit


def _bend_label(d, tok: str) -> str:
    unit, on = tok[:-1], tok[-1] == "+"
    return f"{_unit_name(d, unit)} {'on' if on else 'off'}"


def _bend_phrase(d, tok: str) -> str:
    unit, on = tok[:-1], tok[-1] == "+"
    return f"unit {_unit_name(d, unit)} switches {'on' if on else 'off'}"


def figures(d: Dict[str, Any], A: Dict[str, Any], out: Path) -> Dict[str, str]:
    plt = _setup()
    out.mkdir(parents=True, exist_ok=True)
    grid, L, Prob, labels, raw = A["grid"], A["L"], A["Prob"], A["labels"], A["raw"]
    fams = d["families"]
    inp = d["input"]
    xl = f"{inp['label']} ({inp['unit']})" if inp.get("unit") else inp["label"]
    files = {}
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for e in A["events"]:
        if e.get("visible"):
            by_row.setdefault(e["row"], []).append(e)

    # 1. all ICE curves coloured by family, PD and the median-others curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, Y, ylab, pd_, med in ((axes[0], Prob, "predicted risk", A["pd"], 1 / (1 + np.exp(-A["Lmed"]))),
                                  (axes[1], L, "log-odds", None, A["Lmed"])):
        for fi in range(len(fams)):
            rows = np.nonzero(labels == fi)[0]
            for r in rows:
                ax.plot(grid, Y[r], color=_color(fi), lw=0.6, alpha=0.35)
        for fi in range(len(fams)):   # bends as small dots on the curves
            rows = np.nonzero(labels == fi)[0]
            xs = [e["x"] for r in rows for e in by_row.get(r, [])]
            ys = [np.interp(e["x"], grid, Y[r]) for r in rows for e in by_row.get(r, [])]
            ax.scatter(xs, ys, s=4, color=_color(fi), alpha=0.6, lw=0, zorder=3)
        if pd_ is not None:
            ax.plot(grid, pd_, color=INK, lw=2, label="partial dependence (mean of curves)")
        ax.plot(grid, med, color=INK, lw=2, ls="--", label="others at training median (panel curve)")
        ax.set_xlabel(xl)
        ax.set_ylabel(ylab)
        lo_, hi_ = ax.get_ylim()
        ax.plot(raw, np.full(len(raw), lo_), "|", color=INK2, ms=6, alpha=0.5)
        ax.set_ylim(lo_, hi_)
    handles = [plt.Line2D([], [], color=_color(i), lw=2, label=f"{f['family']} {f['shape_text']} (n={f['n']})")
               for i, f in enumerate(fams)]
    handles += axes[0].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(f"Every training patient's curve for {inp['label']}, coloured by shape family "
                 "(ticks: patients' own values; dots: bends)", color=INK)
    p = out / "ice_families.png"
    fig.savefig(p)
    plt.close(fig)
    files["ice_families"] = p.name

    # 2. small multiples: each family, curves shifted to start at 0 (shape only)
    n = len(fams)
    cols = min(3, n)
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws * 2, cols, figsize=(4.2 * cols, 3.6 * rws),
                             gridspec_kw={"height_ratios": [4, 1] * rws}, squeeze=False)
    for fi, f in enumerate(fams):
        ax = axes[2 * (fi // cols), fi % cols]
        axh = axes[2 * (fi // cols) + 1, fi % cols]
        rows = np.nonzero(labels == fi)[0]
        C = L[rows] - L[rows, :1]
        for c in C:
            ax.plot(grid, c, color=_color(fi), lw=0.6, alpha=0.3)
        ax.plot(grid, np.median(C, axis=0), color=INK, lw=2)
        for pc in f["pieces"][:-1]:
            ax.axvline(pc["to_median"], color=INK2, lw=0.8, ls=":")
            tok = next(iter(pc["ends_with"]), None)
            if tok:
                ax.annotate(f"{tok[:-1]} {'on' if tok[-1] == '+' else 'off'}", (pc["to_median"], ax.get_ylim()[1]),
                            rotation=90, va="top", ha="right", fontsize=7, color=INK2)
        ax.set_title(f"{f['family']} · n={f['n']} · {f['shape_text']}", fontsize=9, color=INK)
        ax.set_ylabel("Δ log-odds from start")
        ax.set_xlim(grid[0], grid[-1])
        ax.tick_params(labelbottom=False)
        axh.hist(raw[rows], bins=np.linspace(grid[0], grid[-1], 27), color=_color(fi), alpha=0.8)
        axh.set_xlim(grid[0], grid[-1])
        axh.set_ylabel("patients", fontsize=7)
        axh.set_xlabel(xl)
    for k in range(n, rws * cols):
        axes[2 * (k // cols), k % cols].axis("off")
        axes[2 * (k // cols) + 1, k % cols].axis("off")
    fig.suptitle("Shape families: member curves (thin), median shape (black), median bend positions "
                 "(unit on/off);\nbelow, where the members' own values are", color=INK)
    fig.tight_layout()
    p = out / "families.png"
    fig.savefig(p)
    plt.close(fig)
    files["families"] = p.name

    # 3. the median-others curve with the unit bends
    fig, ax = plt.subplots(figsize=(8, 3.8))
    med = 1 / (1 + np.exp(-A["Lmed"]))
    ax.plot(grid, med, color=INK, lw=2)
    for i, e in enumerate(sorted(A["med_events"], key=lambda e: e["x"])):
        y = np.interp(e["x"], grid, med)
        ax.axvline(e["x"], color=MUTED, lw=0.8, ls=":")
        ax.annotate(f"{_bend_label(d, e['unit'] + ('+' if e['on'] else '-'))} at {e['x']:.0f}",
                    (e["x"], y), xytext=(-6, 18 + 16 * i), textcoords="offset points", fontsize=7,
                    color=INK2, ha="right" if e["x"] > grid[len(grid) // 2] else "left",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
        ax.scatter([e["x"]], [y], s=18, color=INK, zorder=3)
    ax.set_xlabel(xl)
    ax.set_ylabel("predicted risk")
    ax.set_title("The panel's curve (every other input at its training median), with the unit "
                 "that bends it at each kink", fontsize=9, color=INK)
    p = out / "median_curve.png"
    fig.savefig(p)
    plt.close(fig)
    files["median_curve"] = p.name

    # 4. unit thresholds per patient, by family, over the data distribution
    units = [u for u in d["units"] if u["n_patients_switching_in_range"]]
    fig, axes = plt.subplots(len(units) + 1, 1, figsize=(8, 1.0 + 0.8 * (len(units) + 1)), sharex=True,
                             gridspec_kw={"height_ratios": [1.4] + [1] * len(units)})
    axes[0].hist(raw, bins=np.linspace(grid[0], grid[-1], 40), color=MUTED)
    axes[0].set_ylabel("patients", fontsize=7)
    axes[0].set_title(f"Where each unit switches, patient by patient (colour: family); top: the "
                      f"training patients' own {inp['label']}", fontsize=9, color=INK)
    rng = np.random.default_rng(0)
    for ax, u in zip(axes[1:], units):
        es = [e for e in A["events"] if e["unit"] == u["unit"]]
        xs = np.array([e["x"] for e in es])
        cs = [_color(int(labels[e["row"]])) for e in es]
        ax.scatter(xs, rng.uniform(-0.4, 0.4, len(xs)), s=6, c=cs, alpha=0.7, lw=0)
        ax.set_yticks([])
        ax.set_ylabel(f"{u['unit']}\n{u.get('part') or ''}\n({u['switch']})", rotation=0, ha="right",
                      va="center", fontsize=7)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel(xl)
    p = out / "unit_thresholds.png"
    fig.savefig(p)
    plt.close(fig)
    files["unit_thresholds"] = p.name

    # 5. observed outcome vs model risk by band, per family
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3), sharey=True, squeeze=False)
    for fi, f in enumerate(fams):
        ax = axes[0, fi]
        ob = f["outcome_by_band_train"]
        xs = np.arange(len(ob))
        r = [b["rate"] if b["rate"] is not None else np.nan for b in ob]
        lo_ = [b["rate"] - b["rate_ci95"][0] if b["rate"] is not None else 0 for b in ob]
        hi_ = [b["rate_ci95"][1] - b["rate"] if b["rate"] is not None else 0 for b in ob]
        ax.errorbar(xs, r, yerr=[np.clip(lo_, 0, None), np.clip(hi_, 0, None)], fmt="o", color=_color(fi), ms=4, capsize=2,
                    label="observed rate (95% CI)")
        ax.plot(xs, [b["model_mean_risk"] if b["model_mean_risk"] is not None else np.nan for b in ob],
                color=INK, lw=1.5, marker="_", label="model's mean predicted risk")
        for x, b in zip(xs, ob):
            ax.annotate(f"n={b['n']}", (x, 0.02), ha="center", fontsize=6, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels([b["band"] for b in ob], rotation=45, fontsize=6)
        ax.set_title(f"{f['family']} (n={f['n']})", fontsize=9, color=INK)
        ax.set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("heart disease / predicted risk")
    axes[0, 0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle(f"Training data by {inp['label']} band within each family", color=INK, fontsize=9)
    fig.tight_layout()
    p = out / "outcome_bands.png"
    fig.savefig(p)
    plt.close(fig)
    files["outcome_bands"] = p.name
    return files


# ============================================================================ report

def _rules_text(cond: Dict[str, Any], k: int = 2) -> List[str]:
    out = []
    for r in (cond.get("rules") or [])[:k]:
        out.append(f"{' and '.join(r['if'])} ({r['members']} of the {r['n']} patients this "
                   f"describes are in the family, {r['recall'] * 100:.0f}% of the family)")
    return out


def family_text(d: Dict[str, Any], f: Dict[str, Any]) -> List[str]:
    inp = d["input"]
    u = inp.get("unit") or ""
    L = [f"### {f['family']}: {f['shape_text']} ({f['n']} patients, {f['share'] * 100:.0f}% of training)", ""]
    rules = _rules_text(f["conditions"])
    if rules:
        L.append("**Who.** " + "; or ".join(rules) + ". " +
                 f"(One-vs-rest tree, balanced accuracy {f['conditions'].get('balanced_accuracy_train')} on training.)")
    L.append("")
    words = []
    for pc in f["pieces"]:
        end = next(iter(pc["ends_with"]), None)
        endtxt = (f"; it ends where {_bend_phrase(d, end)} "
                  f"({pc['ends_with'][end]} of {pc['n_exact']})") if end else ""
        inside = ", ".join(f"{_bend_label(d, t)} ({c})" for t, c in pc["bends_inside"].items())
        tm = pc.get("typical_member")
        risk = (f", typical member {_pct(tm['risk_from'])} → {_pct(tm['risk_to'])}" if tm else "")
        words.append(f"{pc['reads']} from about {_num(pc['from_median'])} to {_num(pc['to_median'])} {u} "
                     f"({pc['logodds_change_median']:+.2f} log-odds{risk}){endtxt}"
                     + (f"; smaller bends inside: {inside}" if inside else "")
                     + f"; {pc['members_whose_own_value_is_on_this_piece']} members have their own value on this piece")
    L.append("**Shape (the model).** " + "; then ".join(words) + ".")
    L.append("")
    if f["merged_from"]:
        L.append("Rare shapes merged in: " + ", ".join(f"{' → '.join(m['shape'])} (n={m['n']})"
                                                       for m in f["merged_from"]) + ".")
        L.append("")
    thin = [b for b in f["bends"] if b["thin"] and b["share_of_family"] >= 0.25]
    sup = []
    for b in f["bends"]:
        if b["share_of_family"] < 0.25:
            continue
        sup.append(f"{_bend_label(d, b['bend'])} at {_num(b['x_median'])} {u} (10–90%: {_num(b['x_p10'])}–"
                   f"{_num(b['x_p90'])}; {b['family_patients_with_value_in_region']} family members and "
                   f"{b['all_training_patients_in_region']} training patients have a value there"
                   + ("; **thin**" if b["thin"] else "") + ")")
    L.append("**Data support.** " + "; ".join(sup) + ".")
    if thin:
        L.append(f"Flag: {', '.join(_bend_label(d, b['bend']) for b in thin)} sits where fewer than "
                 f"{d['config']['thin_support']} of this family's patients have their values: that part of "
                 "the shape is the model extrapolating within the observed range, not something the family's "
                 "data shows.")
    L.append("")
    L.append(f"**Typical member** (the patient whose curve is closest to the family median): "
             + "; ".join(f"{k}: {v}" for k, v in f["typical_patient_values"].items()) + ".")
    L.append("")
    L.append(f"| {inp['label']} band | train n | disease | observed rate (95% CI) | model mean risk | "
             f"test n | test rate | test model risk |")
    L.append("|---|---|---|---|---|---|---|---|")
    tb = f.get("outcome_by_band_test") or [{}] * len(f["outcome_by_band_train"])
    for b, t in zip(f["outcome_by_band_train"], tb):
        ci = b["rate_ci95"]
        L.append(f"| {b['band']} | {b['n']} | {b['events']} | {_rate(b['rate'])} "
                 f"({_rate(ci[0])}–{_rate(ci[1])}) | {_pct(b['model_mean_risk'])} | {t.get('n', '–')} | "
                 f"{_rate(t.get('rate'))} | {_pct(t.get('model_mean_risk'))} |")
    L.append("")
    bc = [c for c in f["bend_data_check_train"]
          if next((b for b in f["bends"] if b["bend"] == c["bend"]), {}).get("share_of_family", 0) >= 0.25]
    if bc:
        L.append("Below / above each main bend (training, members' own values): " + "; ".join(
            f"{_bend_label(d, c['bend'])} at {_num(c['at'])}: below {c['below']['events']}/{c['below']['n']} "
            f"({_rate(c['below']['rate'])}, model {_pct(c['below']['model_mean_risk'])}), above "
            f"{c['above']['events']}/{c['above']['n']} ({_rate(c['above']['rate'])}, model "
            f"{_pct(c['above']['model_mean_risk'])})" for c in bc) + ".")
        L.append("")
    return L


def report_md(d: Dict[str, Any], files: Dict[str, str], notes: Optional[str] = None) -> str:
    inp, meta, ck = d["input"], d.get("meta") or {}, d["checks"]
    u = inp.get("unit") or ""
    L = [f"# Feature dossier: {inp['label']} (`{inp['base']}`)"
         + (f" in {meta.get('label')}" if meta.get("label") else ""), ""]
    L.append(f"Genome `{meta.get('genome_id')}`, explanation `{meta.get('explanation_id')}`, experiment "
             f"{(meta.get('experiment') or {}).get('name')}, split `{(meta.get('split') or {}).get('id')}`. "
             f"Training split ({d['data']['n_train']} patients) for everything; test numbers "
             f"({d['data']['n_test']} patients) are labelled and were never used to choose anything. "
             f"Read-only: nothing was written to the database. Built by `explaneat dossier` "
             f"(schema {d['schema_version']}).")
    L.append("")
    L.append("This describes **what the model does** with this input. It is not a claim about blood "
             "pressure and heart disease in the world.")
    L.append("")
    if notes:
        L += [notes.strip(), ""]
    L += ["## The families, in plain words", ""]
    L.append(f"Each training patient's curve is the model's predicted risk as {inp['label']} moves over "
             f"its observed training range ({_num(inp['observed_range_train'][0])}–"
             f"{_num(inp['observed_range_train'][1])} {u}) with everything else about the patient "
             f"kept. The curves fall into {len(d['families'])} families by their shape:")
    L.append("")
    L.append("| family | n | shape | who (first rule) |")
    L.append("|---|---|---|---|")
    for f in d["families"]:
        r = (f["conditions"].get("rules") or [{}])[0]
        L.append(f"| {f['family']} | {f['n']} | {f['shape_text']} | {' and '.join(r.get('if', [])) or '–'} |")
    L.append("")
    L.append(f"![ICE curves by family]({'figures/' + files['ice_families']})")
    L.append("")
    L.append(f"![Families]({'figures/' + files['families']})")
    L.append("")
    for f in d["families"]:
        L += family_text(d, f)
    L.append(f"![Outcome by band]({'figures/' + files['outcome_bands']})")
    L.append("")
    L.append("The observed rates mix everything else that differs between bands; the model's curves hold "
             "the rest of the patient fixed. Agreement or disagreement here is a check on whether the "
             "data could show the bend, not a test of it.")
    L.append("")

    # main curve
    mc = d["median_others_curve"]
    L += ["## The panel's curve", ""]
    L.append("Every other column at its training median, as in the evidence panel's line plot. Each kink "
             "is one hidden unit switching:")
    L.append("")
    L.append(f"![Median curve]({'figures/' + files['median_curve']})")
    L.append("")
    L.append(f"| at ({u}) | unit (part) | switches | change of slope (log-odds per {u}) | risk there |")
    L.append("|---|---|---|---|---|")
    for b in mc["bends"]:
        L.append(f"| {_num(b['x'], 1)} | {_unit_name(d, b['unit'])} | {'on' if b['bend'].endswith('+') else 'off'} "
                 f"| {b['dslope']:+.4f} | {_pct(b['risk_at'])} |")
    L.append("")
    L.append("Held values: " + "; ".join(f"{k}: {v}" for k, v in mc["values_held"].items()) + ".")
    if any(v == "none" for v in mc["values_held"].values()):
        L.append("")
        L.append("Note: the training median of a one-hot column is 0 unless one level holds more than half "
                 "the patients, so on this curve the categories marked 'none' have **no level at all** "
                 "(all their columns 0). The panel's curve is for a patient who does not exist; the "
                 "families above are the curves of real patients.")
    L.append("")

    # units
    L += ["## Units the input reaches", ""]
    L.append(f"Copies of `{inp['base']}` varied: {', '.join(inp['copies_varied'])}"
             + (f" (held: {', '.join(inp['copies_held'])})" if inp["copies_held"] else "") + ". "
             "A unit's threshold is where its pre-activation is zero; it depends on the patient's other "
             "inputs, and the formula changes with the on/off state of the units upstream of it. "
             "Formulas are in real units; [x = level] is 1 when true, 0 otherwise.")
    L.append("")
    L.append(f"![Unit thresholds](figures/{files['unit_thresholds']})")
    L.append("")
    L.append("| unit | part (explanation's name) | reached | switches in range | on throughout | off throughout | "
             "threshold 10–50–90% | bend (log-odds over the range) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for un in d["units"]:
        tq = un["threshold_quantiles"] or {}
        bl = un["bend_logodds_over_range_median"]
        bend = "–" if bl is None else f"{bl:+.2f}"
        reach = ", ".join(f"{r['copy']} (w {r['weight']})" for r in un["reached_directly_from"]) or "–"
        if un["reached_through"]:
            reach += "; via " + ", ".join(un["reached_through"])
        L.append(f"| {un['unit']} | {un.get('part') or '–'} | {reach} | {un['n_patients_switching_in_range']} "
                 f"({un['switch'] or '–'}) | {un['n_on_throughout']} | {un['n_off_throughout']} | "
                 f"{(_num(tq.get('p10')) + '–' + _num(tq.get('median')) + '–' + _num(tq.get('p90'))) if tq else '–'} | "
                 f"{bend} |")
    L.append("")
    for un in d["units"]:
        if not un["patterns"]:
            continue
        hyp = (un["parts"][0].get("hypothesis_first_sentence") if un["parts"] else None)
        L.append(f"**Unit {un['unit']}** ({un.get('part') or 'no part'}"
                 + (f": \"{hyp}\"" if hyp else "") + ")")
        L.append("")
        for p in un["patterns"]:
            if not p["formula"]:
                continue
            conds = [f"{c['unit']} {'on' if c['on'] else 'off'} ({c['holds']})"
                     for c in p["upstream_conditions"] if c["holds"] != "always"]
            L.append(f"- {p['n_crossings']} crossings"
                     + (f" when {'; '.join(conds)}" if conds else "")
                     + f": `{p['formula']['text']}`; the unit is on {p['formula']['unit_on']} "
                     f"(formula vs located crossings: max error {p['max_abs_err_vs_crossings']}).")
        L.append("")

    # methods & checks
    L += ["## Method and checks", ""]
    cfg = d["config"]
    L.append(f"- Curves: {cfg['grid_points']} grid points over the observed training range, all copies of "
             f"the input set together, through the model state (the explanation's operations replayed on the "
             f"phenotype; numpy evaluator `StateEvaluator`). The panel's curve is re-run through "
             f"`StructureNetwork` with `override_output_activation(model_output_activation)`: max abs "
             f"difference {ck['structure_network_median_curve_max_abs_err']}.")
    L.append(f"- Bends: {ck['n_crossings']} unit crossings located on the training curves (grid then "
             f"{cfg['bisect_iters']} bisection steps); every kink of every curve is one of them "
             f"({ck['kinks_unattributed']} unattributed). Threshold formulas reproduce the located "
             f"crossings to {ck['threshold_formula_max_abs_err']} {u}.")
    L.append(f"- A bend counts for the shape if it changes the slope by at least {cfg['min_bend']} log-odds "
             f"over the whole range; a piece reads as flat if its slope would move the log-odds less than "
             f"{cfg['flat_logodds']} over the whole range; pieces shorter than {cfg['min_segment'] * 100:.0f}% "
             f"of the range join a neighbour. Shapes with fewer than {cfg['min_family']} patients are merged "
             f"into the nearest common shape (edit distance), at most {cfg['max_families']} families.")
    km = d["kmeans_check"]
    ft = d["family_tree"]
    L.append(f"- Cross-check: k-means on the curves' slope profiles (k = {km.get('k')}) agrees with the "
             f"shape families at adjusted Rand {km.get('adjusted_rand_vs_families')}.")
    L.append(f"- One multi-class tree for all families ({ft.get('method')}): accuracy {ft.get('accuracy_train')} "
             f"on training, {ft.get('accuracy_cv')} in {ft.get('cv_folds')}-fold cross-validation. The per-family "
             "rules above are one-vs-rest trees. Membership is exactly determined by the unit thresholds "
             "(each a linear formula of the other inputs, above), so the trees are readable approximations.")
    L.append("- Test patients are assigned to families by the same shape rule from their own curves "
             "(nearest shape if theirs is not one of the families); their rates are a labelled check only.")
    L.append("")
    return "\n".join(L)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def write_dossier(d: Dict[str, Any], out_dir, notes: Optional[str] = None,
                  with_figures: bool = True) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    A = d.pop("_arrays", None)
    files = {}
    if with_figures and A is not None:
        files = figures(d, A, out / "figures")
    d["figures"] = {k: f"figures/{v}" for k, v in files.items()}
    (out / "dossier.json").write_text(json.dumps(_jsonable(d), indent=1, ensure_ascii=False))
    if files:
        (out / "report.md").write_text(report_md(d, files, notes))
    return {"dossier": str(out / "dossier.json"), "report": str(out / "report.md") if files else None,
            **{k: str(out / "figures" / v) for k, v in files.items()}}
