"""Evidence plots for the plain-language review (v5e), drawn from the layer only.

Every plotted number comes from the layer's verified claims and recorded evidence, so the plots
and the text agree:

* whole model: an effect overview (each input's change in predicted risk for the typical case,
  sorted, coloured by net direction) and, for a many-level variable, a sorted dot plot of its
  levels with n (small groups grey);
* every numeric input of a part: its response curve on the risk scale for the typical case (bold),
  the 10th-90th percentile band of individual curves, the hinge or step point with its spread
  across patients, and a rug of the observed values;
* interactions with a numeric input: one curve per group;
* categorical inputs of a part: each level's predicted risk for the typical case, with n;
* hidden units: the unit's output against its main numeric input, and how often it is active.

Clean thesis style: no titles, plain axis labels in real units, the diagrams' colours (orange:
raises, teal: lowers, grey: depends). Each figure is written as PDF and PNG with a JSON of the
plotted data next to it; output is deterministic (no timestamps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

UP, DOWN, MIX = "#D95F02", "#1B9E77", "#7F7F7F"
INK = "#222222"
BAND = "#9ECAE1"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "explaneat"})
    return plt


def _save(fig, base: Path, data: Dict[str, Any]) -> Dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base) + ".pdf", bbox_inches="tight", metadata={"CreationDate": None, "Creator": None,
                                                                    "Producer": None})
    fig.savefig(str(base) + ".png", dpi=160, bbox_inches="tight", metadata={"Software": None})
    with open(str(base) + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return {"pdf": str(base) + ".pdf", "png": str(base) + ".png", "json": str(base) + ".json"}


def _risk_label(layer) -> str:
    """Short axis label: 'Predicted risk' or 'Predicted chance'."""
    pred = ((layer.get("outcome") or {}).get("predicted") or "predicted risk").lower()
    return "Predicted chance" if "chance" in pred else "Predicted risk"


def _pc(x):
    return 100.0 * x


def _var_axis(c) -> str:
    v = (c.get("variables") or [{}])[0]
    name = v.get("short") or v.get("label") or v.get("base")
    return f"{name} ({v['unit']})" if v.get("unit") else name


def _short_change(c) -> str:
    v = (c.get("variables") or [{}])[0]
    ch = (c.get("statement") or {}).get("change") or {}
    to = (ch.get("to") or {})
    fr = (ch.get("from") or {})
    if v.get("kind") == "category":
        lab = to.get("label") or ""
        return lab if len(lab) <= 38 else lab[:36] + "..."
    return f"{v.get('short') or v.get('label')}: {fr.get('text', '')} to {to.get('text', '')}"


def _numeric(c) -> bool:
    sh = c.get("shape") or {}
    return bool(sh) and sh.get("class") != "categorical" and sh.get("typical_risk") is not None


# ----------------------------------------------------------------------------- figures
def effect_overview(layer, base: Path, groups) -> Optional[Dict[str, Any]]:
    plt = _plt()
    rows = [c for c in layer["claims"] if c.get("verified") and c["type"] == "direction"
            and c.get("scope") == "model" and (c.get("risk") or {}).get("from") is not None
            and (c.get("variables") or [{}])[0].get("base") not in groups]
    if not rows:
        return None
    rows.sort(key=lambda c: abs(c["risk"]["to"] - c["risk"]["from"]), reverse=True)
    rows = rows[:16][::-1]
    labels = [_short_change(c) for c in rows]
    d = [_pc(c["risk"]["to"] - c["risk"]["from"]) for c in rows]
    col = [UP if x > 0 else DOWN for x in d]
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(rows) + 0.8))
    ax.barh(range(len(rows)), d, color=col, height=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("Change in " + _risk_label(layer).lower() + " for the typical case\n(percentage points)")
    data = {"kind": "effect_overview", "claims": [c["id"] for c in rows], "labels": labels,
            "change_points": [round(x, 3) for x in d],
            "from": [c["risk"]["from"] for c in rows], "to": [c["risk"]["to"] for c in rows]}
    return {**_save(fig, base, data), "data": data}


def level_plot(layer, base: Path, gbase: str, glabel: str) -> Optional[Dict[str, Any]]:
    plt = _plt()
    rows = [c for c in layer["claims"] if c["type"] == "direction" and c.get("scope") == "model"
            and (c.get("variables") or [{}])[0].get("base") == gbase and (c.get("risk") or {}).get("to") is not None]
    if not rows:
        return None
    rows.sort(key=lambda c: c["risk"]["to"])
    ref = rows[0]["risk"]["from"]
    labels = [_short_change(c) for c in rows]
    n = [(c.get("groups") or [{}, {}])[-1].get("n") for c in rows]
    small = [bool(c.get("small_n")) or not c.get("verified") for c in rows]
    x = [_pc(c["risk"]["to"]) for c in rows]
    fig, ax = plt.subplots(figsize=(6.2, 0.2 * len(rows) + 0.8))
    for i, (xi, s) in enumerate(zip(x, small)):
        ax.plot([_pc(ref), xi], [i, i], color="#cccccc", lw=0.8)
        ax.plot(xi, i, "o", ms=4, color=MIX if s else (UP if xi > _pc(ref) else DOWN))
    ax.axvline(_pc(ref), color=INK, lw=0.6, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab} (n = {k})" for lab, k in zip(labels, n)], fontsize=6.5)
    ax.set_xlabel(_risk_label(layer) + f" for the typical case, by {glabel} (%)")
    data = {"kind": "levels", "variable": gbase, "claims": [c["id"] for c in rows], "labels": labels, "n": n,
            "risk": [c["risk"]["to"] for c in rows], "reference_risk": ref, "grey": small}
    return {**_save(fig, base, data), "data": data}


def response_curves(layer, base: Path, claims) -> Optional[Dict[str, Any]]:
    plt = _plt()
    claims = [c for c in claims if _numeric(c)][:4]
    if not claims:
        return None
    k = len(claims)
    fig, axs = plt.subplots(1, k, figsize=(3.1 * k, 2.3), squeeze=False)
    data = {"kind": "response", "panels": []}
    for ax, c in zip(axs[0], claims):
        sh = c["shape"]
        x = sh["grid"]
        y = [_pc(r) for r in sh["typical_risk"]]
        if sh.get("band_risk_p10") and sh.get("band_risk_p90"):
            ax.fill_between(x, [_pc(r) for r in sh["band_risk_p10"]], [_pc(r) for r in sh["band_risk_p90"]],
                            color=BAND, alpha=0.5, lw=0, label="10th–90th percentile of customers"
                            if "customer" in (layer.get("population") or {}).get("plural", "") else
                            "10th–90th percentile of patients")
        d = (c.get("statement") or {}).get("direction")
        ax.plot(x, y, color=UP if d == "raises" else DOWN, lw=2, label="typical case")
        pt = sh.get("point") if sh.get("point_reconciled") else None
        at = (pt or {}).get("value") or (sh.get("points") or {}).get("at")
        if at is not None:
            ax.axvline(at, color=INK, lw=0.7, ls="--")
            if pt and pt.get("spread_p10_p90") and None not in pt["spread_p10_p90"]:
                lo, hi = pt["spread_p10_p90"]
                ax.axvspan(lo, hi, color="#dddddd", alpha=0.5, lw=0)
        rug = sh.get("rug") or []
        if rug:
            y0 = ax.get_ylim()[0]
            ax.plot(rug, [y0] * len(rug), "|", color="#555555", ms=5, mew=0.5)
        ax.set_xlabel(_var_axis(c))
        ax.set_ylabel(_risk_label(layer) + " (%)")
        data["panels"].append({"claim": c["id"], "x": x, "typical_risk": sh["typical_risk"],
                               "band_p10": sh.get("band_risk_p10"), "band_p90": sh.get("band_risk_p90"),
                               "point": at, "spread": (pt or {}).get("spread_p10_p90"), "rug": rug})
    axs[0][0].legend(frameon=False, fontsize=6.5, loc="best")
    fig.tight_layout()
    return {**_save(fig, base, data), "data": data}


def interaction_curves(layer, base: Path, claims) -> Optional[Dict[str, Any]]:
    plt = _plt()
    claims = [c for c in claims if c["type"] == "interaction" and c.get("verified")
              and (c.get("shape") or {}).get("groups")][:3]
    if not claims:
        return None
    k = len(claims)
    fig, axs = plt.subplots(1, k, figsize=(3.1 * k, 2.3), squeeze=False)
    data = {"kind": "interaction", "panels": []}
    for ax, c in zip(axs[0], claims):
        pan = {"claim": c["id"], "groups": []}
        for i, g in enumerate(c["shape"]["groups"]):
            y = [_pc(r) for r in g.get("typical_risk") or []]
            ax.plot(g["grid"], y, lw=2 if i == 0 else 1.2, color=[INK, "#888888", "#bbbbbb"][min(i, 2)],
                    ls="-" if i == 0 else "--", label=g.get("group"))
            pan["groups"].append({"group": g.get("group"), "x": g["grid"], "typical_risk": g.get("typical_risk")})
        ax.set_xlabel(_var_axis(c))
        ax.set_ylabel(_risk_label(layer) + " (%)")
        ax.legend(frameon=False, fontsize=6, loc="best")
        data["panels"].append(pan)
    fig.tight_layout()
    return {**_save(fig, base, data), "data": data}


def category_plot(layer, base: Path, claims) -> Optional[Dict[str, Any]]:
    plt = _plt()
    seen, rows = set(), []
    for c in claims:
        sh = c.get("shape") or {}
        b = (c.get("variables") or [{}])[0].get("base")
        if sh.get("class") == "categorical" and b not in seen and c.get("verified") and c["type"] == "direction":
            seen.add(b)
            rows.append(c)
    rows = rows[:4]
    if not rows:
        return None
    k = len(rows)
    fig, axs = plt.subplots(1, k, figsize=(3.0 * k, 1.9), squeeze=False)
    data = {"kind": "categories", "panels": []}
    for ax, c in zip(axs[0], rows):
        lv = sorted(c["shape"].get("levels") or [], key=lambda x: x.get("risk") or 0)
        ref = next((x.get("risk") for x in lv if x.get("reference")), None)
        ys = list(range(len(lv)))
        for i, x in enumerate(lv):
            r = _pc(x.get("risk") or 0)
            col = INK if x.get("reference") else (UP if ref is not None and x.get("risk", 0) > ref else DOWN)
            if (x.get("n_train") or 0) < 20:
                col = MIX
            ax.plot(r, i, "o", ms=5, color=col)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{(x.get('label') or '')[:30]} (n = {x.get('n_train')})" for x in lv], fontsize=6.5)
        ax.set_xlabel(_risk_label(layer) + " (%)")
        data["panels"].append({"claim": c["id"], "levels": [{"label": x.get("label"), "risk": x.get("risk"),
                                                              "n": x.get("n_train"), "reference": x.get("reference")}
                                                             for x in lv]})
    fig.tight_layout()
    return {**_save(fig, base, data), "data": data}


def activity_plot(layer, base: Path, units, labels) -> Optional[Dict[str, Any]]:
    plt = _plt()
    units = [u for u in units if u.get("activity")][:4]
    if not units:
        return None
    k = len(units)
    fig, axs = plt.subplots(2, k, figsize=(2.6 * k, 3.4), squeeze=False,
                            gridspec_kw={"height_ratios": [2, 1]})
    data = {"kind": "activity", "units": []}
    for j, u in enumerate(units):
        a = u["activity"]
        ax = axs[0][j]
        mi = a.get("main_input")
        if mi:
            ax.plot(mi["x"], mi["y"], ".", ms=2.5, color=INK, alpha=0.5)
            ax.set_xlabel(f"{mi['label']}" + (f" ({mi['unit']})" if mi.get("unit") else ""))
        else:
            ax.text(0.5, 0.5, "no single numeric input", ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, color="#777777")
            ax.set_xticks([])
        ax.set_ylabel(f"{labels.get(u['id'], 'unit')} output")
        hx = axs[1][j]
        e = a["hist_edges"]
        hx.bar(e[:-1], a["hist"], width=[e[i + 1] - e[i] for i in range(len(e) - 1)], align="edge",
               color="#999999")
        hx.set_xlabel(f"{labels.get(u['id'], 'unit')}: active for {a['active_n']} of {a['n']}")
        hx.set_ylabel("count")
        data["units"].append({"unit": labels.get(u["id"]), "active_n": a["active_n"], "n": a["n"],
                              "hist": a["hist"], "hist_edges": e, "main_input": mi})
    fig.tight_layout()
    return {**_save(fig, base, data), "data": data}


# ----------------------------------------------------------------------------- per genome
def make_figures(layer, out_dir) -> Dict[str, List[Dict[str, Any]]]:
    """part id -> [{"pdf", "png", "json", "kind", "caption"}], written to out_dir/figures/<label>/."""
    from .render_structure import grouped_variables, hidden_labels
    from ..analysis.plain_language_store import readdown_order
    out = Path(out_dir) / "figures" / layer["label"]
    groups = {g["base"]: g for g in grouped_variables(layer).values()}
    hl = hidden_labels(layer)
    root = next(p for p in layer["parts"] if p["role"] == "root")
    figs: Dict[str, List[Dict[str, Any]]] = {}
    who = (layer.get("population") or {}).get("plural", "patients")
    for pid in readdown_order(layer):
        lst = []
        mine = [c for c in layer["claims"] if c["annotation"]["id"] == pid and c.get("verified")]
        if pid == root["id"]:
            f = effect_overview(layer, out / f"{pid}-effects", groups)
            if f:
                lst.append({**f, "kind": "effects", "caption": "Each input's change in the predicted value for the "
                            "typical case across the whole model, largest first (orange: raises it, teal: lowers it)."})
            for gb, g in groups.items():
                f = level_plot(layer, out / f"{pid}-levels-{gb}", gb, g["label"])
                if f:
                    lst.append({**f, "kind": "levels", "caption": f"Every level of {g['label']} the model reads: the "
                                f"predicted value for the typical case at that level (dashed: the reference level); "
                                f"n = training {who}; grey: few {who} or not supported."})
        else:
            direction = [c for c in mine if c["type"] in ("direction", "threshold")]
            f = response_curves(layer, out / f"{pid}-response", [c for c in direction if c["type"] == "direction"])
            if f:
                lst.append({**f, "kind": "response", "caption": "Response curves through this part: the typical case "
                            f"(bold), the 10th to 90th percentile of individual {who} (band), the fold or step point "
                            "(dashed) with its spread across them (grey), and the observed values (ticks)."})
            f = interaction_curves(layer, out / f"{pid}-interaction", mine)
            if f:
                lst.append({**f, "kind": "interaction", "caption": "How the effect differs between groups: one "
                            "curve per group, for the typical case of each group."})
            f = category_plot(layer, out / f"{pid}-categories", mine)
            if f:
                lst.append({**f, "kind": "categories", "caption": "The predicted value for the typical case at "
                            f"each level, through this part (n = training {who}; grey: fewer than 20)."})
            units = [n for n in layer["graph"]["nodes"] if n["kind"] == "hidden" and n.get("region") == pid]
            f = activity_plot(layer, out / f"{pid}-units", units, hl)
            if f:
                lst.append({**f, "kind": "activity", "caption": "This part's hidden units: each unit's output "
                            "against its main numeric input (top) and how often it is active (bottom)."})
        for f in lst:     # relative to the review's directory (absolute paths may be rewritten when printed)
            f["rel"] = str(Path(f["pdf"]).relative_to(Path(out_dir)))
        figs[pid] = lst
    return figs
