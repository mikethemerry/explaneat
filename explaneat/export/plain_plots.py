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
from typing import Any, Dict, List, Optional, Tuple

UP, DOWN, MIX = "#D95F02", "#1B9E77", "#7F7F7F"
WIDTH = 6.4          # inches: every figure, printed at the text width (fonts stay at 8 pt or more)
PER_LEVEL = 0.22     # inches per category level
MANY_LEVELS = 6      # a variable with more levels gets its own full-width figure


def _level_label(label, var_label="") -> str:
    """'state other than Alaska, Alabama, ...' -> 'any other state'."""
    label = str(label or "")
    if " other than " in label:
        return "any other " + (var_label or label.split(" other than ")[0])
    return label


def _wrap(text, width=30):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width)) or str(text)


def _grid(plt, k, row_h=2.5, per_row=2):
    """k panels, at most ``per_row`` per row, WIDTH wide."""
    import math
    rows = max(1, math.ceil(k / per_row))
    cols = min(k, per_row)
    fig, axs = plt.subplots(rows, cols, figsize=(WIDTH, row_h * rows), squeeze=False)
    flat = [a for r in axs for a in r]
    for a in flat[k:]:
        a.set_visible(False)
    return fig, flat[:k]


def labels_overlap(fig) -> bool:
    """True when any two tick labels of one axis overlap (bounding boxes, drawn figure)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for ticks in (ax.get_yticklabels(), ax.get_xticklabels()):
            boxes = [t.get_window_extent(renderer=r) for t in ticks if t.get_text() and t.get_visible()]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if boxes[i].overlaps(boxes[j]):
                        return True
    return False
INK = "#222222"
BAND = "#9ECAE1"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "explaneat",
                         "axes.formatter.useoffset": False, "axes.formatter.limits": (-6, 7)})
    return plt


def _save(fig, base: Path, data: Dict[str, Any]) -> Dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:       # a flat curve (e.g. a probability that stays below 1e-50): a readable axis
        lo, hi = ax.get_ylim()
        if ax.get_label() != "<colorbar>" and hi - lo < 1e-9 * max(1.0, abs(hi)):
            ax.set_ylim(min(0.0, lo), max(1.0, hi))
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
    return ("Predicted chance" if "chance" in pred else "Predicted probability" if "probability" in pred
            else "Predicted risk")


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
        lab = _level_label(to.get("label") or "", v.get("label") or "")
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
    fig, ax = plt.subplots(figsize=(WIDTH, PER_LEVEL * 1.2 * len(rows) + 0.9))
    ax.barh(range(len(rows)), d, color=col, height=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color=INK, lw=0.6)
    span = max([abs(x) for x in d] + [0.0])
    if span < 1.0:        # changes under one point: show them on a +-1 point axis, not 1e-7
        ax.set_xlim(-1.0, 1.0)
        ax.text(0.99, 0.01, "all changes under 1 point" if span < 0.5 else "", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#555555")
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
    fig, ax = plt.subplots(figsize=(WIDTH, PER_LEVEL * len(rows) + 0.9))
    for i, (xi, s) in enumerate(zip(x, small)):
        ax.plot([_pc(ref), xi], [i, i], color="#cccccc", lw=0.8)
        ax.plot(xi, i, "o", ms=4, color=MIX if s else (UP if xi > _pc(ref) else DOWN))
    ax.axvline(_pc(ref), color=INK, lw=0.6, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab} (n = {k})" for lab, k in zip(labels, n)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    _pct_axis(ax, x + [_pc(ref)])
    ax.set_xlabel(_risk_label(layer) + f" for the typical case, by {glabel} (%)")
    data = {"kind": "levels", "variable": gbase, "claims": [c["id"] for c in rows], "labels": labels, "n": n,
            "risk": [c["risk"]["to"] for c in rows], "reference_risk": ref, "grey": small}
    return {**_save(fig, base, data), "data": data}


def response_curves(layer, base: Path, claims) -> Optional[Dict[str, Any]]:
    plt = _plt()
    claims = [c for c in claims if _numeric(c)][:4]
    if not claims:
        return None
    fig, axs = _grid(plt, len(claims))
    data = {"kind": "response", "panels": []}
    for ax, c in zip(axs, claims):
        sh = c["shape"]
        x = sh["grid"]
        y = [_pc(r) for r in sh["typical_risk"]]
        if sh.get("band_risk_p10") and sh.get("band_risk_p90"):
            ax.fill_between(x, [_pc(r) for r in sh["band_risk_p10"]], [_pc(r) for r in sh["band_risk_p90"]],
                            color=BAND, alpha=0.5, lw=0, label="10th–90th percentile of "
                            + ((layer.get("population") or {}).get("plural") or "patients"))
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
            hi_x = max(x)
            shown = [r for r in rug if r <= hi_x * 1.05 + 1e-12]
            ax.plot(shown, [y0] * len(shown), "|", color="#555555", ms=5, mew=0.5)
            beyond = len(rug) - len(shown)
            if beyond:       # a long tail (e.g. a capped value) beyond the curve: say so, keep the axis
                ax.text(0.99, 0.02, f"{beyond}% of values above, up to {max(rug):,.6g}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8, color="#555555")
                ax.set_xlim(right=hi_x + 0.03 * (hi_x - min(x)))
        ax.set_xlabel(_var_axis(c))
        ax.set_ylabel(_risk_label(layer) + " (%)")
        data["panels"].append({"claim": c["id"], "x": x, "typical_risk": sh["typical_risk"],
                               "band_p10": sh.get("band_risk_p10"), "band_p90": sh.get("band_risk_p90"),
                               "point": at, "spread": (pt or {}).get("spread_p10_p90"), "rug": rug})
    axs[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    return {**_save(fig, base, data), "data": data}


def interaction_curves(layer, base: Path, claims) -> Optional[Dict[str, Any]]:
    plt = _plt()
    claims = [c for c in claims if c["type"] == "interaction" and c.get("verified")
              and (c.get("shape") or {}).get("groups")][:3]
    if not claims:
        return None
    fig, axs = _grid(plt, len(claims))
    data = {"kind": "interaction", "panels": []}
    for ax, c in zip(axs, claims):
        pan = {"claim": c["id"], "groups": []}
        for i, g in enumerate(c["shape"]["groups"]):
            y = [_pc(r) for r in g.get("typical_risk") or []]
            ax.plot(g["grid"], y, lw=2 if i == 0 else 1.2, color=[INK, "#888888", "#bbbbbb"][min(i, 2)],
                    ls="-" if i == 0 else "--", label=_wrap(g.get("group"), 34))
            pan["groups"].append({"group": g.get("group"), "x": g["grid"], "typical_risk": g.get("typical_risk")})
        ax.set_xlabel(_var_axis(c))
        ax.set_ylabel(_risk_label(layer) + " (%)")
        ax.legend(frameon=False, loc="best")
        data["panels"].append(pan)
    fig.tight_layout()
    return {**_save(fig, base, data), "data": data}


def _cat_rows(claims):
    seen, rows = set(), []
    for c in claims:
        sh = c.get("shape") or {}
        b = (c.get("variables") or [{}])[0].get("base")
        if sh.get("class") == "categorical" and b not in seen and c.get("verified") and c["type"] == "direction":
            seen.add(b)
            rows.append(c)
    return rows


def _pct_axis(ax, xs):
    """A percentage axis inside 0-100 and at least 2 points wide around the plotted values."""
    if not xs:
        return
    lo, hi = min(xs), max(xs)
    pad = max(0.08 * (hi - lo), 1.0)
    a, b = max(0.0, lo - pad), min(100.0, hi + pad)
    if b - a < 2.0:
        a, b = (max(0.0, (lo + hi) / 2 - 1.0), min(100.0, (lo + hi) / 2 + 1.0))
    ax.set_xlim(a, b)


def category_figure(layer, rows, per_row=3):
    """Dot charts of category levels; height grows with the number of levels."""
    plt = _plt()
    k = len(rows)
    nmax = max(len(c["shape"].get("levels") or []) for c in rows)
    per_row = 1 if nmax > MANY_LEVELS else min(per_row, k)
    import math
    nrows = math.ceil(k / per_row)
    h = max(1.6, PER_LEVEL * nmax + 0.9)
    fig, axs = plt.subplots(nrows, per_row, figsize=(WIDTH, h * nrows), squeeze=False)
    flat = [x for r in axs for x in r]
    for x in flat[k:]:
        x.set_visible(False)
    data = {"kind": "categories", "panels": []}
    for ax, c in zip(flat, rows):
        lv = sorted(c["shape"].get("levels") or [], key=lambda x: x.get("risk") or 0)
        ref = next((x.get("risk") for x in lv if x.get("reference")), None)
        for i, x in enumerate(lv):
            r = _pc(x.get("risk") or 0)
            col = INK if x.get("reference") else (UP if ref is not None and x.get("risk", 0) > ref else DOWN)
            if (x.get("n_train") or 0) < 20:
                col = MIX
            ax.plot(r, i, "o", ms=5, color=col)
        ax.set_yticks(range(len(lv)))
        width = 40 if per_row == 1 else 18
        vlab = ((c.get("variables") or [{}])[0].get("label") or "")
        ax.set_yticklabels([_wrap(f"{_level_label(x.get('label'), vlab)} (n = {x.get('n_train')})", width) for x in lv])
        ax.set_ylim(-0.6, len(lv) - 0.4)
        _pct_axis(ax, [_pc(x.get("risk") or 0) for x in lv])
        ax.set_xlabel(_risk_label(layer) + " (%)")
        data["panels"].append({"claim": c["id"], "levels": [{"label": x.get("label"), "risk": x.get("risk"),
                                                              "n": x.get("n_train"), "reference": x.get("reference")}
                                                             for x in lv]})
    fig.tight_layout()
    return fig, data


def category_plots(layer, base: Path, claims) -> List[Dict[str, Any]]:
    """Small variables (at most 6 levels) together, at most 3 per row; each many-level variable
    in its own full-width figure."""
    rows = _cat_rows(claims)
    small = [c for c in rows if len(c["shape"].get("levels") or []) <= MANY_LEVELS]
    big = [c for c in rows if len(c["shape"].get("levels") or []) > MANY_LEVELS]
    out = []
    if small:
        fig, data = category_figure(layer, small[:6])
        out.append({**_save(fig, Path(str(base) + "-small"), data), "data": data})
    for c in big:
        fig, data = category_figure(layer, [c])
        out.append({**_save(fig, Path(str(base) + "-" + c["id"]), data), "data": data})
    return out


def _hinge_of(xs, ys) -> Optional[float]:
    """Where a clipped curve switches on: the first off-to-on (or last on-to-off) change on the
    grid, refined by extending the straight on-side segment back to zero."""
    for i in range(len(xs) - 1):
        a, b = ys[i] > 1e-9, ys[i + 1] > 1e-9
        if a == b:
            continue
        j, k = (i + 1, i + 2) if b else (i, i - 1)      # the two on-side points next to the change
        if 0 <= k < len(xs) and ys[j] != ys[k]:
            x0 = xs[j] - ys[j] * (xs[k] - xs[j]) / (ys[k] - ys[j])
            if min(xs[i], xs[i + 1]) - 1e-12 <= x0 <= max(xs[i], xs[i + 1]) + 1e-12:
                return float(x0)
        return float(xs[j])
    return None


def _titles_of(layer) -> Dict[str, str]:
    try:
        from .render_structure import _titles
        return _titles(layer)
    except Exception:
        return {}


def partial_plot(layer, base: Path, rec, labels) -> Optional[Dict[str, Any]]:
    """A selected node's own output (its activation; other inputs at the typical case): a 1D
    curve with a band across individuals, the hinge and a rug; a heatmap over two interacting
    inputs with the data overlaid and slices; each incoming non-linear unit's own curve; or, for
    category inputs, the output for each combination of levels."""
    plt = _plt()
    name = labels.get(rec["node"], "unit")
    ylab = f"{name} output"
    data = {"kind": "partial", "node": rec["node"], "label": name, "why": rec.get("why")}
    if rec.get("map"):
        mp = rec["map"]
        units = rec.get("units") or []
        k = 2 + len(units)
        fig, axs = _grid(plt, k, row_h=2.6)
        ax = axs[0]
        cs = ax.contourf(mp["x"], mp["y"], mp["z"], levels=12, cmap="Greys")
        ax.contour(mp["x"], mp["y"], mp["z"], levels=[1e-9], colors=[UP], linewidths=1.2)
        ax.plot(mp["points_x"], mp["points_y"], ".", ms=1.5, color=UP, alpha=0.35)
        i1, i2 = rec["inputs"][0], rec["inputs"][1]
        ax.set_xlim(min(mp["x"]), max(mp["x"]))
        ax.set_ylim(min(mp["y"]), max(mp["y"]))
        ax.set_xlabel(_wrap(_ax(i1), 40))
        ax.set_ylabel(_wrap(_ax(i2), 28))
        fig.colorbar(cs, ax=ax, label=ylab)
        ax = axs[1]
        for y, sl, ls in zip(mp["slices_at"], mp["slices"], ["--", "-", ":"]):
            ax.plot(mp["x"], sl, ls, color=INK, lw=1.4, label=f"{y:.3g}")
        ax.set_xlabel(_ax(i1))
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, title=_wrap(i2["label"], 40), title_fontsize=8)
        for ax, u in zip(axs[2:], units):
            ax.plot(u["grid"], u["typical"], color=INK, lw=1.6)
            ax.set_xlabel(_ax(u))
            ax.set_ylabel(f"{labels.get(u['node'], 'unit')} output (incoming unit)")
        data.update({"map": mp, "units": units, "inputs": [i1["variable"], i2["variable"]]})
    elif rec.get("inputs"):
        i1 = rec["inputs"][0]
        fig, axs = _grid(plt, 1 + len(rec.get("units") or []), row_h=2.6)
        ax = axs[0]
        ax.fill_between(i1["grid"], i1["band_p10"], i1["band_p90"], color=BAND, alpha=0.5, lw=0,
                        label="10th–90th percentile across individuals")
        ax.plot(i1["grid"], i1["typical"], color=INK, lw=2, label="typical case")
        hinge = _hinge_of(i1["grid"], i1["typical"])
        if hinge is not None:
            ax.axvline(hinge, color=UP, lw=1, ls="--")
            data["hinge"] = hinge
        y0 = ax.get_ylim()[0]
        ax.plot(i1["rug"], [y0] * len(i1["rug"]), "|", color="#555555", ms=5, mew=0.5)
        ax.set_xlabel(_ax(i1))
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, loc="best")
        for ax, u in zip(axs[1:], rec.get("units") or []):
            ax.plot(u["grid"], u["typical"], color=INK, lw=1.6)
            ax.set_xlabel(_ax(u))
            ax.set_ylabel(f"{labels.get(u['node'], 'unit')} output (incoming unit)")
        data.update({"input": i1, "units": rec.get("units") or []})
    elif rec.get("categories"):
        cat = rec["categories"]
        rows = cat["rows"]
        fig, ax = plt.subplots(figsize=(WIDTH, max(1.6, PER_LEVEL * 1.6 * len(rows) + 0.8)))
        lab = [_wrap("; ".join(r["levels"]), 60) for r in rows]
        ax.barh(range(len(rows)), [r["value"] for r in rows], color=[UP if r["value"] > 1e-9 else "#bbbbbb"
                                                                     for r in rows], height=0.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(lab)
        ax.set_xlabel(ylab + " (other inputs at the typical case)")
        data.update({"categories": cat})
    else:
        return None
    fig.tight_layout()
    return {**_save(fig, base, data), "data": data}


def _ax(i):
    return f"{i['label']} ({i['unit']})" if i.get("unit") else i["label"]


def activity_plot(layer, base: Path, units, labels) -> Optional[Dict[str, Any]]:
    plt = _plt()
    units = [u for u in units if u.get("activity")][:4]
    if not units:
        return None
    k = min(len(units), 2)
    units = units[:2]
    scatter = any(u["activity"].get("main_input") for u in units)
    if scatter:
        fig, axs = plt.subplots(2, k, figsize=(WIDTH, 3.6), squeeze=False,
                                gridspec_kw={"height_ratios": [2, 1]})
    else:      # no numeric input to plot against: how often each unit is active, only
        fig, axs1 = plt.subplots(1, k, figsize=(WIDTH, 1.8), squeeze=False)
        axs = [[None] * k, axs1[0]]
    data = {"kind": "activity", "units": []}
    for j, u in enumerate(units):
        a = u["activity"]
        ax = axs[0][j]
        mi = a.get("main_input")
        if ax is None:
            pass
        elif mi:
            ax.plot(mi["x"], mi["y"], ".", ms=2.5, color=INK, alpha=0.5)
            ax.set_xlabel(f"{mi['label']}" + (f" ({mi['unit']})" if mi.get("unit") else ""))
        else:
            ax.text(0.5, 0.5, "no single numeric input", ha="center", va="center", transform=ax.transAxes,
                    color="#777777")
            ax.set_xticks([])
        if ax is not None:
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


def _boundary_to_axes(b, xg, yg):
    """The traced boundary with its first and last segments extended straight to the edge of the
    plotted rectangle (the map's grid limits), so the line runs to the axes."""
    import numpy as np
    b = np.asarray(b, dtype=float)
    if len(b) < 2:
        return b
    x0, x1, y0, y1 = float(min(xg)), float(max(xg)), float(min(yg)), float(max(yg))

    def ray(p, q):                           # from p through q, on to the edge of the rectangle
        d = q - p
        ts = [((hi if d[k] > 0 else lo) - q[k]) / d[k]
              for k, (lo, hi) in enumerate(((x0, x1), (y0, y1))) if abs(d[k]) > 1e-12]
        t = max(0.0, min(ts)) if ts else 0.0
        return q + t * d
    head = ray(b[1], b[0])
    tail = ray(b[-2], b[-1])
    return np.vstack([head, b, tail])


def _map_panel(ax, mp, field, cmap, outcome_word, show_points=True, show_boundary=True):
    import numpy as np
    Z = np.array(mp[field], dtype=float)
    off = np.array(mp["off"], dtype=bool)
    ax.set_facecolor("#e6e6e6")                         # flat grey: the part is switched off
    Zm = np.ma.masked_where(off, Z) if field == "z" else Z
    cs = ax.contourf(mp["x"]["grid"], mp["y"]["grid"], Zm, levels=12, cmap=cmap)
    if show_boundary and mp.get("boundary"):
        b = _boundary_to_axes(mp["boundary"], mp["x"]["grid"], mp["y"]["grid"])
        ax.plot(b[:, 0], b[:, 1], color=INK, lw=1.6)
        k = len(b) // 2
        ax.annotate("switches on", (b[k, 0], b[k, 1]), xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    if show_points:
        pt = mp["points"]
        x, y, o = np.array(pt["x"]), np.array(pt["y"]), np.array(pt["outcome"]).astype(bool)
        ax.scatter(x[~o], y[~o], s=6, facecolors="none", edgecolors="#555555", linewidths=0.4,
                   label=_wrap(outcome_word[1], 28))
        ax.scatter(x[o], y[o], s=7, color="#111111", label=_wrap(outcome_word[0], 28))
    ax.set_xlim(min(mp["x"]["grid"]), max(mp["x"]["grid"]))
    ax.set_ylim(min(mp["y"]["grid"]), max(mp["y"]["grid"]))
    ax.set_xlabel(_wrap(_ax(mp["x"]), 60))
    ax.set_ylabel(_wrap(_ax(mp["y"]), 30))
    return cs


def outcome_words(layer) -> Tuple[str, str, str]:
    """(event label, no-event label, verb phrase for 'x% of them ...') for the data overlay."""
    pop = (layer.get("population") or {}).get("plural", "")
    if "customer" in pop:
        return "left", "stayed", "left"
    short = ((layer.get("outcome") or {}).get("short") or "the event").strip()
    if short.startswith("'"):
        return short, "not " + short, "were " + short
    return short, ("not " if short.startswith(("an ", "a ")) else "no ") + short, "had " + short


def deep_dive_figures(layer, base: Path, claim) -> List[Dict[str, Any]]:
    """The deep dive of a non-linear part with two or more numeric inputs: the 2D map(s) of its own
    output (off region flat grey, the switch-on boundary, the training data: hollow = no event,
    filled = event), the slices, and the same map on the final risk scale for the typical case."""
    import numpy as np
    plt = _plt()
    maps = (claim.get("evidence") or {}).get("maps") or []
    if not maps:
        return []
    st = claim["statement"]
    word = outcome_words(layer)
    cmap = "Oranges"
    out = []
    # 1. the map(s)
    k = len(maps)
    fig, axs = plt.subplots(1, k, figsize=(WIDTH, 3.3 if k == 1 else 2.6), squeeze=False)
    css = []
    for ax, mp in zip(axs[0], maps):
        css.append(_map_panel(ax, mp, "z", cmap, word))
        if mp.get("held"):
            ax.set_title(", ".join(f"{b} at {h['text']}" for b, h in mp["held"].items()), fontsize=8)
    fig.colorbar(css[0], ax=list(axs[0]), label="part's output (score)", shrink=0.9)
    axs[0][0].legend(frameon=True, loc="upper right", fontsize=8, markerscale=1.5)
    n_on, share = st.get("n_on"), st.get("share_on")
    rate = st.get("outcome_rate_on")
    axs[0][0].text(0.02, 0.02, f"on for {n_on} ({100 * share:.0f}%); {100 * (rate or 0):.0f}% of them "
                   + word[2], transform=axs[0][0].transAxes, fontsize=8,
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    data = {"kind": "deep_map", "claim": claim["id"], "maps": [{"x": m["x"], "y": m["y"], "held": m.get("held")}
                                                                for m in maps],
            "n_on": n_on, "share_on": share, "outcome_rate_on": rate, "rule": (st.get("rule") or {}).get("text")}
    out.append({**_save(fig, Path(str(base) + "-map"), data), "data": data, "kind": "deep_map"})
    # 2. slices of the first map
    mp = maps[0]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0), squeeze=False)
    for sl, ls in zip(mp["slices_x"], ["--", "-", ":"]):
        axs[0][0].plot(mp["x"]["grid"], sl["z"], ls, color=INK, lw=1.4, label=f"{sl['at']:.3g} ({sl['q']}th pct.)")
    axs[0][0].set_xlabel(_ax(mp["x"]))
    axs[0][0].set_ylabel("part's output (score)")
    axs[0][0].legend(frameon=False, fontsize=8, title=_wrap(mp["y"]["label"], 40), title_fontsize=8,
                     loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    for sl, ls in zip(mp["slices_y"], ["--", "-", ":"]):
        axs[0][1].plot(mp["y"]["grid"], sl["z"], ls, color=INK, lw=1.4, label=f"{sl['at']:.3g} ({sl['q']}th pct.)")
    axs[0][1].set_xlabel(_ax(mp["y"]))
    axs[0][1].set_ylabel("part's output (score)")
    axs[0][1].legend(frameon=False, fontsize=8, title=_wrap(mp["x"]["label"], 40), title_fontsize=8,
                     loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    fig.tight_layout()
    d2 = {"kind": "deep_slices", "claim": claim["id"], "slices_x": mp["slices_x"], "slices_y": mp["slices_y"]}
    out.append({**_save(fig, Path(str(base) + "-slices"), d2), "data": d2, "kind": "deep_slices"})
    # 3. the same map on the final risk scale (typical case)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    R = np.array(mp["risk"], dtype=float) * 100
    cs = ax.contourf(mp["x"]["grid"], mp["y"]["grid"], R, levels=12, cmap="Greys")
    if mp.get("boundary"):
        b = _boundary_to_axes(mp["boundary"], mp["x"]["grid"], mp["y"]["grid"])
        ax.plot(b[:, 0], b[:, 1], color=UP, lw=1.6)
    ax.set_xlim(min(mp["x"]["grid"]), max(mp["x"]["grid"]))
    ax.set_ylim(min(mp["y"]["grid"]), max(mp["y"]["grid"]))
    ax.set_xlabel(_wrap(_ax(mp["x"]), 60))
    ax.set_ylabel(_wrap(_ax(mp["y"]), 30))
    fig.colorbar(cs, ax=ax, label=_risk_label(layer) + " for the typical case (%)")
    d3 = {"kind": "deep_risk", "claim": claim["id"], "risk": mp["risk"]}
    out.append({**_save(fig, Path(str(base) + "-risk"), d3), "data": d3, "kind": "deep_risk"})
    return out


def partial_caption(rec, labels, title: Optional[str] = None) -> str:
    name = labels.get(rec["node"], "the unit")
    why = rec.get("why") or ""
    if why.startswith("output of the non-linear part"):
        why = "the output of " + (f"'{title}'" if title else "this non-linear part")
    base = (f"This part's own output, unit {name} ({why}), on its own scale (its activation), with "
            f"every other input at the typical case; it is active for {rec.get('active_n')} of {rec.get('n')}. ")
    if rec.get("map"):
        return base + ("Left: its output over its two main inputs (orange line: where it switches on; dots: the "
                       "data); right: slices at three values of the second input"
                       + ("; then each incoming non-linear unit against its own main input." if rec.get("units") else "."))
    if rec.get("inputs"):
        return base + "Bold: the typical case; band: 10th to 90th percentile across individuals; dashed: where it switches on; ticks: the data."
    return base + "Its output for each combination of the levels it reads."


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
            for f in category_plots(layer, out / f"{pid}-categories", mine):
                lst.append({**f, "kind": "categories", "caption": "The predicted value for the typical case at "
                            f"each level, through this part (n = training {who}; grey: fewer than 20)."})
            bc = next((c for c in layer["claims"] if c["type"] == "boundary" and c["annotation"]["id"] == pid
                       and (c.get("evidence") or {}).get("maps")), None)
            if bc is not None:
                three = len(bc["statement"]["variables"]) >= 3
                caps = {"deep_map": ("The part's own output over " + ("each pair of its three main inputs (the third "
                                     "held at its median, named above each panel; the points are the training rows "
                                     "whose third input is between its 40th and 60th percentile)" if three else
                                     "its two numeric inputs") + ", other inputs at the typical case: colour where it "
                                     "is on, flat grey where it is switched off, the black line where it switches "
                                     f"on; training {who} as hollow (no event) and filled (event) points, so the part "
                                     "can be seen catching the events (the predicted-chance overlay was harder to "
                                     "read)."),
                        "deep_slices": "Slices of the same map: the part's own output against each input at three "
                                       "values of the other, so the tilted boundary shows as a shifting hinge.",
                        "deep_risk": "The same map on the final scale: the predicted value for the typical case, "
                                     "with the switch-on line (orange)."}
                if not bc.get("verified"):
                    caps = {k: v + (" The switch-on line is traced on this map only; the boundary claim ("
                                    + bc["id"] + ") did not pass its check on the data, so no rule is stated.")
                            for k, v in caps.items()}
                for f in deep_dive_figures(layer, out / f"{pid}-deep", bc):
                    lst.append({**f, "caption": caps[f["kind"]]})
            for rec in [r for r in (layer["graph"].get("partials") or []) if r.get("region") == pid]:
                f = partial_plot(layer, out / f"{pid}-partial-{rec['node']}", rec, hl)
                if f:
                    lst.append({**f, "kind": "partial", "node": rec["node"],
                                "caption": partial_caption(rec, hl, _titles_of(layer).get(pid))})
            units = [n for n in layer["graph"]["nodes"] if n["kind"] == "hidden" and n.get("region") == pid]
            f = activity_plot(layer, out / f"{pid}-units", units, hl)
            if f:
                lst.append({**f, "kind": "activity", "caption": "This part's hidden units: each unit's output "
                            "against its main numeric input (top) and how often it is active (bottom)."})
        for f in lst:     # relative to the review's directory (absolute paths may be rewritten when printed)
            f["rel"] = str(Path(f["pdf"]).relative_to(Path(out_dir)))
        figs[pid] = lst
    return figs
