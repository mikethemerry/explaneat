"""Signal-by-seed tick tables (Mike's layout): seeds 0-9 across the top, a tick where the signal
appears in that seed, and a lettered footnote only where a seed realises it differently from most.

Reads the detailed signal tables written by signal_tables.py (tables/exp2/exp2-signals-<set>.tex),
so the presence and realisation codes are exactly those; writes tables/exp2/exp2-signalticks-<set>.tex
and data/signalticks.json (for the HTML preview).

    uv run python thesis/ch7/exp2/signal_ticks.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAB = HERE / "tables" / "exp2"
SETS = {"adult": "Adult", "churn": "Churn", "heart100": "Heart-100"}
LETTERS = "abcdefghijklmnopqrstuvwxyz"


import csv

ROOT = HERE.parents[2]
LIN_NOTE_OPP = "opposite direction to most seeds"
LIN_NOTE_PATTERN = "same overall direction, but its levels are weighted differently (pattern correlation below 0.9)"


def _rows(p):
    return list(csv.DictReader(open(ROOT / p)))


def linear_signals(key):
    """[(short name, [per-seed state])], state in {"ok", "opp", "pattern", None}."""
    out = []
    if key in ("churn", "adult"):
        base = {"churn": "experiments/churn_signals", "adult": "experiments/adult_signals"}[key]
        names = {r["signal"]: re.sub(r"\s*\(.*\)$", "", r["name"]) for r in _rows(f"{base}/signals.csv")}
        per = {}
        for r in _rows(f"{base}/matrix.csv"):
            if not r["signal"].startswith("lin:"):
                continue
            st = None
            if r["used"] == "True":
                c = float(r["corr_with_data"])
                st = "opp" if c <= 0 else "ok"
            per.setdefault(r["signal"], [None] * 10)[int(r["seed"])] = st
        for sig, states in per.items():
            if any(states):
                out.append((names.get(sig, sig), states))
    else:  # heart100: sign of each term's average effect on the score
        short = {"age": "age", "ca": "vessels coloured", "chol": "cholesterol", "cp=1": "typical angina",
                 "cp=2": "atypical angina", "cp=3": "non-anginal pain", "exang=1": "exercise angina",
                 "fbs=1": "fasting sugar above 120", "oldpeak": "ST depression", "restecg=1": "ECG ST-T abnormality",
                 "restecg=2": "ECG LV hypertrophy", "sex=0": "female (vs male)", "slope=2": "flat ST slope",
                 "slope=3": "downsloping ST slope", "thal=4": "thallium not recorded", "thal=6": "thallium fixed defect",
                 "thal=7": "thallium reversible defect", "thalach": "maximum heart rate", "trestbps": "resting blood pressure"}
        eff = {}
        for r in _rows("experiments/heart100_audit/linear_core.csv"):
            seed = int(r["network"].rsplit("-s", 1)[1])
            v = float(r["avg_effect"])
            eff.setdefault(r["term"], [0.0] * 10)[seed] = v
        for term, vals in eff.items():
            signs = [(1 if v > 1e-9 else -1 if v < -1e-9 else 0) for v in vals]
            nz = [x for x in signs if x]
            if not nz:
                continue
            maj = Counter(nz).most_common(1)[0][0]
            out.append((short.get(term, term), [None if x == 0 else ("ok" if x == maj else "opp") for x in signs]))
    return out


import statistics as _st
import sys as _sys

_sys.path.insert(0, str(HERE))
import per_run  # noqa: E402  (same numbers as the ten-run tables)

RUNKEY = {"adult": "adult", "churn": "churn100", "heart100": "heart100"}


def metric_rows(key):
    R = per_run.study_rows(RUNKEY[key])
    auc = lambda x: f"{x:.3f}".lstrip("0")
    spec = [("Test AUC", "test_auc", auc), ("Train AUC", "train_auc", auc),
            ("Inputs used", "inputs", str), ("Hidden nodes", "hidden", str), ("Connections", "conns", str),
            ("Skip connections", "skips", str), ("Skippiness", "skippiness", lambda x: f"{x:.2f}"),
            ("Density", "density", lambda x: f"{x:.2f}"), ("Depth", "depth", str)]
    rows = [{"group": "Model"}]
    for label, k, fmt in spec:
        v = [r[k] for r in R]
        vals = [fmt(x) for x in v]
        if k in ("test_auc", "train_auc"):  # bold the best seed (ties at the displayed precision)
            best = max(vals)
            vals = [rf"\textbf{{{x}}}" if x == best else x for x in vals]
        rows.append({"metric": True, "name": label, "vals": vals, "summary": (fmt(_st.median(v)) if fmt is not str else f"{_st.median(v):g}")})
    parts = [r["leaf"] + r["comp"] for r in R]
    rows.append({"metric": True, "name": "Parts (annotations)", "vals": [str(x) for x in parts],
                 "summary": f"{_st.median(parts):g}"})
    return rows


def linear_rows(key, notes):
    sigs = [(n, st) for n, st in linear_signals(key) if any(st)]
    always = [n for n, st in sigs if all(x == "ok" for x in st)]
    rows = [{"group": "Linear terms present in every seed, same direction"},
            {"name": ", ".join(x[:1].upper() + x[1:] if i == 0 else x[:1].lower() + x[1:] for i, x in enumerate(always)),
             "usual": None, "cells": [{"tick": True, "note": None} for _ in range(10)], "count": 10}]
    rest = [(n, st) for n, st in sigs if n not in always]
    if rest:
        rows.append({"group": "Linear terms present in some seeds only, or with a direction that differs"})
    for n, st in sorted(rest, key=lambda t: -sum(x is not None for x in t[1])):
        cells = []
        for x in st:
            if x is None:
                cells.append({"tick": False, "note": None})
            elif x == "ok":
                cells.append({"tick": True, "note": None})
            else:
                cells.append({"tick": True, "note": notes.setdefault(LIN_NOTE_OPP, LETTERS[len(notes)])})
        rows.append({"name": n[:1].upper() + n[1:], "usual": None, "cells": cells,
                     "count": sum(c["tick"] for c in cells)})
    return rows


def kind(code: str) -> str | None:
    """Map a detailed-table cell to (present?, variant type). None = absent."""
    c = code.strip()
    if not c:
        return None
    if c == r"\checkmark":
        return "std"
    if c == "w":
        return "weak"
    if c == "opp":
        return "opp"
    if c == "H":
        return "H"
    if c == "G":
        return "G"
    if re.fullmatch(r"S\d+", c):
        return "S"
    m = re.fullmatch(r"(\d*)\$?(\\pm|\+|-)\$?", c)
    if m:
        return {"\\pm": "pm", "+": "plus", "-": "minus"}[m.group(2)]
    return "std"


DESC = {
    "weak": "present but weak (0.03 to 0.10 logit)",
    "opp": "opposite direction to most seeds",
    "H": "a single hinge",
    "G": "a single gate that also reads another input",
    "S": "split across several units",
    "plus": "raises risk",
    "minus": "lowers risk",
    "pm": "raises risk for some patients, lowers it for others",
}


def parse(path: Path):
    rows, cur_group = [], None
    for line in path.read_text().splitlines():
        s = line.strip()
        m = re.match(r"\\multicolumn\{12\}\{@\{\}l\}\{\\textit\{(.*)\}\}\s*\\\\", s)
        if m:
            cur_group = m.group(1)
            rows.append({"group": cur_group})
            continue
        if s.startswith(r"\textbf{") and s.endswith(r"\\"):
            cells = [c.strip() for c in s[:-2].split(" & ")]
            if len(cells) < 12:
                continue
            rows.append({"name": cells[0], "codes": cells[1:11], "group_of": cur_group})
    return rows


def build_set(key: str):
    rows = parse(TAB / f"exp2-signals-{key}.tex")
    notes: dict[str, str] = {}  # description -> letter
    out = metric_rows(key) + linear_rows(key, notes)

    def skip(g):
        return g is not None and ("inear core" in g or "no network" in g)

    for r in rows:
        if "group" in r:
            if not skip(r["group"]):
                out.append({"group": r["group"].replace(", strongest in the data first", ", most-used first")})
            continue
        if skip(r["group_of"]):
            continue
        kinds = [kind(c) for c in r["codes"]]
        if r["group_of"] and "no network" in r["group_of"]:
            kinds = [None if k == "weak" else k for k in kinds]
        present = [k for k in kinds if k is not None and k != "weak"]
        mode = Counter(present).most_common(1)[0][0] if present else None
        cells = []
        for k in kinds:
            if k is None:
                cells.append({"tick": False, "note": None})
                continue
            note = None
            if k == "weak" or (k != mode and k != "std"):
                d = DESC.get(k)
                if d:
                    note = notes.setdefault(d, LETTERS[len(notes)])
            cells.append({"tick": True, "note": note})
        usual = DESC.get(mode) if mode not in (None, "std") else None
        if any(c["tick"] for c in cells):  # only signals found in at least one of the ten networks
            out.append({"name": r["name"], "usual": usual, "cells": cells,
                        "count": sum(c["tick"] for c in cells)})
    # within each group, most-used signals first, rarest at the bottom (stable: ties keep the data-strength order)
    grouped, cur = [], None
    for r in out:
        if "group" in r:
            cur = [r]; grouped.append(cur)
        else:
            cur.append(r)
    out = []
    for g in grouped:
        head, body = g[0], g[1:]
        if head["group"] != "Model":
            body = sorted(body, key=lambda r: (r["name"].startswith(r"\textbf{Other"), -r["count"]))
        out += [head, *body]
    # re-letter the notes in reading order so a, b, c appear top to bottom
    order = []
    for r in out:
        for c in r.get("cells", []):
            if c.get("note") and c["note"] not in order:
                order.append(c["note"])
    remap = {old: LETTERS[i] for i, old in enumerate(order)}
    for r in out:
        for c in r.get("cells", []):
            if c.get("note"):
                c["note"] = remap[c["note"]]
    legend = sorted(((remap[v], k) for k, v in notes.items() if v in remap))
    return out, legend


def to_tex(key, rows, legend):
    title = SETS[key]
    body = []
    for r in rows:
        if "group" in r:
            body.append(r"    \midrule" if body else "")
            body.append(rf"    \multicolumn{{12}}{{@{{}}l}}{{\textit{{{r['group']}}}}} \\")
            continue
        if r.get("metric"):
            body.append("    " + " & ".join([r["name"], *r["vals"], r["summary"]]) + r" \\")
            continue
        name = r["name"] + (rf" \emph{{(usually {r['usual']})}}" if r["usual"] else "")
        cells = []
        for c in r["cells"]:
            if not c["tick"]:
                cells.append("")
            else:
                cells.append(r"\checkmark" + (rf"\textsuperscript{{{c['note']}}}" if c["note"] else ""))
        body.append("    " + " & ".join([name, *cells, str(r["count"])]) + r" \\")
    notes = "; ".join(rf"\textsuperscript{{{l}}}\,{d}" for l, d in legend)
    caption = (rf"{title}: the ten networks (columns), their numbers and the signals each carries. "
               r"\checkmark: present in that seed; a letter marks a seed that realises it differently from most"
               + (rf": {notes}." if notes else "."))
    if key == "adult":
        caption += r" The positive class is an income of at most \$50K."
    seeds = " & ".join(f"s{i}" for i in range(10))
    return "\n".join([
        "% Generated by thesis/ch7/exp2/signal_ticks.py -- do not edit by hand.",
        r"\begin{sidewaystable}",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{1.05}",
        r"  \begin{adjustbox}{max totalheight=0.8\textwidth, max width=\textheight}",
        r"  \begin{tabularx}{\textheight}{@{}>{\raggedright\arraybackslash}X*{10}{c}c@{}}",
        r"    \toprule",
        rf"    & {seeds} & \makecell{{median\\ / $n$}} \\",
        *[b for b in body if b != ""],
        r"    \bottomrule",
        r"  \end{tabularx}",
        r"  \end{adjustbox}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:exp2-signalticks-{key}}}",
        r"\end{sidewaystable}",
        "",
    ])


def main():
    data = {}
    for key in SETS:
        rows, legend = build_set(key)
        tex = to_tex(key, rows, legend)
        (TAB / f"exp2-signalticks-{key}.tex").write_text(tex)
        data[key] = {"title": SETS[key], "rows": rows, "legend": legend, "tex": tex}
        print(f"wrote exp2-signalticks-{key}.tex: {sum('name' in r for r in rows)} signals, {len(legend)} note types")
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "signalticks.json").write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
