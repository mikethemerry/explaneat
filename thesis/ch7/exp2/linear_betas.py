"""Linear-term beta tables for Experiment 2: one table per dataset (the linear rows of the signal
tick tables, seeds 0-9 across the top, then the median and the logistic-regression reference) and
appendix tables with every level of the many-level categories.

Reads ``experiments/linear_betas/`` (``betas_<set>.csv``, ``summary_<set>.csv``,
``projection_<set>.csv``, ``voicemail_net_churn.csv``; written by ``experiments/linear_betas/betas.py``)
and writes ``tables/exp2/exp2-betas-<set>.tex`` and ``tables/exp2/exp2-betas-<set>-<group>.tex``.

    uv run python thesis/ch7/exp2/linear_betas.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "experiments/linear_betas"
TAB = HERE / "tables" / "exp2"
SETS = {"adult": "Adult", "churn": "Churn", "heart100": "Heart-100"}
MIN_SHARE = 0.01
TOP_K = {"lin:household": 4, "lin:work": 4, "lin:edulevel": 3, "lin:country": 3, "lin:race": 2, "lin:state": 4}
# appendix groups: (signal ids, file suffix, title)
APPENDIX = {
    "adult": [(["lin:household"], "household", "marital status and household role (each observed pair against "
                                                "married, civilian spouse, husband)"),
              (["lin:work", "lin:race"], "work", "occupation and work class (against private work class, "
                                                 "professional specialty) and race (against White)"),
              (["lin:edulevel"], "education", "education levels (offsets from the straight line in years "
                                              "of schooling)"),
              (["lin:country"], "country", "country of birth (against the United States)")],
    "churn": [(["lin:state"], "states", "state (against West Virginia, the most common state)")],
}
UNIT_TEX = {"per US\\$1,000": r"per US\$1{,}000", "per unit (none recorded)": "per unit (none recorded)",
            "per step (1--16 scale)": "per step (1--16 scale)",
            "offset from the straight line in years of schooling": "offset from the line in years"}
SHORT_UNIT = {"international plan=1": "yes vs no", "voice mail plan=1": "yes vs no", "sex=0": None,
              "fbs=1": "vs 120 mg/dL or less", "exang=1": "vs none on exercise"}
SHORT_REF = {"no chest pain (asymptomatic)": "asymptomatic", "a normal resting ECG": "normal ECG",
             "an upsloping ST segment": "upsloping", "a normal thallium scan": "normal scan"}


def short_unit(term, unit):
    if term in SHORT_UNIT and SHORT_UNIT[term]:
        return SHORT_UNIT[term]
    u = unit
    for a, b in SHORT_REF.items():
        u = u.replace(a, b)
    return u.replace("female vs male", "female vs male").replace("vs married (civilian spouse), husband",
                                                                   "vs married, husband")


def tex(s: str) -> str:
    s = str(s)
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s.replace("'South'", "`South'")


def decimals(x: float) -> int:
    if not x or not math.isfinite(x):
        return 2
    return int(min(4, max(1, 1 - math.floor(math.log10(abs(x))))))


def num(x, d) -> str:
    if x is None or not math.isfinite(x):
        return ""
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
        if float(x) > -0.5 * 10 ** (-d):
            s = s.replace("$-$", "")
    return s


def cap_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def load(key):
    R = pd.read_csv(SRC / f"betas_{key}.csv")
    S = pd.read_csv(SRC / f"summary_{key}.csv")
    P = pd.read_csv(SRC / f"projection_{key}.csv")
    return R, S, P


def seed_cells(R, term, d, marks=True):
    out = []
    g = R[R.term == term].set_index("seed")
    for s in range(10):
        r = g.loc[s]
        if not r.used:
            out.append("")
            continue
        c = num(r.beta, d)
        sup = ""
        if marks and r.direction_vs_most == "opp":
            sup += r"\dag"
        if marks and r.agreement == "differs":
            sup += "*"
        out.append(c + (rf"\textsuperscript{{{sup}}}" if sup else ""))
    return out


def signal_order(key, R):
    """Signals of the tick tables' linear rows, split into 'every seed, same direction' and the rest
    (by number of seeds, as the tick tables)."""
    first = R[R.seed == 0].drop_duplicates("signal").signal.tolist()
    always, rest = [], []
    for sig in first:
        g = R[R.signal == sig].drop_duplicates(["seed"])
        used = g[g.used]
        if used.empty:
            continue
        if key == "heart100":
            ok = len(used) == 10 and (used.direction_vs_most == "ok").all()
        else:
            ok = len(used) == 10 and (used.direction_vs_most == "ok").all()
        (always if ok else rest).append((sig, len(used)))
    rest.sort(key=lambda t: -t[1])
    return [s for s, _ in always], [s for s, _ in rest]


def main_rows(key, R, S):
    always, rest = signal_order(key, R)
    rows = []

    def add_signal(sig):
        sub = S[S.signal == sig]
        if len(sub) == 1:
            t = sub.iloc[0]
            rows.append(dict(term=t.term, label=t["name"] if key != "heart100" else t["name"], unit=t.unit,
                             median=t["median"], lr=t.lr_beta))
            return
        name = sub.iloc[0]["name"]
        if sig == "lin:vmail":
            for _, t in sub.iterrows():
                rows.append(dict(term=t.term, label=t["name"], unit=t.unit, median=t["median"], lr=t.lr_beta))
            rows.append(dict(term="__vmail_net__", label="Voice mail, net for a typical plan holder",
                             unit="plan $+$ median messages among holders", median=None, lr=None))
            return
        cand = sub[sub.share_train >= MIN_SHARE].copy()
        cand["absmed"] = cand["median"].abs()
        cand = cand.sort_values("absmed", ascending=False).head(TOP_K.get(sig, 3))
        units = set(cand.unit)
        rows.append(dict(term="__header__", label=name, unit=cand.unit.iloc[0] if len(units) == 1 else ""))
        for _, t in cand.iterrows():
            rows.append(dict(term=t.term, label=t.level, unit="" if len(units) == 1 else t.unit,
                             median=t["median"], lr=t.lr_beta, indent=True))

    out = [("Linear terms present in every seed, same direction", [])]
    for sig in always:
        add_signal(sig)
    out[0][1].extend(rows)
    rows = []
    for sig in rest:
        add_signal(sig)
    if rows:
        out.append(("Linear terms present in some seeds only, or with a direction that differs", rows))
    return out


def vm_row(R):
    V = pd.read_csv(SRC / "voicemail_net_churn.csv").set_index("seed")
    vals = [V.loc[s, "net_typical_holder"] for s in range(10)]
    med = float(pd.Series(vals).median())
    d = decimals(med)
    cells = [num(v, d) + (r"\textsuperscript{*}" if V.loc[s, "agreement"] == "differs" else "") for s, v in enumerate(vals)]
    S = pd.read_csv(SRC / "summary_churn.csv").set_index("term")
    m = float(V.median_messages_holders.iloc[0])
    lr = S.loc["voice mail plan=1", "lr_beta"] + m * S.loc["number vmail messages", "lr_beta"]
    return cells, num(med, d), num(lr, d), m


def term_label(r):
    unit = short_unit(r["term"], r["unit"]) if r["unit"] else ""
    unit = UNIT_TEX.get(unit, tex(unit))
    lab = tex(cap_first(r["label"]))
    if unit and "vs" not in r["label"]:
        lab += r", \textit{" + unit + "}"
    return lab


def table_main(key):
    R, S, P = load(key)
    groups = main_rows(key, R, S)
    body = []
    for gname, rows in groups:
        if body:
            body.append(r"    \midrule")
        body.append(rf"    \multicolumn{{13}}{{@{{}}l}}{{\textit{{{gname}}}}} \\")
        for r in rows:
            if r["term"] == "__vmail_net__":
                cells, med, lr, m = vm_row(R)
                lab = rf"\quad net for a typical plan holder, \textit{{plan $+$ {m:g} messages}}"
                body.append("    " + " & ".join([lab, *cells, med, lr]) + r" \\")
                continue
            if r["term"] == "__header__":
                body.append("    " + " & ".join([term_label(r)] + [""] * 12) + r" \\")
                continue
            d = decimals(r["median"])
            cells = seed_cells(R, r["term"], d)
            lab = (r"\quad " if r.get("indent") else "") + term_label(r)
            body.append("    " + " & ".join([lab, *cells, num(r["median"], d), num(r["lr"], d)]) + r" \\")
    lo = P.r2_train.min()
    r2 = rf"$R^2 \geq {lo:.3f}$" if lo < 0.9995 else r"$R^2 > 0.999$"
    pos = {"adult": r"an income of at most \$50K (a positive $\beta$ raises the chance of a \emph{low} income)",
           "churn": "churn", "heart100": "heart disease"}[key]
    notes = []
    if (R.used & (R.direction_vs_most == "opp")).any():
        notes.append(r"\textsuperscript{\dag}\,direction opposite to most seeds")
    if (R.used & (R.agreement == "differs")).any():
        notes.append(r"\textsuperscript{*}\,input purely linear in that seed, and the explanation's own linear-path "
                     r"coefficient differs from $\beta$ (by $>$20\% and $>$0.05 logit per training SD or level "
                     r"change: the straight-line fit also gives the input part of other inputs' non-linear terms)")
    extra = ""
    if key == "churn":
        import json
        corr = json.loads((SRC / "meta.json").read_text())["churn"]["voicemail"]["corr"]
        extra = (r" Usage: per minute, the charge moving with it (charge = rate $\times$ minutes, one term). "
                 rf"Voice-mail plan and messages correlate {corr:.2f}: seeds split them differently, the net for "
                 r"a typical plan holder is stable. States: the four with the largest median $|\beta|$ (states "
                 r"with $\geq$1\% of rows); all in Table~\ref{tab:exp2-betas-churn-states}.")
    if key == "adult":
        extra = (r" Household: each observed pair of marital status and household role (read together) against "
                 r"the most common pair. A missing occupation and a missing work class are the same people (one "
                 r"term). Education levels: offsets from the straight line in years. Many-level terms: the levels "
                 r"with the largest median $|\beta|$ among levels with $\geq$1\% of rows; all levels in "
                 r"Tables~\ref{tab:exp2-betas-adult-household}--\ref{tab:exp2-betas-adult-country}.")
    if key == "heart100":
        few = S[(S.share_train.notna()) & (S.share_train < 0.05)]
        lst = ", ".join(f"{tex(t['name'])[:1].lower() + tex(t['name'])[1:]} {100 * t.share_train:.1f}\\%" for _, t in few.iterrows())
        extra = rf" Levels held by few patients have unstable $\beta$ ({lst} of the training rows)."
    cap = (rf"{SETS[key]}: the size of each linear term in each seed's network, $\beta$: the term's coefficient on "
           rf"the linear block of the least-squares projection of the network's logit on a shared basis of data "
           rf"functions (linear terms; bends and interactions, orthogonal to them; training rows, {r2}). Change "
           rf"in the log-odds of {pos} per unit shown; a level against the most common level. Blank: not in that "
           rf"seed's tick. Median over the seeds shown; LR: L2 logistic regression ($C=1$) on the same linear "
           rf"block.{extra}" + (" " + "; ".join(notes) + "." if notes else ""))
    seeds = " & ".join(f"s{i}" for i in range(10))
    lines = [
        "% Generated by thesis/ch7/exp2/linear_betas.py -- do not edit by hand.",
        r"\begin{sidewaystable}",
        r"  \thesistablestyle",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabularx}{\textheight}{@{}>{\raggedright\arraybackslash}X*{12}{r}@{}}",
        r"    \toprule",
        rf"    Term, \textit{{unit}} & {seeds} & Median & LR \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabularx}",
        rf"  \caption{{{cap}}}",
        rf"  \label{{tab:exp2-betas-{key}}}",
        r"\end{sidewaystable}",
        "",
    ]
    (TAB / f"exp2-betas-{key}.tex").write_text("\n".join(lines))
    return len(body)


def table_levels(key, sigs, suffix, title):
    R, S, P = load(key)
    sub = S[S.signal.isin(sigs)].copy()
    body = []
    for sig in sigs:
        g = sub[sub.signal == sig].sort_values("share_train", ascending=False)
        if len(sigs) > 1:
            body.append(rf"    \multicolumn{{14}}{{@{{}}l}}{{\textit{{{tex(g.iloc[0]['name'])}}}}} \\")
        for _, t in g.iterrows():
            cells = seed_cells(R, t.term, 2, marks=False)
            sh = 100 * t.share_train
            body.append("    " + " & ".join([tex(cap_first(t.level)), f"{sh:.1f}" if sh >= 0.05 else "$<$0.1", *cells,
                                             num(t["median"], 2), num(t.lr_beta, 2)]) + r" \\")
    cap = (rf"{SETS[key]}: the size of every level of {title} in each seed, $\beta$ (log-odds of "
           + ("an income of at most \\$50K" if key == "adult" else "churn")
           + r"; definition as in Table~\ref{tab:exp2-betas-" + key + r"}). \%: share of the training rows. "
           r"Blank: the term is not in that seed's tick. Levels with few rows have large, unstable $\beta$ "
           r"(they move the logit for a handful of people only).")
    seeds = " & ".join(f"s{i}" for i in range(10))
    lines = [
        "% Generated by thesis/ch7/exp2/linear_betas.py -- do not edit by hand.",
        r"\begin{table}[p]",
        r"  \thesistablestyle",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2pt}",
        r"  \begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xr*{12}{r}@{}}",
        r"    \toprule",
        rf"    Level & \% & {seeds} & Med. & LR \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabularx}",
        rf"  \caption{{{cap}}}",
        rf"  \label{{tab:exp2-betas-{key}-{suffix}}}",
        r"\end{table}",
        "",
    ]
    (TAB / f"exp2-betas-{key}-{suffix}.tex").write_text("\n".join(lines))
    return len(body)


def build():
    for key in SETS:
        n = table_main(key)
        print(f"wrote exp2-betas-{key}.tex ({n} lines)")
        for sigs, suffix, title in APPENDIX.get(key, []):
            n = table_levels(key, sigs, suffix, title)
            print(f"wrote exp2-betas-{key}-{suffix}.tex ({n} rows)")


if __name__ == "__main__":
    build()
