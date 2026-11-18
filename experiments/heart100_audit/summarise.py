"""Heart-100 audit, stage C: family, recurrence and linear-core summaries from audit.py's CSVs.

No database access. Reads ``features.csv``, ``networks.csv``, ``linear_core.csv`` and
``recurrence.csv``; writes ``families.csv`` (one row per family group x network) and
``summary.md`` (the tables AUDIT.md quotes). Test numbers are reported only (marked TEST).

    uv run python experiments/heart100_audit/summarise.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def s(x):
    return str(x).replace("heart100-", "")


def fmt_ci(g, lo, hi):
    return f"{g:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def thresholds(rows, var):
    out = []
    for t in rows.thresholds:
        out += json.loads(t).get(var, [])
    return out


def families(F):
    U = F[F.level == "unit"].copy()
    rows = []
    for (grp, net), g in U.groupby(["family_group", "network"]):
        rows.append(dict(
            family_group=grp, network=net, units=" ".join(map(str, g.unit)),
            families=" | ".join(g.family), annotations=" | ".join(sorted(set(g.annotation))),
            titles=" | ".join(sorted(set(g.title.fillna("")))),
            drivers=" | ".join(g.drivers.fillna("")), modifiers=" | ".join(g.modifiers.fillna("")),
            regime=" | ".join(f"{r.unit}: {r.regime_side} for {r.n_regime}: {r.regime_rule}" for r in g.itertuples()),
            n_regime=" ".join(str(x) for x in g.n_regime), n_core=" ".join(str(x) for x in g.n_core),
            on_direction=" ".join(g.on_direction),
            age_thresholds=" ".join(f"{x:g}" for x in thresholds(g, "age")),
            bp_thresholds=" ".join(f"{x:g}" for x in thresholds(g, "trestbps")),
            mean_abs_risk_change_regime=" ".join(f"{x:.3f}" for x in g.mean_abs_risk_change_regime),
            max_abs_risk_change=float(g.max_abs_risk_change.max()),
            gain_train=" ".join(f"{x:+.4f}" for x in g.gain_train),
            gain_test=" ".join(fmt_ci(x, lo, hi) for x, lo, hi in zip(g.gain_test, g.gain_test_lo, g.gain_test_hi)),
            n_generalise=int((g.test_verdict == "generalises").sum()),
            n_worse=int((g.test_verdict == "worse on test").sum()),
            seeds_matched=" ".join("–" if pd.isna(x) else str(int(x)) for x in g.seeds_matched),
            heart40_best=" ".join(x for x in g.heart40_best_signal.fillna("") if x),
            small_support=int(g.small_support.sum()), verdicts=" | ".join(g.verdict)))
    return pd.DataFrame(rows)


def where(r):
    """The regime and its largest tree cell (the leaf holding most regime rows)."""
    cells = json.loads(r.cells) if isinstance(r.cells, str) else []
    reg = [c for c in cells if c["n_regime"] * 2 > c["n"]]
    main = max(reg, key=lambda c: c["n_regime"]) if reg else None
    return f"{r.regime_side} for {r.n_regime}" + (f"; mostly {main['rule']} ({main['n_regime']}/{main['n']})" if main else "")


def toks(x):
    return x.split() if isinstance(x, str) else []


def ame(g):
    a = g.avg_effect
    return f"{(a > 0).sum()} / {(a < 0).sum()}, {a.median():+.3g} ({a.min():+.3g} – {a.max():+.3g})"


def involvement(U):
    """Per key input (BP, thal, exang, age): units where it drives the switch or modifies the
    strength, by seed."""
    out = []
    for b, tag in (("trestbps", "BP"), ("thal", "thal"), ("exang", "exang"), ("age", "age")):
        d = U[U.drivers.fillna("").str.split().apply(lambda x: b in x)]
        mo = U[U.modifiers.fillna("").str.split().apply(lambda x: b in x)]
        both = pd.concat([d, mo])
        out.append(dict(input=tag, driver_units=len(d), driver_seeds=" ".join(sorted({s(x) for x in d.network})),
                        modifier_units=len(mo), modifier_seeds=" ".join(sorted({s(x) for x in mo.network})),
                        seeds_any=both.network.nunique(),
                        with_age=int(sum("age" in toks(r.drivers) + toks(r.modifiers) for r in both.itertuples()))
                        if b != "age" else None,
                        raises=int((both.on_direction == "raises").sum()), lowers=int((both.on_direction == "lowers").sum()),
                        regime_ge30=int((both.n_regime >= 30).sum()), generalises=int((both.test_verdict == "generalises").sum()),
                        train_gain_max=float(both.gain_train.max()) if len(both) else 0.0))
    return pd.DataFrame(out)


def main():
    F = pd.read_csv(HERE / "features.csv")
    N = pd.read_csv(HERE / "networks.csv")
    C = pd.read_csv(HERE / "linear_core.csv")
    P = pd.read_csv(HERE / "recurrence.csv")
    FA = families(F)
    FA.to_csv(HERE / "families.csv", index=False)
    U = F[F.level == "unit"]
    Pt = F[F.level == "part"]
    L = []
    # ---------------------------------------------------------------- counts
    def counts(G):
        return dict(n=len(G), generalises=int((G.test_verdict == "generalises").sum()),
                    worse=int((G.test_verdict == "worse on test").sum()),
                    none=int((G.test_verdict == "no measurable test effect").sum()),
                    recurring=int((G.recurrence == "recurring").sum()),
                    seed_specific=int((G.recurrence == "seed-specific").sum()),
                    near_linear=int((G.recurrence == "near-linear").sum()),
                    small=int(G.small_support.sum()), useful=int((G.gain_train >= 0.02).sum()),
                    tiny=int((G.gain_train.abs() < 0.001).sum()),
                    h40=int((G.heart40_best_signal.fillna("") != "").sum()))
    L.append("## Counts\n")
    L.append("| level | total | generalises (TEST CI > 0) | no measurable test effect | worse on test | recurring | seed-specific | near-linear | small support (n<30) | train gain ≥ 0.02 | train gain < 0.001 | matches a 40-gen signal |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lvl, G in (("units", U), ("parts", Pt)):
        c = counts(G)
        L.append(f"| {lvl} | {c['n']} | {c['generalises']} | {c['none']} | {c['worse']} | {c['recurring']} | "
                 f"{c['seed_specific']} | {c['near_linear']} | {c['small']} | {c['useful']} | {c['tiny']} | {c['h40']} |")
    L.append(f"\nBinary one-step units excluded: {int(N.binary_step_units_excluded.sum())}. "
             f"Units exactly affine in the inputs on the data (kept, counted near-linear): "
             f"{int(U.exactly_affine_on_data.sum())}.\n")
    L.append("Generalising units / parts (TEST):\n")
    for r in F[F.test_verdict == "generalises"].itertuples():
        L.append(f"- {s(r.network)} {r.level} {r.unit} ({r.annotation}, {r.family}): train {r.gain_train:+.4f}, "
                 f"TEST {fmt_ci(r.gain_test, r.gain_test_lo, r.gain_test_hi)}; {r.verdict}")
    L.append("")
    # ---------------------------------------------------------------- compact per-unit table
    L.append("## Every non-linear unit (compact; train unless TEST)\n")
    L.append("| seed | part (title) | unit | family | where it switches (train n in regime) | being on | mean / max \\|Δrisk\\| | half the effect on n | train gain | TEST gain [95% CI] | other seeds matched | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in U.sort_values(["network", "annotation", "unit"]).itertuples():
        sm = "–" if pd.isna(r.seeds_matched) else str(int(r.seeds_matched))
        L.append(f"| {s(r.network)} | {r.annotation} ({r.title}) | {r.unit} | {r.family} | {where(r)} | {r.on_direction} | "
                 f"{r.mean_abs_risk_change_regime:.3f} / {r.max_abs_risk_change:.2f} | {r.n_core} | {r.gain_train:+.4f} | "
                 f"{fmt_ci(r.gain_test, r.gain_test_lo, r.gain_test_hi)} | {sm} | {r.verdict} |")
    L.append("")
    L.append("## Every non-linear part (compact; train unless TEST)\n")
    L.append("| seed | part (title) | units | family (dominant unit) | regime n (union) | max \\|Δrisk\\| | train gain | TEST gain [95% CI] | other seeds matched | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in Pt.sort_values(["network", "annotation"]).itertuples():
        sm = "–" if pd.isna(r.seeds_matched) else str(int(r.seeds_matched))
        L.append(f"| {s(r.network)} | {r.annotation} ({r.title}) | {r.unit} | {r.family} | {r.n_regime} | {r.max_abs_risk_change:.2f} | "
                 f"{r.gain_train:+.4f} | {fmt_ci(r.gain_test, r.gain_test_lo, r.gain_test_hi)} | {sm} | {r.verdict} |")
    L.append("")
    # ---------------------------------------------------------------- families
    L.append("## Families (units; train unless TEST)\n")
    L.append("| family group | seeds (units) | being on: raises / lowers risk | regime n (median, range) | age thresholds | BP thresholds | mean \\|Δrisk\\| on regime (median, max) | train gain: median, max | TEST gain CI > 0 / < 0 | recurring | small support |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    order = U.groupby("family_group").network.nunique().sort_values(ascending=False).index
    for grp in order:
        g = U[U.family_group == grp]
        ag, bp = thresholds(g, "age"), thresholds(g, "trestbps")
        L.append(f"| {grp} | {g.network.nunique()}: {' '.join(sorted({s(x) for x in g.network}))} ({len(g)}) | "
                 f"{(g.on_direction == 'raises').sum()} / {(g.on_direction == 'lowers').sum()} | "
                 f"{int(g.n_regime.median())} ({g.n_regime.min()}–{g.n_regime.max()}) | "
                 f"{' '.join(f'{x:g}' for x in sorted(ag)) or '–'} | {' '.join(f'{x:g}' for x in sorted(bp)) or '–'} | "
                 f"{g.mean_abs_risk_change_regime.median():.3f}, {g.max_abs_risk_change.max():.2f} | "
                 f"{g.gain_train.median():+.4f}, {g.gain_train.max():+.4f} | "
                 f"{(g.test_verdict == 'generalises').sum()} / {(g.test_verdict == 'worse on test').sum()} | "
                 f"{(g.recurrence == 'recurring').sum()}/{len(g)} | {int(g.small_support.sum())} |")
    L.append("")
    for grp in [x for x in order if any(k in x for k in ("BP", "thal", "exang", "age"))]:
        g = U[U.family_group == grp]
        L.append(f"### {grp}\n")
        L.append("| seed | part (title) | unit | drivers / modifiers | where it switches (train n) | being on | mean / max \\|Δrisk\\| | half the effect on | age bands in regime: n (mean \\|Δrisk\\|) | train gain | TEST gain [95% CI] | seeds matched | verdict |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in g.sort_values("network").itertuples():
            bands = json.loads(r.bands) if isinstance(r.bands, str) else []
            bt = "; ".join(f"{b['band']}: {b['n']} ({b['mean_abs_risk_change']:.2f})" for b in bands) if r.band_variable == "age" else "–"
            sm = "–" if pd.isna(r.seeds_matched) else str(int(r.seeds_matched))
            L.append(f"| {s(r.network)} | {r.annotation} ({r.title}) | {r.unit} | {' '.join(toks(r.drivers))} / {' '.join(toks(r.modifiers))} | "
                     f"{where(r)} | {r.on_direction} ({r.on_effect_logit:+.2f} logit) | "
                     f"{r.mean_abs_risk_change_regime:.3f} / {r.max_abs_risk_change:.2f} | {r.n_core}: {r.core_rows if isinstance(r.core_rows, str) else ''} | {bt} | {r.gain_train:+.4f} | "
                     f"{fmt_ci(r.gain_test, r.gain_test_lo, r.gain_test_hi)} | {sm} | {r.verdict} |")
        L.append("")
    L.append("## Family tables (condensed, for AUDIT.md)\n")
    for grp in [x for x in order if any(k in x for k in ("BP", "thal", "exang", "age"))]:
        g = U[U.family_group == grp].sort_values("network")
        L.append(f"**{grp}** ({g.network.nunique()} seeds, {len(g)} units)\n")
        L.append("| seed | unit (part title) | where it switches, train n; being on | half the effect on (n) | mean / max \\|Δrisk\\| | train gain | TEST gain [95% CI] | other seeds |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in g.itertuples():
            sm = "–" if pd.isna(r.seeds_matched) else str(int(r.seeds_matched))
            mods = f" (strength: {', '.join(toks(r.modifiers))})" if toks(r.modifiers) else ""
            L.append(f"| {s(r.network)} | {r.unit} ({r.title}) | {where(r)}{mods}; {r.on_direction} risk | {r.n_core} | "
                     f"{r.mean_abs_risk_change_regime:.3f} / {r.max_abs_risk_change:.2f} | {r.gain_train:+.4f} | "
                     f"{fmt_ci(r.gain_test, r.gain_test_lo, r.gain_test_hi)} | {sm} |")
        L.append("")
    L.append("## Key inputs: units where BP / thal / exang / age drive the switch or set its strength\n")
    L.append("| input | units it drives (seeds) | units it modifies (seeds) | seeds with either | of these, units also involving age | being on raises / lowers risk | regime ≥ 30 patients | TEST CI > 0 | max train gain |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in involvement(U).itertuples():
        wa = "–" if r.with_age is None or pd.isna(r.with_age) else int(r.with_age)
        L.append(f"| {r.input} | {r.driver_units} ({r.driver_seeds}) | {r.modifier_units} ({r.modifier_seeds}) | {r.seeds_any} | {wa} | "
                 f"{r.raises} / {r.lowers} | {r.regime_ge30} | {r.generalises} | {r.train_gain_max:+.4f} |")
    L.append("")
    # ---------------------------------------------------------------- recurrence
    L.append("## Recurring signals (strict residual match)\n")
    L.append("| network | level | unit | family | seeds matched (which) | 40-gen seeds reproducing | best 40-gen signal (R²) | train gain | TEST gain |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    R = F[F.recurrence == "recurring"].sort_values(["family_group", "network"])
    for r in R.itertuples():
        L.append(f"| {s(r.network)} | {r.level} | {r.unit} | {r.family} | {int(r.seeds_matched)} ({r.seeds_matched_list}) | "
                 f"{int(r.heart40_seeds_matched)} | {r.heart40_best_signal if isinstance(r.heart40_best_signal, str) else '–'} "
                 f"({r.heart40_best_r2:.2f}) | {r.gain_train:+.4f} | {fmt_ci(r.gain_test, r.gain_test_lo, r.gain_test_hi)} |")
    L.append("")
    L.append(f"Unit-to-unit residual matches across Heart-100 seeds (R² ≥ 0.5): {len(P)}")
    dirs = {f"{r.network}:{r.unit}": r for r in U.itertuples()}
    for r in P.itertuples():
        x, y = dirs[r.a], dirs[r.b]
        L.append(f"- {s(r.a)} ({x.family}, being on {x.on_direction} risk) ~ {s(r.b)} ({y.family}, {y.on_direction}) (R² {r.r2:.2f})")
    L.append("")
    m40 = F[F.heart40_best_signal.fillna("") != ""]
    L.append(f"Units / parts whose residual matches a 40-generation Heart inventory signal (R² ≥ 0.5): {len(m40)}")
    for r in m40.itertuples():
        same = r.heart40_best_signal.split(":")[0].replace("heart-", "") == s(r.network)
        L.append(f"- {s(r.network)} {r.level} {r.unit} ({r.family}) ~ {r.heart40_best_signal} (R² {r.heart40_best_r2:.2f})"
                 + (" [same seed]" if same else " [other seed]"))
    L.append("")
    # ---------------------------------------------------------------- linear core
    L.append("## Linear core: linear-path coefficient across the 10 networks (log-odds per real unit; categories vs the most common level)\n")
    L.append("| term | per | networks with a linear path | sign + / − | median | IQR | min – max | LR on train (reference) | shared? | average effect, all 10: + / −, median (min – max) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for term, g in C.groupby("term", sort=False):
        h = g[g.linear_path]
        lr = g.lr_reference.iloc[0]
        if len(h):
            pos, neg = int((h.coef > 0).sum()), int((h.coef < 0).sum())
            q1, q3 = np.percentile(h.coef, [25, 75])
            agree = max(pos, neg) / len(h)
            shared = ("yes" if len(h) >= 8 and agree == 1 else "mostly" if len(h) >= 6 and agree >= 0.8
                      else "no" if len(h) >= 4 and agree < 0.8 else "rarely used")
            L.append(f"| {term} ({g.label.iloc[0]}) | {g.unit.iloc[0].replace('per ', '')} | {len(h)} | {pos} / {neg} | "
                     f"{h.coef.median():+.3g} | {q1:+.3g} – {q3:+.3g} | {h.coef.min():+.3g} – {h.coef.max():+.3g} | {lr:+.3g} | {shared} | {ame(g)} |")
            rows.append(dict(term=term, n=len(h), pos=pos, neg=neg, median=h.coef.median(), shared=shared))
        else:
            L.append(f"| {term} ({g.label.iloc[0]}) | {g.unit.iloc[0].replace('per ', '')} | 0 | – | – | – | – | {lr:+.3g} | no linear path | {ame(g)} |")
    cls = C.drop_duplicates(["network", "variable"]).groupby(["variable", "var_class"]).size().unstack(fill_value=0)
    L.append("\nVariable classes across the 10 networks (after the binary one-step rule):\n")
    L.append("| variable | " + " | ".join(cls.columns) + " |")
    L.append("|---|" + "---|" * len(cls.columns))
    for v, r in cls.iterrows():
        L.append(f"| {v} | " + " | ".join(str(int(x)) for x in r) + " |")
    (HERE / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
