"""Pooled statistics for studies F and G (GRAFT_F_G_PLAN.md) from their CSVs.

    uv run python experiments/churn/analyze_fg.py      # -> graft_fg_summary.csv (+ printout)

Reads earlygen_{es,e500}.csv (or their _part-* shards), earlygen_selection.csv,
monk2graft_{es,e500}.csv, protect_{c2,d2}_{es,e500}*.csv when present; missing inputs are
reported and skipped.
"""

import glob
from pathlib import Path

import numpy as np

import graft_fg as fg

HERE = fg.HERE
OUT = HERE / "graft_fg_summary.csv"
DS = ["D1", "D2", "D3", "D4", "D5"]
ROWS = []


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load(pattern_full, pattern_parts):
    full = HERE / pattern_full
    if full.exists():
        return fg.read_rows(full), [full.name]
    parts = sorted(glob.glob(str(HERE / pattern_parts)))
    rows = []
    for p in parts:
        rows += fg.read_rows(p)
    return rows, [Path(p).name for p in parts]


def summ(study, regime, statistic, subset, vals, units=None):
    vals = [num(v) for v in vals]
    s = fg.sign_test(vals)
    lo, hi = fg.median_ci(vals)
    row = {"study": study, "regime": regime, "statistic": statistic, "subset": subset,
           "n": s["n"], "median": s["median"], "median_ci_lo": lo, "median_ci_hi": hi,
           "about_zero_0.005": fg.about_zero(lo, hi), "n_positive": s["n_positive"],
           "n_negative": s["n_negative"], "n_zero": s["n_zero"], "sign_test_p": s["sign_test_p"]}
    if units is not None:
        # per unit (recipient / feature): median over its cases, then the sign test
        by = {}
        for u, v in zip(units, vals):
            if np.isfinite(v):
                by.setdefault(u, []).append(v)
        um = [float(np.median(v)) for v in by.values()]
        su = fg.sign_test(um)
        row.update(n_units=su["n"], unit_median=su["median"],
                   unit_n_positive=su["n_positive"], unit_n_negative=su["n_negative"],
                   unit_sign_test_p=su["sign_test_p"])
    ROWS.append(row)
    return row


def rho(study, regime, statistic, subset, x, y, clusters=None):
    r = fg.spearman_ci(x, y)
    row = {"study": study, "regime": regime, "statistic": statistic, "subset": subset,
           "n": r["n"], "spearman_rho": r["rho"], "spearman_p": r["p"], "rho_ci_lo": r["ci_lo"],
           "rho_ci_hi": r["ci_hi"]}
    if clusters is not None:
        rc = fg.spearman_ci(x, y, clusters=clusters)
        row.update(rho_cluster_ci_lo=rc["ci_lo"], rho_cluster_ci_hi=rc["ci_hi"])
    ROWS.append(row)
    return row


def pr(row):
    if "spearman_rho" in row:
        print(f"  {row['statistic']:55s} {row['subset']:22s} n {row['n']:3d} rho "
              f"{row['spearman_rho']:+.3f} [{row['rho_ci_lo']:+.3f},{row['rho_ci_hi']:+.3f}]"
              + (f" cl [{row['rho_cluster_ci_lo']:+.3f},{row['rho_cluster_ci_hi']:+.3f}]"
                 if "rho_cluster_ci_lo" in row else ""))
    else:
        print(f"  {row['statistic']:55s} {row['subset']:22s} n {row['n']:3d} med "
              f"{row['median']:+.4f} [{row['median_ci_lo']:+.4f},{row['median_ci_hi']:+.4f}] "
              f"+{row['n_positive']}/-{row['n_negative']} p {row['sign_test_p']:.3g}"
              + (f" | units {row['n_units']} med {row['unit_median']:+.4f} "
                 f"+{row['unit_n_positive']}/-{row['unit_n_negative']} p "
                 f"{row['unit_sign_test_p']:.3g}" if "n_units" in row else "")
              + (" ~0" if row["about_zero_0.005"] else ""))


def es_counts(study, regime, cond_rows, group_key=None):
    by = {}
    for r in cond_rows:
        if r.get("ft_best_epoch", "") == "":
            continue
        k = (r["condition"], r.get(group_key, "") if group_key else "")
        by.setdefault(k, []).append(int(float(r["ft_best_epoch"])))
    for (cond, g), v in sorted(by.items()):
        ROWS.append({"study": study, "regime": regime, "statistic": "ES best epoch",
                     "subset": f"{cond}" + (f" gen {g}" if g != "" else ""), "n": len(v),
                     "n_best_epoch_0": sum(e == 0 for e in v),
                     "median_best_epoch": float(np.median(v))})
        print(f"  ES best epoch {cond:22s} {('gen ' + str(g)) if g != '' else '':8s} n {len(v):3d}"
              f" best=0 {sum(e == 0 for e in v):3d} median {np.median(v):.0f}")


# ====================================================================== F1

def f1(regime):
    rows, files = load(f"earlygen_{regime}.csv", f"earlygen_{regime}_part-*.csv")
    if not rows:
        print(f"F1 {regime}: no output")
        return
    cases = [r for r in rows if r["kind"] == "case"]
    print(f"\nF1 {regime}: {len(cases)} cases from {files}")
    nov = [num(r["nov"]) for r in cases]
    gen = [num(r["generation"]) for r in cases]
    seed = [r["seed"] for r in cases]
    unit = [f"s{r['seed']}-g{r['generation']}" for r in cases]
    for D in DS:
        y = [num(r.get(D)) for r in cases]
        pr(summ("F1", regime, D, "all cases", y, unit))
        pr(rho("F1", regime, f"{D} vs nov", "all cases", nov, y, clusters=seed))
        pr(rho("F1", regime, f"{D} vs generation", "all cases", gen, y, clusters=seed))
        pc = fg.partial_spearman(nov, y, gen)
        ROWS.append({"study": "F1", "regime": regime, "statistic": f"{D} vs nov | generation",
                     "subset": "all cases", "n": len(y), "partial_spearman": pc})
        print(f"  {D} vs nov | generation: partial Spearman {pc:+.3f}")
        for arm in ("maxinc", "random"):
            sel = [i for i, r in enumerate(cases) if r["arm"] == arm]
            pr(summ("F1", regime, D, f"arm {arm}", [y[i] for i in sel]))
            pr(rho("F1", regime, f"{D} vs nov", f"arm {arm}", [nov[i] for i in sel],
                   [y[i] for i in sel]))
        for g in sorted(set(gen)):
            sel = [i for i in range(len(cases)) if gen[i] == g]
            pr(summ("F1", regime, D, f"generation {int(g)}", [y[i] for i in sel]))
        for cut in (0.1, 0.15):
            sel = [i for i in range(len(cases)) if nov[i] < cut]
            pr(summ("F1", regime, D, f"nov < {cut}", [y[i] for i in sel],
                    [unit[i] for i in sel]))
        sel = [i for i in range(len(cases)) if nov[i] >= 0.1]
        pr(summ("F1", regime, D, "nov >= 0.1", [y[i] for i in sel]))
    # early vs late (D1): per seed median over gens 0, 5, 10 minus median at gen 99
    for D in ("D1", "D3"):
        diffs = []
        for s in sorted(set(seed)):
            early = [num(r[D]) for r in cases if r["seed"] == s and int(r["generation"]) <= 10]
            late = [num(r[D]) for r in cases if r["seed"] == s and int(r["generation"]) == 99]
            if early and late:
                diffs.append(float(np.median(early) - np.median(late)))
        pr(summ("F1", regime, f"{D} early (gen 0-10) minus gen 99, per seed", "seeds", diffs))
    # G context 3
    for lab in ("G_graft", "G_sham", "D4_p25", "D3_p25"):
        y = [num(r.get(lab)) for r in cases]
        if any(np.isfinite(y)):
            pr(summ("G3", regime, lab, "gens 0/10/99", y, unit))
            for g in (0, 10, 99):
                sel = [i for i in range(len(cases)) if gen[i] == g]
                pr(summ("G3", regime, lab, f"generation {g}", [y[i] for i in sel]))
    dd = [num(r.get("D4_p25")) - num(r.get("D4")) for r in cases]
    pr(summ("G3", regime, "D4_p25 - D4 (graft - sham, protected minus not)", "gens 0/10/99",
            dd, unit))
    for role, a, b in (("graft", "surv_graft-refit-ft-p25", "surv_graft-refit-ft"),
                       ("sham", "surv_sham-refit-ft-p25", "surv_sham-refit-ft")):
        v = [num(r.get(a)) - num(r.get(b)) for r in cases]
        pr(summ("G3", regime, f"survival r P25 - P0 ({role})", "gens 0/10/99", v))
        pr(summ("G3", regime, f"survival r P0 ({role})", "all cases",
                [num(r.get(b)) for r in cases]))
        pr(summ("G3", regime, f"survival r P25 ({role})", "gens 0/10/99",
                [num(r.get(a)) for r in cases]))
    if regime == "es":
        es_counts("F1", regime, [r for r in rows if r["kind"] == "condition"], "generation")


def f1_selection():
    p = HERE / "earlygen_selection.csv"
    if not p.exists():
        return
    rows = fg.read_rows(p)
    pairs = [r for r in rows if r["kind"] == "pair"]
    print(f"\nF1 selection (train only): {len(pairs)} pairs")
    pr(rho("F1-train", "", "inc vs nov (train)", "all pairs", [num(r["nov"]) for r in pairs],
           [num(r["inc"]) for r in pairs], clusters=[r["seed"] for r in pairs]))
    for g in sorted({int(r["generation"]) for r in pairs}):
        sel = [r for r in pairs if int(r["generation"]) == g]
        nv = [num(r["nov"]) for r in sel]
        ROWS.append({"study": "F1-train", "regime": "", "statistic": "nov by generation",
                     "subset": f"generation {g}", "n": len(nv), "median": float(np.median(nv)),
                     "min": float(np.min(nv)), "max": float(np.max(nv)),
                     "inc_median": float(np.median([num(r["inc"]) for r in sel]))})
        print(f"  gen {g:2d}: nov median {np.median(nv):.3f} [{np.min(nv):.3f}, "
              f"{np.max(nv):.3f}] inc median {np.median([num(r['inc']) for r in sel]):+.4f}")
        pr(rho("F1-train", "", "inc vs nov (train)", f"generation {g}", nv,
               [num(r["inc"]) for r in sel]))


# ====================================================================== F2

def f2(regime):
    p = HERE / f"monk2graft_{regime}.csv"
    if not p.exists():
        print(f"F2 {regime}: no output")
        return
    rows = fg.read_rows(p)
    cases = [r for r in rows if r["kind"] == "case"]
    print(f"\nF2 {regime}: {len(cases)} cases")
    nov = [num(r["nov"]) for r in cases]
    unit = [r["seed"] for r in cases]
    for D in DS:
        y = [num(r.get(D)) for r in cases]
        pr(summ("F2", regime, D, "all pairs", y, unit))
        pr(summ("F2", regime, D, "maxinc", [y[i] for i, r in enumerate(cases)
                                            if r["maxinc"] == "True"]))
        pr(rho("F2", regime, f"{D} vs nov", "all pairs", nov, y, clusters=unit))
    if regime == "es":
        es_counts("F2", regime, [r for r in rows if r["kind"] == "condition"])


# ====================================================================== G contexts 1, 2

def g12(context, regime):
    rows, files = load(f"protect_{context}_{regime}_P0-25-100.csv",
                       f"protect_{context}_{regime}_P0-25-100_part-*.csv")
    if not rows:
        rows, files = load(f"protect_{context}_{regime}.csv",
                           f"protect_{context}_{regime}_part-*.csv")
    if not rows:
        print(f"G {context} {regime}: no output")
        return
    cases = [r for r in rows if r["kind"] == "case"]
    print(f"\nG {context} {regime}: {len(cases)} cases from {files}")
    unit = [r["unit"] for r in cases]
    roles = ["graft", "sham"] if context == "c2" else ["alias", "sham", "wrong"]
    st = "G1" if context == "c2" else "G2"
    for role in roles:
        pr(summ(st, regime, f"{role} P25 - P0", "cases", [r.get(f"{role}_p25_minus_p0")
                                                         for r in cases], unit))
        if any(r.get(f"{role}_p100_minus_p0", "") != "" for r in cases):
            pr(summ(st, regime, f"{role} P100 - P0", "cases",
                    [r.get(f"{role}_p100_minus_p0") for r in cases], unit))
        for P in (0, 25, 100):
            pr(summ(st, regime, f"survival r {role} P{P}", "cases",
                    [r.get(f"surv_{role}_p{P}") for r in cases]))
        for P in (25, 100):
            pr(summ(st, regime, f"survival r {role} P{P} - P0", "cases",
                    [num(r.get(f"surv_{role}_p{P}")) - num(r.get(f"surv_{role}_p0"))
                     for r in cases], unit))
        bip = [r.get(f"{role}_p25_ft_best_in_phase") for r in cases]
        if any(b not in ("", None) for b in bip):
            n_in = sum(b == "True" for b in bip)
            ROWS.append({"study": st, "regime": regime, "statistic": "ES best epoch <= P",
                         "subset": f"{role} P25", "n": len(bip), "n_best_in_phase": n_in})
            b0 = sum(num(r.get(f"{role}_p0_ft_best_epoch")) == 0 for r in cases)
            ROWS.append({"study": st, "regime": regime, "statistic": "ES best epoch = 0",
                         "subset": f"{role} P0", "n": len(cases), "n_best_epoch_0": b0})
            print(f"  ES: {role} P25 best epoch <= 25 in {n_in}/{len(bip)}; P0 best epoch 0 "
                  f"in {b0}/{len(cases)}")
    if context == "c2":
        comps = [("graft_minus_sham", "graft - sham"), ("graft_minus_ftonly", "graft - ftonly")]
    else:
        comps = [("alias_minus_sham", "alias - sham"), ("alias_minus_wrong", "alias - wrong"),
                 ("alias_minus_koft", "alias - relearn (KO + ft)")]
    for key, lab in comps:
        for P in (0, 25, 100):
            pr(summ(st, regime, f"{lab} at P{P}", "cases", [r.get(f"{key}_p{P}") for r in cases],
                    unit))
        for P in (25, 100):
            dd = [num(r.get(f"{key}_p{P}")) - num(r.get(f"{key}_p0")) for r in cases]
            pr(summ(st, regime, f"{lab}: P{P} minus P0", "cases", dd, unit))


def main():
    f1_selection()
    for regime in fg.REGIMES:
        f1(regime)
    for regime in fg.REGIMES:
        f2(regime)
    for context in ("c2", "d2"):
        for regime in fg.REGIMES:
            g12(context, regime)
    fg.write_rows(OUT, ROWS)


if __name__ == "__main__":
    main()
