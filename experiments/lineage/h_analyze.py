"""Study H, stage 2: tests 1-3 and 5, CSVs and figures (PLAN_H.md).

Reads the per-lineage caches written by h_build.py (and, for Test 5, the DB,
read-only). It writes, in experiments/lineage/:
- h_steps.csv, h_test1.csv, h_test2.csv, h_emergence.csv, h_leaves.csv,
  h_test3.csv, h_monk2.csv;
- figures/h_lineages_<set>.{pdf,png} and figures/h_emergence_<set>.{pdf,png}.

    uv run python experiments/lineage/h_analyze.py
"""
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE / "replay_db" / "h_cache"
FIG = HERE / "figures"
SETS = ["churn100", "churn200", "heart100"]
SEEDS = list(range(10))
N_PERM = N_BOOT = 10_000
SEED = 42
PRIMARY = "q95"
THRESHOLDS = ["q90", "q95", "q99", "abs0.005", "abs0.01", "abs0.02", "elite99"]


def load(label, seed):
    p = CACHE / f"{label}-s{seed}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0])
    for r in rows[1:]:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ===================================================================== steps
def steps_of(c, lineage=None):
    rows = c["rows"] if lineage is None else lineage
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        S = a["structure_signature"] != b["structure_signature"]
        typ = "elite" if b["is_elite_copy"] else ("child-structural" if S else "child-weights")
        d = b.get("diff") if lineage is None else None
        r = dict(set=c["label"], seed=c["seed"], generation=b["generation"],
                 delta=b["fitness"] - a["fitness"], S=int(S), step_type=typ,
                 elite=int(b["is_elite_copy"]))
        if lineage is None:
            r.update(delta_test=b["test_auc"] - a["test_auc"], train_auc=b["fitness"],
                     test_auc=b["test_auc"],
                     n_added_nodes=len(d["added_nodes"]), n_removed_nodes=len(d["removed_nodes"]),
                     n_added_conns=len(d["added_connections"]),
                     n_removed_conns=len(d["removed_connections"]),
                     n_enabled_changed=len(d["enabled_changed"]))
        out.append(r)
    return out


def thresholds(steps):
    d = np.array([s["delta"] for s in steps])
    de = np.array([s["delta"] for s in steps if s["elite"]])
    return {"q90": float(np.quantile(d, 0.90)), "q95": float(np.quantile(d, 0.95)),
            "q99": float(np.quantile(d, 0.99)), "abs0.005": 0.005, "abs0.01": 0.01,
            "abs0.02": 0.02, "elite99": float(np.quantile(de, 0.99)) if len(de) else np.nan}


def mark_jumps(steps, taus):
    for s in steps:
        for k, t in taus.items():
            s[f"jump_{k}"] = int(s["delta"] >= t and s["delta"] > 0)


# ===================================================================== test 1
def rd(J, S):
    J, S = np.asarray(J, float), np.asarray(S, bool)
    if S.all() or (~S).all():
        return np.nan, np.nan
    p1, p0 = J[S].mean(), J[~S].mean()
    return p1 - p0, (p1 / p0 if p0 > 0 else np.inf)


def test1(steps, jcol, block=None, rng_seed=SEED):
    by = defaultdict(list)
    for s in steps:
        by[s["seed"]].append(s)
    seeds = sorted(by)
    J = {k: np.array([s[jcol] for s in by[k]]) for k in seeds}
    S = {k: np.array([s["S"] for s in by[k]]) for k in seeds}
    G = {k: np.array([s["generation"] for s in by[k]]) for k in seeds}
    allJ = np.concatenate([J[k] for k in seeds])
    allS = np.concatenate([S[k] for k in seeds])
    obs, rr = rd(allJ, allS)
    rng = np.random.default_rng(rng_seed)
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        parts = []
        for k in seeds:
            if block is None:
                parts.append(rng.permutation(S[k]))
            else:
                x = S[k].copy()
                blk = (G[k] - 1) // block
                for b in np.unique(blk):
                    m = blk == b
                    x[m] = rng.permutation(x[m])
                parts.append(x)
        perm[i] = rd(allJ, np.concatenate(parts))[0]
    p_one = (1 + np.sum(perm >= obs)) / (N_PERM + 1)
    p_two = (1 + np.sum(np.abs(perm) >= abs(obs))) / (N_PERM + 1)
    rngb = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rngb.choice(seeds, size=len(seeds), replace=True)
        boots.append(rd(np.concatenate([J[k] for k in pick]), np.concatenate([S[k] for k in pick]))[0])
    boots = np.array(boots)
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    nJ = int(allJ.sum())
    return dict(n_steps=len(allJ), n_structural=int(allS.sum()), n_jumps=nJ,
                p_jump_given_S1=float(allJ[allS == 1].mean()) if allS.any() else np.nan,
                p_jump_given_S0=float(allJ[allS == 0].mean()) if (allS == 0).any() else np.nan,
                RD=obs, RD_lo=lo, RD_hi=hi, RR=rr,
                p_S1_given_jump=float(allS[allJ == 1].mean()) if nJ else np.nan,
                p_S1_overall=float(allS.mean()), p_perm_one_sided=p_one, p_perm_two_sided=p_two,
                n_lineages_with_jump=int(sum(J[k].any() for k in seeds)))


# ===================================================================== test 2
def added_genes(c, jumps_by_gen, final_nodes, final_conns_enabled):
    out = []
    gm = c["gene_map"] or {}
    for row in c["rows"][1:]:
        d = row["diff"]
        g = row["generation"]
        for n in d["added_nodes"]:
            e = gm.get(str(n), {})
            out.append(dict(set=c["label"], seed=c["seed"], generation=g, gene=str(n), type="node",
                            jump=jumps_by_gen.get(g, 0), in_leaf=int(bool(e.get("containing_leaves"))),
                            in_any=int(bool(e.get("containing"))),
                            survives=int(n in final_nodes)))
        for k in d["added_connections"]:
            a, b = (int(x) for x in k.rsplit("_", 1))
            e = gm.get(f"{a}_{b}", {})
            out.append(dict(set=c["label"], seed=c["seed"], generation=g, gene=k,
                            type="connection", jump=jumps_by_gen.get(g, 0),
                            in_leaf=int(bool(e.get("containing_leaves"))),
                            in_any=int(bool(e.get("containing"))),
                            survives=int((a, b) in final_conns_enabled)))
    return out


def mh_diff(rows, ycol):
    num = den = 0.0
    per = {}
    for t in ("node", "connection"):
        r = [x for x in rows if x["type"] == t]
        j = np.array([x[ycol] for x in r if x["jump"]])
        n = np.array([x[ycol] for x in r if not x["jump"]])
        if len(j) and len(n):
            d = j.mean() - n.mean()
            w = len(j) * len(n) / (len(j) + len(n))
            num += w * d
            den += w
            per[t] = (d, j.mean(), n.mean(), len(j), len(n))
        else:
            per[t] = (np.nan, j.mean() if len(j) else np.nan, n.mean() if len(n) else np.nan,
                      len(j), len(n))
    return (num / den if den else np.nan), per


def test2(rows, ycol):
    obs, per = mh_diff(rows, ycol)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["seed"], r["type"])].append(i)
    jump = np.array([r["jump"] for r in rows])
    rng = np.random.default_rng(SEED)
    perm = np.empty(N_PERM)
    rows_c = [dict(r) for r in rows]
    for i in range(N_PERM):
        jp = jump.copy()
        for idx in groups.values():
            jp[idx] = rng.permutation(jp[idx])
        for r, v in zip(rows_c, jp):
            r["jump"] = v
        perm[i] = mh_diff(rows_c, ycol)[0]
    p_one = (1 + np.sum(perm >= obs)) / (N_PERM + 1)
    p_two = (1 + np.sum(np.abs(perm) >= abs(obs))) / (N_PERM + 1)
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    seeds = sorted(by_seed)
    rngb = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rngb.choice(seeds, size=len(seeds), replace=True)
        boots.append(mh_diff([r for k in pick for r in by_seed[k]], ycol)[0])
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    out = dict(outcome=ycol, MH_diff=obs, lo=lo, hi=hi, p_perm_one_sided=p_one,
               p_perm_two_sided=p_two)
    for t, (d, pj, pn, nj, nn) in per.items():
        out.update({f"{t}_diff": d, f"{t}_p_jump": pj, f"{t}_p_nonjump": pn,
                    f"{t}_n_jump": nj, f"{t}_n_nonjump": nn})
    return out


# ===================================================================== test 3
def leaf_rows(c, steps_by_gen):
    rows, curves = [], []
    gens = [r["generation"] for r in c["rows"]]
    nodes_by = [r["nodes"] for r in c["rows"]]
    conns_by = [{k for k, e in r["conns"].items() if e} for r in c["rows"]]
    for lf in c["leaves"]:
        name = lf["name"]
        d = np.array([r["d"].get(name, np.nan) for r in c["rows"]], dtype=float)
        for g, v in zip(gens, d):
            curves.append(dict(set=c["label"], seed=c["seed"], leaf=name, generation=g, d=v))
        base = dict(set=c["label"], seed=c["seed"], leaf=name, signal=lf["kind"],
                    reason=lf["reason"], s_sd=lf["s_sd"],
                    G_nodes=len(lf["G_nodes"]), G_conns=len(lf["G_conns"]))
        if lf["kind"] is None or not np.isfinite(d[-1]) or (lf["s_sd"] or 0) <= 1e-12:
            rows.append(dict(base, status="excluded"))
            continue
        ok = np.isfinite(d)
        dmin = float(np.nanmin(d))
        R = float(d[-1] - dmin)
        rises = np.full(len(d), np.nan)
        for i in range(1, len(d)):
            if ok[i] and ok[i - 1]:
                rises[i] = d[i] - d[i - 1]
        i_star = int(np.nanargmax(rises)) if np.isfinite(rises).any() else None
        max_rise = float(rises[i_star]) if i_star is not None else np.nan
        g_star = gens[i_star] if i_star is not None else None
        # g_c: earliest generation from which every G_L gene is present to the end.
        Gn, Gc = set(lf["G_nodes"]), set(map(tuple, lf["G_conns"]))
        present = [Gn <= nodes_by[i] and Gc <= conns_by[i] for i in range(len(gens))]
        g_c = None
        if present[-1] and (Gn or Gc):
            i = len(gens) - 1
            while i > 0 and present[i - 1]:
                i -= 1
            g_c = gens[i]
        T = int(np.sum(rises[np.isfinite(rises)] > 0))
        # A uniform draw among the positive-rise steps hits g_c with probability 1/T
        # if g_c is itself a positive-rise step, else 0 (the permutation null in PLAN_H).
        i_c = gens.index(g_c) if g_c is not None else None
        chance = (1.0 / T) if (T and i_c is not None and np.isfinite(rises[i_c])
                               and rises[i_c] > 0) else 0.0
        at = i_star if i_star is not None else len(gens) - 1
        st = steps_by_gen.get(g_star, {})
        added_at = set()
        if i_star is not None and i_star > 0:
            dd = c["rows"][i_star]["diff"]
            added_at = set(dd["added_nodes"]) | {tuple(int(x) for x in k.rsplit("_", 1))
                                                 for k in dd["added_connections"]}
        rows.append(dict(
            base, status="tested" if R >= 0.10 else "below_rise_cutoff",
            d_first=float(d[0]), d_min=dmin, d_final=float(d[-1]), R=R, max_rise=max_rise,
            share=max_rise / R if R > 0 else np.nan,
            step_like=int(R > 0 and max_rise >= 0.5 * R),
            step_like_33=int(R > 0 and max_rise >= (1 / 3) * R),
            step_like_67=int(R > 0 and max_rise >= (2 / 3) * R),
            g_star=g_star, g_c=g_c, g_star_eq_g_c=int(g_c is not None and g_star == g_c),
            g_star_within2=int(g_c is not None and g_star is not None and abs(g_star - g_c) <= 2),
            T_positive=T, chance=chance,
            step_structural=st.get("S"), step_type=st.get("step_type"),
            step_delta=st.get("delta"), step_jump=st.get(f"jump_{PRIMARY}"),
            G_genes_added_at_g_star=int(bool(added_at & (Gn | Gc))),
            G_nodes_present_at_g_star=len(Gn & nodes_by[at]),
            G_conns_present_at_g_star=len(Gc & conns_by[at])))
    return rows, curves


def test3(leaves):
    t = [r for r in leaves if r["status"] == "tested"]
    out = dict(n_leaves=len(leaves), n_excluded=sum(r["status"] == "excluded" for r in leaves),
               n_below_cutoff=sum(r["status"] == "below_rise_cutoff" for r in leaves),
               n_tested=len(t))
    if not t:
        return out
    by = defaultdict(list)
    for r in t:
        by[r["seed"]].append(r)
    seeds = sorted(by)
    rngb = np.random.default_rng(SEED)
    for col in ("step_like", "step_like_33", "step_like_67", "g_star_eq_g_c", "g_star_within2",
                "step_structural", "G_genes_added_at_g_star", "step_jump"):
        vals = [r[col] for r in t if r[col] is not None]
        out[col] = float(np.mean(vals)) if vals else np.nan
        boots = []
        for _ in range(N_BOOT):
            pick = rngb.choice(seeds, size=len(seeds), replace=True)
            v = [r[col] for k in pick for r in by[k] if r[col] is not None]
            boots.append(np.mean(v) if v else np.nan)
        out[f"{col}_lo"], out[f"{col}_hi"] = np.nanquantile(boots, [0.025, 0.975])
    # alignment vs chance: draw g* uniformly among each leaf's positive-rise steps
    al = [r for r in t if r["g_c"] is not None and r["T_positive"]]
    out["n_alignable"] = len(al)
    obs = sum(r["g_star_eq_g_c"] for r in al)
    out["n_g_star_eq_g_c"] = obs
    out["expected_chance"] = float(sum(r["chance"] for r in al))
    rng = np.random.default_rng(SEED)
    p = np.array([r["chance"] for r in al])
    sims = (rng.random((N_PERM, len(al))) < p).sum(axis=1) if len(al) else np.zeros(N_PERM)
    out["p_perm_alignment"] = float((1 + np.sum(sims >= obs)) / (N_PERM + 1))
    return out


# ===================================================================== figures
def figures(caches, steps_all, curves, leaves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    ink, muted, accent, test_c, grid = "#1f2937", "#9ca3af", "#c2410c", "#2563eb", "#e5e7eb"
    for label in SETS:
        cs = [c for c in caches if c["label"] == label]
        if not cs:
            continue
        fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharey=True)
        for ax, c in zip(axes.ravel(), cs):
            g = [r["generation"] for r in c["rows"]]
            ax.plot(g, [r["fitness"] for r in c["rows"]], color=ink, lw=1.3, label="train AUC")
            ax.plot(g, [r["test_auc"] for r in c["rows"]], color=test_c, lw=1.0, alpha=0.8,
                    label="test AUC")
            st = [s for s in steps_all if s["set"] == label and s["seed"] == c["seed"]]
            ymin = min(r["test_auc"] for r in c["rows"])
            sg = [s["generation"] for s in st if s["S"]]
            ax.plot(sg, [ymin - 0.01] * len(sg), "|", color=muted, ms=6, label="structural step")
            jg = [s for s in st if s[f"jump_{PRIMARY}"]]
            fit = {r["generation"]: r["fitness"] for r in c["rows"]}
            ax.scatter([s["generation"] for s in jg], [fit[s["generation"]] for s in jg],
                       s=22, facecolors="none", edgecolors=accent, lw=1.2, zorder=3,
                       label="jump (Δ ≥ τ95)")
            ax.set_title(f"seed {c['seed']}", fontsize=9, color=ink)
            ax.grid(color=grid, lw=0.6)
            ax.tick_params(labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
        fig.supxlabel("generation (lineage of the explained genome, fitter-parent)", fontsize=9)
        fig.supylabel("AUC", fontsize=9)
        fig.suptitle(f"{label}: lineage fitness, jumps and structural steps", fontsize=11)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"h_lineages_{label}.{ext}", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharey=True)
        lv = {(r["seed"], r["leaf"]): r for r in leaves if r["set"] == label}
        for ax, c in zip(axes.ravel(), cs):
            cur = defaultdict(list)
            for r in curves:
                if r["set"] == label and r["seed"] == c["seed"]:
                    cur[r["leaf"]].append((r["generation"], r["d"]))
            for name, pts in cur.items():
                info = lv.get((c["seed"], name), {})
                tested = info.get("status") == "tested"
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=accent if tested else muted, lw=1.0 if tested else 0.7,
                        alpha=0.9 if tested else 0.6)
                if tested and info.get("g_star") is not None:
                    gi = xs.index(info["g_star"])
                    ax.plot(info["g_star"], ys[gi], "o", color=accent, ms=3)
                if tested and info.get("g_c") is not None:
                    ax.axvline(info["g_c"], color=accent, lw=0.5, ls=":", alpha=0.6)
            ax.set_title(f"seed {c['seed']}", fontsize=9, color=ink)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(color=grid, lw=0.6)
            ax.tick_params(labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.supxlabel("generation", fontsize=9)
        fig.supylabel("decodability d_L(g) = 1 − novelty", fontsize=9)
        fig.suptitle(f"{label}: emergence of each final leaf's signal along the lineage "
                     "(orange = rise ≥ 0.10, dot = largest step, dotted = completion g_c)",
                     fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"h_emergence_{label}.{ext}", dpi=150)
        plt.close(fig)


# ===================================================================== main
def main():
    caches, steps_all, t1, genes_all, t2, leaves_all, curves_all, t3 = [], [], [], [], [], [], [], []
    for label in SETS:
        cs = [c for c in (load(label, s) for s in SEEDS) if c is not None]
        if not cs:
            continue
        if len(cs) < len(SEEDS):
            print(f"WARNING {label}: {len(cs)}/{len(SEEDS)} lineages", file=sys.stderr)
        caches += cs
        steps = [s for c in cs for s in steps_of(c)]
        taus = thresholds(steps)
        mark_jumps(steps, taus)
        steps_all += steps
        # parent1-mode lineage, same thresholds (sensitivity)
        steps_p1 = [s for c in cs for s in steps_of(c, c["lineage_parent1"])]
        mark_jumps(steps_p1, taus)
        for k in THRESHOLDS:
            jc = f"jump_{k}"
            variants = [("primary", steps, None), ("no_elite_steps", [s for s in steps if not s["elite"]], None),
                        ("generations_11_plus", [s for s in steps if s["generation"] > 10], None)]
            if k == PRIMARY:
                variants += [("block10_permutation", steps, 10), ("parent1_lineage", steps_p1, None)]
            for vname, st, block in variants:
                r = test1(st, jc, block=block)
                t1.append(dict(set=label, threshold=k, tau=taus[k], variant=vname, **r))
        mean_by_type = defaultdict(list)
        for s in steps:
            mean_by_type[s["step_type"]].append(s["delta"])
        for typ, v in sorted(mean_by_type.items()):
            t1.append(dict(set=label, threshold="-", variant=f"mean_delta_{typ}", n_steps=len(v),
                           RD=float(np.mean(v)), RD_lo=float(np.median(v))))
        # Test 2
        jumps_by = {(s["seed"], s["generation"]): s[f"jump_{PRIMARY}"] for s in steps}
        genes = []
        for c in cs:
            if not c["has_explanation"]:
                continue
            fin = c["rows"][-1]
            genes += added_genes(c, {g: j for (sd, g), j in jumps_by.items() if sd == c["seed"]},
                                 fin["nodes"], {k for k, e in fin["conns"].items() if e})
        genes_all += genes
        for ycol, sub in (("in_leaf", genes), ("in_leaf", [g for g in genes if g["survives"]]),
                          ("survives", genes)):
            r = test2(sub, ycol)
            t2.append(dict(set=label, population="all_added" if sub is genes else "survivors",
                           n_genes=len(sub), **r))
        # Test 3
        sb = {}
        for s in steps:
            sb.setdefault(s["seed"], {})[s["generation"]] = s
        for c in cs:
            lr, cv = leaf_rows(c, sb.get(c["seed"], {}))
            leaves_all += lr
            curves_all += cv
        lv = [r for r in leaves_all if r["set"] == label]
        t3.append(dict(set=label, rise_cutoff=0.10, **test3(lv)))
        for cut in (0.05, 0.2):
            lv2 = [dict(r, status=("tested" if r.get("R", -1) >= cut else "below_rise_cutoff")
                        if r["status"] != "excluded" else "excluded") for r in lv]
            t3.append(dict(set=label, rise_cutoff=cut, **test3(lv2)))
        print(f"{label}: {len(cs)} lineages, {len(steps)} steps, tau95={taus['q95']:.4f}", flush=True)

    write_csv(HERE / "h_steps.csv", steps_all)
    write_csv(HERE / "h_test1.csv", t1)
    write_csv(HERE / "h_test2_genes.csv", genes_all)
    write_csv(HERE / "h_test2.csv", t2)
    write_csv(HERE / "h_leaves.csv", leaves_all)
    write_csv(HERE / "h_emergence.csv", curves_all)
    write_csv(HERE / "h_test3.csv", t3)
    figures(caches, steps_all, curves_all, leaves_all)


if __name__ == "__main__":
    main()
