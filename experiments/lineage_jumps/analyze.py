"""Jumps and plateaus, stage 2: jumps, enrichment, lag (RESULTS.md section 1).

Reads the per-run caches written by build.py and writes, in this directory:
- jumps.csv: every event (jump or climb) of every run and threshold variant;
- runs.csv: per-run summary (primary threshold);
- enrichment.csv: per study and pooled, P(flag | jump) vs P(flag | non-jump), RD, CI, p;
- lag.csv: lag distribution at jumps and at non-jump generations;
- analysis.json: everything the figures and RESULTS.md need.

    uv run python experiments/lineage_jumps/analyze.py
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get(
    "JUMPS_SCRATCH",
    "/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad"))
CACHE = SCRATCH / "runs"

VARIANTS = {"primary": (3, 0.10), "strict": (5, 0.20), "loose": (2, 0.05)}
FLAGS = ["structural", "node_added", "conn_added", "conn_enabled", "crossover",
         "from_other_parent", "elite_only"]
EARLY = 10
PLATEAU = 5
N_BOOT = int(os.environ.get("JUMPS_NBOOT", 10_000))
N_PERM = int(os.environ.get("JUMPS_NPERM", 10_000))

# Studies pooled for the lineage analysis (prefix runs excluded; section 1.1).
POOLED = ["heart100", "monk2", "churn200", "shape-A", "shape-B", "shape-C-g300", "shape-C-sig",
          "churn-case", "old"]


def study_of(run):
    if run.startswith("heart100-"):
        return "heart100"
    if run.startswith("heart-"):
        return "heart (40, prefix)"
    if run.startswith("monk2-"):
        return "monk2"
    if run.startswith("churn200-"):
        return "churn200"
    if run.startswith("churn100-"):
        return "churn100 (prefix)"
    if run.startswith("shape-C-sig-"):
        return "shape-C-sig"
    if run.startswith("shape-C-") and run.endswith("-g300"):
        return "shape-C-g300"
    if run.startswith("shape-C-"):
        return "shape-C (100, prefix)"
    if run.startswith("shape-A-"):
        return "shape-A"
    if run.startswith("shape-B-"):
        return "shape-B"
    if run.startswith("churn-case"):
        return "churn-case"
    if run.startswith("10seed-adult-"):
        return "adult"
    return "old"


# ----------------------------------------------------------------- jumps
def events(B, f, k_m, k_T):
    G = len(B)
    m = float(np.median(np.abs(np.diff(f)))) if G > 1 else 0.0
    T = float(B[-1] - B[0])
    tau = max(k_m * m, k_T * T)
    if T <= 0 or tau <= 0:
        return [], dict(m=m, T=T, tau=tau)
    rising = np.zeros(G, bool)
    rising[1:] = np.diff(B) > 0
    ivs = []
    for a in range(G - 1):
        for b in range(a + 1, min(a + 3, G - 1) + 1):
            if B[b] - B[a] >= tau:
                gs = [g for g in range(a + 1, b + 1) if rising[g]]
                if gs:
                    ivs.append((gs[0], gs[-1]))
                break  # the shortest window from a is enough
    ivs.sort()
    merged = []
    for o, e in ivs:
        if merged and o <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([o, e])
    out = []
    for o, e in merged:
        size = float(B[e] - B[o - 1])
        after = float(B[min(e + PLATEAU, G - 1)] - B[e])
        plateau = (e == G - 1) or after < tau / 2
        out.append(dict(onset=o, end=e, width=e - o + 1, size=size, after=after,
                        plateau=bool(plateau), kind="jump" if plateau else "climb",
                        early=o <= EARLY))
    return out, dict(m=m, T=T, tau=tau)


# ----------------------------------------------------------------- lineage arrays
def lineage_arrays(lin):
    """Per generation g (1..Gc): step flags of L_{g-1} -> L_g, and lineage fitness."""
    Gc = lin[-1]["generation"]
    assert [r["generation"] for r in lin] == list(range(Gc + 1))
    F = np.array([r["fitness"] for r in lin], float)
    A = {fl: np.zeros(Gc + 1, bool) for fl in FLAGS}
    unknown_other = 0
    for r in lin[1:]:
        g = r["generation"]
        for fl in FLAGS:
            if fl == "elite_only":
                A[fl][g] = r["elite"]
            elif fl == "crossover":
                A[fl][g] = bool(r["crossover"]) and not r["elite"]
            else:
                v = r.get(fl)
                if v is None:
                    unknown_other += fl == "from_other_parent"
                A[fl][g] = bool(v) and not r["elite"]
    return Gc, F, A, unknown_other


def window_any(A, fl, o, e):
    if fl == "elite_only":
        return bool(A[fl][o:e + 1].all())
    return bool(A[fl][o:e + 1].any())


def null_windows(Gc, w, evs):
    blocked = np.zeros(Gc + 2, bool)
    for ev in evs:
        blocked[ev["onset"]:ev["end"] + 1] = True
    return [s for s in range(1, Gc - w + 2) if not blocked[s:s + w].any()]


def last_flag(A, fl, e):
    idx = np.nonzero(A[fl][1:e + 1])[0]
    return int(idx[-1] + 1) if len(idx) else None


# ----------------------------------------------------------------- per run
def analyse_run(run, variant):
    k_m, k_T = VARIANTS[variant]
    traj = run["trajectory"]
    f = np.array([t["best"] for t in traj], float)
    B = np.maximum.accumulate(f)
    evs, th = events(B, f, k_m, k_T)
    res = dict(run=run["run"], study=study_of(run["run"]), variant=variant, G=len(B), **th,
               events=evs)
    # Generation-best line (every run).
    sig = [t.get("best_signature") for t in traj]
    for ev in evs:
        a, b = ev["onset"] - 1, ev["end"]
        ev["genbest_changed"] = (sig[a] != sig[b]) if sig[a] and sig[b] else None
    gb_null = {}
    for ev in evs:
        w = ev["width"]
        if w not in gb_null:
            pos = null_windows(len(B) - 1, w, evs)
            vals = [sig[s - 1] != sig[s + w - 1] for s in pos if sig[s - 1] and sig[s + w - 1]]
            gb_null[w] = float(np.mean(vals)) if vals else np.nan
        ev["genbest_null"] = gb_null[w]
    lin = run.get("lineage")
    res["has_lineage"] = bool(lin)
    if not lin:
        return res
    Gc, F, A, unknown_other = lineage_arrays(lin)
    res["Gc"] = Gc
    res["unknown_other"] = unknown_other
    test = [r.get("test_auc") for r in lin]
    nullcache = {}
    for ev in evs:
        o, e = ev["onset"], ev["end"]
        ev["on_lineage_span"] = e <= Gc
        if e > Gc:
            ev["carried"] = False
            continue
        gain = float(F[o:e + 1].max() - F[o - 1])
        ev["lineage_gain"] = gain
        ev["carried"] = bool(gain >= 0.5 * ev["size"])
        for fl in FLAGS:
            ev[fl] = window_any(A, fl, o, e)
        w = ev["width"]
        if w not in nullcache:
            pos = null_windows(Gc, w, evs)
            nullcache[w] = dict(n=len(pos), **{fl: (float(np.mean([window_any(A, fl, s, s + w - 1)
                                                                  for s in pos])) if pos else np.nan)
                                              for fl in FLAGS})
        ev["null"] = nullcache[w]
        # Exploratory (not in section 1): generation-matched null, windows starting within
        # +-10 generations of the onset.
        loc = [x for x in null_windows(Gc, w, evs) if abs(x - o) <= 10]
        ev["null_local"] = dict(n=len(loc), **{fl: (float(np.mean([window_any(A, fl, x, x + w - 1)
                                                                  for x in loc])) if loc else np.nan)
                                              for fl in FLAGS})
        gs_ = last_flag(A, "structural", e)
        gn = last_flag(A, "node_added", e)
        ev["lag"] = None if gs_ is None else max(0, o - gs_)
        ev["lag_node"] = None if gn is None else max(0, o - gn)
        if test[o - 1] is not None and test[e] is not None:
            ev["test_delta_lineage"] = float(test[e] - test[o - 1])
        ev["train_delta_lineage"] = float(F[e] - F[o - 1])
        # Genes the lineage gained over the window that survive into the champion.
        ev["added_nodes"] = sorted({n for r in lin[o:e + 1] for n in r.get("added_nodes", [])})
        ev["added_conns"] = sorted({c for r in lin[o:e + 1] for c in r.get("added_conns", [])})
        ev["enabled_conns"] = sorted({c for r in lin[o:e + 1] for c in r.get("enabled_conns", [])})
    # Null lag at every non-jump generation.
    blocked = np.zeros(Gc + 1, bool)
    for ev in evs:
        blocked[ev["onset"]:min(ev["end"], Gc) + 1] = True
    lags, lags_node = [], []
    for g in range(1, Gc + 1):
        if blocked[g]:
            continue
        gs_ = last_flag(A, "structural", g)
        gn = last_flag(A, "node_added", g)
        lags.append(None if gs_ is None else g - gs_)
        lags_node.append(None if gn is None else g - gn)
    res["null_lag"] = lags
    res["null_lag_node"] = lags_node
    res["step_share"] = {fl: float(A[fl][1:].mean()) for fl in FLAGS}
    res["perm_inputs"] = dict(Gc=Gc, A={fl: A[fl].tolist() for fl in FLAGS})
    return res


# ----------------------------------------------------------------- enrichment
def jump_set(res_list, carried_only=True, late_only=False):
    out = []
    for r in res_list:
        for ev in r["events"]:
            if ev["kind"] != "jump" or "structural" not in ev:
                continue
            if carried_only and not ev["carried"]:
                continue
            if late_only and ev["early"]:
                continue
            out.append((r, ev))
    return out


NULL_KEY = "null"


def rd_stat(pairs_by_run, fl):
    n = obs = exp = 0.0
    for evs in pairs_by_run:
        for ev in evs:
            if np.isnan(ev[NULL_KEY][fl]):
                continue
            n += 1
            obs += ev[fl]
            exp += ev[NULL_KEY][fl]
    if n == 0:
        return np.nan, np.nan, 0
    return obs / n, exp / n, int(n)


def placed_share(r, evs, fl, rng):
    """One random re-placement of a run's jump windows (same widths, non-overlapping)."""
    Gc = r["perm_inputs"]["Gc"]
    A = np.array(r["perm_inputs"]["A"][fl], bool)
    taken = np.zeros(Gc + 2, bool)
    hits = 0
    for ev in sorted(evs, key=lambda x: -x["width"]):
        w = ev["width"]
        for _ in range(200):
            s = int(rng.integers(1, Gc - w + 2))
            if not taken[s:s + w].any():
                break
        taken[s:s + w] = True
        win = A[s:s + w]
        hits += bool(win.all()) if fl == "elite_only" else bool(win.any())
    return hits


def enrichment(res_list, fl, carried_only=True, late_only=False, seed=42):
    by_run = defaultdict(list)
    runs = {}
    for r, ev in jump_set(res_list, carried_only, late_only):
        by_run[r["run"]].append(ev)
        runs[r["run"]] = r
    keys = sorted(by_run)
    pj, pn, n = rd_stat([by_run[k] for k in keys], fl)
    out = dict(flag=fl, n_jumps=n, n_runs=len(keys), p_jump=pj, p_null=pn,
               rd=pj - pn if n else np.nan)
    if n == 0:
        return out
    # Bootstrap over runs (stratified by study when pooled).
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for k in keys:
        strata[runs[k]["study"]].append(k)
    rds = []
    for _ in range(N_BOOT):
        sample = []
        for st, ks in strata.items():
            sample += [by_run[ks[i]] for i in rng.integers(0, len(ks), len(ks))]
        a, b, m = rd_stat(sample, fl)
        if m:
            rds.append(a - b)
    out["rd_lo"], out["rd_hi"] = (float(np.percentile(rds, 2.5)), float(np.percentile(rds, 97.5)))
    # Bootstrap CIs of the two shares, for the chart.
    rng = np.random.default_rng(seed)
    pjs, pns = [], []
    for _ in range(2000):
        sample = []
        for st, ks in strata.items():
            sample += [by_run[ks[i]] for i in rng.integers(0, len(ks), len(ks))]
        a, b, m = rd_stat(sample, fl)
        if m:
            pjs.append(a)
            pns.append(b)
    out["p_jump_lo"], out["p_jump_hi"] = float(np.percentile(pjs, 2.5)), float(np.percentile(pjs, 97.5))
    out["p_null_lo"], out["p_null_hi"] = float(np.percentile(pns, 2.5)), float(np.percentile(pns, 97.5))
    if fl not in ("structural", "node_added", "elite_only") or NULL_KEY != "null":
        out["p_perm"] = None
        return out
    # Permutation: re-place each run's jump windows at random.
    rng = np.random.default_rng(seed)
    obs_hits = sum(ev[fl] for k in keys for ev in by_run[k])
    ge = 0
    for _ in range(N_PERM):
        h = sum(placed_share(runs[k], by_run[k], fl, rng) for k in keys)
        ge += (h <= obs_hits) if fl == "elite_only" else (h >= obs_hits)
    out["p_perm"] = (ge + 1) / (N_PERM + 1)
    return out


def main():
    runs = []
    for p in sorted(CACHE.glob("*.json")):
        runs.append(json.loads(p.read_text()))
    print(f"{len(runs)} runs", flush=True)
    results = {v: [analyse_run(r, v) for r in runs] for v in VARIANTS}

    # ---- jumps.csv
    with open(HERE / "jumps.csv", "w", newline="") as fh:
        cols = ["variant", "study", "run", "kind", "onset", "end", "width", "size", "early",
                "carried", "lineage_gain", "train_delta_lineage", "test_delta_lineage", "lag",
                "lag_node"] + FLAGS + [f"null_{fl}" for fl in FLAGS] + \
               ["genbest_changed", "genbest_null", "added_nodes", "added_conns", "enabled_conns"]
        w = csv.writer(fh)
        w.writerow(cols)
        for v, rl in results.items():
            for r in rl:
                for ev in r["events"]:
                    row = dict(variant=v, study=r["study"], run=r["run"], **ev)
                    for fl in FLAGS:
                        row[f"null_{fl}"] = ev.get("null", {}).get(fl)
                    w.writerow([";".join(map(str, row[c])) if isinstance(row.get(c), list)
                                else row.get(c) for c in cols])

    # ---- runs.csv (primary)
    with open(HERE / "runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["study", "run", "G", "champion_generation", "m", "T", "tau", "n_jumps", "n_climbs",
                "n_jumps_late", "n_carried", "gain_share_jumps", "gens_in_jumps_share",
                "gens_flat_share", "step_share_structural", "step_share_node_added",
                "step_share_elite", "test_auc_valid"]
        w.writerow(cols)
        for r, run in zip(results["primary"], runs):
            evs = r["events"]
            jumps = [e for e in evs if e["kind"] == "jump"]
            f = np.array([t["best"] for t in run["trajectory"]])
            B = np.maximum.accumulate(f)
            flat = float(np.mean(np.diff(B) <= 0)) if len(B) > 1 else np.nan
            r["gain_share_jumps"] = (sum(e["size"] for e in jumps) / r["T"]) if r["T"] > 0 else np.nan
            r["gens_in_jumps_share"] = sum(e["width"] for e in jumps) / max(1, r["G"] - 1)
            r["gens_flat_share"] = flat
            ss = r.get("step_share", {})
            w.writerow([r["study"], r["run"], r["G"], run["champion"]["generation"],
                        r["m"], r["T"], r["tau"], len(jumps), len(evs) - len(jumps),
                        sum(not e["early"] for e in jumps),
                        sum(bool(e.get("carried")) for e in jumps), r["gain_share_jumps"],
                        r["gens_in_jumps_share"], flat, ss.get("structural"),
                        ss.get("node_added"), ss.get("elite_only"), run.get("test_auc_valid")])

    # ---- enrichment
    studies = sorted({r["study"] for r in results["primary"] if r["has_lineage"]})
    enr_rows = []
    for v in VARIANTS:
        rl = [r for r in results[v] if r["has_lineage"]]
        for scope_late in (False, True):
            for carried in (True, False):
                if v != "primary" and not carried:
                    continue
                groups = [(st, [r for r in rl if r["study"] == st]) for st in studies] + \
                         [("pooled", [r for r in rl if r["study"] in POOLED])]
                for st, sub in groups:
                    for fl in FLAGS:
                        if v != "primary" and fl not in ("structural", "node_added", "elite_only"):
                            continue
                        e = enrichment(sub, fl, carried_only=carried, late_only=scope_late)
                        e.update(variant=v, study=st, late_only=scope_late, carried_only=carried)
                        enr_rows.append(e)
                        print(v, st, "late" if scope_late else "all",
                              "carried" if carried else "all-jumps", fl, e.get("n_jumps"),
                              None if not e.get("n_jumps") else
                              f"{e['p_jump']:.2f} vs {e['p_null']:.2f} RD {e['rd']:+.2f} "
                              f"[{e['rd_lo']:+.2f},{e['rd_hi']:+.2f}] p={e['p_perm']}",
                              flush=True)
    # Exploratory: generation-matched null (primary threshold, carried jumps).
    global NULL_KEY
    NULL_KEY = "null_local"
    rl = [r for r in results["primary"] if r["has_lineage"]]
    for scope_late in (False, True):
        for st, sub in [(st, [r for r in rl if r["study"] == st]) for st in studies] + \
                [("pooled", [r for r in rl if r["study"] in POOLED])]:
            for fl in ("structural", "node_added", "elite_only"):
                e = enrichment(sub, fl, carried_only=True, late_only=scope_late)
                e.pop("p_perm", None)
                e.update(variant="primary-localnull", study=st, late_only=scope_late,
                         carried_only=True)
                enr_rows.append(e)
                if st == "pooled":
                    print("localnull", st, scope_late, fl, e.get("n_jumps"), e.get("p_jump"),
                          e.get("p_null"), e.get("rd"), e.get("rd_lo"), e.get("rd_hi"), flush=True)
    NULL_KEY = "null"
    cols = ["variant", "study", "late_only", "carried_only", "flag", "n_runs", "n_jumps", "p_jump",
            "p_jump_lo", "p_jump_hi", "p_null", "p_null_lo", "p_null_hi", "rd", "rd_lo", "rd_hi",
            "p_perm"]
    with open(HERE / "enrichment.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for e in enr_rows:
            w.writerow([e.get(c) for c in cols])

    # ---- generation-best line (all runs)
    gb_rows = []
    for st in sorted({r["study"] for r in results["primary"]}):
        sub = [r for r in results["primary"] if r["study"] == st]
        evs = [ev for r in sub for ev in r["events"] if ev["kind"] == "jump"
               and ev.get("genbest_changed") is not None and not np.isnan(ev["genbest_null"])]
        late = [ev for ev in evs if not ev["early"]]
        gb_rows.append(dict(study=st, n_runs=len(sub), n_jumps=len(evs),
                            p_changed=float(np.mean([e["genbest_changed"] for e in evs])) if evs else None,
                            p_null=float(np.mean([e["genbest_null"] for e in evs])) if evs else None,
                            n_late=len(late),
                            p_changed_late=float(np.mean([e["genbest_changed"] for e in late])) if late else None,
                            p_null_late=float(np.mean([e["genbest_null"] for e in late])) if late else None))
    with open(HERE / "genbest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gb_rows[0]))
        w.writeheader()
        w.writerows(gb_rows)

    # ---- lag
    lag_rows = []
    rl = [r for r in results["primary"] if r["has_lineage"] and r["study"] in POOLED]
    for kind in ("lag", "lag_node"):
        jl = [ev[kind] for _, ev in jump_set(rl) if ev[kind] is not None]
        nl = [x for r in rl for x in r["null_" + kind] if x is not None]
        jl_late = [ev[kind] for _, ev in jump_set(rl, late_only=True) if ev[kind] is not None]
        lag_rows.append(dict(kind=kind, jumps=jl, jumps_late=jl_late, null=nl))
    with open(HERE / "lag.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "set", "lag", "count", "share"])
        for lr in lag_rows:
            for nm in ("jumps", "jumps_late", "null"):
                xs = lr[nm]
                vals, cnt = np.unique(np.minimum(xs, 20), return_counts=True) if xs else ([], [])
                for v, c in zip(vals, cnt):
                    w.writerow([lr["kind"], nm, int(v), int(c), c / len(xs)])

    out = dict(variants=VARIANTS, studies=studies, pooled=POOLED,
               results={v: [{k: x for k, x in r.items() if k not in ("perm_inputs",)}
                            for r in rl_] for v, rl_ in results.items()},
               enrichment=enr_rows, genbest=gb_rows, lag=lag_rows)
    (SCRATCH / "analysis.json").write_text(json.dumps(out, default=lambda o: None
                                                      if isinstance(o, float) and np.isnan(o) else
                                                      (o.item() if hasattr(o, "item") else str(o))))
    print("done", flush=True)


if __name__ == "__main__":
    main()
