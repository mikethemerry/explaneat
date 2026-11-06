"""Study G (GRAFT_F_G_PLAN.md): a protection phase for grafts, contexts 1 (C2 judged arm)
and 2 (D2 alias cases). Context 3 (F1 recipients) is run by run_earlygen.py.

For epochs 1..P of fine-tuning the graft's entry -> graft and internal connection weights and
its node biases get zero gradient (genome_surgery.graft_protect_keys / finetune protect_*);
the readout trains throughout. P = 0 is the unprotected reference and must reproduce the
earlier CSVs (context 1 E500: transplant2_judged.csv; context 2: rescue2.csv / rescue2_es.csv),
otherwise the run stops.

    uv run python experiments/churn/run_protect.py --context c2 --regime e500 --P 0,25
    uv run python experiments/churn/run_protect.py --context d2 --regime es --P 0,25 [--shard 1/2]
No DB writes.
"""

import argparse
import time

import numpy as np
import torch

import ext_common as ec
import graft_fg as fg
import run_rescue2 as r2
import run_transplant as rt
import run_transplant2 as rt2
from ext_common import gs

TOL = 1e-9


def pname(cond, P):
    return f"{cond}-p{P}"


def check_ref(label, got, ref):
    if ref is None:
        return ""
    err = abs(got - float(ref))
    if err > TOL:
        raise SystemExit(f"P = 0 does not reproduce the earlier result for {label}: "
                         f"{got} vs {ref} (|diff| {err:.3g}); stopping")
    return err


# ====================================================================== context 1 (C2 judged)

def run_c2(data, rd, Ps, boot):
    Xtr, ytr = data["X_train"], data["y_train"]
    runs = ec.read_runs()
    cand_rows, cache = rt.candidates(runs, data)
    by_key = {(c["seed"], c["name"]): c for c in cand_rows}
    pairs = rt2.read_judged_pairs(rt2.JUDGED_PAIRS)
    ref = {}
    if rd["regime"] == "e500":
        for r in fg.read_rows(rt2.OUT_JUDGED):
            if r["kind"] == "condition":
                ref[(int(r["recipient_seed"]), r["condition"])] = r["test_auc"]
    rows, timer = [], []
    for rs, ds, name, _ in pairs:
        t0 = time.time()
        c = by_key[(ds, name)]
        assert c["qualifies"], c
        sub, s, F, t0_ = rt2.donor_signals([c], cache, Xtr)[(ds, name)]
        rg, rcfg, rinfo = cache[rs]
        donor = {"genome": cache[ds][0], "sub": sub, "s": s, "seed": ds, "name": name}
        G = fg.make_grafts(rg, rcfg, donor, rs, Xtr)
        (gg, rec), (sh, srec) = G["graft"], G["sham"]
        gr, _ = fg.refit(gg, rec, rcfg, rd["X_refit"], rd["y_refit"])
        sr, _ = fg.refit(sh, srec, rcfg, rd["X_refit"], rd["y_refit"])
        s_sham0 = fg.contribution(sr, rcfg, srec, Xtr)
        genomes, ft = {}, {}
        genomes["ftonly"], ft["ftonly"] = fg.finetune(rg, rcfg, rd, rs)
        for P in Ps:
            for cond, start, rc in (("graft-refit-ft", gr, rec), ("sham-refit-ft", sr, srec)):
                t = time.time()
                genomes[pname(cond, P)], ft[pname(cond, P)] = fg.finetune(start, rcfg, rd, rs,
                                                                          rec=rc, P=P)
                timer.append(time.time() - t)
        p_test, auc = {}, {}
        for cond, g in genomes.items():
            m, p_test[cond] = ec.evaluate(g, rcfg, data, boot)
            auc[cond] = m["test_auc"]
            row = {"kind": "condition", "context": "c2", "regime": rd["regime"],
                   "case": f"r{rs}-d{ds}-{name}", "recipient_seed": rs, "condition": cond, **m}
            row.update(fg.ft_fields("ft", ft[cond], ft[cond].get("protect_epochs", 0)))
            ref_cond = cond[:-3] if cond.endswith("-p0") else (cond if cond == "ftonly"
                                                               else None)
            if ref_cond is not None and (rs, ref_cond) in ref:
                row["ref_abs_diff"] = check_ref(f"c2 r{rs} {cond}", m["test_auc"],
                                                ref[(rs, ref_cond)])
            rows.append(row)
        case = {"kind": "case", "context": "c2", "regime": rd["regime"],
                "case": f"r{rs}-d{ds}-{name}", "unit": f"r{rs}", "recipient_seed": rs,
                "donor_seed": ds, "donor_annotation": name,
                "nov": gs.linear_novelty(s, gs.signal_basis(rg, rcfg, Xtr)[0])["nov"],
                "auc_ftonly": auc["ftonly"]}
        for P in Ps:
            case[f"auc_graft_p{P}"] = auc[pname("graft-refit-ft", P)]
            case[f"auc_sham_p{P}"] = auc[pname("sham-refit-ft", P)]
            for lab, a, b in ((f"graft_minus_sham_p{P}", pname("graft-refit-ft", P),
                               pname("sham-refit-ft", P)),
                              (f"graft_minus_ftonly_p{P}", pname("graft-refit-ft", P), "ftonly")):
                d, lo, hi = ec.paired_delta(data, p_test[a], p_test[b], boot)
                case.update({lab: d, f"{lab}_ci_lo": lo, f"{lab}_ci_hi": hi})
            case[f"surv_graft_p{P}"] = fg.survival_r(genomes[pname("graft-refit-ft", P)], rcfg,
                                                     rec, Xtr, s)
            case[f"surv_sham_p{P}"] = fg.survival_r(genomes[pname("sham-refit-ft", P)], rcfg,
                                                    srec, Xtr, s_sham0)
            for role, cond in (("graft", "graft-refit-ft"), ("sham", "sham-refit-ft")):
                fi = ft[pname(cond, P)]
                case.update({f"{role}_p{P}_{k}": v for k, v in fg.ft_fields("ft", fi, P).items()})
            if P:
                for role, cond in (("graft", "graft-refit-ft"), ("sham", "sham-refit-ft")):
                    lab = f"{role}_p{P}_minus_p0"
                    d, lo, hi = ec.paired_delta(data, p_test[pname(cond, P)],
                                                p_test[pname(cond, 0)], boot)
                    case.update({lab: d, f"{lab}_ci_lo": lo, f"{lab}_ci_hi": hi})
        rows.append(case)
        print(f"c2 r{rs} <- s{ds} {name} ({time.time() - t0:.0f}s): " + " ".join(
            f"P{P} g-s {case[f'graft_minus_sham_p{P}']:+.4f}" for P in Ps), flush=True)
    return rows


# ====================================================================== context 2 (D2)

def run_d2(data, rd, Ps, boot, shard, out=None):
    Xtr = data["X_train"]
    leaves = r2.load_pool(ec.read_runs(), data)
    leaf_rows, pair_rows, summary, prepared = r2.select(leaves, data)
    r2.check_against_committed(pair_rows, r2.committed_selection_rows())
    print("D2 selection matches the committed rescue2_selection.csv")
    cases = [r["case"] for r in pair_rows if r["include"]]
    feats = sorted({(prepared[c]["R"]["seed"], prepared[c]["R"]["name"]) for c in cases})
    k, n = shard
    mine = set(feats[k - 1::n])
    cases = [c for c in cases if (prepared[c]["R"]["seed"], prepared[c]["R"]["name"]) in mine]
    print(f"shard {k}/{n}: {len(cases)} cases over {len(mine)} recipient features")
    ref_csv = r2.OUT_ES if rd["regime"] == "es" else r2.OUT
    ref = {(r["case"], r["condition"]): r["test_auc"] for r in fg.read_rows(ref_csv)
           if r["kind"] == "condition" and r.get("status") == "ok"}
    if rd["regime"] == "es":
        for c in cases:
            r2.refit_on_fit_rows(prepared[c], rd)
    rows, koft, timer = [], {}, []
    t_start = time.time()
    n_ft = len(mine) + sum(len(prepared[c]["grafts"]) * len(Ps) for c in cases)
    for i, case in enumerate(cases):
        P_ = prepared[case]
        R, K, G = P_["R"], P_["K"], P_["grafts"]
        cfg, rs = R["config"], R["seed"]
        fk = (rs, R["name"])
        if fk not in koft:
            t = time.time()
            koft[fk] = fg.finetune(K["ko"], cfg, rd, rs)
            timer.append(time.time() - t)
        genomes, ft = {"ko-ft": koft[fk][0]}, {"ko-ft": koft[fk][1]}
        s0 = {}
        for role, gr in G.items():
            cond = f"ko-{role}-refit-ft"
            s0[role] = (gr["donor"]["s"] if role != "sham"
                        else fg.contribution(gr["refit"], cfg, gr["rec"], Xtr))
            for P in Ps:
                t = time.time()
                genomes[pname(cond, P)], ft[pname(cond, P)] = fg.finetune(
                    gr["refit"], cfg, rd, rs, rec=gr["rec"], P=P)
                timer.append(time.time() - t)
        p_test, auc = {}, {}
        for cond, g in genomes.items():
            m, p_test[cond] = ec.evaluate(g, cfg, data, boot)
            auc[cond] = m["test_auc"]
            row = {"kind": "condition", "context": "d2", "regime": rd["regime"], "case": case,
                   "recipient_seed": rs, "recipient_feature": R["name"], "condition": cond, **m}
            row.update(fg.ft_fields("ft", ft[cond], ft[cond].get("protect_epochs", 0)))
            ref_cond = cond[:-3] if cond.endswith("-p0") else (cond if cond == "ko-ft" else None)
            if ref_cond is not None:
                row["ref_abs_diff"] = check_ref(f"d2 {case} {cond}", m["test_auc"],
                                                ref.get((case, ref_cond)))
            rows.append(row)
        cr = {"kind": "case", "context": "d2", "regime": rd["regime"], "case": case,
              "unit": f"r{rs}-{R['name']}", "recipient_seed": rs, "recipient_feature": R["name"],
              "alias_seed": P_["A"]["seed"], "alias_feature": P_["A"]["name"],
              "alias_r": P_["row"]["r"], "auc_ko-ft": auc["ko-ft"]}
        for P in Ps:
            for role in G:
                cond = pname(f"ko-{role}-refit-ft", P)
                cr[f"auc_{role}_p{P}"] = auc[cond]
                cr[f"surv_{role}_p{P}"] = fg.survival_r(genomes[cond], cfg, G[role]["rec"], Xtr,
                                                        s0[role])
                cr.update({f"{role}_p{P}_{k}": v for k, v in
                           fg.ft_fields("ft", ft[cond], P).items()})
                if P:
                    lab = f"{role}_p{P}_minus_p0"
                    d, lo, hi = ec.paired_delta(data, p_test[cond],
                                                p_test[pname(f"ko-{role}-refit-ft", 0)], boot)
                    cr.update({lab: d, f"{lab}_ci_lo": lo, f"{lab}_ci_hi": hi})
            a = pname("ko-alias-refit-ft", P)
            if a not in p_test:  # alias graft not runnable (collision), as in D2
                cr["status"] = P_["row"].get("alias_graft_status", "alias not run")
                continue
            for other, b in (("sham", pname("ko-sham-refit-ft", P)),
                             ("wrong", pname("ko-wrong-refit-ft", P)), ("koft", "ko-ft")):
                if b in p_test:
                    lab = f"alias_minus_{other}_p{P}"
                    d, lo, hi = ec.paired_delta(data, p_test[a], p_test[b], boot)
                    cr.update({lab: d, f"{lab}_ci_lo": lo, f"{lab}_ci_hi": hi})
        rows.append(cr)
        if out is not None:
            fg.write_rows(out, rows)  # rewritten after every case
        el = time.time() - t_start
        print(f"[{i + 1}/{len(cases)}] {case} ({len(timer)}/{n_ft} fine-tunes, est. remaining "
              f"{el / max(len(timer), 1) * (n_ft - len(timer)) / 3600:.2f} h): " + " ".join(
                  f"P{P} a-s {cr.get(f'alias_minus_sham_p{P}', float('nan')):+.4f}" for P in Ps),
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Study G contexts 1 and 2")
    ap.add_argument("--context", choices=["c2", "d2"], required=True)
    ap.add_argument("--regime", choices=list(fg.REGIMES), required=True)
    ap.add_argument("--P", default="0,25")
    ap.add_argument("--shard", default="1/1")
    args = ap.parse_args()
    Ps = [int(p) for p in args.P.split(",")]
    if Ps[0] != 0:
        raise SystemExit("--P must start with 0 (the unprotected reference)")
    k, n = (int(v) for v in args.shard.split("/"))
    torch.set_num_threads(1)
    from explaneat.db.base import db
    db.init_db()
    commit = fg.require_clean_tree()
    print(f"commit {commit}; context {args.context} regime {args.regime} P {Ps} shard {k}/{n}")
    data = ec.load_data()
    tag = "" if n == 1 else f"_part-{k}-of-{n}"
    ptag = "" if Ps == [0, 25] else "_P" + "-".join(map(str, Ps))
    out = fg.HERE / f"protect_{args.context}_{args.regime}{ptag}{tag}.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    val = ec.load_val_split(data)[:4] if args.regime == "es" else None
    rd = fg.regime_data(data, args.regime, val)
    boot = ec.bootstrap_indices(len(data["y_test"]))
    t0 = time.time()
    if args.context == "c2":
        rows = run_c2(data, rd, Ps, boot)
    else:
        rows = run_d2(data, rd, Ps, boot, (k, n), out)
    fg.write_rows(out, rows)
    print(f"done ({(time.time() - t0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
