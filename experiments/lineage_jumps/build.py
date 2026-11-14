"""Jumps and plateaus, stage 1: per-run trajectory and champion lineage (read-only).

For one run this writes ``<cache>/<run>.json`` with:
- per generation: best, mean and size of the population's training fitness, the key and
  structure signature of the generation's best genome (the "generation-best line");
- the champion (best-by-train over all generations: highest fitness, earliest generation,
  lowest key) and its fitter-parent lineage, one row per generation, each step classified
  from the gene diff against the followed parent (RESULTS.md section 1.4), with the genes it
  added and, where a split is linked and the stored fitness reproduces as train AUC, the
  ancestor's test AUC.

Sources:
- DB runs (genomes kept): populations/genomes in explaneat_dev, parents from the DB columns.
- Pruned runs (heart100, monk2, churn200): fitness and parents from the verified replay CSV
  (experiments/lineage/replay_db), genomes from replay_dump.py's dump (scratch).

    uv run python experiments/lineage_jumps/build.py --db shape-C-s1-g300
    uv run python experiments/lineage_jumps/build.py --replay heart100 3
"""
import argparse
import csv
import gzip
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments" / "churn"), str(REPO / "experiments" / "10seed")):
    if p not in sys.path:
        sys.path.insert(0, p)

from threadpoolctl import threadpool_limits  # noqa: E402

threadpool_limits(1)
import torch  # noqa: E402

torch.set_num_threads(1)

from sqlalchemy import text  # noqa: E402

from explaneat.analysis import lineage as L  # noqa: E402
from explaneat.db import db  # noqa: E402

SCRATCH = Path(os.environ.get(
    "JUMPS_SCRATCH",
    "/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad"))
CACHE = SCRATCH / "runs"
DUMPS = SCRATCH / "dump"
REPLAY_DB = Path("/Users/mike/dev/explaneat/experiments/lineage/replay_db")
REPLAY_EXP = {"heart100": "10seed-heart100-s{s}", "monk2": "10seed-monk2-s{s}",
              "churn200": "churn200-s{s}"}


class RO:
    def __enter__(self):
        self.s = db.Session()
        self.s.execute(text("SET TRANSACTION READ ONLY"))
        return self.s

    def __exit__(self, *exc):
        self.s.rollback()
        self.s.close()
        db.Session.remove()
        return False


def gene_keys(data):
    nodes = {int(k) for k in data.get("nodes", {})}
    conns = {(int(c["in_node"]), int(c["out_node"])): bool(c["enabled"])
             for c in data.get("connections", {}).values()}
    return nodes, conns


def classify(child, parent, other):
    """Step class of child vs the followed parent (RESULTS.md 1.4)."""
    d = L.diff_genome_data(parent, child)
    cn, cc = gene_keys(child)
    pn, pc = gene_keys(parent)
    from_other = []
    if other is not None:
        on, oc = gene_keys(other)
        from_other = [str(n) for n in sorted(cn - pn) if n in on] + \
                     [f"{a}_{b}" for (a, b) in sorted(set(cc) - set(pc)) if (a, b) in oc]
    added_nodes = sorted(cn - pn)
    added_conns = sorted(set(cc) - set(pc))
    enabled = [f"{a}_{b}" for (a, b) in sorted(set(cc) & set(pc)) if cc[(a, b)] and not pc[(a, b)]]
    disabled = [f"{a}_{b}" for (a, b) in sorted(set(cc) & set(pc)) if pc[(a, b)] and not cc[(a, b)]]
    structural = bool(d["structural_change"] or from_other)
    return dict(structural=structural, node_added=bool(added_nodes), conn_added=bool(added_conns),
                conn_enabled=bool(enabled), conn_disabled=bool(disabled),
                removed=bool(d["removed_nodes"] or d["removed_connections"]),
                from_other_parent=bool(from_other), n_from_other_parent=len(from_other),
                added_nodes=added_nodes, added_conns=[f"{a}_{b}" for a, b in added_conns],
                enabled_conns=enabled, max_abs_weight_change=d["summary"]["max_abs_weight_change"])


def walk(rows, parents, champ):
    """Fitter-parent lineage, as lineage.get_lineage: ties to parent1, elites to the same key."""
    chain, gk = [], champ
    while True:
        g, k = gk
        p1, p2, elite = parents.get(gk, (None, None, False))
        if g == 0 or (p1 is None and p2 is None):
            chain.append((gk, None))
            break
        if elite:
            nxt, fol = (g - 1, k), "elite"
        elif p2 is None:
            nxt, fol = (g - 1, p1), "parent1"
        else:
            f1 = rows.get((g - 1, p1), -np.inf)
            f2 = rows.get((g - 1, p2), -np.inf)
            f1 = -np.inf if f1 is None else f1
            f2 = -np.inf if f2 is None else f2
            nxt, fol = ((g - 1, p2), "parent2") if f2 > f1 else ((g - 1, p1), "parent1")
        chain.append((gk, fol))
        gk = nxt
    return chain[::-1]


def trajectory(rows):
    gens = sorted({g for g, _ in rows})
    out = []
    for g in gens:
        fs = [(f, k) for (gg, k), f in rows.items() if gg == g and f is not None]
        best = max(fs, key=lambda t: (t[0], -t[1]))
        out.append(dict(generation=g, best=best[0], best_key=best[1],
                        mean=float(np.mean([f for f, _ in fs])), n=len(fs)))
    return out


def champion(rows):
    return max(((f, -g, -k, (g, k)) for (g, k), f in rows.items() if f is not None))[3]


def test_auc_fn(session, exp_row):
    """Callable genome_data -> (train_auc, test_auc), or None when there is no split."""
    if exp_row["split_id"] is None:
        return None
    try:
        from compute_runs import load_split
        import genome_surgery as gs
        from sklearn.metrics import roc_auc_score
        from explaneat.core.config_utils import load_neat_config
        from explaneat.db.serialization import deserialize_genome
        data = load_split(session, str(exp_row["split_id"]))
        config = load_neat_config(exp_row["neat_config_text"] or "", exp_row["config_json"])
    except Exception as e:  # noqa: BLE001
        print(f"  no test AUC: {type(e).__name__}: {e}", flush=True)
        return None

    def f(gd):
        g = deserialize_genome(gd, config)
        tr = float(roc_auc_score(data["y_train"], gs.predict_proba(g, config, data["X_train"])))
        te = float(roc_auc_score(data["y_test"], gs.predict_proba(g, config, data["X_test"])))
        return tr, te
    return f


def exp_row(session, name):
    r = session.execute(text(
        "select id, name, split_id, neat_config_text, config_json, created_at, hardware_info "
        "from experiments where name = :n order by created_at"), {"n": name}).mappings().all()
    return r


def build_db(exp_id, label=None):
    with RO() as s:
        e = s.execute(text("select id, name, split_id, neat_config_text, config_json, created_at "
                           "from experiments where id = :i"), {"i": exp_id}).mappings().one()
        q = s.execute(text(
            "select p.generation, g.genome_id, g.id, g.fitness, g.parent1_id, g.parent2_id "
            "from genomes g join populations p on g.population_id = p.id "
            "where p.experiment_id = :i"), {"i": exp_id}).all()
        rows, ids, rawp = {}, {}, {}
        for gen, k, rid, fit, p1, p2 in q:
            rows[(gen, k)] = fit
            ids[(gen, k)] = rid
            rawp[(gen, k)] = (p1, p2)
        id2gk = {str(v): kk for kk, v in ids.items()}
        has_parents = any(p1 is not None for p1, _ in rawp.values())
        parents = None
        if has_parents:
            parents = {}
            for (g, k), (p1, p2) in rawp.items():
                if p1 is None and p2 is None and g > 0 and (g - 1, k) in rows:
                    # Old runs (2025-10, 2026-04) recorded no parent for elite copies.
                    parents[(g, k)] = (k, None, True)
                    continue
                k1 = id2gk.get(str(p1), (None, None))[1] if p1 is not None else None
                k2 = id2gk.get(str(p2), (None, None))[1] if p2 is not None else None
                parents[(g, k)] = (k1, k2, k1 == k and p2 is None)
        traj = trajectory(rows)
        need = {(t["generation"], t["best_key"]) for t in traj}
        champ = champion(rows)
        chain = walk(rows, parents, champ) if parents else None
        if chain:
            for (g, k), fol in chain:
                need.add((g, k))
                p1, p2, el = parents.get((g, k), (None, None, False))
                for pk in (p1, p2):
                    if pk is not None and (g - 1, pk) in ids:
                        need.add((g - 1, pk))
        need_ids = [ids[x] for x in need if x in ids]
        gd = {}
        for i in range(0, len(need_ids), 500):
            for rid, data in s.execute(text("select id, genome_data from genomes where id = any(:ids)"),
                                       {"ids": need_ids[i:i + 500]}):
                gd[str(rid)] = data
        get = lambda gk: gd.get(str(ids[gk])) if gk in ids else None  # noqa: E731
        tfn = test_auc_fn(s, e)
        out = assemble(e["name"], label or e["name"], traj, rows, parents, champ, chain, get, tfn,
                       source="db", genome_id=lambda gk: str(ids[gk]))
        out["experiment_id"] = str(e["id"])
        out["created_at"] = str(e["created_at"])
    return out


PREFIX = {"churn100": ("churn200", 100, "churn100-s{s}"),
          "heart": ("heart100", 40, "10seed-heart-s{s}")}


def build_replay(study, seed):
    max_gen, label = None, study
    if study in PREFIX:  # bit-identical prefix of a longer replayed run
        study, max_gen, pat = PREFIX[label]
        name = pat.format(s=seed)
    else:
        name = REPLAY_EXP[study].format(s=seed)
    rows, parents = {}, {}
    with gzip.open(REPLAY_DB / f"{study}-s{seed}.replay.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            gk = (int(r["generation"]), int(r["genome_key"]))
            if max_gen is not None and gk[0] >= max_gen:
                continue
            rows[gk] = float(r["fitness"])
            p1 = int(r["parent1_key"]) if r["parent1_key"] else None
            p2 = int(r["parent2_key"]) if r["parent2_key"] else None
            parents[gk] = (p1, p2, r["is_elite_copy"] == "1")
    traj = trajectory(rows)
    champ = champion(rows)
    chain = walk(rows, parents, champ)
    need = {(t["generation"], t["best_key"]) for t in traj}
    for (g, k), fol in chain:
        need.add((g, k))
        p1, p2, el = parents[(g, k)]
        for pk in (p1, p2):
            if pk is not None:
                need.add((g - 1, pk))
    gd = {}
    with gzip.open(DUMPS / f"{study}-s{seed}.genomes.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if max_gen is not None and r["g"] >= max_gen:
                break
            if (r["g"], r["k"]) in need:
                gd[(r["g"], r["k"])] = r["d"]
    with RO() as s:
        e = exp_row(s, name)[-1]
        tfn = test_auc_fn(s, e)
        # DB ids of the stored genomes (final generation and referenced ones survive pruning).
        idmap = {(g, k): str(i) for g, k, i in s.execute(text(
            "select p.generation, g.genome_id, g.id from genomes g join populations p on "
            "g.population_id = p.id where p.experiment_id = :i"), {"i": e["id"]}).all()}
        out = assemble(name, f"{label}-s{seed}", traj, rows, parents, champ, chain, gd.get, tfn,
                       source="replay", genome_id=lambda gk: idmap.get(gk))
        out["experiment_id"] = str(e["id"])
        out["created_at"] = str(e["created_at"])
    return out


def assemble(name, label, traj, rows, parents, champ, chain, get, tfn, source, genome_id):
    for t in traj:
        d = get((t["generation"], t["best_key"]))
        t["best_signature"] = L.structure_signature(d) if d else None
    lin = None
    test_ok = None
    lineage_error = None
    if chain and (chain[0][0][0] != 0 or any(get(gk) is None for gk, _ in chain)):
        lineage_error = ("chain does not reach generation 0" if chain[0][0][0] != 0
                         else "lineage genome missing")
        chain = None
    if chain:
        lin = []
        prev = None
        for (g, k), fol in chain:
            d = get((g, k))
            p1, p2, elite = parents.get((g, k), (None, None, False))
            n, c = gene_keys(d)
            row = dict(generation=g, key=k, fitness=rows[(g, k)], followed=fol, elite=bool(elite),
                       parent1=p1, parent2=p2, crossover=bool(p2 is not None and p2 != p1),
                       n_nodes=len(n), n_conns=len(c), n_enabled=sum(c.values()),
                       signature=L.structure_signature(d), genome_id=genome_id((g, k)),
                       pop_best=next(t["best"] for t in traj if t["generation"] == g))
            if prev is not None:
                if elite:
                    cl = classify(d, get(prev), None)
                    cl["structural"] = cl["structural"]  # elite copies have no genetic change
                else:
                    other_key = p2 if fol == "parent1" else p1
                    other = get((g - 1, other_key)) if (other_key is not None and other_key != prev[1]) \
                        else None
                    cl = classify(d, get(prev), other)
                    if row["crossover"] and other is None:
                        cl["from_other_parent"] = None  # other parent's genome not available
                row.update(cl)
            if tfn is not None:
                tr, te = tfn(d)
                row["train_auc_recomputed"], row["test_auc"] = tr, te
            lin.append(row)
            prev = (g, k)
        if tfn is not None:
            err = max(abs(r["train_auc_recomputed"] - r["fitness"]) for r in lin)
            test_ok = bool(err < 1e-6)
            if not test_ok:
                for r in lin:
                    r.pop("test_auc", None)
            for r in lin:
                r.pop("train_auc_recomputed", None)
        else:
            err = None
    return dict(run=label, experiment=name, source=source, trajectory=traj,
                champion=dict(generation=champ[0], key=champ[1], fitness=rows[champ],
                              genome_id=genome_id(champ)),
                lineage=lin, test_auc_valid=test_ok, lineage_error=lineage_error)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="*", default=[], help="experiment names or ids")
    ap.add_argument("--replay", nargs="*", default=[], help="study seed [seed ...]")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    db.init_db()
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.replay:
        study, seeds = args.replay[0], [int(x) for x in args.replay[1:]]
        jobs += [("replay", study, s) for s in seeds]
    for nm in args.db:
        jobs.append(("db", nm, None))
    for kind, a, b in jobs:
        if kind == "replay":
            out_path = CACHE / f"{a}-s{b}.json"
            if out_path.exists() and not args.force:
                continue
            out = build_replay(a, b)
        else:
            try:
                uuid.UUID(a)
                eid = a
                with RO() as s:
                    nm = s.execute(text("select name, created_at from experiments where id = :i"),
                                   {"i": a}).one()
                label = f"old:{nm[0]}:{nm[1]:%Y-%m-%d}:{a[:8]}"
            except ValueError:
                with RO() as s:
                    r = exp_row(s, a)
                assert len(r) == 1, (a, len(r))
                eid, label = str(r[0]["id"]), a
            out_path = CACHE / f"{label.replace(' ', '_').replace(':', '__')}.json"
            if out_path.exists() and not args.force:
                continue
            out = build_db(eid, label)
        out_path.write_text(json.dumps(out))
        lin = out["lineage"]
        print(f"{out['run']}: {len(out['trajectory'])} generations, champion gen "
              f"{out['champion']['generation']} fitness {out['champion']['fitness']:.4f}, "
              f"lineage {len(lin) if lin else 'none'}, test AUC valid {out['test_auc_valid']}",
              flush=True)


if __name__ == "__main__":
    main()
