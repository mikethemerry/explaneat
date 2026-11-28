"""The champion's lineage: fitter-parent chain, structural change per step, held-out score.

From ``champion.pkl`` (evaluate.py) walk back through ``parents.csv``: neat-python's crossover puts
the fitter parent first, so parent1 is followed (elites keep their id across generations, so each
genome is taken at its birth generation). For each ancestor: birth generation, phenotype size,
connections added/removed versus its parent (phenotype, enabled, input->output paths), hidden
nodes added/removed, the RMS change of shared weights, and the mean return on 20 validation seeds.

Writes lineage.csv into the run folder. A "jump" here = a step whose validation mean rises by more
than --jump (default 50) over its parent's.
"""
import argparse
import csv
import gzip
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
from evaluate import _init, _score  # noqa: E402
from policy import ACTION_NAMES, INPUT_NAMES, Policy  # noqa: E402
from run import VAL_BASE, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--val", type=int, default=20)
    ap.add_argument("--jump", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=str(STUDY / "configs" / "lander.ini"))
    args = ap.parse_args()
    run = Path(args.run_dir)
    config = load_config(args.config)
    gc = config.genome_config
    name = {k: INPUT_NAMES[i] for i, k in enumerate(gc.input_keys)}
    name.update({k: ACTION_NAMES[i] for i, k in enumerate(gc.output_keys)})

    parents, birth = {}, {}
    with open(run / "parents.csv") as f:
        for r in csv.DictReader(f):
            gid, g = int(r["genome_id"]), int(r["generation"])
            if gid not in birth:
                birth[gid] = g
                parents[gid] = int(r["parent1"]) if r["parent1"] else None
    champ = pickle.loads((run / "champion.pkl").read_bytes())
    chain = [champ.key]
    while parents.get(chain[-1]) is not None:
        chain.append(parents[chain[-1]])
    chain.reverse()

    cache = {}

    def genome(gid):
        g = birth[gid]
        if g not in cache:
            with gzip.open(run / "genomes" / f"gen_{g:04d}.pkl.gz", "rb") as f:
                cache[g] = pickle.load(f)
        return cache[g][gid]

    gens = [genome(k) for k in chain]
    seeds = [VAL_BASE + i for i in range(args.val)]
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.config,))
    val = dict(pool.map(_score, [(i, gn, seeds) for i, gn in enumerate(gens)], chunksize=1))
    pool.close()
    pool.join()

    def pheno(gn):
        p = Policy(gn, config)
        conns = {(a, b): w for b, lst in p.incoming.items() for a, w in lst}
        return p, conns

    def label(k):
        return name.get(k, f"h{k}")

    rows, prev = [], None
    for i, (gid, gn) in enumerate(zip(chain, gens)):
        p, conns = pheno(gn)
        v = float(np.mean(val[i]))
        row = dict(step=i, genome_id=gid, birth_generation=birth[gid],
                   train_fitness=f"{gn.fitness:.2f}", val_mean=f"{v:.2f}",
                   hidden=len(p.hidden), conns=len(conns))
        if prev is not None:
            pp, pc, pv = prev
            add = sorted(set(conns) - set(pc))
            rem = sorted(set(pc) - set(conns))
            shared = set(conns) & set(pc)
            rms = np.sqrt(np.mean([(conns[k] - pc[k]) ** 2 for k in shared])) if shared else 0.0
            row.update(
                d_val=f"{v - pv:.2f}",
                conns_added=" ".join(f"{label(a)}>{label(b)}" for a, b in add),
                conns_removed=" ".join(f"{label(a)}>{label(b)}" for a, b in rem),
                hidden_added=" ".join(label(h) for h in sorted(set(p.hidden) - set(pp.hidden))),
                hidden_removed=" ".join(label(h) for h in sorted(set(pp.hidden) - set(p.hidden))),
                weight_rms_change=f"{rms:.3f}",
                jump=int(v - pv > args.jump))
        rows.append(row)
        prev = (p, conns, v)
    cols = ["step", "genome_id", "birth_generation", "train_fitness", "val_mean", "d_val",
            "jump", "hidden", "conns", "conns_added", "conns_removed", "hidden_added",
            "hidden_removed", "weight_rms_change"]
    with open(run / "lineage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    for r in rows:
        if r.get("jump"):
            print(r)
    print(f"{len(rows)} ancestors, {sum(int(r.get('jump', 0)) for r in rows)} jumps > {args.jump}")


if __name__ == "__main__":
    main()
