"""Evolve LunarLander-v3 policies by pure neuroevolution (neat-python + ExplaNEAT semantics).

Fitness = mean episode return over E seeded episodes. The seeds change every generation
(generation g uses TRAIN_BASE + g*E + i, i < E) and are the same for every genome within it.
Validation seeds (VAL_BASE + i) and test seeds (TEST_BASE + i) are never used here.

Writes to --out:
  generations.csv        per generation: best/mean/median fitness, best genome id and size, species
  parents.csv            generation, genome_id, parent1, parent2 (neat-python's reproduction record)
  genomes/gen_XXXX.pkl.gz  the whole evaluated population of each generation (for lineage analysis)
  champions.pkl          {generation: best genome of that generation (by training fitness)}
  run.json               arguments and seeds

Deterministic: python ``random`` seeded, seeded env resets, numpy forward pass; evaluation order
does not matter (pure function of genome and seeds). Run with PYTHONHASHSEED=0.
"""
import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

import neat
import numpy as np

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
from policy import Policy, mean_return  # noqa: E402

TRAIN_BASE = 1_000_000
VAL_BASE = 2_000_000
TEST_BASE = 3_000_000


def load_config(path):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))


_W = {}


def _init(config_path):
    import gymnasium as gym
    _W["config"] = load_config(config_path)
    _W["env"] = gym.make("LunarLander-v3")


def _eval(task):
    gid, genome, seeds = task
    m, rets = mean_return(genome, _W["config"], seeds, env=_W["env"])
    return gid, m, rets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=str(STUDY / "configs" / "lander.ini"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--stop-at", type=float, default=None,
                    help="stop when the generation's best training fitness reaches this")
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("run with PYTHONHASHSEED=0")

    out = Path(args.out or STUDY / "results" / f"seed{args.seed}")
    (out / "genomes").mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    pop = neat.Population(config)
    (out / "run.json").write_text(json.dumps(dict(
        vars(args), train_seeds=f"{TRAIN_BASE} + gen*{args.episodes} + i",
        config_text=Path(args.config).read_text()), indent=1))

    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                        initargs=(args.config,))
    gen_f = open(out / "generations.csv", "w", newline="")
    gw = csv.writer(gen_f)
    gw.writerow(["generation", "best_fitness", "mean_fitness", "median_fitness", "best_genome",
                 "best_hidden", "best_conns", "mean_hidden", "mean_conns", "n_species",
                 "eval_seconds"])
    par_f = open(out / "parents.csv", "w", newline="")
    pw = csv.writer(par_f)
    pw.writerow(["generation", "genome_id", "parent1", "parent2"])
    champions = {}

    def fitness(genomes, cfg):
        g = pop.generation
        seeds = [TRAIN_BASE + g * args.episodes + i for i in range(args.episodes)]
        t0 = time.time()
        res = pool.map(_eval, [(gid, gn, seeds) for gid, gn in genomes], chunksize=1)
        by = {gid: m for gid, m, _ in res}
        for gid, gn in genomes:
            gn.fitness = by[gid]
        dt = time.time() - t0
        fits = np.array([gn.fitness for _, gn in genomes])
        best_gid, best = max(genomes, key=lambda t: (t[1].fitness, -t[0]))
        sizes = np.array([Policy(gn, cfg).size() for _, gn in genomes])
        bh, bc = Policy(best, cfg).size()
        gw.writerow([g, f"{best.fitness:.4f}", f"{fits.mean():.4f}", f"{np.median(fits):.4f}",
                     best_gid, bh, bc, f"{sizes[:, 0].mean():.3f}", f"{sizes[:, 1].mean():.3f}",
                     len(pop.species.species) if pop.species else 0, f"{dt:.1f}"])
        gen_f.flush()
        anc = pop.reproduction.ancestors
        for gid, _ in genomes:
            p = anc.get(gid, tuple())
            pw.writerow([g, gid, p[0] if len(p) > 0 else "", p[1] if len(p) > 1 else ""])
        par_f.flush()
        with gzip.open(out / "genomes" / f"gen_{g:04d}.pkl.gz", "wb") as f:
            pickle.dump(dict(genomes), f)
        champions[g] = pickle.loads(pickle.dumps(best))
        with open(out / "champions.pkl", "wb") as f:
            pickle.dump(champions, f)
        print(f"gen {g:4d} best {best.fitness:8.2f} mean {fits.mean():8.2f} "
              f"size h{bh}/c{bc} species {len(pop.species.species) if pop.species else 0} "
              f"{dt:.1f}s", flush=True)
        if args.stop_at is not None and best.fitness >= args.stop_at:
            raise StopIteration

    try:
        pop.run(fitness, args.generations)
    except StopIteration:
        print("stop-at reached", flush=True)
    finally:
        pool.close()
        pool.join()
        gen_f.close()
        par_f.close()


if __name__ == "__main__":
    main()
