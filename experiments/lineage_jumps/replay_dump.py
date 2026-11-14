"""Replay a pruned 10-seed / churn run in memory and dump every genome (scratch only).

The 9-10 Nov pruning deleted intermediate-generation genomes of the 10seed-heart, heart100,
monk2, churn100 and churn200 runs. The runs replay bit for bit (experiments/lineage/
replay_verification.csv), so this script reruns one exactly as experiments/lineage/
replay_lineage.py does (same RNG path, the database stubbed out, nothing written to any
database) and writes every saved genome's serialised form to a gzipped JSON-lines file
outside the repo, checking each genome's md5 against the verified replay CSV
(experiments/lineage/replay_db/<label>-s<seed>.replay.csv.gz).

    PYTHONHASHSEED=0 uv run python experiments/lineage_jumps/replay_dump.py monk2 3 --out DIR
"""
import argparse
import csv
import gzip
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("replay_lineage",
                                               REPO / "experiments/lineage/replay_lineage.py")
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)

import neat  # noqa: E402

from explaneat.db.serialization import serialize_genome  # noqa: E402

REPLAY_DB = Path(os.environ.get("LINEAGE_REPLAY_DB",
                                "/Users/mike/dev/explaneat/experiments/lineage/replay_db"))


class DumpPopulation(rl.ReplayPopulation):
    fh = None

    def _save_genomes(self, population_id, ancestry_reporter=None):
        gen = population_id
        for key, g in self.population.items():
            data = serialize_genome(g)
            self.captured.append((gen, key, data["fitness"], rl.genome_md5(data)))
            self.fh.write(json.dumps({"g": gen, "k": key, "d": data}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(rl.run_seed.DATASETS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")
    spec = rl.run_seed.DATASETS[args.dataset]
    n_gen = args.generations or spec["n_generations"]
    label = f"churn{n_gen}" if args.dataset == "churn" else args.dataset
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}-s{args.seed}.genomes.jsonl.gz"

    rl.real_db.init_db()
    X_train, y_train, _ = rl.run_seed.load_training_data(spec["dataset_id"], spec["split_id"])
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(spec["neat_config"]))
    rl.run_seed.seed_everything(args.seed)
    t0 = time.time()
    with gzip.open(out, "wt") as fh:
        DumpPopulation.fh = fh
        pop = DumpPopulation(config, X_train, y_train, mask_phantom=False)
        rl.popmod.db = rl._NullDB()
        pop.run(rl.auc_fitness, n=n_gen, nEpochs=rl.run_seed.N_EPOCHS_BACKPROP)

    ref = {}
    with gzip.open(REPLAY_DB / f"{label}-s{args.seed}.replay.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            ref[(int(r["generation"]), int(r["genome_key"]))] = r["genome_md5"]
    same = sum(1 for g, k, _, h in pop.captured if ref.get((g, k)) == h)
    status = "OK" if same == len(pop.captured) == len(ref) else "MISMATCH"
    print(json.dumps({"dataset": args.dataset, "seed": args.seed, "generations": n_gen,
                      "rows": len(pop.captured), "md5_identical": same, "ref_rows": len(ref),
                      "status": status, "wall_s": round(time.time() - t0, 1),
                      "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
