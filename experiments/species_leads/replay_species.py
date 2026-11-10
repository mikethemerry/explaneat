"""Replay a recorded run in memory and capture NEAT's species membership.

Exploratory, not pre-registered (species-leads study, thesis Ch 7).

Species were never stored (the ``species`` table is empty, ``genomes.species_id``
is NULL, ``populations.num_species`` is 0: the runner reads a ``species_id``
attribute NEAT genomes do not have). The runs are bit-identical on replay
(experiments/lineage/replay_verification.csv), so this script reuses the
lineage replay (``experiments/lineage/replay_lineage.py``: the real
``DatabaseBackpropPopulation.run`` loop with the database stubbed out) and,
additively, reads ``self.species`` at the moment each generation is saved.
Nothing in the training path is changed and nothing reads an RNG.

Which species object belongs to the evaluated population
--------------------------------------------------------
In ``DatabaseBackpropPopulation.run`` one iteration for generation ``g`` is::

    backprop + fitness(population_g)       # evaluation
    _save_genomes(...)                     # <- captured here
    population_{g+1} = reproduce(species, ...)
    species.speciate(population_{g+1}, generation=g)

``population_0`` is speciated in ``BackpropPopulation.__init__`` with
``generation=0``. So when ``_save_genomes`` runs for generation ``g`` the
species set holds the partition of exactly ``population_g`` (computed from
its pre-backprop genomes at the end of iteration ``g-1``, or in ``__init__``
for ``g=0``), and that partition is what NEAT's ``reproduce`` then uses for
fitness sharing, stagnation and elitism of ``population_g``. The speciate call
after the last generation (labelling ``population_n``) partitions a population
that is never evaluated and is ignored here.

NEAT's ``Species.created`` is the ``generation`` argument of the speciate
call that created it, i.e. ``g-1`` for a species first holding members in
evaluated generation ``g`` (and ``0`` for both generations 0 and 1). The
unambiguous ``first_generation`` (first evaluated generation in which the
species id holds members) is recorded as well.

Outputs (local, not committed), under ``experiments/species_leads/replay_db/``:

- ``<label>-s<seed>.replay.csv.gz``: ``generation, genome_key, fitness,
  genome_md5`` for every saved genome of every generation, in the format
  ``experiments/lineage/verify_replay.py`` reads.
- ``<label>-s<seed>.species.json``: the final generation's membership
  ``{genome_key: species_id}`` and, per species, representative key, NEAT
  ``created``/``last_improved``, first evaluated generation and size.

    PYTHONHASHSEED=0 uv run python experiments/species_leads/replay_species.py heart 0
    PYTHONHASHSEED=0 uv run python experiments/species_leads/replay_species.py churn 0 --generations 100
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

_spec = importlib.util.spec_from_file_location(
    "replay_lineage", REPO / "experiments/lineage/replay_lineage.py")
replay_lineage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_lineage)

import neat  # noqa: E402

run_seed = replay_lineage.run_seed
OUT_DIR = HERE / "replay_db"


def species_snapshot(species_set, population_keys, generation):
    """Membership of the evaluated population, read from a DefaultSpeciesSet.

    Pure read (no RNG, no mutation). Asserts that the species set partitions
    exactly ``population_keys``.

    Returns ``(membership, species)`` where ``membership`` is
    ``{genome_key: species_id}`` and ``species`` is
    ``{species_id: {representative_key, created, last_improved, size}}``.
    """
    keys = set(population_keys)
    membership = {}
    species = {}
    for sid, s in species_set.species.items():
        members = set(s.members)
        if not members:
            continue
        for k in members:
            assert k not in membership, f"gen {generation}: genome {k} in two species"
            membership[k] = sid
        rep = s.representative.key
        assert rep in members, f"gen {generation}: species {sid} representative not a member"
        species[sid] = dict(representative_key=rep, created=s.created,
                            last_improved=s.last_improved, size=len(members))
    assert set(membership) == keys, (
        f"gen {generation}: species partition {len(membership)} keys != population {len(keys)}")
    g2s = {k: species_set.genome_to_species[k] for k in keys}
    assert g2s == membership, f"gen {generation}: genome_to_species disagrees with members"
    return membership, species


class SpeciesReplayPopulation(replay_lineage.ReplayPopulation):
    """Lineage replay that also records species membership at each save."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.species_by_gen = {}  # gen -> (membership, species)

    def _save_genomes(self, population_id, ancestry_reporter=None):
        super()._save_genomes(population_id, ancestry_reporter)
        gen = population_id  # replay's _save_population_state returns the generation
        self.species_by_gen[gen] = species_snapshot(self.species, self.population.keys(), gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(run_seed.DATASETS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    spec = run_seed.DATASETS[args.dataset]
    n_gen = args.generations or spec["n_generations"]
    label = args.label or (f"churn{n_gen}" if args.dataset == "churn" else args.dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{label}-s{args.seed}"

    replay_lineage.real_db.init_db()
    X_train, y_train, _ = run_seed.load_training_data(spec["dataset_id"], spec["split_id"])

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(spec["neat_config"]))
    run_seed.seed_everything(args.seed)
    t0 = time.time()
    pop = SpeciesReplayPopulation(config, X_train, y_train, mask_phantom=False)
    replay_lineage.popmod.db = replay_lineage._NullDB()
    pop.run(replay_lineage.auc_fitness, n=n_gen, nEpochs=run_seed.N_EPOCHS_BACKPROP)

    with gzip.open(f"{stem}.replay.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "fitness", "genome_md5"])
        for gen, key, fit, h in pop.captured:
            w.writerow([gen, key, repr(fit), h])

    first_gen = {}
    for gen in sorted(pop.species_by_gen):
        for sid in pop.species_by_gen[gen][1]:
            first_gen.setdefault(sid, gen)
    final = max(pop.species_by_gen)
    membership, species = pop.species_by_gen[final]
    for sid, s in species.items():
        s["first_generation"] = first_gen[sid]
    out = dict(dataset=args.dataset, seed=args.seed, label=label, generations=n_gen,
               final_generation=final,
               n_species_by_generation={g: len(v[1]) for g, v in sorted(pop.species_by_gen.items())},
               membership={str(k): v for k, v in sorted(membership.items())},
               species={str(k): v for k, v in sorted(species.items())})
    Path(f"{stem}.species.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"label": label, "seed": args.seed, "final_generation": final,
                      "n_species": len(species), "rows": len(pop.captured),
                      "wall_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
