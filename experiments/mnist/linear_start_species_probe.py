"""Tiny no-training diagnostic for neat-python 0.92's overspeciation freeze.

    uv run python experiments/mnist/linear_start_species_probe.py

Uses 2 inputs, 1 output and a 6-genome target. Does not touch real runs.
The fresh-species probe illustrates consolidation only; it is not a production
controller and discards species identity/history. Full retraining is deferred.
"""
import json
import random
import tempfile
from pathlib import Path

import neat
import linear_start as LS


def main():
    random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        cp = LS.write_config(Path(tmp), ["NEAT.pop_size=6", "DefaultGenome.num_inputs=2",
            "DefaultGenome.num_outputs=1", "DefaultSpeciesSet.compatibility_threshold=0.0"])
        cfg = LS.load_config(cp, False)
    reporters = neat.reporting.ReporterSet()
    stagnation = cfg.stagnation_type(cfg.stagnation_config, reporters)
    rep = cfg.reproduction_type(cfg.reproduction_config, reporters, stagnation)
    pop = rep.create_new(cfg.genome_type, cfg.genome_config, cfg.pop_size)
    species = cfg.species_set_type(cfg.species_set_config, reporters)
    species.speciate(cfg, pop, 0)
    initial_species = len(species.species)
    births = []
    for generation in range(3):
        for g in pop.values():
            g.fitness = float(generation)
        old = set(pop)
        pop = rep.reproduce(cfg, species, cfg.pop_size, generation)
        cfg.species_set_config.compatibility_threshold = 1e9
        species.speciate(cfg, pop, generation + 1)
        births.append(dict(generation=generation + 1, species=len(species.species),
                           population=len(pop), new_keys=len(set(pop) - old)))
    assert initial_species == 6
    assert births[-1] == dict(generation=3, species=6, population=12, new_keys=0)
    fresh = cfg.species_set_type(cfg.species_set_config, reporters)
    fresh.speciate(cfg, pop, 3)
    assert len(fresh.species) == 1
    for g in pop.values():
        g.fitness = 4.0
    old = set(pop)
    consolidated = rep.reproduce(cfg, fresh, cfg.pop_size, 4)
    assert len(consolidated) == 6 and len(set(consolidated) - old) == 4
    out = dict(dependency="neat-python==0.92", target_population=6,
               initial_species=initial_species, huge_threshold_generations=births,
               fresh_species_count=1, after_fresh_reproduction=dict(population=6, new_keys=4),
               limitation="Fresh rebuild discards species history; this is a mechanism probe only.")
    (LS.LOG_DIR / "species_probe.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
