"""Species capture for the species-leads replay (experiments/species_leads/).

The capture must be a pure read of NEAT's species set: same partition as
``genome_to_species``, no RNG consumed.
"""
import importlib.util
import random
from pathlib import Path

import neat
import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "experiments/10seed/configs/heart.ini"


def _load():
    spec = importlib.util.spec_from_file_location(
        "replay_species", REPO / "experiments/species_leads/replay_species.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def _speciated(seed=0, threshold=None):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(CONFIG))
    if threshold is not None:
        config.species_set_config.compatibility_threshold = threshold
    random.seed(seed)
    reporters = neat.reporting.ReporterSet()
    stagnation = config.stagnation_type(config.stagnation_config, reporters)
    repro = config.reproduction_type(config.reproduction_config, reporters, stagnation)
    pop = repro.create_new(config.genome_type, config.genome_config, 30)
    for g in pop.values():  # mutate so some genomes fall in different species
        for _ in range(3):
            g.mutate(config.genome_config)
    species = config.species_set_type(config.species_set_config, reporters)
    species.speciate(config, pop, 0)
    return pop, species


def test_snapshot_matches_genome_to_species(mod):
    pop, species = _speciated(threshold=0.3)
    membership, info = mod.species_snapshot(species, pop.keys(), 0)
    assert membership == {k: species.genome_to_species[k] for k in pop}
    assert sum(v["size"] for v in info.values()) == len(pop)
    assert len(info) == len(species.species) > 1
    for sid, v in info.items():
        assert membership[v["representative_key"]] == sid
        assert v["created"] == 0


def test_snapshot_consumes_no_randomness(mod):
    pop, species = _speciated()
    state = random.getstate()
    mod.species_snapshot(species, pop.keys(), 0)
    assert random.getstate() == state


def test_snapshot_rejects_partition_of_another_population(mod):
    pop, species = _speciated()
    with pytest.raises(AssertionError):
        mod.species_snapshot(species, list(pop.keys())[:-1], 0)
