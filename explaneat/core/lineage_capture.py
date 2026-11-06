"""Parent capture for one NEAT generation (RNG-neutral).

neat-python's ``DefaultReproduction.ancestors`` maps a genome key to the keys
of its parents, ``(parent1_key, parent2_key)``, and is cumulative over the
whole run (it is never cleared). The parent keys name genomes of the
*previous* generation's population. Two cases need care:

- **Elites.** ``reproduce`` carries each species' elites into the next
  generation as the same object under the same key. They get no new
  ``ancestors`` entry, so ``ancestors[key]`` still holds the parents from
  when the genome was first created, possibly many generations back. An elite
  copy's parent is therefore the *same key one generation earlier*: we record
  ``parent1_key = key``, ``parent2_key = None`` and ``is_elite_copy = True``.
  (The copy is not identical to its parent row: backprop trains it again in
  the new generation, so its weights and fitness move.)
- **Generation 0.** ``create_new`` records ``ancestors[key] = ()``, so both
  parents are ``None``.

A crossover child whose two parents are the same genome has
``parent1_key == parent2_key``; that is a clone under a new key, not an elite.

Nothing here touches any RNG, so capturing parents cannot change a run.
"""
from typing import Dict, Iterable, Optional, Tuple

ParentRecord = Tuple[Optional[int], Optional[int], bool]


def generation_parents(population_keys: Iterable[int],
                       previous_keys: Iterable[int],
                       ancestors: Dict[int, tuple]) -> Dict[int, ParentRecord]:
    """Parent keys of every genome in one generation.

    Args:
        population_keys: keys of the generation's genomes.
        previous_keys: keys of the previous generation's genomes (empty for
            generation 0 or an unknown predecessor, e.g. a resumed run).
        ancestors: ``reproduction.ancestors``.

    Returns:
        ``{key: (parent1_key, parent2_key, is_elite_copy)}``.
    """
    prev = set(previous_keys)
    out: Dict[int, ParentRecord] = {}
    for key in population_keys:
        if key in prev:
            out[key] = (key, None, True)
            continue
        parents = tuple(ancestors.get(key) or ())
        p1 = parents[0] if len(parents) > 0 else None
        p2 = parents[1] if len(parents) > 1 else None
        out[key] = (p1, p2, False)
    return out


def parent_rows(generation: int, records: Dict[int, ParentRecord]):
    """Rows ``(generation, genome_key, parent1_key, parent2_key, is_elite_copy)``."""
    return [(generation, k, p1, p2, el)
            for k, (p1, p2, el) in sorted(records.items())]
