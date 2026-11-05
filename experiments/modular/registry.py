"""Module library (PLAN.md "Module library").

A module is an alias family of chunks. Each family has a module id; each
member (a *variant*) is one chunk found in one island's best genome at one
checkpoint, with its weights and its contribution on the task's probe set.

Alignment rule (pre-registered): a new chunk joins the existing module whose
variant has the highest Pearson r between contributions on the probe set,
if that r >= ALIAS_R (0.9); ties -> lowest module id. Otherwise it founds a
new module id (ids count up from 1, in the order chunks are added).

The library keeps, per module id, the latest variant from each island (a
newer checkpoint replaces that island's older variant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from chunks import Chunk

ALIAS_R = 0.9


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Variant:
    module_id: int
    island: int
    generation: int
    genome_key: int
    chunk: Chunk
    variant_id: int

    def to_json(self) -> dict:
        return {"module_id": self.module_id, "island": self.island,
                "generation": self.generation, "genome_key": self.genome_key,
                "variant_id": self.variant_id, "chunk": self.chunk.to_json()}


@dataclass
class ModuleLibrary:
    alias_r: float = ALIAS_R
    modules: Dict[int, Dict[int, Variant]] = field(default_factory=dict)  # mid -> island -> v
    history: List[dict] = field(default_factory=list)
    _next_id: int = 1
    _next_variant: int = 1

    def align(self, s) -> Tuple[Optional[int], float]:
        """(module id, r) of the best-correlated module with r >= alias_r, else (None, best r)."""
        best_m, best_r = None, -np.inf
        for mid in sorted(self.modules):
            for isl in sorted(self.modules[mid]):
                r = pearson(s, self.modules[mid][isl].chunk.s_probe)
                if np.isfinite(r) and r > best_r:
                    best_m, best_r = mid, r
        if best_m is not None and best_r >= self.alias_r:
            return best_m, float(best_r)
        return None, float(best_r) if np.isfinite(best_r) else float("nan")

    def add(self, chunk: Chunk, island: int, generation: int, genome_key: int) -> Variant:
        mid, r = self.align(chunk.s_probe)
        new_family = mid is None
        if new_family:
            mid = self._next_id
            self._next_id += 1
            self.modules[mid] = {}
        v = Variant(mid, island, generation, genome_key, chunk, self._next_variant)
        self._next_variant += 1
        self.modules[mid][island] = v
        self.history.append({"generation": generation, "island": island,
                             "genome_key": genome_key, "module_id": mid,
                             "variant_id": v.variant_id, "new_module": new_family,
                             "align_r": r, "gain": chunk.gain,
                             "n_hidden": len(chunk.hidden_nodes),
                             "n_entries": len(chunk.entry_inputs),
                             "entries": list(chunk.entry_inputs),
                             "n_exits": len(chunk.exit_nodes)})
        return v

    def module_ids(self) -> List[int]:
        return sorted(self.modules)

    def variants(self, mid: int) -> List[Variant]:
        return [self.modules[mid][i] for i in sorted(self.modules[mid])]

    def best_variant(self, mid: int, exclude_island=None, exclude_genome_nodes=None):
        """Highest-gain variant of module ``mid`` not from ``exclude_island``
        (ties: lowest island). None if there is none."""
        vs = [v for v in self.variants(mid) if v.island != exclude_island]
        if not vs:
            return None
        return sorted(vs, key=lambda v: (-v.chunk.gain, v.island))[0]
