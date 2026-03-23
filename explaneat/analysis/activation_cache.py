"""LRU cache for activation data to avoid redundant forward passes."""
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np


class ActivationCache:
    """LRU cache for activation results keyed by (genome_id, split_id, annotation_id).

    Thread-safe is not guaranteed; intended for single-request-at-a-time usage.
    """

    def __init__(self, max_entries: int = 50):
        self._cache: OrderedDict[tuple, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._max_entries = max_entries

    def get(
        self, genome_id: str, split_id: str, annotation_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached (entry_acts, exit_acts) or None."""
        key = (genome_id, split_id, annotation_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(
        self,
        genome_id: str,
        split_id: str,
        annotation_id: str,
        entry_acts: np.ndarray,
        exit_acts: np.ndarray,
    ) -> None:
        """Store activation data."""
        key = (genome_id, split_id, annotation_id)
        self._cache[key] = (entry_acts, exit_acts)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def invalidate_genome(self, genome_id: str) -> None:
        """Remove all entries for a given genome."""
        keys_to_remove = [k for k in self._cache if k[0] == genome_id]
        for key in keys_to_remove:
            del self._cache[key]

    def clear(self) -> None:
        """Remove all entries."""
        self._cache.clear()


# Global singleton
activation_cache = ActivationCache()
