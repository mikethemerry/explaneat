"""Concept probe under both poolings, and class selectivity.

Synthetic task: ``explaneat.conv.analysis.concept_probe`` (ROC-AUC
selectivity ``s_c = 2 AUC - 1`` of the pooled map for each planted concept;
mono / poly / conjunctive / none at tau = 0.5), run once with spatial-max
pooling and once with global-average pooling, since the categories depend on
the pooling (RESULTS.md section 4).

Digits: no concept labels are available and stroke concepts were not built
(skipped: they would need hand-labelled or synthesised stroke annotations).
Instead each node's **class selectivity** (2 AUC - 1 of the pooled map for
class 1 vs class 0, e.g. "8" vs "3") is reported under both poolings.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..analysis import STRONG, _auc, concept_probe
from ..genome import ConvGenome


def probe_both(g: ConvGenome, maps: Dict[int, np.ndarray], C: np.ndarray, y: np.ndarray,
               names: List[str]) -> Dict[int, dict]:
    pm = concept_probe(g, maps, C, y, names, "max")
    pa = concept_probe(g, maps, C, y, names, "mean")
    out = {}
    for k in pm:
        out[k] = {"max": pm[k], "mean": pa[k]}
    return out


def class_selectivity(maps: Dict[int, np.ndarray], y: np.ndarray, kinds: Dict[int, str]) -> Dict[int, dict]:
    out = {}
    for k, m in maps.items():
        if kinds[k] == "input":
            continue
        a = m.reshape(len(y), -1)
        smax = 2 * _auc(a.max(1), y) - 1
        smean = 2 * _auc(a.mean(1), y) - 1
        out[k] = {"max": smax, "mean": smean,
                  "category": ("class-1" if smax >= STRONG else "class-0" if smax <= -STRONG else "none")}
    return out
