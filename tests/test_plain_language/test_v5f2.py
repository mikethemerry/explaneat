"""v5f2: in the overview each cluster's input copies sit directly beneath that cluster."""

import re

import pytest

from explaneat.analysis.plain_language_store import load_layer
from explaneat.export.render_structure import _tname, overview


def _positions(tikz):
    return {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"\\node\[[^\]]*\] \((\w+)\) at \((-?[\d.]+),(-?[\d.]+)\)", tikz)}


@pytest.mark.parametrize("label", ["heart-paper", "churn100-s4"])
def test_inputs_sit_under_their_cluster(label):
    lay = load_layer(label)
    if lay is None:
        pytest.skip(f"{label} layer missing")
    tk, meta = overview(lay)
    pos = _positions(tk)
    cl_of = meta["clusters_of"]
    band = meta["band_of"]
    # inputs of different clusters never interleave along x
    seq = [cl_of[n] for n in sorted(cl_of, key=lambda n: pos[_tname(n)][0])]
    blocks = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
    assert len(blocks) == len(set(blocks))
    # each cluster's inputs lie within the x-range between its neighbours' inputs and units
    order = list(dict.fromkeys(blocks))
    for i, c in enumerate(order):
        xs = [pos[_tname(n)][0] for n in cl_of if cl_of[n] == c]
        mem = [pos[_tname(n)][0] for n in band if band[n] == c and n not in cl_of and _tname(n) in pos]
        for d in order[i + 1:]:
            other = [pos[_tname(n)][0] for n in band if band[n] == d and _tname(n) in pos]
            assert max(xs + mem) < min(other)
    # every edge from an input goes upward, into its own cluster
    for u, v, _ in meta["merged"]:
        if u in cl_of:
            assert pos[_tname(v)][1] > pos[_tname(u)][1]
            assert band[v] == cl_of[u]
