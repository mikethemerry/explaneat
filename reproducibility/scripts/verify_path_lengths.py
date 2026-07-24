"""Verify 'maximum active path length' for all three paper models under ONE
metric, and reconcile the Adult capital-gain cascade description in Section 5.2.

Metric (stated once, applied uniformly): the maximum number of EDGES (hops) on
a directed path from any input node to the output node, over enabled
connections in the evolved phenotype (the active input->output subgraph).

Paper/draft claims to check: Adult 4, Heart 3, MONK 3.

Run: uv run python scripts/aaai27/verify_path_lengths.py
"""
import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from aaai27.common import ADULT, HEART, load_model

from explaneat.db import db
from explaneat.db.models import Experiment, Population, Genome
from explaneat.core.config_utils import load_neat_config
from explaneat.core.explaneat import ExplaNEAT

MONK = {"label": "monk2", "genome_id": "23dde0fa-a635-4974-91aa-c05266ea0747",
        "experiment_id": "45e998d7-dac1-4e6b-953e-3f1d32cbc59e"}
CLAIMS = {"adult": 4, "heart": 3, "monk2": 3}


def longest_path_hops(pheno):
    """Longest input->output path length (edges) over enabled phenotype edges."""
    succ = {}
    preds = {}
    for c in pheno.connections:
        if not c.enabled:
            continue
        succ.setdefault(c.from_node, []).append(c.to_node)
        preds.setdefault(c.to_node, []).append(c.from_node)
    inputs = set(pheno.input_node_ids)
    outputs = set(pheno.output_node_ids)

    # longest distance from any input, via DFS with memo (DAG)
    memo = {}

    def dist(node):
        if node in memo:
            return memo[node]
        if node in inputs or node not in preds:
            memo[node] = 0
            return 0
        memo[node] = 1 + max(dist(p) for p in preds[node])
        return memo[node]

    # also recover one longest chain to the output for reporting
    out = next(iter(outputs))
    d = dist(out)

    chain = [out]
    cur = out
    while cur not in inputs and cur in preds:
        cur = max(preds[cur], key=dist)
        chain.append(cur)
    return d, list(reversed(chain))


def load_genome(session, spec):
    exp = session.query(Experiment).filter_by(id=uuid.UUID(spec["experiment_id"])).first()
    g = session.query(Genome).filter_by(id=uuid.UUID(spec["genome_id"])).first()
    config = load_neat_config(exp.neat_config_text or "", exp.config_json)
    return g.to_neat_genome(config), config


def main():
    print("metric: max EDGES (hops) on a directed input->output path in the "
          "evolved phenotype (enabled edges).\n")
    with db.session_scope() as s:
        for spec, key in [(ADULT, "adult"), (HEART, "heart"), (MONK, "monk2")]:
            if key in ("adult", "heart"):
                genome, config, *_ = load_model(s, spec)
            else:
                genome, config = load_genome(s, spec)
            pheno = ExplaNEAT(genome, config).get_phenotype_network()
            hops, chain = longest_path_hops(pheno)
            claim = CLAIMS[key]
            flag = "OK" if hops == claim else f"MISMATCH (paper says {claim})"
            print(f"{key:6}: longest active path = {hops} hops  [{flag}]")
            print(f"        chain: {' -> '.join(str(c) for c in chain)}")

        # ---- Adult capital-gain cascade reconciliation ----
        print("\n=== Adult capital-gain pathway (Section 5.2 reconciliation) ===")
        genome, config, *_ = load_model(s, ADULT)
        CASCADE = [3446, 4739, 2129, 655, 313, 5102, 6421]   # paper's 'seven neurons'
        present = [n for n in CASCADE if n in genome.nodes]
        print(f"pathway neurons present in genome: {len(present)}/7  {present}")
        # edges into the output (node 0) from the pathway, with signs
        print("enabled connections INTO output (node 0):")
        into_out = [(f, t, c.weight) for (f, t), c in genome.connections.items()
                    if t == 0 and c.enabled]
        for f, t, w in sorted(into_out, key=lambda r: r[0]):
            tag = "  [cascade readout]" if f in CASCADE else ""
            print(f"   {f:>6} -> 0   w = {w:+.4f}{tag}")
        readout_edges = [(f, w) for f, t, w in into_out if f in CASCADE]
        print(f"cascade readout edges into output: {len(readout_edges)} "
              f"(paper says 'four connections of opposing sign')")
        pos = [f for f, w in readout_edges if w > 0]
        neg = [f for f, w in readout_edges if w < 0]
        print(f"   positive: {pos}   negative: {neg}")


if __name__ == "__main__":
    main()
