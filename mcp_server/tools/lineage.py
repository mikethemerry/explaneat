"""Lineage tools: ancestor chains, fitness jumps and genome diffs.

Thin JSON wrappers over ``explaneat.analysis.lineage``. Parents come from
``Genome.parent1_id``/``parent2_id`` when recorded, else from the replay
sidecar ``experiments/lineage/parents/<experiment>.csv.gz``.
"""
import json
from typing import Optional

from explaneat.analysis import lineage as L

from mcp_server.server import get_db


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def _err(e: Exception) -> str:
    return _dump({"error": f"{type(e).__name__}: {e}"})


def get_lineage(genome_id: str, mode: str = "fitter_parent", compact: bool = False) -> str:
    """Ancestor chain of a genome from generation 0 to the genome, one row per generation.

    mode: "fitter_parent" (at each crossover follow the fitter parent, the one
    NEAT takes disjoint/excess genes from) or "parent1". Elite copies (same NEAT
    key carried into the next generation) follow their own key one generation back.

    Each row: generation, genome_id, genome_key, fitness, train_auc, n_nodes,
    n_connections, n_enabled_connections, parent1_id, parent2_id, parent1_key,
    parent2_key, is_elite_copy, followed, structure_signature (changes only when
    nodes/connections/enabled flags change). compact=True keeps only generation,
    genome_id, genome_key, fitness, n_nodes, n_connections, is_elite_copy and
    structural_change versus the previous row.
    """
    try:
        lin = L.get_lineage(genome_id, mode=mode, db=get_db())
    except Exception as e:  # noqa: BLE001 - reported to the caller
        return _err(e)
    if compact:
        out, prev = [], None
        for r in lin:
            out.append({k: r[k] for k in ("generation", "genome_id", "genome_key", "fitness",
                                          "n_nodes", "n_connections", "is_elite_copy")}
                       | {"structural_change": prev is not None and
                          r["structure_signature"] != prev["structure_signature"]})
            prev = r
        lin = out
    return _dump({"genome_id": genome_id, "mode": mode, "length": len(lin), "lineage": lin})


def get_ancestor(genome_id: str, generation: int, mode: str = "fitter_parent") -> str:
    """The lineage ancestor of a genome at a given generation, with that
    generation's best genome overall for comparison (fitness gap, the
    ancestor's rank in its generation)."""
    try:
        return _dump(L.ancestor_at_generation(genome_id, int(generation), mode=mode, db=get_db()))
    except Exception as e:  # noqa: BLE001
        return _err(e)


def diff_genomes(genome_a_id: str, genome_b_id: str, include_weights: bool = True) -> str:
    """Gene-level diff from genome A to genome B: added/removed nodes,
    added/removed connections (keys "<in>_<out>"), enabled flips, and weight and
    bias changes as key -> [old, new]. include_weights=False drops the
    weight/bias maps and keeps their counts in summary."""
    try:
        d = L.diff_genomes(genome_a_id, genome_b_id, db=get_db())
    except Exception as e:  # noqa: BLE001
        return _err(e)
    if not include_weights:
        d.pop("weight_changes")
        d.pop("bias_changes")
    return _dump(d)


def detect_jumps(genome_id: str, metric: str = "fitness", min_delta: Optional[float] = None,
                 method: str = "threshold", k: int = 5, mode: str = "fitter_parent") -> str:
    """Fitness jumps along a genome's lineage.

    method="threshold": steps with delta >= min_delta (default: 95th percentile
    of the lineage's positive deltas). method="top_k": the k largest positive
    deltas. Each jump: generation, delta, before_genome_id, after_genome_id,
    structural_change (did nodes/connections/enabled flags change at that step),
    is_elite_copy. Use diff_genomes on before/after to see the genes involved.
    """
    try:
        lin = L.get_lineage(genome_id, mode=mode, db=get_db())
        jumps = L.detect_jumps(lin, metric=metric, min_delta=min_delta, method=method, k=int(k))
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return _dump({"genome_id": genome_id, "mode": mode, "metric": metric, "method": method,
                  "min_delta": min_delta, "n_steps": len(lin) - 1, "jumps": jumps})


def register(mcp) -> None:
    """Register lineage tools with the MCP server."""
    mcp.tool()(get_lineage)
    mcp.tool()(get_ancestor)
    mcp.tool()(diff_genomes)
    mcp.tool()(detect_jumps)
