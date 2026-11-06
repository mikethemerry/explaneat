"""Lineage routes: ancestor chain with population context, fitness jumps
(with the annotations of the descendant's explanation that contain the genes
each jump added), genome diffs, the diff as a drawable union graph, and the
ancestor at a generation. Thin wrappers over ``explaneat.analysis.lineage``.

Read-only: these routes never create Explanation rows.
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ...analysis import lineage as L
from ...db import db
from ...db.models import Experiment, Genome, Population

router = APIRouter()

NOT_RECORDED = (
    "Lineage not recorded for this run (the genomes have no complete parent ids and there is no "
    "replay sidecar): run the replay/backfill (experiments/lineage/replay_lineage.py, "
    "backfill_parents.py) to reconstruct it."
)


def _population_stats(experiment_id) -> List[Dict[str, Any]]:
    with db.session_scope() as s:
        rows = (
            s.query(Population.generation, Population.best_fitness, Population.mean_fitness,
                    Population.stdev_fitness, Population.population_size)
            .filter(Population.experiment_id == uuid.UUID(str(experiment_id)))
            .order_by(Population.generation)
            .all()
        )
        return [dict(generation=g, best_fitness=b, mean_fitness=m, stdev_fitness=sd,
                     population_size=n) for g, b, m, sd, n in rows]


def _lineage_or_none(genome_id: str, mode: str):
    try:
        return L.get_lineage(genome_id, mode=mode, db=db), None
    except LookupError as e:
        L.clear_cache()  # so a request after the backfill sees the new parents
        return None, str(e)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _experiment_of(genome_id: str):
    with db.session_scope() as s:
        g = s.get(Genome, uuid.UUID(genome_id))
        if g is None:
            raise HTTPException(status_code=404, detail="Genome not found")
        p = s.get(Population, g.population_id)
        e = s.get(Experiment, p.experiment_id)
        return str(e.id), e.name, p.generation


@router.get("/{genome_id}/lineage")
async def get_lineage(genome_id: str, mode: str = Query("fitter_parent", pattern="^(fitter_parent|parent1)$")):
    """Ancestor chain from generation 0, plus per-generation population best and
    mean fitness. ``available`` is false (with a message) when the run has no
    recorded parents."""
    exp_id, exp_name, gen = _experiment_of(genome_id)
    lineage, err = _lineage_or_none(genome_id, mode)
    prev = None
    for r in lineage or []:
        r["structural_change"] = prev is not None and r.get("structure_signature") != prev.get("structure_signature")
        prev = r
    return {
        "genome_id": genome_id,
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "generation": gen,
        "mode": mode,
        "available": lineage is not None,
        "message": None if lineage is not None else f"{NOT_RECORDED} ({err})",
        "lineage": lineage or [],
        "population": _population_stats(exp_id),
    }


def _added_genes(diff: dict) -> List[Any]:
    genes: List[Any] = [int(n) for n in diff["added_nodes"]]
    genes += list(diff["added_connections"])
    genes += [c for c, (old, new) in diff["enabled_changed"].items() if new and not old]
    return genes


@router.get("/{genome_id}/lineage/jumps")
async def get_jumps(
    genome_id: str,
    mode: str = Query("fitter_parent", pattern="^(fitter_parent|parent1)$"),
    metric: str = "fitness",
    min_delta: Optional[float] = None,
    method: str = Query("threshold", pattern="^(threshold|top_k)$"),
    k: int = Query(5, ge=1, le=50),
):
    """Fitness jumps along the lineage. Each jump carries the gene diff of its
    step and, for the genes it added, the annotations of ``genome_id``'s
    explanation that contain them (``map_genes_to_annotations``)."""
    lineage, err = _lineage_or_none(genome_id, mode)
    if lineage is None:
        return {"available": False, "message": f"{NOT_RECORDED} ({err})", "jumps": []}
    jumps = L.detect_jumps(lineage, metric=metric, min_delta=min_delta, method=method, k=k)
    for j in jumps:
        d = L.diff_genomes(j["before_genome_id"], j["after_genome_id"], db=db)
        j["diff_summary"] = d["summary"]
        j["added_nodes"] = d["added_nodes"]
        j["added_connections"] = d["added_connections"]
        genes = _added_genes(d)
        j["annotations"] = {}
        if genes:
            try:
                m = L.map_genes_to_annotations(genes, genome_id, db=db)
                j["gene_annotations"] = m["genes"]
                names = sorted({n for g in m["genes"].values() for n in g["containing"]})
                j["annotations"] = {n: [lab for lab, g in m["genes"].items() if n in g["containing"]]
                                    for n in names}
                j["genes_not_in_final"] = [lab for lab, g in m["genes"].items() if not g["containing"]]
            except Exception as e:  # no explanation for the descendant, etc.
                j["annotation_error"] = str(e)
    return {"available": True, "metric": metric, "method": method, "jumps": jumps}


@router.get("/{genome_a}/diff/{genome_b}")
async def diff(genome_a: str, genome_b: str):
    """Gene-level diff from genome A to genome B (keys as in the serialised genome)."""
    try:
        d = L.diff_genomes(genome_a, genome_b, db=db)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # JSON keys must be strings; tuples become lists
    d["enabled_changed"] = {k: list(v) for k, v in d["enabled_changed"].items()}
    d["weight_changes"] = {k: list(v) for k, v in d["weight_changes"].items()}
    d["bias_changes"] = {str(k): list(v) for k, v in d["bias_changes"].items()}
    d["other_node_changes"] = {str(k): v for k, v in d["other_node_changes"].items()}
    return d


def _phenotype(session, genome_id: str):
    from ...core.config_utils import load_neat_config
    from ...core.explaneat import ExplaNEAT

    g = session.get(Genome, uuid.UUID(genome_id))
    if g is None:
        raise HTTPException(status_code=404, detail=f"Genome {genome_id} not found")
    exp = g.population.experiment
    config = load_neat_config(exp.neat_config_text or "", exp.config_json)
    return ExplaNEAT(g.to_neat_genome(config), config).get_phenotype_network(), g


@router.get("/{genome_a}/diff/{genome_b}/graph")
async def diff_graph(genome_a: str, genome_b: str, labels_from: Optional[str] = None):
    """Union of the two phenotypes for drawing: every node and connection has
    ``status`` "common", "added" (only in B) or "removed" (only in A); common
    connections carry ``weight_a``/``weight_b``. Node display names come from
    ``labels_from``'s explanation renames (default: B), if any."""
    from ...core.model_state import ModelStateEngine
    from ...db.models import Explanation

    with db.session_scope() as s:
        pa, ga = _phenotype(s, genome_a)
        pb, gb = _phenotype(s, genome_b)
        names: Dict[str, str] = {}
        src = labels_from or genome_b
        ex = s.query(Explanation).filter(Explanation.genome_id == uuid.UUID(src)).first()
        renames = [op for op in (ex.operations if ex else []) or [] if op.get("type") == "rename_node"]
        for op in renames:
            p = op.get("params", {})
            names[str(p.get("node_id"))] = p.get("display_name")
        fa, fb = ga.fitness, gb.fitness
    na = {n.id: n for n in pa.nodes}
    nb = {n.id: n for n in pb.nodes}
    nodes = []
    for nid in sorted(set(na) | set(nb), key=lambda x: (int(x) if x.lstrip("-").isdigit() else 0, x)):
        n = nb.get(nid) or na[nid]
        status = "common" if nid in na and nid in nb else ("added" if nid in nb else "removed")
        nodes.append(dict(id=nid, type=n.type.value, bias=n.bias, activation=n.activation,
                          response=n.response, aggregation=n.aggregation,
                          display_name=names.get(nid), status=status,
                          bias_a=na[nid].bias if nid in na else None,
                          bias_b=nb[nid].bias if nid in nb else None))
    ca = {(c.from_node, c.to_node): c for c in pa.connections if c.enabled}
    cb = {(c.from_node, c.to_node): c for c in pb.connections if c.enabled}
    conns = []
    for key in sorted(set(ca) | set(cb)):
        c = cb.get(key) or ca[key]
        status = "common" if key in ca and key in cb else ("added" if key in cb else "removed")
        conns.append({"from": key[0], "to": key[1], "weight": c.weight, "enabled": True,
                      "status": status,
                      "weight_a": ca[key].weight if key in ca else None,
                      "weight_b": cb[key].weight if key in cb else None})
    inputs = sorted(set(pa.input_node_ids) | set(pb.input_node_ids))
    outputs = sorted(set(pa.output_node_ids) | set(pb.output_node_ids))
    return {"genome_a": genome_a, "genome_b": genome_b, "fitness_a": fa, "fitness_b": fb,
            "nodes": nodes, "connections": conns,
            "metadata": {"input_nodes": inputs, "output_nodes": outputs, "is_original": True}}


@router.get("/{genome_id}/ancestor")
async def ancestor(genome_id: str, generation: int = Query(..., ge=0),
                   mode: str = Query("fitter_parent", pattern="^(fitter_parent|parent1)$")):
    """The lineage ancestor at ``generation`` and that generation's best genome."""
    try:
        return L.ancestor_at_generation(genome_id, generation, mode=mode, db=db)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=f"{NOT_RECORDED} ({e})")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
