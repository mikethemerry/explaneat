"""Lineage of a genome: its ancestor chain, fitness jumps, genome diffs, and
which annotations of an explanation contain given genes.

Parents come from ``Genome.parent1_id``/``parent2_id`` when the run recorded
them (``DatabaseBackpropPopulation`` does by default since 2026-11-07), and
otherwise from a replay sidecar ``experiments/lineage/parents/<experiment
name>.csv.gz`` with columns ``generation, genome_key, parent1_key,
parent2_key, is_elite_copy`` (keys are NEAT genome keys; parent keys name
genomes of generation - 1). See ``experiments/lineage/replay_lineage.py``.

Elite copies: NEAT carries a species' elites into the next generation under
the same key. Their parent is the same key one generation earlier
(``is_elite_copy`` True, ``parent2`` None). Backprop trains the copy again,
so its weights and fitness can still change.

Public interface (stable; used by the API/UI and the MCP tools):

- ``get_lineage(genome_id, mode="fitter_parent" | "parent1", db=None)``
- ``detect_jumps(lineage, metric="fitness", min_delta=None, method="threshold" | "top_k", k=5)``
- ``diff_genomes(genome_a_id, genome_b_id, db=None)``
- ``ancestor_at_generation(genome_id, generation, mode="fitter_parent", db=None)``
- ``map_genes_to_annotations(gene_keys, genome_id, explanation=None, db=None)``

Everything here only reads the database.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SIDECAR_DIR = REPO / "experiments" / "lineage" / "parents"
MODES = ("fitter_parent", "parent1")

__all__ = [
    "get_lineage", "detect_jumps", "diff_genomes", "ancestor_at_generation",
    "map_genes_to_annotations", "sidecar_dir", "load_sidecar", "structure_signature",
    "base_node_id",
]


# --------------------------------------------------------------------- plumbing
def _db(db=None):
    if db is not None:
        return db
    from explaneat.db import db as global_db
    if getattr(global_db, "engine", None) is None:
        global_db.init_db()
    return global_db


def _uuid(x) -> uuid.UUID:
    return x if isinstance(x, uuid.UUID) else uuid.UUID(str(x))


def sidecar_dir() -> Path:
    return Path(os.environ.get("EXPLANEAT_LINEAGE_DIR", DEFAULT_SIDECAR_DIR))


def load_sidecar(experiment_name: str, directory: Optional[Path] = None
                 ) -> Optional[Dict[Tuple[int, int], Tuple[Optional[int], Optional[int], bool]]]:
    """``{(generation, key): (parent1_key, parent2_key, is_elite_copy)}`` or None."""
    d = Path(directory) if directory else sidecar_dir()
    for path in (d / f"{experiment_name}.csv.gz", d / f"{experiment_name}.csv"):
        if path.exists():
            opener = gzip.open if path.suffix == ".gz" else open
            out = {}
            with opener(path, "rt") as f:
                for r in csv.DictReader(f):
                    p1 = int(r["parent1_key"]) if r["parent1_key"] not in ("", None) else None
                    p2 = int(r["parent2_key"]) if r["parent2_key"] not in ("", None) else None
                    el = str(r.get("is_elite_copy", "0")).strip().lower() in ("1", "true")
                    out[(int(r["generation"]), int(r["genome_key"]))] = (p1, p2, el)
            return out
    return None


def structure_signature(genome_data: dict) -> str:
    """md5 of the genome's structure: node keys and (connection key, enabled).
    Weights and biases are excluded, so equal signatures mean no structural change."""
    nodes = sorted(int(k) for k in genome_data.get("nodes", {}))
    conns = sorted((int(c["in_node"]), int(c["out_node"]), bool(c["enabled"]))
                   for c in genome_data.get("connections", {}).values())
    return hashlib.md5(json.dumps([nodes, conns]).encode()).hexdigest()


class _ExperimentIndex:
    """Light per-experiment index of genome rows and parents."""

    def __init__(self, experiment_id, experiment_name, fitness_is_auc, rows, parents, source):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.fitness_is_auc = fitness_is_auc
        self.rows = rows  # (gen, key) -> dict(id, fitness, n_nodes, n_connections, n_enabled)
        self.by_id = {r["id"]: gk for gk, r in rows.items()}
        self.parents = parents  # (gen, key) -> (p1_key, p2_key, elite) or None
        self.parent_source = source
        self.gen_best = {}
        for (g, k), r in rows.items():
            f = r["fitness"] if r["fitness"] is not None else -np.inf
            cur = self.gen_best.get(g)
            if cur is None or f > cur[0] or (f == cur[0] and k < cur[1]):
                self.gen_best[g] = (f, k)


def _build_index(session, experiment_id) -> _ExperimentIndex:
    from explaneat.db.models import Experiment, Genome, Population

    exp = session.get(Experiment, _uuid(experiment_id))
    if exp is None:
        raise KeyError(f"experiment not found: {experiment_id}")
    hw = exp.hardware_info or {}
    fitness_fn = (hw.get("run_params") or {}).get("fitness_function")
    q = (session.query(Population.generation, Genome.genome_id, Genome.id, Genome.fitness,
                       Genome.num_nodes, Genome.num_connections, Genome.num_enabled_connections,
                       Genome.parent1_id, Genome.parent2_id)
         .join(Genome, Genome.population_id == Population.id)
         .filter(Population.experiment_id == exp.id))
    rows, raw_parents = {}, {}
    for g, k, rid, fit, nn, nc, ne, p1, p2 in q:
        rows[(g, k)] = dict(id=str(rid), fitness=fit, n_nodes=nn, n_connections=nc,
                            n_enabled_connections=ne)
        raw_parents[(g, k)] = (p1, p2)
    id_to_gk = {r["id"]: gk for gk, r in rows.items()}
    has_db = any(p1 is not None for (p1, _) in raw_parents.values())
    if has_db:
        parents, source = {}, "db"
        for (g, k), (p1, p2) in raw_parents.items():
            if p1 is None and p2 is None:
                parents[(g, k)] = (None, None, False)
                continue
            k1 = id_to_gk.get(str(p1))[1] if p1 is not None and str(p1) in id_to_gk else None
            k2 = id_to_gk.get(str(p2))[1] if p2 is not None and str(p2) in id_to_gk else None
            parents[(g, k)] = (k1, k2, k1 == k and p2 is None)
    else:
        parents = load_sidecar(exp.name)
        source = "sidecar" if parents is not None else None
    return _ExperimentIndex(str(exp.id), exp.name, fitness_fn == "auc", rows, parents, source)


_INDEX_CACHE: Dict[Tuple[int, str], _ExperimentIndex] = {}


def _index_for(session, db, experiment_id) -> _ExperimentIndex:
    key = (id(db), str(experiment_id))
    if key not in _INDEX_CACHE:
        if len(_INDEX_CACHE) > 8:
            _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
        _INDEX_CACHE[key] = _build_index(session, experiment_id)
    return _INDEX_CACHE[key]


def clear_cache():
    _INDEX_CACHE.clear()


def _locate(session, genome_id):
    from explaneat.db.models import Genome, Population
    row = (session.query(Genome.id, Genome.genome_id, Population.generation,
                         Population.experiment_id)
           .join(Population, Genome.population_id == Population.id)
           .filter(Genome.id == _uuid(genome_id)).first())
    if row is None:
        raise KeyError(f"genome not found: {genome_id}")
    return row


# --------------------------------------------------------------------- lineage
def get_lineage(genome_id, mode: str = "fitter_parent", db=None,
                include_structure: bool = True) -> List[dict]:
    """Ancestor chain from generation 0 to ``genome_id``, one row per generation.

    ``mode``:
      - ``"fitter_parent"``: at each crossover step follow the parent with the
        higher (train) fitness; ties go to parent1. NEAT's crossover takes the
        fitter parent's disjoint/excess genes, so this is the structural line.
      - ``"parent1"``: always follow parent1.
    Elite copies follow the same key one generation earlier.

    Row keys: ``generation, genome_id, genome_key, fitness, train_auc`` (the
    fitness when the run's fitness function is AUC, else None), ``n_nodes,
    n_connections, n_enabled_connections, parent1_id, parent2_id,
    parent1_key, parent2_key, is_elite_copy, followed`` ("parent1",
    "parent2", "elite" or None at the root), ``parent_source`` ("db" or
    "sidecar") and, with ``include_structure``, ``structure_signature``
    (see ``structure_signature``).

    Raises LookupError when no parent information exists for the experiment.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    db = _db(db)
    with db.session_scope() as s:
        loc = _locate(s, genome_id)
        idx = _index_for(s, db, loc.experiment_id)
        if idx.parents is None:
            raise LookupError(
                f"experiment {idx.experiment_name}: no parent ids in the DB and no sidecar "
                f"{sidecar_dir() / (idx.experiment_name + '.csv.gz')}")
        chain = []
        gk = (loc.generation, loc.genome_id)
        while gk is not None:
            g, k = gk
            p1, p2, elite = idx.parents.get(gk, (None, None, False))
            chain.append((gk, p1, p2, elite))
            if g == 0 or (p1 is None and p2 is None):
                break
            if elite:
                nxt, followed = (g - 1, k), "elite"
            elif mode == "parent1" or p2 is None:
                nxt, followed = (g - 1, p1), "parent1"
            else:
                f1 = idx.rows.get((g - 1, p1), {}).get("fitness")
                f2 = idx.rows.get((g - 1, p2), {}).get("fitness")
                f1 = -np.inf if f1 is None else f1
                f2 = -np.inf if f2 is None else f2
                nxt, followed = ((g - 1, p2), "parent2") if f2 > f1 else ((g - 1, p1), "parent1")
            if nxt not in idx.rows:
                raise LookupError(f"parent {nxt} of {gk} not found in {idx.experiment_name}")
            chain[-1] = chain[-1] + (followed,)
            gk = nxt
        chain.reverse()

        sigs = {}
        if include_structure:
            from explaneat.db.models import Genome
            ids = [_uuid(idx.rows[c[0]]["id"]) for c in chain]
            for rid, data in s.query(Genome.id, Genome.genome_data).filter(Genome.id.in_(ids)):
                sigs[str(rid)] = structure_signature(data)

    out = []
    for i, c in enumerate(chain):
        (g, k), p1, p2, elite = c[:4]
        r = idx.rows[(g, k)]
        p1_id = idx.rows.get((g - 1, p1), {}).get("id") if p1 is not None else None
        p2_id = idx.rows.get((g - 1, p2), {}).get("id") if p2 is not None else None
        # "followed" is stored on the child row: how the walk reached the previous row.
        followed = c[4] if len(c) > 4 else None
        row = dict(generation=g, genome_id=r["id"], genome_key=k, fitness=r["fitness"],
                   train_auc=r["fitness"] if idx.fitness_is_auc else None,
                   n_nodes=r["n_nodes"], n_connections=r["n_connections"],
                   n_enabled_connections=r["n_enabled_connections"],
                   parent1_id=p1_id, parent2_id=p2_id, parent1_key=p1, parent2_key=p2,
                   is_elite_copy=bool(elite), followed=followed,
                   parent_source=idx.parent_source,
                   experiment_id=idx.experiment_id, experiment_name=idx.experiment_name)
        if include_structure:
            row["structure_signature"] = sigs.get(r["id"])
        out.append(row)
    return out


def detect_jumps(lineage: Sequence[dict], metric: str = "fitness",
                 min_delta: Optional[float] = None, method: str = "threshold",
                 k: int = 5) -> List[dict]:
    """Steps of a lineage where ``metric`` rises sharply.

    ``delta`` is ``metric[i] - metric[i-1]`` between consecutive lineage rows.
    - ``method="threshold"``: steps with ``delta >= min_delta``. When
      ``min_delta`` is None it defaults to the 95th percentile of this
      lineage's positive deltas (a per-lineage rule; Study H pre-registers a
      pooled threshold instead, experiments/lineage/PLAN_H.md).
    - ``method="top_k"``: the ``k`` largest positive deltas (``min_delta``, if
      given, is also applied).

    ``structural_change`` compares ``structure_signature`` (node keys and
    connection keys with enabled flags) of the two rows; when the rows carry no
    signature it falls back to comparing node and connection counts.
    Returned in generation order.
    """
    if method not in ("threshold", "top_k"):
        raise ValueError("method must be 'threshold' or 'top_k'")
    steps = []
    for a, b in zip(lineage[:-1], lineage[1:]):
        va, vb = a.get(metric), b.get(metric)
        if va is None or vb is None:
            continue
        sa, sb = a.get("structure_signature"), b.get("structure_signature")
        if sa is not None and sb is not None:
            structural = sa != sb
        else:
            structural = (a.get("n_nodes"), a.get("n_connections"), a.get("n_enabled_connections")) != \
                         (b.get("n_nodes"), b.get("n_connections"), b.get("n_enabled_connections"))
        steps.append(dict(generation=b["generation"], delta=float(vb - va),
                          before_genome_id=a["genome_id"], after_genome_id=b["genome_id"],
                          structural_change=bool(structural),
                          is_elite_copy=bool(b.get("is_elite_copy", False))))
    if not steps:
        return []
    if method == "threshold":
        thr = min_delta
        if thr is None:
            pos = [s["delta"] for s in steps if s["delta"] > 0]
            if not pos:
                return []
            thr = float(np.quantile(pos, 0.95))
        return [s for s in steps if s["delta"] >= thr and s["delta"] > 0]
    pos = [s for s in steps if s["delta"] > 0 and (min_delta is None or s["delta"] >= min_delta)]
    top = sorted(pos, key=lambda s: -s["delta"])[:k]
    return sorted(top, key=lambda s: s["generation"])


# --------------------------------------------------------------------- diffs
def _genome_data(session, genome_id) -> dict:
    from explaneat.db.models import Genome
    row = session.get(Genome, _uuid(genome_id))
    if row is None:
        raise KeyError(f"genome not found: {genome_id}")
    return row.genome_data


def diff_genome_data(a: dict, b: dict) -> dict:
    """Diff of two serialised genomes (``serialize_genome`` format), a -> b."""
    na, nb = a.get("nodes", {}), b.get("nodes", {})
    ca, cb = a.get("connections", {}), b.get("connections", {})
    added_nodes = sorted((int(n) for n in set(nb) - set(na)))
    removed_nodes = sorted((int(n) for n in set(na) - set(nb)))
    ck = lambda key: tuple(int(x) for x in key.rsplit("_", 1)) if key.count("_") == 1 else \
        (int(cb.get(key, ca.get(key))["in_node"]), int(cb.get(key, ca.get(key))["out_node"]))
    added_c = sorted(set(cb) - set(ca), key=ck)
    removed_c = sorted(set(ca) - set(cb), key=ck)
    shared_c = sorted(set(ca) & set(cb), key=ck)
    enabled_changed = {c: (bool(ca[c]["enabled"]), bool(cb[c]["enabled"]))
                       for c in shared_c if bool(ca[c]["enabled"]) != bool(cb[c]["enabled"])}
    weight_changes = {c: (ca[c]["weight"], cb[c]["weight"])
                      for c in shared_c if ca[c]["weight"] != cb[c]["weight"]}
    bias_changes = {int(n): (na[n]["bias"], nb[n]["bias"])
                    for n in sorted(set(na) & set(nb), key=int) if na[n]["bias"] != nb[n]["bias"]}
    other_node_changes = {int(n): {f: (na[n].get(f), nb[n].get(f))
                                   for f in ("activation", "aggregation", "response")
                                   if na[n].get(f) != nb[n].get(f)}
                          for n in sorted(set(na) & set(nb), key=int)}
    other_node_changes = {n: v for n, v in other_node_changes.items() if v}
    wd = [abs(n - o) for o, n in weight_changes.values() if o is not None and n is not None]
    structural = bool(added_nodes or removed_nodes or added_c or removed_c or enabled_changed)
    return dict(added_nodes=added_nodes, removed_nodes=removed_nodes,
                added_connections=added_c, removed_connections=removed_c,
                enabled_changed=enabled_changed, weight_changes=weight_changes,
                bias_changes=bias_changes, other_node_changes=other_node_changes,
                structural_change=structural,
                summary=dict(n_added_nodes=len(added_nodes), n_removed_nodes=len(removed_nodes),
                             n_added_connections=len(added_c),
                             n_removed_connections=len(removed_c),
                             n_enabled_changed=len(enabled_changed),
                             n_weight_changes=len(weight_changes),
                             n_bias_changes=len(bias_changes),
                             max_abs_weight_change=max(wd) if wd else 0.0,
                             fitness_a=a.get("fitness"), fitness_b=b.get("fitness")))


def diff_genomes(genome_a_id, genome_b_id, db=None) -> dict:
    """Gene-level diff from genome A to genome B.

    Connections are keyed as in the serialised genome, ``"<in>_<out>"`` (e.g.
    ``"-3_17"``). ``enabled_changed``, ``weight_changes`` and ``bias_changes``
    map a key to ``(old, new)``. Input nodes are not genes, so node lists hold
    hidden and output node keys only.
    """
    db = _db(db)
    with db.session_scope() as s:
        a = _genome_data(s, genome_a_id)
        b = _genome_data(s, genome_b_id)
    out = diff_genome_data(a, b)
    out.update(genome_a_id=str(genome_a_id), genome_b_id=str(genome_b_id))
    return out


def ancestor_at_generation(genome_id, generation: int, mode: str = "fitter_parent",
                           db=None) -> dict:
    """The lineage ancestor of ``genome_id`` at ``generation``, and that
    generation's best genome overall (highest fitness, ties to the lower key)."""
    db = _db(db)
    lineage = get_lineage(genome_id, mode=mode, db=db)
    match = [r for r in lineage if r["generation"] == generation]
    if not match:
        raise ValueError(f"generation {generation} not in lineage "
                         f"(0-{lineage[-1]['generation']})")
    anc = match[0]
    with db.session_scope() as s:
        loc = _locate(s, genome_id)
        idx = _index_for(s, db, loc.experiment_id)
    f_best, k_best = idx.gen_best[generation]
    best = idx.rows[(generation, k_best)]
    fits = sorted((r["fitness"] for (g, _), r in idx.rows.items()
                   if g == generation and r["fitness"] is not None), reverse=True)
    rank = 1 + sum(1 for f in fits if anc["fitness"] is not None and f > anc["fitness"])
    return dict(generation=generation, ancestor=anc,
                generation_best=dict(genome_id=best["id"], genome_key=k_best,
                                     fitness=best["fitness"],
                                     train_auc=best["fitness"] if idx.fitness_is_auc else None,
                                     n_nodes=best["n_nodes"],
                                     n_connections=best["n_connections"]),
                ancestor_is_generation_best=anc["genome_key"] == k_best,
                ancestor_rank=rank, population_size=len(fits),
                fitness_gap=(best["fitness"] - anc["fitness"])
                if best["fitness"] is not None and anc["fitness"] is not None else None)


# --------------------------------------------------------------------- annotations
_BASE_RE = re.compile(r"^(-?\d+)(?:_[a-z]+)?$")


def base_node_id(node_id: Union[str, int]) -> Optional[int]:
    """NEAT node key a model-state node derives from: ``"5"``, ``"5_a"``,
    ``"-71_ab"`` (split/consolidated) -> 5 / -71. Identity and other added
    nodes return None (they are not genes; see ``_resolver``)."""
    m = _BASE_RE.match(str(node_id))
    return int(m.group(1)) if m else None


def _parse_gene(g) -> Tuple[str, Union[int, Tuple[int, int]]]:
    if isinstance(g, (list, tuple)) and len(g) == 2:
        return "connection", (int(g[0]), int(g[1]))
    if isinstance(g, (int, np.integer)):
        return "node", int(g)
    s = str(g).strip()
    m = re.match(r"^\(?\s*(-?\d+)\s*[,_ ]\s*(-?\d+)\s*\)?$", s)
    if m and not re.match(r"^-?\d+$", s):
        return "connection", (int(m.group(1)), int(m.group(2)))
    return "node", int(s)


def _gene_label(kind, key):
    return f"{key[0]}_{key[1]}" if kind == "connection" else str(key)


class _Resolver:
    """Maps model-state node/connection ids back to NEAT genes using the ops."""

    def __init__(self, operations):
        self.identity_target = {}   # identity node -> node it feeds
        self.identity_sources = {}  # identity node -> intercepted source nodes
        for op in operations:
            if op.get("type") == "add_identity_node":
                p, r = op.get("params", {}), op.get("result") or {}
                created = (r.get("created_nodes") or [p.get("new_node_id")])
                nid = p.get("new_node_id") or (created[0] if created else None)
                if nid is None:
                    continue
                self.identity_target[str(nid)] = str(p.get("target_node"))
                self.identity_sources[str(nid)] = [str(c[0]) for c in p.get("connections", [])]

    def node(self, nid) -> Optional[int]:
        return base_node_id(nid)

    def sources(self, nid, depth=0) -> List[Optional[int]]:
        nid = str(nid)
        if nid in self.identity_sources and depth < 50:
            out = []
            for s in self.identity_sources[nid]:
                out.extend(self.sources(s, depth + 1))
            return out
        return [base_node_id(nid)]

    def target(self, nid, depth=0) -> Optional[int]:
        nid = str(nid)
        if nid in self.identity_target and depth < 50:
            return self.target(self.identity_target[nid], depth + 1)
        return base_node_id(nid)

    def connection(self, u, v) -> List[Tuple[int, int]]:
        """Base connection genes a model-state edge (u, v) stands for. An edge
        into an identity node stands for the intercepted edge to its target; an
        edge out of one stands for every intercepted edge."""
        t = self.target(v)
        if t is None:
            return []
        if str(u) in self.identity_sources:
            srcs = self.sources(u)
        else:
            srcs = [base_node_id(u)]
        return [(s, t) for s in srcs if s is not None]


def _load_operations(session, genome_id, explanation):
    from explaneat.db.models import Explanation
    if explanation is None:
        rows = session.query(Explanation).filter(Explanation.genome_id == _uuid(genome_id)).all()
        if not rows:
            raise LookupError(f"genome {genome_id} has no Explanation in the DB")
        if len(rows) > 1:
            raise LookupError(f"genome {genome_id} has {len(rows)} explanations; pass one")
        return list(rows[0].operations or []), str(rows[0].id)
    if isinstance(explanation, (str, Path)) and Path(str(explanation)).exists():
        explanation = json.loads(Path(str(explanation)).read_text())
    if isinstance(explanation, (str, uuid.UUID)):
        row = session.get(Explanation, _uuid(explanation))
        if row is None:
            raise LookupError(f"explanation not found: {explanation}")
        return list(row.operations or []), str(row.id)
    if isinstance(explanation, dict):
        if "operations" not in explanation:
            raise LookupError("explanation dict has no 'operations' (the records in "
                              "experiments/10seed/explanations/*.json do not carry them; "
                              "the operations live in the DB Explanation)")
        return list(explanation["operations"]), explanation.get("id")
    return list(explanation), None


def annotations_from_operations(operations) -> List[dict]:
    """Annotation dicts (name, entry/exit/subgraph nodes and connections,
    children) from an operations stream, undone ops removed."""
    out = []
    for op in operations:
        if op.get("type") != "annotate" or op.get("undone"):
            continue
        p = op.get("params", {})
        out.append(dict(name=p.get("name"), entry_nodes=[str(x) for x in p.get("entry_nodes", [])],
                        exit_nodes=[str(x) for x in p.get("exit_nodes", [])],
                        subgraph_nodes=[str(x) for x in p.get("subgraph_nodes", [])],
                        subgraph_connections=[(str(c[0]), str(c[1]))
                                              for c in p.get("subgraph_connections", [])],
                        children=list(p.get("child_annotation_ids") or [])))
    return out


def map_genes_to_annotations(gene_keys: Iterable, genome_id, explanation=None,
                             db=None) -> dict:
    """Which annotations of ``genome_id``'s explanation contain the given genes.

    ``gene_keys``: node keys (int or "17") and/or connection keys
    (``(-3, 17)``, ``[-3, 17]`` or ``"-3_17"``), as NEAT genes of the
    *original* genome. ``explanation``: None (the genome's single DB
    Explanation), an Explanation id, a list of operations, a dict with
    ``"operations"``, or a JSON file path holding one.

    Model-state ids are mapped back to genes: split/consolidated nodes
    (``"5_a"``, ``"-71_ab"``) to their base node; an edge into an identity
    node (``"identity_3"``, ``"lin_agg"``) to the intercepted edge to the
    identity's target; an edge out of an identity node to every intercepted
    edge. A node gene is in an annotation when any subgraph node derives from
    it; a connection gene when any subgraph connection stands for it.
    Composition annotations list only their own glue nodes and edges, so a
    gene is usually in one leaf; ``containing_leaves`` and ``containing`` are
    reported separately, and ``ancestors`` gives each annotation's chain of
    compositions up to the root.
    """
    db = _db(db)
    with db.session_scope() as s:
        ops, expl_id = _load_operations(s, genome_id, explanation)
    anns = annotations_from_operations(ops)
    res = _Resolver(ops)
    parent_of = {}
    for a in anns:
        for c in a["children"]:
            parent_of[c] = a["name"]
    leaf = {a["name"]: not a["children"] for a in anns}

    def ancestors(name):
        chain, seen = [], set()
        while name in parent_of and name not in seen:
            seen.add(name)
            name = parent_of[name]
            chain.append(name)
        return chain

    ann_nodes, ann_conns = {}, {}
    for a in anns:
        ann_nodes[a["name"]] = {res.node(n) for n in a["subgraph_nodes"]} - {None}
        conns = set()
        for u, v in a["subgraph_connections"]:
            conns.update(res.connection(u, v))
        ann_conns[a["name"]] = conns

    genes = {}
    for g in gene_keys:
        kind, key = _parse_gene(g)
        pool = ann_nodes if kind == "node" else ann_conns
        names = [a["name"] for a in anns if key in pool[a["name"]]]
        genes[_gene_label(kind, key)] = dict(
            kind=kind, key=list(key) if kind == "connection" else key,
            containing=names, containing_leaves=[n for n in names if leaf.get(n)],
            ancestors={n: ancestors(n) for n in names})
    return dict(genome_id=str(genome_id), explanation_id=expl_id, genes=genes,
                annotations=[dict(name=a["name"], is_leaf=leaf[a["name"]],
                                  children=a["children"], parent=parent_of.get(a["name"]),
                                  n_gene_nodes=len(ann_nodes[a["name"]]),
                                  n_gene_connections=len(ann_conns[a["name"]]))
                             for a in anns])
