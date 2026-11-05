"""Genome-level operations for the churn extension studies (EXTENSIONS_PLAN.md).

Everything here acts on plain neat-python genomes, so every result is still a
valid NEAT genome that NeuralNeat can build, that can be stored, re-explained
and evolved further.

- ``finetune``: the PropNEAT backprop primitive (NeuralNeat -> Adadelta lr 1.5
  -> full-batch BCE -> ``update_genome_weights``) applied to one genome, with
  optional gradient masking ("freezing").
- ``disable_connections``: copy with some connection genes disabled.
- ``annotation_to_genome_subgraph``: resolve an explanation annotation (whose
  node ids live in the *model state*, i.e. after splits / identity nodes) back
  to original genome node ids and connection keys.
- ``graft``: copy a donor leaf annotation's subgraph into a recipient genome.
- ``save_derived_genome``: store a derived genome as its own Experiment.

Library code: no DB access except in ``save_derived_genome``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from neat.genes import DefaultConnectionGene, DefaultNodeGene

from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.model_state import ModelStateEngine
from explaneat.core.neuralneat import NeuralNeat

ConnKey = Tuple[int, int]

# The PropNEAT training primitive (explaneat/core/backproppop.py::backpropagate).
FT_OPTIMIZER = "Adadelta"
FT_LR = 1.5
FT_LOSS = "BCELoss"

REPO = Path(__file__).resolve().parents[2]


# ====================================================================== forward

def _as_tensors(X, y=None):
    xs = torch.tensor(np.asarray(X, dtype=np.float64), dtype=torch.float64)
    if y is None:
        return xs
    ys = torch.tensor(np.asarray(y, dtype=np.float64).reshape(-1, 1), dtype=torch.float64)
    return xs, ys


def build_net(genome, config) -> NeuralNeat:
    """NeuralNeat for ``genome``; raises instead of exiting on a bad genome.

    (NeuralNeat calls ``sys.exit(1)`` when layer parsing fails.)
    """
    try:
        return NeuralNeat(genome, config, criterion=nn.BCELoss())
    except SystemExit as e:  # pragma: no cover - defensive
        raise ValueError(f"NeuralNeat could not build genome {genome.key}") from e


def predict_proba(genome, config, X) -> np.ndarray:
    """P(y=1) from NeuralNeat, float64, shape (n,)."""
    net = build_net(genome, config)
    with torch.no_grad():
        return net.forward(_as_tensors(X)).detach().numpy().ravel()


# ====================================================================== finetune

def _param_masks(net: NeuralNeat, genome, frozen_conn_keys, frozen_node_keys,
                 mask_phantom: bool = False):
    """Gradient masks (1 = trainable, 0 = frozen) per layer weight/bias tensor.

    Genome location -> tensor index comes from the same maps that
    ``NeuralNeat.update_genome_weights`` writes back through:
    connection (a, b) -> ``layers[L]["input_map"][(a, b)]`` = (row, col) of
    ``net.weights[L]``; node n's bias -> ``net.biases[L][layers[L]["nodes"][n]["layer_index"]]``.

    A frozen node also has the *phantom* entries of its weight column masked:
    NeuralNeat's dense layer matrices hold a zero for every (earlier node,
    this node) pair without a connection gene, and those zeros are trainable
    (they are dropped again at write-back). Masking them keeps the frozen
    unit's function fixed throughout training. Real connections into a frozen
    node are frozen only if listed in ``frozen_conn_keys``.

    ``mask_phantom`` masks every phantom entry of every layer, so only genome
    weights and biases train and the written-back genome is exactly the
    trained network.
    """
    frozen_conn_keys = {tuple(k) for k in frozen_conn_keys}
    frozen_node_keys = set(frozen_node_keys)
    w_masks = {L: torch.ones_like(w) for L, w in net.weights.items()}
    b_masks = {L: torch.ones_like(b) for L, b in net.biases.items()}
    located_conns, located_nodes = set(), set()
    for L, layer in net.layers.items():
        real_entries = {loc: key for key, loc in layer["input_map"].items()}
        if mask_phantom and w_masks[L].ndim == 2 and layer["layer_type"] != "INPUT":
            w_masks[L].zero_()
            for (r, c) in real_entries:
                w_masks[L][r, c] = 1.0
        for key, (r, c) in layer["input_map"].items():
            if key in frozen_conn_keys:
                w_masks[L][r, c] = 0.0
                located_conns.add(key)
        for nid, node in layer["nodes"].items():
            if nid not in frozen_node_keys:
                continue
            col = node["layer_index"]
            if b_masks[L].numel():
                b_masks[L][col] = 0.0
            located_nodes.add(nid)
            if w_masks[L].ndim == 2 and w_masks[L].shape[1] > col:
                for r in range(w_masks[L].shape[0]):
                    if (r, col) not in real_entries:
                        w_masks[L][r, col] = 0.0
    missing = [k for k in frozen_conn_keys - located_conns
               if k in genome.connections and genome.connections[k].enabled]
    if missing:
        raise KeyError(f"frozen connections not found in the network: {sorted(missing)}")
    if frozen_conn_keys - set(genome.connections):
        raise KeyError(f"frozen connections not in genome: "
                       f"{sorted(frozen_conn_keys - set(genome.connections))}")
    missing_nodes = frozen_node_keys - located_nodes
    if missing_nodes:
        raise KeyError(f"frozen nodes not in the network: {sorted(missing_nodes)}")
    return w_masks, b_masks


def finetune(genome, config, X, y, epochs: int, seed: int,
             frozen_conn_keys: Iterable[ConnKey] = (),
             frozen_node_keys: Iterable[int] = (),
             mask_phantom: bool = False,
             return_info: bool = False):
    """Fine-tune a deep copy of ``genome`` with the PropNEAT backprop primitive.

    Exactly ``BackpropPopulation.backpropagate`` for one genome: NeuralNeat with
    BCELoss, ``Adadelta(lr=1.5)``, full batch, ``epochs`` steps, no early
    stopping, then ``update_genome_weights``. ``torch.manual_seed(seed)`` is set
    first. Frozen connection weights and frozen node biases get zero gradient
    (Adadelta then leaves them bit-identical); see ``_param_masks``.

    ``mask_phantom=False`` (the literal primitive) also trains NeuralNeat's
    phantom zero entries, which ``update_genome_weights`` then drops, so the
    stored genome differs from the trained network (``info`` reports by how
    much). ``mask_phantom=True`` trains only the genome's own weights and
    biases.

    Returns the new genome, or (genome, info) with ``return_info``.
    """
    g = copy.deepcopy(genome)
    frozen_conn_keys = [tuple(k) for k in frozen_conn_keys]
    frozen_nodes = set(frozen_node_keys)
    torch.manual_seed(seed)
    xs, ys = _as_tensors(X, y)
    net = build_net(g, config)
    w_masks, b_masks = _param_masks(net, g, frozen_conn_keys, frozen_nodes, mask_phantom)

    optimizer = optim.Adadelta(net.parameters(), lr=FT_LR)
    optimizer.zero_grad()
    bce = nn.BCELoss()
    with torch.no_grad():
        loss_before = float(bce(net.forward(xs), ys))
    losses = []
    for _ in range(epochs):
        preds = net.forward(xs)
        loss = bce(preds, ys)
        loss.backward()
        for L, w in net.weights.items():
            if w.grad is not None:
                w.grad.mul_(w_masks[L])
        for L, b in net.biases.items():
            if b.grad is not None:
                b.grad.mul_(b_masks[L])
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss))
    with torch.no_grad():
        preds_trained = net.forward(xs).detach().numpy().ravel()
        loss_trained = float(bce(net.forward(xs), ys))

    phantom_into_frozen = 0.0
    for L, layer in net.layers.items():
        real = set(layer["input_map"].values())
        for nid, node in layer["nodes"].items():
            if nid in frozen_nodes and net.weights[L].ndim == 2:
                col = node["layer_index"]
                for r in range(net.weights[L].shape[0]):
                    if (r, col) not in real:
                        phantom_into_frozen = max(phantom_into_frozen,
                                                  abs(float(net.weights[L][r, col])))

    net.update_genome_weights()
    if not return_info:
        return g
    p_back = predict_proba(g, config, X)
    eps = 1e-12
    yv = np.asarray(y, dtype=np.float64).ravel()
    pc = np.clip(p_back, eps, 1 - eps)
    info = {
        "epochs": epochs, "seed": seed, "optimizer": FT_OPTIMIZER, "lr": FT_LR,
        "loss": FT_LOSS, "full_batch": True, "mask_phantom": mask_phantom,
        "n_frozen_connections": len(set(frozen_conn_keys)),
        "n_frozen_nodes": len(frozen_nodes),
        "loss_before": loss_before, "losses": losses,
        "loss_trained_net": loss_trained,
        "loss_after_written_back": float(-np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc))),
        # NeuralNeat trains phantom zero entries too and drops them at
        # write-back, so the stored genome is not exactly the trained net.
        "max_abs_pred_diff_trained_vs_written_back": float(np.max(np.abs(preds_trained - p_back))),
        "max_abs_phantom_into_frozen": phantom_into_frozen,
    }
    return g, info


# ====================================================================== disable

def disable_connections(genome, keys: Iterable[ConnKey]):
    """Deep copy of ``genome`` with the given connection genes disabled."""
    g = copy.deepcopy(genome)
    for k in keys:
        k = tuple(k)
        if k not in g.connections:
            raise KeyError(f"connection {k} not in genome {genome.key}")
        g.connections[k].enabled = False
    return g


def enabled_connections_from(genome, sources: Iterable[int]) -> List[ConnKey]:
    """Sorted keys of enabled connections whose source is in ``sources``."""
    sources = set(sources)
    return sorted(k for k, c in genome.connections.items() if c.enabled and k[0] in sources)


# ====================================================================== id mapping

class NonInputEntryError(ValueError):
    """An annotation entry does not resolve to a genome input node."""


class GraftCollisionError(ValueError):
    """A connection added by a graft already exists in the recipient."""


@dataclass
class DonorSubgraph:
    """A leaf annotation expressed in original genome ids (see
    ``annotation_to_genome_subgraph``)."""

    annotation_name: str
    genome_key: int
    hidden_nodes: List[int]
    entry_inputs: List[int]
    exit_nodes: List[int]
    # input->hidden and hidden->hidden edges inside the annotation
    internal_connections: List[ConnKey]
    # hidden exit -> output edges
    readout_connections: List[ConnKey]
    # input -> output edges inside the annotation (only possible through an
    # identity exit that collects direct input edges); a graft must add them
    # to the recipient, where the same key may already exist
    input_output_connections: List[ConnKey] = field(default_factory=list)
    # exit -> hidden edges outside the annotation (not copied by a graft)
    egress_connections: List[ConnKey] = field(default_factory=list)
    model_state_exit_nodes: List[str] = field(default_factory=list)
    model_state_entry_nodes: List[str] = field(default_factory=list)

    @property
    def all_connections(self) -> List[ConnKey]:
        return sorted(set(self.internal_connections) | set(self.readout_connections)
                      | set(self.input_output_connections))

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        for k in ("internal_connections", "readout_connections",
                  "input_output_connections", "egress_connections"):
            d[k] = [list(c) for c in d[k]]
        return d


class ModelStateIdResolver:
    """Map model-state node ids back to original genome node ids.

    Built from the operation stream (not by parsing id strings: identity
    nodes such as ``identity_lin`` would parse as split copies of
    ``identity``):
    - ``split_node``: each created copy (``-37_a``) aliases the split node;
    - ``consolidate_node``: the merged copy (``-37_cd``) aliases its members;
    - ``add_identity_node``: the identity node is a weight-1 passthrough to its
      ``target_node`` (``(u, identity)`` carries the original ``(u, target)``
      weight).
    Other structural operations are rejected.
    """

    _NO_ID_CHANGE = {"annotate", "rename_node"}

    def __init__(self, operations: Sequence[dict]):
        self.alias: Dict[str, str] = {}
        self.identity_target: Dict[str, str] = {}
        for op in operations:
            t, p = op.get("type"), op.get("params") or {}
            res = op.get("result") or {}
            if t in self._NO_ID_CHANGE:
                continue
            if t == "split_node":
                created = res.get("created_nodes")
                if created is None:
                    raise ValueError(f"split_node op {op.get('seq')} has no result; "
                                     "replay it first (ModelStateEngine)")
                for c in created:
                    self.alias[str(c)] = str(p["node_id"])
            elif t == "consolidate_node":
                created = res.get("created_nodes")
                if created is None:
                    raise ValueError(f"consolidate_node op {op.get('seq')} has no result")
                for c in created:
                    self.alias[str(c)] = str(p["node_ids"][0])
            elif t == "add_identity_node":
                self.identity_target[str(p["new_node_id"])] = str(p["target_node"])
            else:
                raise NotImplementedError(f"operation type {t!r} changes ids in a way "
                                          "this resolver does not handle")

    def base(self, node_id: str) -> str:
        """Follow split/consolidate aliases to a pre-split id (genome or identity)."""
        seen = set()
        n = str(node_id)
        while n in self.alias:
            if n in seen:
                raise ValueError(f"alias cycle at {n}")
            seen.add(n)
            n = self.alias[n]
        return n

    def is_identity(self, node_id: str) -> bool:
        return self.base(node_id) in self.identity_target

    def genome_node(self, node_id: str) -> int:
        """Genome id of a non-identity model-state node."""
        b = self.base(node_id)
        if b in self.identity_target:
            raise ValueError(f"{node_id} is an identity node, not a genome node")
        return int(b)

    def through(self, node_id: str) -> int:
        """Genome node a connection into ``node_id`` really feeds (follows identity chains)."""
        b = self.base(node_id)
        seen = set()
        while b in self.identity_target:
            if b in seen:
                raise ValueError(f"identity cycle at {b}")
            seen.add(b)
            b = self.base(self.identity_target[b])
        return int(b)


def replay_model_state(explanation_ops: Sequence[dict], genome, config):
    """(ModelStateEngine, replayed operation dicts with results) for ``genome``."""
    phenotype = ExplaNEAT(genome, config).get_phenotype_network()
    engine = ModelStateEngine(phenotype)
    engine.load_operations({"operations": copy.deepcopy(list(explanation_ops))})
    _ = engine.current_state  # replay; fills op.result
    return engine, [op.to_dict() for op in engine.operations]


def _find_annotation(explanation_ops, name):
    for op in explanation_ops:
        if op.get("type") == "annotate" and (op.get("params") or {}).get("name") == name:
            return op["params"]
    raise KeyError(f"annotation {name!r} not in the explanation")


def annotation_to_genome_subgraph(explanation_ops: Sequence[dict], annotation_name: str,
                                  genome, config) -> DonorSubgraph:
    """Resolve a leaf annotation to original genome ids and connection keys.

    The operations are replayed on ``genome``'s phenotype to recover the model
    state (needed for split results and for the exit's readout edges, which lie
    outside the annotation). Every resolved connection is checked to exist,
    enabled, in ``genome``, with the model-state weight.

    Raises KeyError (unknown name), ValueError (not a leaf),
    NonInputEntryError (an entry is not an input node).
    """
    params = _find_annotation(explanation_ops, annotation_name)
    if params.get("child_annotation_ids"):
        raise ValueError(f"{annotation_name!r} is a composition, not a leaf annotation")
    engine, ops = replay_model_state(explanation_ops, genome, config)
    state = engine.current_state
    res = ModelStateIdResolver(ops)
    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)

    entries = [str(n) for n in params.get("entry_nodes", [])]
    exits = [str(n) for n in params.get("exit_nodes", [])]
    sub_nodes = {str(n) for n in params.get("subgraph_nodes", [])} | set(exits) | set(entries)

    bad = [n for n in entries if res.is_identity(n) or res.genome_node(n) not in inputs]
    if bad:
        raise NonInputEntryError(
            f"annotation {annotation_name!r}: entries {bad} are not input nodes "
            "(donors must be leaves whose entries are all inputs)")
    entry_inputs = sorted({res.genome_node(n) for n in entries})

    weights = {(c.from_node, c.to_node): c.weight for c in state.connections if c.enabled}
    mapped: Dict[ConnKey, float] = {}

    def add(u, v):
        if res.is_identity(u):
            return  # identity -> x edges are the weight-1 passthrough
        key = (res.genome_node(u), res.through(v))
        w = weights.get((u, v))
        if w is None:
            raise ValueError(f"connection {(u, v)} not enabled in the model state")
        if key in mapped and mapped[key] != w:
            raise ValueError(f"{key}: inconsistent weights {mapped[key]} vs {w}")
        mapped[key] = w

    for u, v in params.get("subgraph_connections", []):
        add(str(u), str(v))
    # readout / egress edges of non-identity exits (outside the annotation)
    egress = []
    for x in exits:
        if res.is_identity(x):
            continue
        for c in state.connections:
            if c.enabled and c.from_node == x and c.to_node not in sub_nodes:
                key = (res.genome_node(x), res.through(c.to_node))
                if key[1] in outputs:
                    add(x, c.to_node)
                else:
                    egress.append(key)

    hidden = sorted({res.genome_node(n) for n in sub_nodes if not res.is_identity(n)} - inputs)
    internal, readout, io = [], [], []
    for (a, b) in sorted(mapped):
        if b in outputs:
            (io if a in inputs else readout).append((a, b))
        else:
            if b not in hidden:
                raise ValueError(f"{(a, b)} leaves the annotation to a non-output node")
            internal.append((a, b))
    for key, w in mapped.items():
        gc = genome.connections.get(key)
        if gc is None or not gc.enabled:
            raise ValueError(f"resolved connection {key} is not an enabled genome connection")
        if gc.weight != w:
            raise ValueError(f"{key}: genome weight {gc.weight} != model-state weight {w}")
    exit_genome = sorted({a for a, _ in readout}
                         | {res.genome_node(x) for x in exits if not res.is_identity(x)})
    return DonorSubgraph(
        annotation_name=annotation_name, genome_key=genome.key, hidden_nodes=hidden,
        entry_inputs=entry_inputs, exit_nodes=exit_genome,
        internal_connections=internal, readout_connections=readout,
        input_output_connections=io, egress_connections=sorted(set(egress)),
        model_state_exit_nodes=exits, model_state_entry_nodes=entries)


# ====================================================================== graft

def is_acyclic(genome) -> bool:
    """True if the enabled connections of ``genome`` form a DAG."""
    succ: Dict[int, List[int]] = {}
    for (a, b), c in genome.connections.items():
        if c.enabled:
            succ.setdefault(a, []).append(b)
    state: Dict[int, int] = {}
    for start in list(succ):
        if state.get(start):
            continue
        stack = [(start, iter(succ.get(start, [])))]
        state[start] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[node] = 2
                stack.pop()
                continue
            s = state.get(nxt, 0)
            if s == 1:
                return False
            if s == 0:
                state[nxt] = 1
                stack.append((nxt, iter(succ.get(nxt, []))))
    return True


def graft(recipient, recipient_config, donor_genome, donor_subgraph: DonorSubgraph,
          sham_seed: Optional[int] = None):
    """Copy a donor leaf subgraph into a deep copy of ``recipient``.

    - donor hidden nodes get fresh ids above the recipient's maximum node id
      (ascending donor id order), copying bias, response, activation and
      aggregation;
    - internal connections (input->hidden, hidden->hidden), readout
      connections (exit->output) and any input->output connections of the
      annotation are added, enabled, with the donor's weights. Input and
      output ids are shared by every network on the dataset and kept as is.
    - ``sham_seed``: same topology, but every grafted weight and grafted-node
      bias is redrawn from N(0, 1) with ``numpy.random.default_rng(sham_seed)``
      (biases in ascending new-node order, then weights in ascending new-key
      order).

    A connection key that already exists in the recipient (only possible for
    input->output keys) raises GraftCollisionError. The result is checked for
    acyclicity and that NeuralNeat builds it.

    Returns (new_genome, graft_record).
    """
    new = copy.deepcopy(recipient)
    start = max(max(new.nodes), 0) + 1
    id_map = {d: start + i for i, d in enumerate(sorted(donor_subgraph.hidden_nodes))}
    inputs = set(recipient_config.genome_config.input_keys)
    outputs = set(recipient_config.genome_config.output_keys)

    def m(n):
        if n in inputs or n in outputs:
            return n
        return id_map[n]

    planned = [(k, (m(k[0]), m(k[1]))) for k in donor_subgraph.all_connections]
    collisions = sorted(nk for _, nk in planned if nk in new.connections)
    if collisions:
        raise GraftCollisionError(
            f"graft of {donor_subgraph.annotation_name!r} would add connection(s) "
            f"{', '.join(str(c) for c in collisions)} that already exist in recipient "
            f"{recipient.key}")

    rng = np.random.default_rng(sham_seed) if sham_seed is not None else None
    for d in sorted(id_map):
        src = donor_genome.nodes[d]
        ng = DefaultNodeGene(id_map[d])
        ng.bias, ng.response = src.bias, src.response
        ng.activation, ng.aggregation = src.activation, src.aggregation
        if rng is not None:
            ng.bias = float(rng.normal(0.0, 1.0))
        new.nodes[ng.key] = ng
    added = []
    for dk, nk in sorted(planned, key=lambda t: t[1]):
        cg = DefaultConnectionGene(nk)
        cg.weight = donor_genome.connections[dk].weight
        cg.enabled = True
        if rng is not None:
            cg.weight = float(rng.normal(0.0, 1.0))
        new.connections[nk] = cg
        added.append(nk)

    acyclic = is_acyclic(new)
    if not acyclic:
        raise ValueError("graft produced a cycle")
    build_net(new, recipient_config)
    record = {
        "annotation_name": donor_subgraph.annotation_name,
        "donor_genome_key": donor_genome.key,
        "recipient_genome_key": recipient.key,
        "id_map": dict(id_map),
        "added_connections": added,
        "connection_map": {str(list(dk)): list(nk) for dk, nk in planned},
        "entry_inputs": list(donor_subgraph.entry_inputs),
        "readout_connections": [(m(a), b) for a, b in donor_subgraph.readout_connections],
        "sham_seed": sham_seed,
        "acyclic": acyclic,
    }
    return new, record


# ====================================================================== storage

def code_commit() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        # modified tracked sources only (tracked .pyc files change on every run)
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", "--",
             "explaneat", "experiments/churn", "experiments/10seed", "mcp_server"],
            cwd=REPO, text=True)
        dirty = any(line.strip().endswith(".py") for line in status.splitlines())
    except Exception:  # pragma: no cover
        sha, dirty = None, None
    return {"code_commit": sha, "dirty": dirty}


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda v: v.tolist() if hasattr(v, "tolist")
                                 else str(v)))


def derived_experiment_payload(genome, config, name, provenance, split_id, dataset_id,
                               source_config_json=None, neat_config_text=None) -> dict:
    """Column values for the Experiment / Population rows of a derived genome."""
    from explaneat.db.serialization import serialize_population_config

    cfg = dict(source_config_json if source_config_json is not None
               else serialize_population_config(config))
    cfg["provenance"] = _jsonable(provenance)
    sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:40]
    commit = (provenance or {}).get("code_commit")
    return {
        "experiment": dict(
            experiment_sha=sha, name=name,
            description=f"Derived genome ({name}); see config_json.provenance",
            config_json=cfg, neat_config_text=neat_config_text or "",
            status="completed", random_seed=None, split_id=split_id,
            dataset_id=dataset_id, git_commit_sha=commit,
            hardware_info={"derived": True}),
        "population": dict(generation=0, population_size=1, num_species=1,
                           best_fitness=genome.fitness, mean_fitness=genome.fitness,
                           config_json=cfg),
    }


def save_derived_genome(genome, config, name, provenance, split_id, dataset_id,
                        source_config_json=None, neat_config_text=None, database=None):
    """Store ``genome`` as its own completed Experiment with one Population
    (generation 0) and one Genome row. Refuses an existing experiment name.

    Returns {"experiment_id", "population_id", "genome_id"} (UUIDs).
    """
    import uuid as _uuid

    from explaneat.db.base import db as default_db
    from explaneat.db.models import Experiment, Genome, Population

    database = database or default_db
    genome = copy.copy(genome)  # plain-float fitness for the DB adapters
    genome.fitness = None if genome.fitness is None else float(genome.fitness)
    payload = derived_experiment_payload(genome, config, name, provenance, split_id,
                                         dataset_id, source_config_json, neat_config_text)
    e = payload["experiment"]
    for k in ("split_id", "dataset_id"):
        if isinstance(e[k], str):
            e[k] = _uuid.UUID(e[k])
    now = datetime.utcnow()
    with database.session_scope() as s:
        if s.query(Experiment).filter(Experiment.name == name).count():
            raise ValueError(f"experiment {name!r} already exists")
        exp = Experiment(start_time=now, end_time=now, **e)
        s.add(exp)
        s.flush()
        pop = Population(experiment_id=exp.id, **payload["population"])
        s.add(pop)
        s.flush()
        row = Genome.from_neat_genome(genome, pop.id)
        s.add(row)
        s.flush()
        return {"experiment_id": exp.id, "population_id": pop.id, "genome_id": row.id}


# ====================================================================== base features
# (study C amendment 2a: complementary donor selection)

_USAGE_FEATURE = re.compile(r"^total (day|eve|night|intl) (minutes|charge)$")


def base_feature_name(feature_name: str) -> str:
    """Base feature of a churn input column: ``total <period> minutes`` and
    ``total <period> charge`` collapse to ``<period> usage`` (charge is an exact
    linear function of minutes), and every one-hot column ``<feature>:<level>``
    (e.g. ``state:16``, ``area code:415``) collapses to its source ``<feature>``
    (amendment 3: per-level tokens inflated the Jaccard union and let a donor
    duplicating the recipient's mechanism pass). Other features are their own
    base feature."""
    m = _USAGE_FEATURE.match(feature_name)
    if m:
        return f"{m.group(1)} usage"
    return feature_name.split(":", 1)[0] if ":" in feature_name else feature_name


def entry_base_features(entry_nodes: Iterable, resolver: "ModelStateIdResolver",
                        feature_names: Sequence[str], input_keys: Iterable[int]) -> frozenset:
    """Base-feature set of an annotation's entry nodes.

    Split copies (``-61_a``) collapse to their input through ``resolver``;
    input id ``-(i+1)`` is data column ``i``, mapped with ``base_feature_name``.
    A non-input entry is kept as its own token (``node:<genome id>`` or
    ``identity:<id>``), so it never matches an input feature.
    """
    inputs = set(input_keys)
    out = set()
    for n in entry_nodes:
        n = str(n)
        if resolver.is_identity(n):
            out.add(f"identity:{resolver.base(n)}")
            continue
        g = resolver.genome_node(n)
        if g in inputs:
            out.add(base_feature_name(feature_names[-g - 1]))
        else:
            out.add(f"node:{g}")
    return frozenset(out)


def jaccard(a: Iterable, b: Iterable) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def complementary_donor(candidates: Sequence[dict], recipient_seed: int,
                        max_jaccard: float = 0.5, min_gain: float = 0.01):
    """Amendment 2a: the complementary donor for ``recipient_seed``.

    ``candidates``: one dict per leaf annotation of every explanation, with
    ``seed``, ``name``, ``train_restore_gain``, ``nonlinear``,
    ``entries_all_inputs``, ``qualifies`` (graftable: entries all inputs,
    collapse preconditions, a readout edge) and ``entry_base_features``.

    Among leaves of *other* seeds that qualify, have all-input entries and
    train restore gain > ``min_gain``, keep those whose base-feature entry set
    has Jaccard < ``max_jaccard`` with the base-feature entry set of every
    nonlinear leaf of the recipient; return the one with the highest gain
    (ties: lowest seed, then name), or None. Also returns, per eligible
    candidate, its maximum Jaccard against the recipient's nonlinear leaves.
    """
    rec_nl = [c for c in candidates if c["seed"] == recipient_seed and c["nonlinear"]]
    scored = []
    for c in candidates:
        if c["seed"] == recipient_seed or not c.get("qualifies") \
                or not c.get("entries_all_inputs") or not c["train_restore_gain"] > min_gain:
            continue
        mj = max((jaccard(c["entry_base_features"], r["entry_base_features"])
                  for r in rec_nl), default=0.0)
        scored.append((c, mj))
    ok = [(c, mj) for c, mj in scored if mj < max_jaccard]
    if not ok:
        return None, scored
    best = sorted(ok, key=lambda t: (-t[0]["train_restore_gain"], t[0]["seed"], t[0]["name"]))[0]
    return best, scored


# ====================================================================== readout refit
# (study C amendment 2b)

def _output_layer(net: NeuralNeat, output_key: int):
    for L, layer in net.layers.items():
        if output_key in layer["nodes"] and layer["layer_type"] == "OUTPUT":
            return L
    raise KeyError(f"output node {output_key} not in the network")


def output_preactivation_terms(genome, config, X, output_key: int = 0):
    """NeuralNeat's output-layer input matrix, the output's weight column and
    bias, and its pre-activation z (P = sigmoid(z)), all float64 numpy."""
    net = build_net(genome, config)
    with torch.no_grad():
        net.forward(_as_tensors(X))
        L = _output_layer(net, output_key)
        inp = torch.cat([net._outputs[ii] for ii in net.layer_inputs[L]], dim=1).numpy()
        col = net.layers[L]["nodes"][output_key]["layer_index"]
        w = net.weights[L].detach().numpy()[:, col].copy()
        b = float(net.biases[L].detach().numpy()[col])
    return {"net": net, "layer": L, "col": col, "inputs": inp, "w": w, "b": b,
            "z": inp @ w + b}


def _logistic_newton(offset, F, y, theta0, l2, penalise, max_iter=100, tol=1e-12):
    """argmin mean BCE(sigmoid(offset + F theta), y) + l2/2 * sum(theta[penalise]^2),
    damped Newton with backtracking from theta0. Returns (theta, n_iter, converged)."""
    pen = np.asarray(penalise, dtype=np.float64)
    n = len(y)

    def obj(t):
        z = offset + F @ t
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * np.sum(pen * t * t))

    t = np.asarray(theta0, dtype=np.float64).copy()
    f = obj(t)
    for it in range(1, max_iter + 1):
        z = offset + F @ t
        p = 1.0 / (1.0 + np.exp(-z))
        g = F.T @ (p - y) / n + l2 * pen * t
        H = (F * (p * (1 - p))[:, None]).T @ F / n + np.diag(l2 * pen)
        H += 1e-12 * np.eye(len(t))
        step = np.linalg.solve(H, g)
        a = 1.0
        while True:
            t_new = t - a * step
            f_new = obj(t_new)
            if f_new <= f or a < 1e-10:
                break
            a *= 0.5
        if f_new > f:  # no descent possible: at the optimum to precision
            return t, it, True
        done = f - f_new < tol and np.max(np.abs(g)) < 1e-8
        t, f = t_new, f_new
        if done:
            return t, it, True
    return t, max_iter, False


REFIT_L2 = 1e-6


def refit_readout(genome, config, X, y, readout_keys: Iterable[ConnKey],
                  scaled_groups: Sequence[Sequence[ConnKey]] = (), output_key: int = 0,
                  l2: float = REFIT_L2, max_iter: int = 100):
    """Refit only the given readout weight(s) and the output bias by logistic
    regression on (X, y), every other weight fixed (study C amendment 2b).

    The network's output pre-activation is z = offset + b + sum_k w_k a_k, with
    a_k the activation of readout source k (from NeuralNeat's output-layer
    input matrix) and ``offset`` the rest of z, which the refit holds fixed.
    ``scaled_groups``: groups of input->output keys that together form one
    identity exit's readout; each group gets one scale factor (its exit's
    readout weight) applied to all its weights.

    L2 of ``l2`` (default 1e-6, for stability when an exit is dead on X) on the
    readout parameters only; the output bias is unpenalised. Damped Newton from
    the current values, so the penalised train log-loss never increases.

    Returns (new genome, info).
    """
    readout_keys = [tuple(k) for k in readout_keys]
    groups = [[tuple(k) for k in g] for g in scaled_groups if len(g)]
    terms = output_preactivation_terms(genome, config, X, output_key)
    net, L = terms["net"], terms["layer"]
    imap = net.layers[L]["input_map"]

    def row(k):
        if k not in imap:
            raise KeyError(f"readout connection {k} not in the network's output layer")
        r, c = imap[k]
        assert c == terms["col"], k
        return r

    inp, w = terms["inputs"], terms["w"]
    feats, theta0 = [np.ones(len(inp))], [terms["b"]]
    explained = terms["b"] * np.ones(len(inp))
    for k in readout_keys:
        r = row(k)
        feats.append(inp[:, r])
        theta0.append(w[r])
        explained = explained + w[r] * inp[:, r]
    for g in groups:
        f = sum(w[row(k)] * inp[:, row(k)] for k in g)
        feats.append(f)
        theta0.append(1.0)
        explained = explained + f
    F = np.column_stack(feats)
    offset = terms["z"] - explained
    yv = np.asarray(y, dtype=np.float64).ravel()
    penalise = [0.0] + [1.0] * (F.shape[1] - 1)
    theta0 = np.asarray(theta0)

    def bce(t):
        z = offset + F @ t
        return float(np.mean(np.logaddexp(0.0, z) - yv * z))

    theta, n_iter, converged = _logistic_newton(offset, F, yv, theta0, l2, penalise, max_iter)

    g = copy.deepcopy(genome)
    g.nodes[output_key].bias = float(theta[0])
    for i, k in enumerate(readout_keys):
        g.connections[k].weight = float(theta[1 + i])
    scale = theta[1 + len(readout_keys):]
    for s, grp in zip(scale, groups):
        for k in grp:
            g.connections[k].weight = float(genome.connections[k].weight * s)
    p_back = predict_proba(g, config, X)
    p_fit = 1.0 / (1.0 + np.exp(-(offset + F @ theta)))
    info = {
        "method": "logistic regression with fixed offset (damped Newton)",
        "l2": l2, "l2_on": "readout parameters only (output bias unpenalised)",
        "n_iter": n_iter, "converged": converged, "n_rows": len(yv),
        "readout_connections": [list(k) for k in readout_keys],
        "scaled_groups": [[list(k) for k in grp] for grp in groups],
        "output_bias_before": float(theta0[0]), "output_bias_after": float(theta[0]),
        "readout_weights_before": [float(v) for v in theta0[1:1 + len(readout_keys)]],
        "readout_weights_after": [float(v) for v in theta[1:1 + len(readout_keys)]],
        "group_scales_after": [float(v) for v in scale],
        "train_logloss_before": bce(theta0), "train_logloss_after": bce(theta),
        "max_abs_pred_diff_fit_vs_genome": float(np.max(np.abs(p_fit - p_back))),
    }
    return g, info


# ====================================================================== signal novelty
# (transplant study C2, TRANSPLANT2_PLAN.md)

def signal_basis(genome, config, X):
    """Recipient signal basis B_r(x): the network inputs plus the activation of
    every hidden node NeuralNeat computes for ``genome``, on ``X``.

    Returns (B, labels): B is (n, n_inputs + n_hidden) float64, labels are
    ``in:<id>`` (columns in input-key order, i.e. data columns) then
    ``h:<node id>`` (ascending id). Hidden activations are read from
    NeuralNeat's per-layer outputs (the same forward pass as ``predict_proba``).
    """
    net = build_net(genome, config)
    Xa = np.asarray(X, dtype=np.float64)
    with torch.no_grad():
        net.forward(_as_tensors(Xa))
        hidden = {}
        for L, layer in net.layers.items():
            if layer["layer_type"] in ("INPUT", "OUTPUT") or L not in net._outputs:
                continue
            out = net._outputs[L].detach().numpy()
            for nid, node in layer["nodes"].items():
                hidden[nid] = out[:, node["layer_index"]].copy()
    in_keys = list(config.genome_config.input_keys)
    special = set(in_keys) | set(config.genome_config.output_keys)
    every = {nid for layer in net.layers.values() for nid in layer["nodes"]} - special
    if every != set(hidden):
        raise ValueError(f"hidden nodes outside CONNECTED layers: {sorted(every - set(hidden))}")
    cols = [Xa[:, -k - 1] for k in in_keys]
    labels = [f"in:{k}" for k in in_keys]
    for nid in sorted(hidden):
        cols.append(hidden[nid])
        labels.append(f"h:{nid}")
    return np.column_stack(cols), labels


def readout_contribution(genome, config, X, readout_keys: Iterable[ConnKey],
                         io_keys: Iterable[ConnKey] = (), output_key: int = 0):
    """A leaf's logit contribution s_d(x) on ``X`` computed with ``genome``:
    sum over its readout edges of readout weight x exit activation, plus, for an
    identity exit, its input->output edges (weight x input).

    Returns (s, columns, theta0): ``columns`` are the refit features in
    ``refit_readout`` order (one activation column per readout key, then the
    summed input->output group if any) and ``theta0`` their current weights
    (the group's scale starts at 1), so ``s == columns @ theta0``.
    """
    terms = output_preactivation_terms(genome, config, X, output_key)
    imap = terms["net"].layers[terms["layer"]]["input_map"]
    inp, w = terms["inputs"], terms["w"]
    cols, theta0 = [], []
    for k in [tuple(k) for k in readout_keys]:
        r, c = imap[k]
        assert c == terms["col"], k
        cols.append(inp[:, r])
        theta0.append(w[r])
    io_keys = [tuple(k) for k in io_keys]
    if io_keys:
        g = np.zeros(len(inp))
        for k in io_keys:
            r, c = imap[k]
            assert c == terms["col"], k
            g = g + w[r] * inp[:, r]
        cols.append(g)
        theta0.append(1.0)
    F = np.column_stack(cols) if cols else np.zeros((len(inp), 0))
    theta0 = np.asarray(theta0, dtype=np.float64)
    return F @ theta0, F, theta0


def linear_novelty(s, B, rel_tol: float = 1e-12) -> dict:
    """nov = 1 - R^2 of the OLS regression of ``s`` on ``B`` with an intercept.

    Equivalent formulation used for conditioning: centre ``s`` and every
    column of ``B`` (that is the intercept), drop columns with zero variance,
    scale the rest to unit norm, then ``numpy.linalg.lstsq`` (SVD, minimum norm,
    so rank-deficient bases such as one-hot groups or duplicated activations
    are handled; the fitted values are the orthogonal projection either way).

    A (numerically) constant ``s`` has no variance to explain: nov = NaN with
    ``reason`` set. Returns {"nov", "r2", "rank", "n_cols", "n_cols_used", "reason"}.
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    sc = s - s.mean()
    sst = float(sc @ sc)
    out = {"n_cols": int(B.shape[1]), "rank": 0, "n_cols_used": 0, "reason": ""}
    if not np.isfinite(sst) or sst <= rel_tol * max(1.0, float(s @ s)):
        return dict(out, nov=float("nan"), r2=float("nan"),
                    reason="donor signal constant on the data")
    Bc = B - B.mean(axis=0)
    norms = np.sqrt(np.sum(Bc * Bc, axis=0))
    keep = norms > rel_tol * max(1.0, float(np.max(norms)) if norms.size else 1.0)
    Bk = Bc[:, keep] / norms[keep]
    out["n_cols_used"] = int(keep.sum())
    if Bk.shape[1] == 0:
        return dict(out, nov=1.0, r2=0.0)
    coef, _, rank, _ = np.linalg.lstsq(Bk, sc, rcond=None)
    resid = sc - Bk @ coef
    r2 = 1.0 - float(resid @ resid) / sst
    out["rank"] = int(rank)
    return dict(out, nov=1.0 - r2, r2=r2)


def refit_added_signal(z_recipient, output_bias, F, theta0, y, l2: float = REFIT_L2,
                       max_iter: int = 100):
    """Train incremental-value refit (C2 ``inc``): the logit of the recipient
    with the donor contribution added, z = z_recipient + F theta0, refit exactly
    as ``refit_readout`` would refit the grafted genome: the readout
    parameters (columns of ``F``) and the output bias by logistic regression,
    the rest of the recipient logit a fixed offset, L2 ``l2`` on the readout
    parameters only, damped Newton from the current values.

    Returns (z_refit, info).
    """
    z_r = np.asarray(z_recipient, dtype=np.float64).ravel()
    F = np.asarray(F, dtype=np.float64).reshape(len(z_r), -1)
    yv = np.asarray(y, dtype=np.float64).ravel()
    Fa = np.column_stack([np.ones(len(z_r)), F])
    t0 = np.concatenate([[float(output_bias)], np.asarray(theta0, dtype=np.float64)])
    offset = z_r - float(output_bias)
    pen = [0.0] + [1.0] * F.shape[1]
    theta, n_iter, converged = _logistic_newton(offset, Fa, yv, t0, l2, pen, max_iter)
    z = offset + Fa @ theta
    return z, {"l2": l2, "n_iter": n_iter, "converged": converged,
               "output_bias_before": float(t0[0]), "output_bias_after": float(theta[0]),
               "readout_params_before": [float(v) for v in t0[1:]],
               "readout_params_after": [float(v) for v in theta[1:]]}


def select_novel_donor(pairs: Sequence[dict], recipient_seed: int, min_nov: float = 0.5):
    """C2 selection for one recipient. ``pairs``: one dict per (donor
    candidate, recipient) with ``donor_seed``, ``donor_annotation``,
    ``recipient_seed``, ``nov`` and ``inc``.

    Donors from other seeds with ``nov >= min_nov`` (NaN never qualifies); the
    largest ``inc`` wins, ties -> lowest donor seed, then annotation name.
    Returns (best pair or None, max novelty over the recipient's other-seed
    pairs (NaN if none is finite)).
    """
    mine = [p for p in pairs if p["recipient_seed"] == recipient_seed
            and p["donor_seed"] != recipient_seed]
    finite = [p["nov"] for p in mine if np.isfinite(p["nov"])]
    max_nov = max(finite) if finite else float("nan")
    ok = [p for p in mine if np.isfinite(p["nov"]) and p["nov"] >= min_nov
          and np.isfinite(p["inc"])]
    if not ok:
        return None, max_nov
    best = sorted(ok, key=lambda p: (-p["inc"], p["donor_seed"], p["donor_annotation"]))[0]
    return best, max_nov


# ====================================================================== knockout and rescue
# (study D, RESCUE_PLAN.md)

def feature_knockout(genome, config, sub: DonorSubgraph, output_key: int = 0):
    """Knock a leaf feature out of a deep copy of ``genome``: disable every
    connection of its resolved subgraph ``sub`` (entry -> feature and internal
    connections, the readout edges to the output, and any input -> output
    edges of an identity exit). Every one must be an enabled genome
    connection.

    Nothing else is touched. Enabled edges between a feature node and a node
    outside the feature that are not part of ``sub`` are reported, not
    disabled: ``other_outgoing`` (feature node -> non-output node outside the
    feature), ``other_to_output`` (feature node -> output not in ``sub``) and
    ``external_incoming`` (node outside the feature -> feature node).

    Returns (knocked-out genome, record).
    """
    keys = [tuple(k) for k in sub.all_connections]
    bad = [k for k in keys if k not in genome.connections or not genome.connections[k].enabled]
    if bad:
        raise ValueError(f"knockout of {sub.annotation_name!r}: connection(s) {bad} are not "
                         "enabled genome connections")
    outputs = set(config.genome_config.output_keys)
    feat = set(sub.hidden_nodes)
    kset = set(keys)
    enabled = [k for k, c in genome.connections.items() if c.enabled and k not in kset]
    other_out = sorted(k for k in enabled if k[0] in feat and k[1] not in feat
                       and k[1] not in outputs)
    other_to_output = sorted(k for k in enabled if k[0] in feat and k[1] in outputs)
    external_in = sorted(k for k in enabled if k[1] in feat and k[0] not in feat)
    ko = disable_connections(genome, keys)
    record = {"annotation_name": sub.annotation_name, "genome_key": genome.key,
              "disabled_connections": sorted(keys), "n_disabled": len(keys),
              "feature_hidden_nodes": sorted(feat),
              "readout_connections": sorted(tuple(k) for k in sub.readout_connections),
              "input_output_connections": sorted(tuple(k) for k in
                                                 sub.input_output_connections),
              "other_outgoing": other_out, "other_to_output": other_to_output,
              "external_incoming": external_in}
    return ko, record


def enabled_feature_readout(genome, sub: DonorSubgraph):
    """(readout keys, input -> output keys) of ``sub`` that are enabled in
    ``genome`` (after a knockout: none)."""
    def on(k):
        k = tuple(k)
        return k in genome.connections and genome.connections[k].enabled
    return ([tuple(k) for k in sub.readout_connections if on(k)],
            [tuple(k) for k in sub.input_output_connections if on(k)])


def feature_contribution(genome, config, X, sub: DonorSubgraph, output_key: int = 0):
    """The feature's logit contribution in ``genome`` on ``X``, over the
    readout (and input -> output) edges of ``sub`` that are enabled in
    ``genome``; identically zero when none is enabled (a knocked-out feature)."""
    ro, io = enabled_feature_readout(genome, sub)
    if not ro and not io:
        return np.zeros(len(np.asarray(X)))
    s, _, _ = readout_contribution(genome, config, X, ro, io, output_key)
    return s


def graft_collisions(recipient, recipient_config, donor_subgraph: DonorSubgraph):
    """Connection keys a ``graft`` of ``donor_subgraph`` into ``recipient``
    would add that already exist in ``recipient`` (enabled or disabled), with
    the same id mapping as ``graft``. Only input -> output keys can collide
    (grafted hidden nodes get fresh ids). Returns
    [{"key": (a, b), "recipient_enabled": bool}] sorted by key."""
    start = max(max(recipient.nodes), 0) + 1
    id_map = {d: start + i for i, d in enumerate(sorted(donor_subgraph.hidden_nodes))}
    special = (set(recipient_config.genome_config.input_keys)
               | set(recipient_config.genome_config.output_keys))

    def m(n):
        return n if n in special else id_map[n]

    out = []
    for k in donor_subgraph.all_connections:
        nk = (m(k[0]), m(k[1]))
        if nk in recipient.connections:
            out.append({"key": nk, "recipient_enabled": bool(recipient.connections[nk].enabled)})
    return sorted(out, key=lambda d: d["key"])


def graft_into_knockout(ko_genome, config, donor_genome, donor_subgraph: DonorSubgraph,
                        knocked_out_keys: Iterable[ConnKey] = (),
                        sham_seed: Optional[int] = None):
    """``graft`` of a donor feature into a knocked-out recipient, with a clear
    failure when a key the donor needs already exists in the recipient.

    The recipient's own gene is never re-enabled or overwritten: any collision
    (e.g. the donor needs input -> output edge (-3, 0) and the recipient holds
    that key, enabled, disabled by the knockout, or disabled originally)
    raises ``GraftCollisionError`` naming each key and its state, before
    ``graft`` runs. Returns ``graft``'s (new genome, record).
    """
    ko_keys = {tuple(k) for k in knocked_out_keys}
    col = graft_collisions(ko_genome, config, donor_subgraph)
    if col:
        desc = []
        for c in col:
            if c["recipient_enabled"]:
                state = "enabled in the recipient"
            elif c["key"] in ko_keys:
                state = "disabled by the knockout"
            else:
                state = "disabled in the original recipient"
            desc.append(f"{c['key']} ({state})")
        raise GraftCollisionError(
            f"graft of {donor_subgraph.annotation_name!r} into knocked-out recipient "
            f"{ko_genome.key} needs connection key(s) the recipient already holds: "
            f"{', '.join(desc)}. Re-enabling or overwriting the recipient's own gene is "
            "not allowed; the case fails.")
    return graft(ko_genome, config, donor_genome, donor_subgraph, sham_seed=sham_seed)


# ====================================================================== algorithmic aliases
# (study D2, RESCUE2_REINIT_PLAN.md)

def pearson_r(a, b) -> float:
    """Pearson correlation of two signals; nan when either is constant."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"signal shapes differ: {a.shape} vs {b.shape}")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def representation_differs(n_hidden_a: int, base_features_a, n_hidden_b: int,
                           base_features_b) -> bool:
    """D's check 2: the features differ in hidden-node count or in their entry
    base-feature set."""
    return n_hidden_a != n_hidden_b or frozenset(base_features_a) != frozenset(base_features_b)


def alias_pairs(leaves: Sequence[dict], min_r: float = 0.9, min_ko_drop: float = 0.005):
    """Every ordered pair (recipient F, donor G) of candidate leaves from
    different seeds, with the D2 alias rule (train quantities):

    1. Pearson r between F's and G's logit contributions (``s``, each computed
       with its own genome on the same inputs) >= ``min_r``;
    2. F and G differ in hidden-node count or entry base-feature set;
    3. knocking out F costs >= ``min_ko_drop`` train AUC (``ko_train_auc_drop``
       of F; nan / None = knockout not possible -> fails).

    ``leaves``: dicts with seed, name, s, n_hidden, base_features,
    ko_train_auc_drop. Returns one dict per ordered pair, sorted by
    (F seed, F name, G seed, G name), with the three checks and ``alias``
    (all three hold).
    """
    out = []
    key = lambda L: (L["seed"], L["name"])  # noqa: E731
    for F in sorted(leaves, key=key):
        drop = F.get("ko_train_auc_drop")
        drop = float("nan") if drop is None else float(drop)
        for G in sorted(leaves, key=key):
            if G["seed"] == F["seed"]:
                continue
            r = pearson_r(F["s"], G["s"])
            diff = representation_differs(F["n_hidden"], F["base_features"], G["n_hidden"],
                                          G["base_features"])
            r_ok = bool(np.isfinite(r) and r >= min_r)
            ko_ok = bool(np.isfinite(drop) and drop >= min_ko_drop)
            out.append({"recipient_seed": F["seed"], "recipient_feature": F["name"],
                        "donor_seed": G["seed"], "donor_feature": G["name"],
                        "r": r, "r_ok": r_ok, "different_representation": diff,
                        "ko_train_auc_drop": drop, "knockout_cost_ok": ko_ok,
                        "alias": r_ok and diff and ko_ok})
    return out


def wrong_family_donor(leaves: Sequence[dict], recipient: dict, exclude_seeds,
                       max_abs_r: float = 0.3):
    """D2's wrong-family donor for recipient feature ``recipient``: among the
    candidate ``leaves`` from seeds not in ``exclude_seeds`` (the recipient's
    and the alias donor's), those with |Pearson r| < ``max_abs_r`` between
    their contribution and the recipient feature's; the one with the highest
    ``train_restore_gain`` (ties: lowest seed, then name).

    Returns (leaf or None, its r, number of eligible leaves).
    """
    excl = set(exclude_seeds)
    elig = []
    for L in leaves:
        if L["seed"] in excl:
            continue
        r = pearson_r(recipient["s"], L["s"])
        if np.isfinite(r) and abs(r) < max_abs_r:
            elig.append((L, r))
    if not elig:
        return None, float("nan"), 0
    L, r = sorted(elig, key=lambda t: (-t[0]["train_restore_gain"], t[0]["seed"],
                                       t[0]["name"]))[0]
    return L, r, len(elig)
