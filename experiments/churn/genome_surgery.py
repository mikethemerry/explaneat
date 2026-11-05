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
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--",
                                              "explaneat", "experiments/churn"],
                                             cwd=REPO, text=True).strip())
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
