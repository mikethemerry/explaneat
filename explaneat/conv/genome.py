"""ConvNEAT genome: single-channel feature-map nodes, k x k kernel connections.

Prototype (research code). Nothing outside ``explaneat.conv`` depends on it.

Encoding
--------
* ``NodeGene``: one single-channel 2-D feature map. ``kind`` is ``input``
  (one per image channel, key -1, -2, ...), ``output`` (key 0..n_out-1, the
  maps read by the fixed head) or ``hidden`` (ReLU). ``level`` r is the
  resolution level: the map is ``size(H, r) x size(W, r)`` where
  ``size(n, r) = max(1, n // 2**r)``. Inputs and outputs sit at level 0.
* ``ConnGene``: one k x k kernel from ``src``'s map to ``dst``'s map, stride
  1, "same" padding, then a *derived* resampling from ``src.level`` to
  ``dst.level`` (adaptive average pooling when going down, nearest-neighbour
  upsampling when going up).  Nothing about the shape is evolved, so every
  genome builds (the genotype-to-phenotype map is total); the only structural
  constraint is that the *enabled* edges form a DAG.

Node ``j`` computes ``y_j = act(b_j + sum_i resample(W_ji * x_i))``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConvNeatConfig:
    """Everything that is fixed for a run. All fields are recorded in results."""

    in_channels: int = 1
    image_hw: Tuple[int, int] = (16, 16)
    n_classes: int = 1  # 1 = binary (sigmoid head), >1 = softmax head
    max_level: int = 2
    head: str = "gap"  # "gap": global average pool per output map; "grid2": 2x2 average pool (4 cells per map)
    kernel_sizes: Tuple[int, ...] = (1, 3, 5)
    init_kernel_size: int = 3
    init_kernel_std: float = 0.3

    # mutation probabilities (applied independently per offspring)
    p_add_node: float = 0.25
    p_add_conn: float = 0.35
    p_kernel_size: float = 0.10
    p_toggle_enable: float = 0.05
    p_level: float = 0.05
    p_weight: float = 0.6
    weight_sigma: float = 0.05
    weight_frac: float = 0.3
    p_coarse_split: float = 0.25  # add-node puts N one level coarser than A

    # crossover / speciation (NEAT formula, kernel term instead of weight term)
    p_crossover: float = 0.3
    p_inherit_disabled: float = 0.75
    c_excess: float = 1.0
    c_disjoint: float = 1.0
    c_kernel: float = 2.0
    compat_threshold: float = 1.0
    target_species: int = 4

    # population / training
    pop_size: int = 24
    generations: int = 12
    epochs_per_gen: int = 2
    batch_size: int = 128
    lr: float = 0.01
    complexity_penalty: float = 0.002
    elitism: int = 1
    survival_threshold: float = 0.4

    @property
    def n_out_maps(self) -> int:
        return self.n_classes

    @property
    def head_cells(self) -> int:
        """Pooled values per output map read by the linear head."""
        return {"gap": 1, "grid2": 4}[getattr(self, "head", "gap")]

    def level_size(self, r: int) -> Tuple[int, int]:
        h, w = self.image_hw
        return max(1, h // (2 ** r)), max(1, w // (2 ** r))


@dataclass
class NodeGene:
    key: int
    kind: str  # "input" | "hidden" | "output"
    level: int = 0
    bias: float = 0.0


@dataclass
class ConnGene:
    innov: int
    src: int
    dst: int
    kernel: np.ndarray  # (k, k) float32, k odd
    enabled: bool = True

    @property
    def k(self) -> int:
        return int(self.kernel.shape[0])


class InnovationTracker:
    """Historical markings shared by the whole population.

    A connection (src, dst) always gets the same innovation number; splitting
    connection ``innov`` always yields the same new node id (unless the genome
    already has that node, e.g. after re-enabling and splitting again).
    """

    def __init__(self, n_outputs: int):
        self.conn_innov: Dict[Tuple[int, int], int] = {}
        self.split_node: Dict[int, int] = {}
        self.next_innov = 0
        self.next_node = n_outputs

    def get_conn(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.conn_innov:
            self.conn_innov[key] = self.next_innov
            self.next_innov += 1
        return self.conn_innov[key]

    def get_split_node(self, innov: int, existing: set) -> int:
        nid = self.split_node.get(innov)
        if nid is None or nid in existing:
            nid = self.next_node
            self.next_node += 1
            if innov not in self.split_node:
                self.split_node[innov] = nid
        return nid


@dataclass
class ConvGenome:
    key: int
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    conns: Dict[int, ConnGene] = field(default_factory=dict)
    head_w: Optional[np.ndarray] = None  # (n_classes, n_out_maps)
    head_b: Optional[np.ndarray] = None  # (n_classes,)
    fitness: Optional[float] = None
    disable_log: List[dict] = field(default_factory=list)
    history: List[str] = field(default_factory=list)  # "mutation:add_node", ...
    metrics: dict = field(default_factory=dict)

    # ---- structure helpers -------------------------------------------------
    def input_keys(self) -> List[int]:
        return sorted((k for k, n in self.nodes.items() if n.kind == "input"), reverse=True)

    def output_keys(self) -> List[int]:
        return sorted(k for k, n in self.nodes.items() if n.kind == "output")

    def hidden_keys(self) -> List[int]:
        return sorted(k for k, n in self.nodes.items() if n.kind == "hidden")

    def enabled_conns(self) -> List[ConnGene]:
        return [c for c in self.conns.values() if c.enabled]

    def complexity(self) -> int:
        """Hidden nodes + enabled edges (the parsimony term)."""
        return len(self.hidden_keys()) + len(self.enabled_conns())

    def n_params(self) -> int:
        p = sum(c.k * c.k for c in self.enabled_conns())
        p += sum(1 for n in self.nodes.values() if n.kind != "input")
        if self.head_w is not None:
            p += self.head_w.size + self.head_b.size
        return p

    def log_disable(self, innov: int, cause: str, detail: str = "") -> None:
        """Record every disable. cause in {mutation, recombination, repair}."""
        assert cause in ("mutation", "recombination", "repair"), cause
        self.disable_log.append({"innov": int(innov), "cause": cause, "detail": detail})

    def copy(self, new_key: Optional[int] = None) -> "ConvGenome":
        g = copy.deepcopy(self)
        if new_key is not None:
            g.key = new_key
        g.fitness = None
        g.metrics = {}
        return g


# ---- graph utilities -------------------------------------------------------

def creates_cycle(edges: List[Tuple[int, int]], new: Tuple[int, int]) -> bool:
    """True if adding ``new`` (src, dst) to ``edges`` would create a cycle."""
    src, dst = new
    if src == dst:
        return True
    adj: Dict[int, List[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    stack, seen = [dst], set()
    while stack:
        n = stack.pop()
        if n == src:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, []))
    return False


def topo_order(genome: ConvGenome) -> List[int]:
    """Topological order of all nodes over enabled edges (raises on a cycle)."""
    indeg = {k: 0 for k in genome.nodes}
    adj: Dict[int, List[int]] = {k: [] for k in genome.nodes}
    for c in genome.enabled_conns():
        adj[c.src].append(c.dst)
        indeg[c.dst] += 1
    ready = sorted(k for k, d in indeg.items() if d == 0)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort()
    if len(order) != len(genome.nodes):
        raise ValueError("enabled connections contain a cycle")
    return order


def is_valid(genome: ConvGenome, cfg: ConvNeatConfig) -> Tuple[bool, str]:
    """Structural validity: the only possible failure is a cycle / bad gene."""
    for c in genome.conns.values():
        if c.src not in genome.nodes or c.dst not in genome.nodes:
            return False, f"conn {c.innov} references a missing node"
        if c.k % 2 != 1 or c.k not in cfg.kernel_sizes:
            return False, f"conn {c.innov} has kernel size {c.k}"
        if genome.nodes[c.dst].kind == "input" or genome.nodes[c.src].kind == "output":
            return False, f"conn {c.innov} enters an input or leaves an output"
    for n in genome.nodes.values():
        if not 0 <= n.level <= cfg.max_level:
            return False, f"node {n.key} level {n.level}"
    try:
        topo_order(genome)
    except ValueError as e:
        return False, str(e)
    return True, ""


def random_kernel(k: int, std: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal((k, k)) * std).astype(np.float32)


def minimal_genome(key: int, cfg: ConvNeatConfig, tracker: InnovationTracker,
                   rng: np.random.Generator) -> ConvGenome:
    """NEAT-style minimal start: every input map connected to every output map."""
    g = ConvGenome(key=key)
    for c in range(cfg.in_channels):
        g.nodes[-(c + 1)] = NodeGene(-(c + 1), "input", 0, 0.0)
    for o in range(cfg.n_out_maps):
        g.nodes[o] = NodeGene(o, "output", 0, 0.0)
    for i in g.input_keys():
        for o in g.output_keys():
            inn = tracker.get_conn(i, o)
            g.conns[inn] = ConnGene(inn, i, o, random_kernel(cfg.init_kernel_size, cfg.init_kernel_std, rng))
    n, cells = cfg.n_out_maps, cfg.head_cells
    # identity on the map means: with grid2 each map's 4 cells get weight 1/4, so the start equals GAP
    g.head_w = (np.repeat(np.eye(cfg.n_classes, n, dtype=np.float32), cells, axis=1) / cells).astype(np.float32)
    g.head_b = np.zeros(cfg.n_classes, dtype=np.float32)
    return g
