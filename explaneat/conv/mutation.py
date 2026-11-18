"""ConvNEAT mutations. Each returns a short event string (or None if no-op)
and appends ``"mutation:<name>"`` to ``genome.history``.

Function preservation (see tests/test_conv/test_preservation.py):

* add_node: the split edge A -> B (kernel W, resample-then-conv) becomes
  A -> N (3x3 delta kernel) and N -> B (kernel W), N with bias 0.
  Default rule: N's level is A's or B's level (chosen at random). Both are
  EXACT when A's map is non-negative (A a hidden ReLU node, or an input with
  pixels >= 0; all datasets here are in [0, 1]) and the resamplings compose:
    - N at A's level: N = relu(A) = A, then N -> B resamples A to B's level
      exactly as the old edge did, and applies W there.
    - N at B's level: N = relu(resample(A)) = resample(A) (avg-pooling and
      nearest-upsampling keep non-negativity), then N -> B is same-level, W.
  More generally any level between (or finer than) both is exact when every
  map size divides by 2 at each level (adaptive avg-pool == avg-pool, and
  "nearest up then avg down" is the identity); with non-dividing sizes
  (e.g. a 7x9 image) adaptive pooling makes it approximate.
  APPROXIMATE cases: A can be negative (relu clips it), and the optional
  "coarse split" (probability p_coarse_split) that puts N one level coarser
  than both A and B: the map passes through a lower-resolution bottleneck
  (pool then upsample), which loses detail.
* add_conn: EXACT (zero kernel).
* kernel_size: EXACT when growing (odd k -> larger odd k, zero-padded and
  centred, "same" padding keeps alignment). APPROXIMATE when shrinking (crop).
* level: not function preserving (the map is pooled / upsampled).
* weight / toggle_enable: change the function by design.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .genome import (ConnGene, ConvGenome, ConvNeatConfig, InnovationTracker, NodeGene,
                     creates_cycle, random_kernel)


def delta_kernel(k: int) -> np.ndarray:
    d = np.zeros((k, k), dtype=np.float32)
    d[k // 2, k // 2] = 1.0
    return d


def reembed_kernel(kernel: np.ndarray, new_k: int) -> np.ndarray:
    """Centre ``kernel`` in a new_k x new_k kernel (zero-pad, or crop if smaller)."""
    k = kernel.shape[0]
    out = np.zeros((new_k, new_k), dtype=np.float32)
    if new_k >= k:
        o = (new_k - k) // 2
        out[o:o + k, o:o + k] = kernel
    else:
        o = (k - new_k) // 2
        out[:] = kernel[o:o + new_k, o:o + new_k]
    return out


def mutate_add_node(g: ConvGenome, cfg: ConvNeatConfig, tr: InnovationTracker,
                    rng: np.random.Generator, coarse: Optional[bool] = None) -> Optional[str]:
    enabled = g.enabled_conns()
    if not enabled:
        return None
    c = enabled[rng.integers(len(enabled))]
    a, b = g.nodes[c.src], g.nodes[c.dst]
    if coarse is None:
        coarse = rng.random() < cfg.p_coarse_split
    if coarse:
        level = min(max(a.level, b.level) + 1, cfg.max_level)
    else:
        level = a.level if rng.random() < 0.5 else b.level
    nid = tr.get_split_node(c.innov, set(g.nodes))
    c.enabled = False
    g.log_disable(c.innov, "mutation", f"add_node split -> node {nid}")
    g.nodes[nid] = NodeGene(nid, "hidden", level, 0.0)
    i1 = tr.get_conn(a.key, nid)
    i2 = tr.get_conn(nid, b.key)
    g.conns[i1] = ConnGene(i1, a.key, nid, delta_kernel(cfg.init_kernel_size))
    g.conns[i2] = ConnGene(i2, nid, b.key, c.kernel.copy())
    name = "add_node" + ("_coarse" if level > max(a.level, b.level) else "")
    g.history.append("mutation:" + name)
    return name


def _all_edges(g: ConvGenome):
    return [(c.src, c.dst) for c in g.conns.values()]


def _enabled_edges(g: ConvGenome):
    return [(c.src, c.dst) for c in g.conns.values() if c.enabled]


def mutate_add_conn(g: ConvGenome, cfg: ConvNeatConfig, tr: InnovationTracker,
                    rng: np.random.Generator, k: Optional[int] = None) -> Optional[str]:
    existing = {(c.src, c.dst) for c in g.conns.values()}
    srcs = [n for n in g.nodes.values() if n.kind != "output"]
    dsts = [n for n in g.nodes.values() if n.kind != "input"]
    edges = _all_edges(g)  # DAG over *all* genes, so re-enabling is usually safe
    cands = [(s.key, d.key) for s in srcs for d in dsts
             if s.key != d.key and (s.key, d.key) not in existing]
    rng.shuffle(cands)
    for s, d in cands:
        if not creates_cycle(edges, (s, d)):
            kk = int(k if k is not None else rng.choice(cfg.kernel_sizes))
            inn = tr.get_conn(s, d)
            g.conns[inn] = ConnGene(inn, s, d, np.zeros((kk, kk), dtype=np.float32))
            g.history.append("mutation:add_conn")
            return "add_conn"
    return None


def mutate_kernel_size(g: ConvGenome, cfg: ConvNeatConfig, rng: np.random.Generator,
                       new_k: Optional[int] = None, innov: Optional[int] = None) -> Optional[str]:
    conns = list(g.conns.values())
    if not conns:
        return None
    c = g.conns[innov] if innov is not None else conns[rng.integers(len(conns))]
    choices = [k for k in cfg.kernel_sizes if k != c.k]
    if not choices:
        return None
    nk = int(new_k if new_k is not None else rng.choice(choices))
    c.kernel = reembed_kernel(c.kernel, nk)
    g.history.append("mutation:kernel_size")
    return "kernel_size"


def mutate_toggle_enable(g: ConvGenome, cfg: ConvNeatConfig, rng: np.random.Generator) -> Optional[str]:
    conns = list(g.conns.values())
    if not conns:
        return None
    c = conns[rng.integers(len(conns))]
    if c.enabled:
        c.enabled = False
        g.log_disable(c.innov, "mutation", "toggle_enable")
        g.history.append("mutation:disable")
        return "disable"
    # re-enable: validate first
    if creates_cycle(_enabled_edges(g), (c.src, c.dst)):
        g.log_disable(c.innov, "repair", "re-enable refused: would create a cycle")
        g.history.append("mutation:enable_refused")
        return "enable_refused"
    c.enabled = True
    g.history.append("mutation:enable")
    return "enable"


def mutate_level(g: ConvGenome, cfg: ConvNeatConfig, rng: np.random.Generator) -> Optional[str]:
    hidden = g.hidden_keys()
    if not hidden or cfg.max_level == 0:
        return None
    n = g.nodes[hidden[rng.integers(len(hidden))]]
    step = int(rng.choice([-1, 1]))
    new = int(np.clip(n.level + step, 0, cfg.max_level))
    if new == n.level:
        new = int(np.clip(n.level - step, 0, cfg.max_level))
    n.level = new
    g.history.append("mutation:level")
    return "level"


def mutate_weights(g: ConvGenome, cfg: ConvNeatConfig, rng: np.random.Generator) -> Optional[str]:
    """Small Gaussian perturbation; backprop does the heavy lifting."""
    for c in g.enabled_conns():
        if rng.random() < cfg.weight_frac:
            c.kernel = (c.kernel + rng.standard_normal(c.kernel.shape) * cfg.weight_sigma).astype(np.float32)
    for n in g.nodes.values():
        if n.kind != "input" and rng.random() < cfg.weight_frac:
            n.bias = float(n.bias + rng.standard_normal() * cfg.weight_sigma)
    g.history.append("mutation:weight")
    return "weight"


def _n_times(rate: float, rng: np.random.Generator) -> int:
    """How many times a structural operator fires: ``rate`` is its expected count per offspring.
    For rate <= 1 this is the usual Bernoulli(rate) (one rng draw, as before); a scaled rate such
    as 1.4 fires once for sure and a second time with probability 0.4."""
    whole = int(np.floor(rate)) if rate > 1 else 0
    return whole + int(rng.random() < rate - whole)


def mutate(g: ConvGenome, cfg: ConvNeatConfig, tr: InnovationTracker,
           rng: np.random.Generator) -> list:
    """Apply each mutation operator independently with its probability."""
    events = []
    for _ in range(_n_times(cfg.p_add_node, rng)):
        events.append(mutate_add_node(g, cfg, tr, rng))
    for _ in range(_n_times(cfg.p_add_conn, rng)):
        events.append(mutate_add_conn(g, cfg, tr, rng))
    for _ in range(_n_times(cfg.p_kernel_size, rng)):
        events.append(mutate_kernel_size(g, cfg, rng))
    if rng.random() < cfg.p_toggle_enable:
        events.append(mutate_toggle_enable(g, cfg, rng))
    if rng.random() < cfg.p_level:
        events.append(mutate_level(g, cfg, rng))
    if rng.random() < cfg.p_weight:
        events.append(mutate_weights(g, cfg, rng))
    return [e for e in events if e]


def random_genome_by_mutation(key: int, cfg: ConvNeatConfig, tr: InnovationTracker,
                              rng: np.random.Generator, n_steps: int):
    """For the totality test: a minimal genome plus ``n_steps`` random mutations."""
    from .genome import minimal_genome
    g = minimal_genome(key, cfg, tr, rng)
    ops = ["add_node", "add_conn", "kernel_size", "toggle", "level", "weight"]
    for _ in range(n_steps):
        op = ops[rng.integers(len(ops))]
        if op == "add_node":
            mutate_add_node(g, cfg, tr, rng)
        elif op == "add_conn":
            mutate_add_conn(g, cfg, tr, rng)
        elif op == "kernel_size":
            mutate_kernel_size(g, cfg, rng)
        elif op == "toggle":
            mutate_toggle_enable(g, cfg, rng)
        elif op == "level":
            mutate_level(g, cfg, rng)
        else:
            mutate_weights(g, cfg, rng)
        if rng.random() < 0.3:  # make zero kernels non-zero so outputs are non-trivial
            for c in g.conns.values():
                if not c.kernel.any():
                    c.kernel = random_kernel(c.k, 0.3, rng)
    return g
