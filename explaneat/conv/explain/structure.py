"""Structural identification for ConvNEAT networks: find the larger subnetworks first, separate
them, and only then read them (ExplaNEAT's decomposition and recomposition on the conv DAG).

1. **Attribution flow.** For every class c and every enabled edge e = i -> j, the flow
   ``F_e^c = mean_images sum_pixels z_e * dz_c / dpre_j`` (the edge's term times the gradient of
   the class score with respect to the target's pre-activation: gradient x input at the edge).
   Node attribution ``A_n^c = mean sum h_n * dz_c / dh_n``. |flow| summed over classes weights the
   graph; the signed per-class flows give each digit's pathway.
2. **Modules.** Greedy agglomerative grouping of the functional hidden maps: two groups merge
   when they exchange attribution flow (normalised by the smaller group's total flow) or hold maps
   with near-identical pooled activations (correlation >= ``SIM``), as long as the groups still form
   a DAG (so they compose as function nodes) and their number stays above ``k_min``. Merging
   continues while there are more than ``k_max`` groups or the best score is >= ``tau``.
   Inactive maps (zero on every image) join the group of their main consumer.
3. **Separation by node splits.** A map that feeds both its own group and other groups (an
   intermediate node with external outputs) is split into two copies with the same inputs, kernels
   and bias: the original keeps the internal edges, the copy (``split_of``) carries the external
   edges and is the group's exported feature. This is the identity operation of
   ``docs/annotation_collapsing_model.md`` (node splits, exit-only egress); the function is
   unchanged, which the pipeline checks on every evaluation image.
4. **Motifs.** Resolution stages, serial chains, parallel branches into one node, skip edges
   (spanning two or more depth steps), and bottleneck maps (highest share of all attribution flow
   passing through them).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..genome import ConnGene, ConvGenome, NodeGene, topo_order
from .engine import ExplainNet, depth_map, functional_nodes
from .parts import _quotient_acyclic

SIM = 0.8


# ---------------------------------------------------------------------------- attribution flow
def attribution_flow(net: ExplainNet, X: np.ndarray, k: int, batch: int = 250) -> dict:
    """Per-edge and per-node gradient x input attributions to every class score, averaged over
    images: signed means and mean absolute values. Returns numpy arrays of shape (k,)."""
    g = net.g
    edges = [c.innov for c in g.enabled_conns()]
    nodes = [n for n in g.nodes if g.nodes[n].kind != "input"]
    sF = {e: np.zeros(k) for e in edges}
    aF = {e: np.zeros(k) for e in edges}
    sA = {n: np.zeros(k) for n in nodes}
    aA = {n: np.zeros(k) for n in nodes}
    dst = {c.innov: c.dst for c in g.enabled_conns()}
    N = 0
    for i in range(0, len(X), batch):
        x = torch.as_tensor(X[i:i + batch], dtype=torch.float64).clone().requires_grad_(True)
        f = net.forward(x, keep_contrib=True)
        for t in list(f.maps.values()) + list(f.pre.values()) + list(f.contrib.values()):
            if t.requires_grad:
                t.retain_grad()
        b = len(x)
        for c in range(k):
            for t in list(f.maps.values()) + list(f.pre.values()):
                if t.grad is not None:
                    t.grad = None
            f.logits[:, c].sum().backward(retain_graph=c < k - 1)
            for e in edges:
                gp = f.pre[dst[e]].grad
                if gp is None:
                    continue
                v = (f.contrib[e] * gp).sum(dim=(1, 2, 3)).detach().numpy()
                sF[e][c] += v.sum()
                aF[e][c] += np.abs(v).sum()
            for n in nodes:
                gh = f.maps[n].grad
                if gh is None:
                    continue
                v = (f.maps[n] * gh).sum(dim=(1, 2, 3)).detach().numpy()
                sA[n][c] += v.sum()
                aA[n][c] += np.abs(v).sum()
        N += b
    return {"edge_signed": {e: v / N for e, v in sF.items()}, "edge_abs": {e: v / N for e, v in aF.items()},
            "node_signed": {n: v / N for n, v in sA.items()}, "node_abs": {n: v / N for n, v in aA.items()}}


# ---------------------------------------------------------------------------- modules
def group_structures(g: ConvGenome, flow_abs: Dict[int, np.ndarray], pooled: Dict[int, np.ndarray],
                     inactive: set, k_min: int = 5, k_max: int = 8, tau: float = 0.35) -> Tuple[List[List[int]], List[dict]]:
    """Greedy agglomerative grouping (see the module docstring). Returns (groups in topological
    order, merge log)."""
    fn = functional_nodes(g)
    order = {n: i for i, n in enumerate(topo_order(g))}
    units = [n for n in topo_order(g) if n in fn and g.nodes[n].kind == "hidden" and n not in inactive]
    w = {}
    for c in g.enabled_conns():
        if c.src in units and c.dst in units:
            key = (c.src, c.dst)
            w[key] = w.get(key, 0.0) + float(flow_abs[c.innov].sum())
    tot = {n: 0.0 for n in units}
    for (a, b), v in w.items():
        tot[a] += v
        tot[b] += v
    corr = {}
    P = {n: pooled[n] for n in units}
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            pa, pb = P[a], P[b]
            if pa.std() > 0 and pb.std() > 0:
                r = float(np.corrcoef(pa, pb)[0, 1])
                if r >= SIM:
                    corr[(a, b)] = r
    group_of = {n: n for n in units}
    members = {n: [n] for n in units}
    log = []

    def score(A, B):
        ma, mb = members[A], members[B]
        f = sum(w.get((x, y), 0.0) + w.get((y, x), 0.0) for x in ma for y in mb)
        ta = sum(tot[x] for x in ma)
        tb = sum(tot[y] for y in mb)
        s_flow = f / max(1e-12, min(ta, tb))
        s_sim = max([corr.get((x, y), corr.get((y, x), 0.0)) for x in ma for y in mb], default=0.0)
        return s_flow + (s_sim if s_sim >= SIM else 0.0), s_flow, s_sim

    while len(members) > k_min:
        best = None
        keys = list(members)
        for i, A in enumerate(keys):
            for B in keys[i + 1:]:
                sc, sf, ss = score(A, B)
                if sc <= 0 or (best is not None and sc <= best[0]):
                    continue
                trial = dict(group_of)
                for n in members[B]:
                    trial[n] = A
                if not _quotient_acyclic(g, trial):
                    continue
                best = (sc, A, B, sf, ss, trial)
        if best is None or (len(members) <= k_max and best[0] < tau):
            break
        sc, A, B, sf, ss, trial = best
        group_of = trial
        members[A] = members[A] + members.pop(B)
        log.append({"merged": [int(x) for x in members[A]], "score": sc, "flow_share": sf, "similarity": ss})
    # inactive maps join their main consumer's group (if acyclic), else stay alone
    for n in sorted(inactive & set(n for n in fn if g.nodes[n].kind == "hidden"), key=order.get):
        cons = [c.dst for c in g.enabled_conns() if c.src == n and c.dst in group_of]
        placed = False
        for d in cons:
            trial = dict(group_of)
            trial[n] = group_of[d]
            if _quotient_acyclic(g, trial):
                group_of = trial
                members[group_of[d]].append(n)
                placed = True
                break
        if not placed:
            group_of[n] = n
            members[n] = [n]
    groups = [sorted(m, key=order.get) for m in members.values()]
    gid = {n: i for i, p in enumerate(groups) for n in p}
    deps = {i: {gid[c.src] for c in g.enabled_conns() if c.dst in p and c.src in gid and gid[c.src] != i}
            for i, p in enumerate(groups)}
    out, done = [], set()
    while len(out) < len(groups):
        ready = [i for i in range(len(groups)) if i not in done and deps[i] <= done]
        i = min(ready, key=lambda i: order[groups[i][0]])
        done.add(i)
        out.append(groups[i])
    return out, log


# ---------------------------------------------------------------------------- separation by splits
def split_mixed_roles(g: ConvGenome, groups: List[List[int]]) -> Tuple[ConvGenome, List[List[int]], Dict[int, int]]:
    """For every member with both internal and external consumers, add a copy that carries the
    external edges (same inputs, kernels, bias). Returns (new genome, new groups, {copy: original})."""
    g2 = copy.deepcopy(g)
    gid = {n: i for i, p in enumerate(groups) for n in p}
    nxt_node = max(g2.nodes) + 1
    nxt_inn = max(g2.conns) + 1
    split_of: Dict[int, int] = {}
    new_groups = [list(p) for p in groups]
    for i, p in enumerate(groups):
        for n in p:
            outs = [c for c in g2.enabled_conns() if c.src == n]
            internal = [c for c in outs if gid.get(c.dst) == i]
            external = [c for c in outs if gid.get(c.dst) != i]
            if not internal or not external:
                continue
            cp = nxt_node
            nxt_node += 1
            nd = g2.nodes[n]
            g2.nodes[cp] = NodeGene(cp, nd.kind, nd.level, nd.bias)
            for c in [c for c in g2.enabled_conns() if c.dst == n]:
                g2.conns[nxt_inn] = ConnGene(nxt_inn, c.src, cp, c.kernel.copy(), True)
                nxt_inn += 1
            for c in external:
                del g2.conns[c.innov]
                g2.conns[nxt_inn] = ConnGene(nxt_inn, cp, c.dst, c.kernel.copy(), True)
                nxt_inn += 1
            split_of[cp] = n
            new_groups[i].append(cp)
            gid[cp] = i
    order = {n: j for j, n in enumerate(topo_order(g2))}
    return g2, [sorted(p, key=order.get) for p in new_groups], split_of


# ---------------------------------------------------------------------------- motifs
def motifs(g: ConvGenome, groups: List[List[int]], flow_abs: Dict[int, np.ndarray], cfg) -> dict:
    fn = functional_nodes(g)
    depth = depth_map(g, fn)
    hid = [n for n in fn if g.nodes[n].kind == "hidden"]
    stages = {}
    for n in hid:
        stages.setdefault(g.nodes[n].level, []).append(n)
    skips = [c.innov for c in g.enabled_conns() if c.src in depth and c.dst in depth and depth[c.dst] - depth[c.src] >= 3]
    fan_in = {n: [c.src for c in g.enabled_conns() if c.dst == n and c.src in fn] for n in hid}
    parallel = [n for n, s in fan_in.items() if len(s) >= 3]
    # serial chains: maximal paths of nodes with exactly one functional input and one output
    succ = {n: [c.dst for c in g.enabled_conns() if c.src == n and c.dst in fn] for n in fn}
    pred = {n: [c.src for c in g.enabled_conns() if c.dst == n and c.src in fn] for n in fn}
    chains, seen = [], set()
    for n in sorted(hid, key=lambda n: depth[n]):
        if n in seen or not (len(pred[n]) == 1 and len(succ[n]) == 1):
            continue
        ch = [n]
        seen.add(n)
        while True:
            nx = succ[ch[-1]][0]
            if nx in hid and len(pred[nx]) == 1 and len(succ[nx]) == 1 and nx not in seen:
                ch.append(nx)
                seen.add(nx)
            else:
                break
        if len(ch) >= 2:
            chains.append(ch)
    through = {n: 0.0 for n in hid}
    total = 0.0
    for c in g.enabled_conns():
        v = float(flow_abs[c.innov].sum()) if c.innov in flow_abs else 0.0
        total += v
        if c.dst in through:
            through[c.dst] += v
    bott = sorted(hid, key=lambda n: -through[n])[:3]
    return {"stages": {int(l): [int(x) for x in v] for l, v in sorted(stages.items())},
            "stage_sizes": {int(l): list(cfg.level_size(l)) for l in stages},
            "skip_edges": [int(e) for e in skips], "n_skip_edges": len(skips), "parallel_into": [int(n) for n in parallel],
            "chains": [[int(x) for x in ch] for ch in chains],
            "bottlenecks": [{"node": int(n), "share": through[n] / total if total else 0.0} for n in bott],
            "max_depth": int(max(depth.values())) if depth else 0}
