"""Study E (RESCUE2_REINIT_PLAN.md "E"): re-initialisation consistency.

"Given a topology, how consistent are the weights?"

E2 (network level, ``--stage network``): for each churn100 best genome
(runs_churn100.csv, 10 topologies) keep the topology and draw K = 10
re-initialisations of every enabled connection weight and every node bias
from N(0, 1), seeds 1000-1009 (``numpy.random.default_rng(seed)``: node biases
in ascending node id, then enabled connection weights in ascending key; the
same convention as ``genome_surgery.graft``'s sham draw). Disabled genes are
untouched. Then masked backprop on train (the PropNEAT primitive,
``genome_surgery.finetune(mask_phantom=True)`` re-implemented here only to
take a checkpoint mid-run: Adadelta lr 1.5, full batch BCE, torch seed = the
init seed), checkpoints at E = 500 (primary) and E = 2000 (secondary, the same
run continued). The evolved weights are the reference.

E1 (feature level, ``--stage feature``): for each D2 case
(rescue2_selection.csv, inclusion flag) plus D's confirmed cases B1, B2, C1, C2
(RESCUE_PLAN.md) not already included: knock the recipient feature out, graft
the alias topology with every grafted node bias and connection weight drawn
from N(0, 1) (``graft_into_knockout(sham_seed=seed)``, seeds 1000-1009), study
C's readout refit on train, then masked fine-tune E = 500 (ft seed = recipient
seed, as study D). Reference: the same pipeline with the donor's own weights
(D's condition 5).

Metrics (train for similarity, test for performance):
- test AUC (+ bootstrap CI and paired delta vs the reference, 10,000
  resamples, seed 42);
- logit Pearson r (E2) or feature logit-contribution r (E1) on train, for
  every pair of inits and each init vs the reference;
- symmetry-aware weight similarity: every hidden ReLU unit's incoming
  weight+bias vector is scaled to unit norm and its outgoing weights by the
  inverse (positive homogeneity of ReLU, so the function is unchanged; done in
  topological order so a unit's incoming vector already carries its
  predecessors' rescaling). Units are matched one-to-one by the Hungarian
  algorithm on cosine within groups of identical topological position (same
  input and output node sets), groups in topological order (a matched
  permutation is carried into later units' incoming vectors). Median matched
  cosine and share >= 0.9;
- "equivalent": logit r >= 0.99 and median matched cosine >= 0.9 (vs the
  reference);
- dead units: hidden units whose ReLU output is 0 on every train row.

No genome is saved to the DB (allowed for E); final weights are written as
JSON to experiments/churn/reinit_weights/.

    uv run python experiments/churn/reinit.py --stage network            # E2 -> reinit_network.csv
    uv run python experiments/churn/reinit.py --stage feature            # E1 -> reinit_feature.csv
    uv run python experiments/churn/reinit.py --stage network --dry-run  # 1 topology, 2 inits, E = 3,6
"""

from __future__ import annotations

import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

import genome_surgery as gs

HERE = Path(__file__).resolve().parent
INIT_SEEDS = list(range(1000, 1010))
E_PRIMARY = 500
E_SECONDARY = 2000
EQUIV_R = 0.99
EQUIV_COS = 0.9
COS_HIGH = 0.9
NET_OUT = HERE / "reinit_network.csv"
FEAT_OUT = HERE / "reinit_feature.csv"
WEIGHTS_DIR = HERE / "reinit_weights"
SELECTION = HERE / "rescue2_selection.csv"
D_CONFIRMED = ["B1", "B2", "C1", "C2"]  # RESCUE_PLAN.md / rescue_confirm.csv include_primary

ConnKey = Tuple[int, int]


# ====================================================================== re-initialisation

def reinit_all(genome, seed: int):
    """Deep copy of ``genome`` with every node bias and every *enabled*
    connection weight redrawn from N(0, 1) (``default_rng(seed)``; biases in
    ascending node id, then weights in ascending connection key). Topology,
    disabled genes, responses and activations are unchanged."""
    g = copy.deepcopy(genome)
    rng = np.random.default_rng(seed)
    for nid in sorted(g.nodes):
        g.nodes[nid].bias = float(rng.normal(0.0, 1.0))
    for k in sorted(k for k, c in g.connections.items() if c.enabled):
        g.connections[k].weight = float(rng.normal(0.0, 1.0))
    return g


def train_masked(genome, config, X, y, checkpoints: Sequence[int], seed: int):
    """``genome_surgery.finetune(mask_phantom=True)`` with checkpoints: one
    run of max(checkpoints) Adadelta steps; at each checkpoint the trained
    weights are written back to a copy of the genome. A checkpoint at E equals
    ``finetune(epochs=E, mask_phantom=True)`` exactly (tested).

    Returns {E: (genome, info)} with info: loss_before, loss_at (train BCE of
    the written-back genome)."""
    cps = sorted(set(int(e) for e in checkpoints))
    g = copy.deepcopy(genome)
    torch.manual_seed(seed)
    xs, ys = gs._as_tensors(X, y)
    net = gs.build_net(g, config)
    w_masks, b_masks = gs._param_masks(net, g, (), (), mask_phantom=True)
    optimizer = optim.Adadelta(net.parameters(), lr=gs.FT_LR)
    optimizer.zero_grad()
    bce = nn.BCELoss()
    with torch.no_grad():
        loss_before = float(bce(net.forward(xs), ys))
    out = {}
    for step in range(1, cps[-1] + 1):
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
        if step in cps:
            net.update_genome_weights()  # writes into g (net.genome)
            snap = copy.deepcopy(g)
            with torch.no_grad():
                loss_at = float(bce(net.forward(xs), ys))
            out[step] = (snap, {"epochs": step, "seed": seed, "loss_before": loss_before,
                                "loss_at": loss_at, "optimizer": gs.FT_OPTIMIZER,
                                "lr": gs.FT_LR, "mask_phantom": True, "full_batch": True})
    return out


# ====================================================================== network parameters

def network_params(genome, config):
    """(units, conns, bias) of the network NeuralNeat builds for ``genome``:
    hidden units (excluding inputs and outputs), every connection in a layer
    ``input_map`` (enabled genes between valid nodes) with its weight, and the
    bias of every hidden and output node."""
    net = gs.build_net(genome, config)
    special = set(config.genome_config.input_keys) | set(config.genome_config.output_keys)
    nodes = {nid for layer in net.layers.values() for nid in layer["nodes"]}
    units = sorted(nodes - special)
    keys = sorted({tuple(k) for layer in net.layers.values() for k in layer["input_map"]})
    conns = {k: float(genome.connections[k].weight) for k in keys}
    bias = {n: float(genome.nodes[n].bias) for n in sorted(nodes)
            if n in genome.nodes and n not in config.genome_config.input_keys}
    return units, conns, bias


def topo_units(units: Iterable[int], conns: Dict[ConnKey, float]) -> List[int]:
    """``units`` in a topological order of ``conns`` (Kahn, ties by id)."""
    units = list(units)
    uset = set(units)
    pred = {u: {a for (a, b) in conns if b == u and a in uset} for u in units}
    order, done = [], set()
    while len(order) < len(units):
        ready = sorted(u for u in units if u not in done and pred[u] <= done)
        if not ready:
            raise ValueError("cycle among units")
        order.append(ready[0])
        done.add(ready[0])
    return order


def normalise_units(units, conns: Dict[ConnKey, float], bias: Dict[int, float]):
    """Symmetry normalisation: in topological order, each unit's incoming
    weight+bias vector is divided by its L2 norm c and every outgoing weight
    multiplied by c. ReLU(c t) = c ReLU(t) for c > 0, so the network function
    is unchanged. Returns (conns, bias, scales) as new dicts."""
    w, b = dict(conns), dict(bias)
    scales = {}
    for h in topo_units(units, w):
        inc = sorted(k for k in w if k[1] == h)
        v = np.array([w[k] for k in inc] + [b[h]], dtype=np.float64)
        c = float(np.linalg.norm(v))
        scales[h] = c
        if c <= 0:
            continue
        for k in inc:
            w[k] = w[k] / c
        b[h] = b[h] / c
        for k in [k for k in w if k[0] == h]:
            w[k] = w[k] * c
    return w, b, scales


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def position_groups(units, conns):
    """Units grouped by identical (input node set, output node set), groups in
    topological order (by their first member)."""
    order = topo_units(units, conns)
    pos = {u: i for i, u in enumerate(order)}
    key = {u: (frozenset(a for (a, b) in conns if b == u),
               frozenset(b for (a, b) in conns if a == u)) for u in units}
    groups: Dict[tuple, List[int]] = {}
    for u in order:
        groups.setdefault(key[u], []).append(u)
    return sorted(groups.values(), key=lambda g: pos[g[0]])


def match_units(units, conns_a, bias_a, conns_b, bias_b, normalise: bool = True):
    """Symmetry-aware matching of the hidden units of two networks with the
    same topology (same unit ids and connection keys).

    Both are symmetry-normalised (``normalise_units``); within each position
    group (``position_groups`` on network A) units are matched one-to-one by
    the Hungarian algorithm maximising the cosine of incoming weight+bias
    vectors. Groups are processed in topological order and B's matched
    permutation is applied to later units' incoming vectors.

    Returns {"pairs": [(unit_a, unit_b, cosine)], "cosines", "median", "share_ge",
    "n_groups", "max_group_size", "n_nontrivial_groups"}.
    """
    if set(conns_a) != set(conns_b):
        raise ValueError("networks differ in topology")
    if normalise:
        wa, ba, _ = normalise_units(units, conns_a, bias_a)
        wb, bb, _ = normalise_units(units, conns_b, bias_b)
    else:
        wa, ba, wb, bb = conns_a, bias_a, conns_b, bias_b
    groups = position_groups(units, wa)
    inv = {}  # A-role node -> B node (identity unless permuted)
    pairs = []
    for grp in groups:
        srcs = sorted(a for (a, b) in wa if b == grp[0])

        def vec_a(u):
            return np.array([wa[(s, u)] for s in srcs] + [ba[u]])

        def vec_b(u):
            return np.array([wb[(inv.get(s, s), u)] for s in srcs] + [bb[u]])

        if len(grp) == 1:
            u = grp[0]
            pairs.append((u, u, _cos(vec_a(u), vec_b(u))))
            continue
        C = np.array([[_cos(vec_a(a), vec_b(b)) for b in grp] for a in grp])
        cost = -np.nan_to_num(C, nan=-2.0)
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            inv[grp[i]] = grp[j]
            pairs.append((grp[i], grp[j], float(C[i, j])))
    cos = np.array([p[2] for p in pairs], dtype=np.float64)
    fin = cos[np.isfinite(cos)]
    return {"pairs": pairs, "cosines": cos,
            "median": float(np.median(fin)) if fin.size else float("nan"),
            "share_ge": float(np.mean(fin >= COS_HIGH)) if fin.size else float("nan"),
            "min": float(np.min(fin)) if fin.size else float("nan"),
            "n_groups": len(groups), "max_group_size": max((len(g) for g in groups), default=0),
            "n_nontrivial_groups": sum(len(g) > 1 for g in groups)}


def hidden_activations(genome, config, X) -> Dict[int, np.ndarray]:
    """{hidden node id: ReLU output on X} from NeuralNeat's forward pass."""
    net = gs.build_net(genome, config)
    with torch.no_grad():
        net.forward(gs._as_tensors(X))
        out = {}
        for L, layer in net.layers.items():
            if layer["layer_type"] in ("INPUT", "OUTPUT"):
                continue
            a = net._outputs[L].detach().numpy()
            for nid, node in layer["nodes"].items():
                out[nid] = a[:, node["layer_index"]].copy()
    return out


def dead_units(acts: Dict[int, np.ndarray], units: Iterable[int]) -> List[int]:
    """Units whose ReLU output is exactly 0 on every row (never active)."""
    return sorted(u for u in units if not np.any(acts[u] > 0))


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def logits(genome, config, X):
    return gs.output_preactivation_terms(genome, config, X)["z"]


def genome_weights_json(genome, keys=None):
    """Compact weights: {"bias": {node: b}, "conn": [[a, b, w], ...]} (enabled genes)."""
    ks = sorted(k for k, c in genome.connections.items() if c.enabled) if keys is None else keys
    return {"bias": {str(n): genome.nodes[n].bias for n in sorted(genome.nodes)},
            "conn": [[int(k[0]), int(k[1]), genome.connections[tuple(k)].weight] for k in ks]}


def _nanmed(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=np.float64)
    return float(np.median(v)) if v.size else float("nan")


# ====================================================================== E2 runner

NET_COLUMNS = [
    "kind", "topology_seed", "genome_id", "init_seed", "epochs", "n_hidden",
    "n_network_connections", "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi",
    "evolved_test_auc", "test_auc_minus_evolved", "delta_vs_evolved_ci_lo",
    "delta_vs_evolved_ci_hi", "train_loss_before", "train_loss",
    "logit_r_vs_evolved", "logit_r_pairwise_median", "logit_r_pairwise_min",
    "cos_median_vs_evolved", "cos_share_ge_0.9_vs_evolved", "cos_min_vs_evolved",
    "cos_median_pairwise_median", "cos_share_ge_0.9_pairwise_median", "equivalent",
    "n_dead", "dead_share", "dead_units", "n_groups", "max_group_size",
    "n_nontrivial_groups", "weights_file",
    # summary / pooled rows
    "n_inits", "init_test_auc_median", "init_test_auc_min", "init_test_auc_max",
    "n_init_test_auc_ge_evolved", "n_init_test_auc_ge_evolved_ci_lo",
    "median_logit_r_vs_evolved", "median_logit_r_pairwise", "min_logit_r_pairwise",
    "median_cos_vs_evolved", "median_cos_share_ge_0.9_vs_evolved", "median_cos_pairwise",
    "n_equivalent", "init_dead_share_pooled", "evolved_dead_share", "n_topologies", "note",
]


def run_network(dry_run=False, only_seed=None):
    import ext_common as ec
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(1)
    from explaneat.db.base import db
    db.init_db()
    data = ec.load_data()
    runs = ec.read_runs()
    boot = ec.bootstrap_indices(len(data["y_test"]))
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    seeds = INIT_SEEDS[:2] if dry_run else INIT_SEEDS
    cps = [3, 6] if dry_run else [E_PRIMARY, E_SECONDARY]
    topo = sorted(runs) if only_seed is None else [only_seed]
    if dry_run:
        topo = topo[:1]
    WEIGHTS_DIR.mkdir(exist_ok=True)
    rows, summaries = [], []
    import time
    for ts in topo:
        t0 = time.time()
        g0, cfg, info = ec.load_genome(runs[ts]["genome_id"])
        units, conns0, bias0 = network_params(g0, cfg)
        p_ev_te = gs.predict_proba(g0, cfg, Xte)
        ev_test = float(roc_auc_score(yte, p_ev_te))
        assert abs(ev_test - float(runs[ts]["test_auc"])) < 1e-9, (ev_test, runs[ts]["test_auc"])
        z_ev = logits(g0, cfg, Xtr)
        m0, _ = ec.evaluate(g0, cfg, data, boot)
        acts0 = dead_units(hidden_activations(g0, cfg, Xtr), units)
        ref = {"kind": "evolved", "topology_seed": ts, "genome_id": info["genome_id"],
               "init_seed": "", "epochs": "", "n_hidden": len(units),
               "n_network_connections": len(conns0), **m0, "evolved_test_auc": ev_test,
               "test_auc_minus_evolved": 0.0, "logit_r_vs_evolved": 1.0,
               "cos_median_vs_evolved": 1.0, "cos_share_ge_0.9_vs_evolved": 1.0,
               "n_dead": len(acts0), "dead_share": len(acts0) / max(len(units), 1),
               "dead_units": acts0}
        grp = match_units(units, conns0, bias0, conns0, bias0)
        ref.update(n_groups=grp["n_groups"], max_group_size=grp["max_group_size"],
                   n_nontrivial_groups=grp["n_nontrivial_groups"])
        rows.append(ref)
        wjson = {"topology_seed": ts, "genome_id": info["genome_id"],
                 "init_convention": "numpy default_rng(seed).normal(0,1): biases of all "
                                    "genome nodes ascending, then enabled conns ascending",
                 "evolved": genome_weights_json(g0), "inits": {}}
        per_cp = {E: [] for E in cps}
        for s in seeds:
            gi = reinit_all(g0, s)
            trained = train_masked(gi, cfg, Xtr, ytr, cps, seed=s)
            wjson["inits"][str(s)] = {"initial": genome_weights_json(gi)}
            for E in cps:
                gE, tinfo = trained[E]
                wjson["inits"][str(s)][f"E{E}"] = genome_weights_json(gE)
                u, cE, bE = network_params(gE, cfg)
                assert u == units and set(cE) == set(conns0)
                m, p_te = ec.evaluate(gE, cfg, data, boot)
                d, lo, hi = ec.paired_delta(data, p_te, p_ev_te, boot)
                z = logits(gE, cfg, Xtr)
                mt = match_units(units, conns0, bias0, cE, bE)
                dead = dead_units(hidden_activations(gE, cfg, Xtr), units)
                r_ev = pearson(z, z_ev)
                row = {"kind": "init", "topology_seed": ts, "genome_id": info["genome_id"],
                       "init_seed": s, "epochs": E, "n_hidden": len(units),
                       "n_network_connections": len(cE), **m, "evolved_test_auc": ev_test,
                       "test_auc_minus_evolved": d, "delta_vs_evolved_ci_lo": lo,
                       "delta_vs_evolved_ci_hi": hi, "train_loss_before": tinfo["loss_before"],
                       "train_loss": tinfo["loss_at"], "logit_r_vs_evolved": r_ev,
                       "cos_median_vs_evolved": mt["median"],
                       "cos_share_ge_0.9_vs_evolved": mt["share_ge"],
                       "cos_min_vs_evolved": mt["min"],
                       "equivalent": bool(r_ev >= EQUIV_R and mt["median"] >= EQUIV_COS),
                       "n_dead": len(dead), "dead_share": len(dead) / max(len(units), 1),
                       "dead_units": dead, "n_groups": mt["n_groups"],
                       "max_group_size": mt["max_group_size"],
                       "n_nontrivial_groups": mt["n_nontrivial_groups"],
                       "weights_file": f"reinit_weights/network_s{ts}.json"}
                per_cp[E].append((row, z, cE, bE))
                rows.append(row)
            print(f"s{ts} init {s}: " + "  ".join(
                f"E{E} test {per_cp[E][-1][0]['test_auc']:.4f} r {per_cp[E][-1][0]['logit_r_vs_evolved']:.3f} "
                f"cos {per_cp[E][-1][0]['cos_median_vs_evolved']:.3f} dead {per_cp[E][-1][0]['n_dead']}"
                for E in cps), flush=True)
        for E in cps:
            items = per_cp[E]
            pr, pc = [], []
            pair_r = {i: [] for i in range(len(items))}
            pair_c = {i: [] for i in range(len(items))}
            pair_s = {i: [] for i in range(len(items))}
            for i, j in combinations(range(len(items)), 2):
                r = pearson(items[i][1], items[j][1])
                mt = match_units(units, items[i][2], items[i][3], items[j][2], items[j][3])
                pr.append(r)
                pc.append(mt["median"])
                for k in (i, j):
                    pair_r[k].append(r)
                    pair_c[k].append(mt["median"])
                    pair_s[k].append(mt["share_ge"])
            for k, (row, *_rest) in enumerate(items):
                row["logit_r_pairwise_median"] = _nanmed(pair_r[k])
                row["logit_r_pairwise_min"] = float(np.nanmin(pair_r[k])) if pair_r[k] else ""
                row["cos_median_pairwise_median"] = _nanmed(pair_c[k])
                row["cos_share_ge_0.9_pairwise_median"] = _nanmed(pair_s[k])
            ir = [it[0] for it in items]
            ta = np.array([r["test_auc"] for r in ir])
            n_units_tot = sum(r["n_hidden"] for r in ir)
            summaries.append({
                "kind": "summary", "topology_seed": ts, "genome_id": info["genome_id"],
                "epochs": E, "n_hidden": len(units), "n_network_connections": len(conns0),
                "n_inits": len(ir), "evolved_test_auc": ev_test,
                "init_test_auc_median": float(np.median(ta)), "init_test_auc_min": float(ta.min()),
                "init_test_auc_max": float(ta.max()),
                "n_init_test_auc_ge_evolved": int((ta >= ev_test).sum()),
                "n_init_test_auc_ge_evolved_ci_lo": int((ta >= ref["test_auc_ci_lo"]).sum()),
                "median_logit_r_vs_evolved": _nanmed([r["logit_r_vs_evolved"] for r in ir]),
                "median_logit_r_pairwise": _nanmed(pr),
                "min_logit_r_pairwise": float(np.nanmin(pr)) if pr else "",
                "median_cos_vs_evolved": _nanmed([r["cos_median_vs_evolved"] for r in ir]),
                "median_cos_share_ge_0.9_vs_evolved": _nanmed(
                    [r["cos_share_ge_0.9_vs_evolved"] for r in ir]),
                "median_cos_pairwise": _nanmed(pc),
                "n_equivalent": int(sum(r["equivalent"] for r in ir)),
                "init_dead_share_pooled": sum(r["n_dead"] for r in ir) / max(n_units_tot, 1),
                "evolved_dead_share": ref["dead_share"],
                "max_group_size": ref["max_group_size"],
                "n_nontrivial_groups": ref["n_nontrivial_groups"],
                "_pairs_r": pr, "_pairs_c": pc})
        (WEIGHTS_DIR / f"network_s{ts}.json").write_text(json.dumps(wjson, separators=(",", ":")))
        for sm in summaries[-len(cps):]:
            print(f"s{ts} E{sm['epochs']}: evolved {ev_test:.4f} init median "
                  f"{sm['init_test_auc_median']:.4f} [{sm['init_test_auc_min']:.4f},"
                  f"{sm['init_test_auc_max']:.4f}] r_ev {sm['median_logit_r_vs_evolved']:.3f} "
                  f"r_pair {sm['median_logit_r_pairwise']:.3f} cos {sm['median_cos_vs_evolved']:.3f} "
                  f"equiv {sm['n_equivalent']}/{sm['n_inits']} dead {sm['init_dead_share_pooled']:.3f} "
                  f"(evolved {sm['evolved_dead_share']:.3f}) [{time.time() - t0:.0f}s]", flush=True)
    pooled = []
    for E in cps:
        ir = [r for r in rows if r["kind"] == "init" and r["epochs"] == E]
        sm = [s for s in summaries if s["epochs"] == E]
        d = np.array([r["test_auc_minus_evolved"] for r in ir])
        pooled.append({
            "kind": "pooled", "epochs": E, "n_topologies": len(sm), "n_inits": len(ir),
            "init_test_auc_median": float(np.median([r["test_auc"] for r in ir])),
            "init_test_auc_min": float(np.min([r["test_auc"] for r in ir])),
            "init_test_auc_max": float(np.max([r["test_auc"] for r in ir])),
            "test_auc_minus_evolved": float(np.median(d)),
            "n_init_test_auc_ge_evolved": int(sum(s["n_init_test_auc_ge_evolved"] for s in sm)),
            "n_init_test_auc_ge_evolved_ci_lo": int(sum(s["n_init_test_auc_ge_evolved_ci_lo"]
                                                        for s in sm)),
            "median_logit_r_vs_evolved": _nanmed([r["logit_r_vs_evolved"] for r in ir]),
            "median_logit_r_pairwise": _nanmed([x for s in sm for x in s["_pairs_r"]]),
            "min_logit_r_pairwise": float(np.nanmin([x for s in sm for x in s["_pairs_r"]])),
            "median_cos_vs_evolved": _nanmed([r["cos_median_vs_evolved"] for r in ir]),
            "median_cos_share_ge_0.9_vs_evolved": _nanmed(
                [r["cos_share_ge_0.9_vs_evolved"] for r in ir]),
            "median_cos_pairwise": _nanmed([x for s in sm for x in s["_pairs_c"]]),
            "n_equivalent": int(sum(r["equivalent"] for r in ir)),
            "init_dead_share_pooled": sum(r["n_dead"] for r in ir) / max(
                sum(r["n_hidden"] for r in ir), 1),
            "evolved_dead_share": sum(r["n_dead"] for r in rows if r["kind"] == "evolved") / max(
                sum(r["n_hidden"] for r in rows if r["kind"] == "evolved"), 1),
            "note": "medians over all inits / all within-topology pairs; "
                    "test_auc_minus_evolved = median per-init delta"})
    for p in pooled:
        print(f"POOLED E{p['epochs']}: init test median {p['init_test_auc_median']:.4f} "
              f"delta median {p['test_auc_minus_evolved']:+.4f} >=evolved "
              f"{p['n_init_test_auc_ge_evolved']}/{p['n_inits']} r_ev "
              f"{p['median_logit_r_vs_evolved']:.3f} r_pair {p['median_logit_r_pairwise']:.3f} "
              f"cos {p['median_cos_vs_evolved']:.3f} cos_pair {p['median_cos_pairwise']:.3f} "
              f"equiv {p['n_equivalent']} dead {p['init_dead_share_pooled']:.3f} "
              f"(evolved {p['evolved_dead_share']:.3f})")
    if dry_run:
        print("DRY RUN: nothing written (weights JSON written to reinit_weights/ is removed)")
        (WEIGHTS_DIR / f"network_s{topo[0]}.json").unlink()
        return
    _write(NET_OUT, rows + summaries + pooled, NET_COLUMNS)


def _fmt(v):
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return v


def _write(path, rows, columns):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in columns})
    print(f"wrote {path} ({len(rows)} rows)")


# ====================================================================== CLI

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Study E: re-initialisation consistency")
    ap.add_argument("--stage", choices=["network", "feature"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="network: one topology only (debug)")
    args = ap.parse_args()
    if args.stage == "network":
        if args.seed is not None and not args.dry_run:
            raise SystemExit("--seed is for debugging with --dry-run only")
        run_network(dry_run=args.dry_run, only_seed=args.seed)
    else:
        run_feature(dry_run=args.dry_run)


def run_feature(dry_run=False):  # filled in below (E1)
    raise SystemExit("E1 not implemented yet")


if __name__ == "__main__":
    main()
