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

Early stopping (EARLY_STOPPING_PLAN.md, ``--early-stopping``; additive, the
fixed-epoch outputs are untouched): the same init draws and seeds, trained on
the fit rows of val_split.json with early stopping on its validation rows
(patience 50, min_delta 1e-5, max 5000, best weights restored); E1's readout
refit on the fit rows. E2 also early-stop fine-tunes the evolved weights
(``evolved_esft`` rows, a second reference). Stop and best epochs recorded.

    uv run python experiments/churn/reinit.py --stage network --early-stopping  # -> reinit_network_es.csv
    uv run python experiments/churn/reinit.py --stage feature --early-stopping  # -> reinit_feature_es.csv
    uv run python experiments/churn/reinit.py --stage network --early-stopping --dry-run  # max 8 epochs, no output
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
    ap.add_argument("--case", default=None, help="feature --dry-run: one E1 case id")
    ap.add_argument("--early-stopping", action="store_true",
                    help="EARLY_STOPPING_PLAN.md: train on the fit rows with early stopping on "
                         "val_split.json -> reinit_network_es.csv / reinit_feature_es.csv "
                         "(the fixed-epoch outputs are not touched)")
    args = ap.parse_args()
    if args.early_stopping:
        if (args.seed is not None or args.case is not None) and not args.dry_run:
            raise SystemExit("--seed / --case are for --dry-run only")
        if args.stage == "network":
            run_network_es(dry_run=args.dry_run, only_seed=args.seed)
        else:
            run_feature_es(dry_run=args.dry_run, only_case=args.case)
        return
    if args.stage == "network":
        if args.seed is not None and not args.dry_run:
            raise SystemExit("--seed is for debugging with --dry-run only")
        run_network(dry_run=args.dry_run, only_seed=args.seed)
    else:
        if args.case is not None and not args.dry_run:
            raise SystemExit("--case is for --dry-run only")
        run_feature(dry_run=args.dry_run, only_case=args.case)


# ====================================================================== E1 runner

# D's confirmed cases (RESCUE_PLAN.md, rescue_confirm.csv include_primary):
# case: (recipient seed, recipient feature, donor seed, donor feature)
D_CASES = {
    "B1": (4, "day_eve_minutes_hinge", 6, "heavy_day_eve_usage"),
    "B2": (6, "heavy_day_eve_usage", 4, "day_eve_minutes_hinge"),
    "C1": (1, "dayUsageHinges", 0, "day_charge_no_voicemail"),
    "C2": (0, "day_charge_no_voicemail", 1, "dayUsageHinges"),
}

FEAT_COLUMNS = [
    "kind", "case", "case_source", "recipient_seed", "recipient_genome_id",
    "recipient_feature", "donor_seed", "donor_genome_id", "donor_feature", "init_seed",
    "epochs", "ft_seed", "n_grafted_units", "excluded_units", "n_grafted_connections", "graft_status",
    "refit_converged", "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi",
    "ref_test_auc", "test_auc_minus_ref", "delta_vs_ref_ci_lo", "delta_vs_ref_ci_hi",
    "base_test_auc", "ft_loss_before", "ft_loss_after",
    "contrib_r_vs_donor_orig", "contrib_r_vs_recipient_orig", "contrib_r_vs_ref_ft",
    "contrib_r_pairwise_median", "contrib_r_pairwise_min", "logit_r_vs_ref_ft",
    "cos_median_vs_donor_orig", "cos_share_ge_0.9_vs_donor_orig", "cos_min_vs_donor_orig",
    "cos_median_vs_ref_ft", "cos_share_ge_0.9_vs_ref_ft", "cos_median_pairwise_median",
    "equivalent_vs_donor_orig", "n_dead", "dead_share", "dead_units", "n_groups",
    "max_group_size", "n_nontrivial_groups", "weights_file",
    # summary / pooled
    "n_inits", "init_test_auc_median", "init_test_auc_min", "init_test_auc_max",
    "n_init_test_auc_ge_ref", "median_contrib_r_vs_donor_orig",
    "median_contrib_r_vs_recipient_orig", "median_contrib_r_vs_ref_ft",
    "median_contrib_r_pairwise", "min_contrib_r_pairwise", "median_cos_vs_donor_orig",
    "median_cos_share_ge_0.9_vs_donor_orig", "median_cos_vs_ref_ft", "median_cos_pairwise",
    "n_equivalent_vs_donor_orig", "init_dead_share_pooled", "ref_dead_share",
    "donor_orig_dead_share", "ref_contrib_r_vs_donor_orig", "ref_contrib_r_vs_recipient_orig",
    "ref_cos_median_vs_donor_orig", "n_cases", "note",
]


def e1_cases():
    """[(case id, source, rs, rfeat, ds, dfeat)]: D2's included cases from the
    committed rescue2_selection.csv, then D's B1, B2, C1, C2 not among them."""
    import csv
    import subprocess
    rel = SELECTION.relative_to(HERE.parents[1])
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)],
                             cwd=HERE.parents[1], capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(rel)],
                           cwd=HERE.parents[1], capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        raise SystemExit(f"{rel} must exist and be committed (D2's selection)")
    out, seen = [], set()
    with open(SELECTION) as f:
        for r in csv.DictReader(f):
            if r["kind"] != "pair" or r["include"] != "True":
                continue
            key = (int(r["recipient_seed"]), r["recipient_feature"], int(r["donor_seed"]),
                   r["donor_feature"])
            seen.add(key)
            out.append((r["case"], "D2", *key))
    for cid, key in D_CASES.items():
        if key not in seen:
            out.append((cid, "D (not in D2)", *key))
    return out


def load_leaf(ec, runs, seed, name, Xtr, cache):
    if seed not in cache:
        g, cfg, info = ec.load_genome(runs[seed]["genome_id"])
        ec.explained_state(g, cfg, info["operations"], Xtr)  # function preservation
        cache[seed] = (g, cfg, info)
    g, cfg, info = cache[seed]
    sub = gs.annotation_to_genome_subgraph(info["operations"], name, g, cfg)
    s, _, _ = gs.readout_contribution(g, cfg, Xtr, sub.readout_connections,
                                      sub.input_output_connections)
    return {"seed": seed, "name": name, "genome": g, "config": cfg, "info": info,
            "sub": sub, "s": s}


def graft_params(genome, sub, rec):
    """(conns, bias) of a grafted subgraph in the donor's ids: every connection
    of ``sub`` read from ``genome`` through the graft's connection map, and the
    bias of every grafted unit."""
    cmap = {tuple(json.loads(k)): tuple(v) for k, v in rec["connection_map"].items()}
    idm = {int(k): int(v) for k, v in rec["id_map"].items()}
    conns = {tuple(dk): float(genome.connections[cmap[tuple(dk)]].weight)
             for dk in sub.all_connections}
    bias = {d: float(genome.nodes[idm[d]].bias) for d in sub.hidden_nodes}
    return conns, bias


def donor_params(donor_genome, sub):
    conns = {tuple(k): float(donor_genome.connections[tuple(k)].weight)
             for k in sub.all_connections}
    bias = {d: float(donor_genome.nodes[d].bias) for d in sub.hidden_nodes}
    return conns, bias


def grafted_contribution(genome, cfg, X, rec):
    outputs = set(cfg.genome_config.output_keys)
    inputs = set(cfg.genome_config.input_keys)
    ro = [tuple(k) for k in rec["readout_connections"]]
    io = [tuple(k) for k in rec["added_connections"] if k[0] in inputs and k[1] in outputs]
    s, _, _ = gs.readout_contribution(genome, cfg, X, ro, io)
    return s


def run_feature(dry_run=False, only_case=None):
    import time

    import ext_common as ec
    from run_rescue import refit_graft

    torch.set_num_threads(1)
    from explaneat.db.base import db
    db.init_db()
    data = ec.load_data()
    runs = ec.read_runs()
    boot = ec.bootstrap_indices(len(data["y_test"]))
    Xtr, ytr = data["X_train"], data["y_train"]
    if dry_run:
        cases = ([c for c in e1_cases() if c[0] == only_case] if only_case
                 else [("B1", "D dry run", *D_CASES["B1"])])
        seeds, E = INIT_SEEDS[:2], 3
    else:
        cases = e1_cases()
        seeds, E = INIT_SEEDS, E_PRIMARY
    print(f"E1: {len(cases)} cases, {len(seeds)} inits each, E = {E}", flush=True)
    for c in cases:
        print(f"   {c[0]} ({c[1]})")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    cache, rows, summaries, wjson = {}, [], [], {}
    for cid, source, rs, rfeat, ds, dfeat in cases:
        t0 = time.time()
        R = load_leaf(ec, runs, rs, rfeat, Xtr, cache)
        A = load_leaf(ec, runs, ds, dfeat, Xtr, cache)
        cfg = R["config"]
        units = list(A["sub"].hidden_nodes)
        common = {"case": cid, "case_source": source, "recipient_seed": rs,
                  "recipient_genome_id": R["info"]["genome_id"], "recipient_feature": rfeat,
                  "donor_seed": ds, "donor_genome_id": A["info"]["genome_id"],
                  "donor_feature": dfeat, "n_grafted_units": len(units),
                  "n_grafted_connections": len(A["sub"].all_connections)}
        ko, korec = gs.feature_knockout(R["genome"], cfg, R["sub"])
        ko_keys = korec["disabled_connections"]
        m_base, _ = ec.evaluate(R["genome"], cfg, data, boot)

        def pipeline(sham_seed):
            gg, rec = gs.graft_into_knockout(ko, cfg, A["genome"], A["sub"],
                                             knocked_out_keys=ko_keys, sham_seed=sham_seed)
            refit, rinf = refit_graft(gg, rec, cfg, Xtr, ytr)
            gft, finf = gs.finetune(refit, cfg, Xtr, ytr, E, rs, mask_phantom=True,
                                    return_info=True)
            return gft, rec, rinf, finf

        try:
            g_ref, rec_ref, rinf_ref, finf_ref = pipeline(None)
        except gs.GraftCollisionError as e:
            rows.append(dict(common, kind="reference", graft_status=f"FAILED: {e}"))
            print(f"{cid}: graft failed: {e}")
            continue
        # Grafted units outside the network NeuralNeat builds for the graft (no
        # path to the output inside the graft, e.g. a unit whose only outgoing
        # edge was egress outside the annotation) compute nothing and get no
        # gradient: excluded from the unit analysis and reported.
        idm_ref = {int(k): int(v) for k, v in rec_ref["id_map"].items()}
        acts_ref = hidden_activations(g_ref, cfg, Xtr)
        excluded = sorted(d for d in units if idm_ref[d] not in acts_ref)
        units = [d for d in units if d not in excluded]
        keep = set(units) | set(cfg.genome_config.input_keys) | set(
            cfg.genome_config.output_keys)

        def restrict(cb):
            c_, b_ = cb
            return ({k: w for k, w in c_.items() if k[0] in keep and k[1] in keep},
                    {n: v for n, v in b_.items() if n in keep})

        common["excluded_units"] = excluded
        common["n_grafted_units"] = len(units)
        dc, db_ = restrict(donor_params(A["genome"], A["sub"]))
        rc, rb = restrict(graft_params(g_ref, A["sub"], rec_ref))
        m_ref, p_ref = ec.evaluate(g_ref, cfg, data, boot)
        s_ref = grafted_contribution(g_ref, cfg, Xtr, rec_ref)
        z_ref = logits(g_ref, cfg, Xtr)
        mt_ref = match_units(units, dc, db_, rc, rb)
        dead_ref = sorted(d for d in units if not np.any(acts_ref[idm_ref[d]] > 0))
        acts_donor = hidden_activations(A["genome"], cfg, Xtr)
        dead_donor = dead_units(acts_donor, units)
        ref_row = dict(common, kind="reference", init_seed="donor weights", epochs=E,
                       ft_seed=rs, graft_status="ok", refit_converged=rinf_ref["converged"],
                       **m_ref, ref_test_auc=m_ref["test_auc"], test_auc_minus_ref=0.0,
                       base_test_auc=m_base["test_auc"],
                       ft_loss_before=finf_ref["loss_before"],
                       ft_loss_after=finf_ref["loss_after_written_back"],
                       contrib_r_vs_donor_orig=pearson(s_ref, A["s"]),
                       contrib_r_vs_recipient_orig=pearson(s_ref, R["s"]),
                       contrib_r_vs_ref_ft=1.0, logit_r_vs_ref_ft=1.0,
                       cos_median_vs_donor_orig=mt_ref["median"],
                       cos_share_ge_0_9_vs_donor_orig=mt_ref["share_ge"],
                       cos_min_vs_donor_orig=mt_ref["min"],
                       n_dead=len(dead_ref), dead_share=len(dead_ref) / max(len(units), 1),
                       dead_units=dead_ref, n_groups=mt_ref["n_groups"],
                       max_group_size=mt_ref["max_group_size"],
                       n_nontrivial_groups=mt_ref["n_nontrivial_groups"],
                       weights_file="reinit_weights/feature.json",
                       note="D condition 5 (alias with donor weights, refit + ft); "
                            f"donor-original dead units {dead_donor}")
        ref_row["cos_share_ge_0.9_vs_donor_orig"] = ref_row.pop("cos_share_ge_0_9_vs_donor_orig")
        rows.append(ref_row)
        wjson[cid] = {"donor_orig": {"conn": [[*k, w] for k, w in sorted(dc.items())],
                                     "bias": {str(k): v for k, v in db_.items()}},
                      "reference_ft": {"conn": [[*k, w] for k, w in sorted(rc.items())],
                                       "bias": {str(k): v for k, v in rb.items()}},
                      "inits": {}, "ids": "donor genome ids (graft id_map in reference)",
                      "id_map": {str(k): v for k, v in idm_ref.items()}}
        items = []
        for s in seeds:
            gk, rec, rinf, finf = pipeline(s)
            kc, kb = restrict(graft_params(gk, A["sub"], rec))
            m, p_te = ec.evaluate(gk, cfg, data, boot)
            d, lo, hi = ec.paired_delta(data, p_te, p_ref, boot)
            sk = grafted_contribution(gk, cfg, Xtr, rec)
            mt_d = match_units(units, dc, db_, kc, kb)
            mt_r = match_units(units, rc, rb, kc, kb)
            idm = {int(k): int(v) for k, v in rec["id_map"].items()}
            acts = hidden_activations(gk, cfg, Xtr)
            dead = sorted(dd for dd in units if not np.any(acts[idm[dd]] > 0))
            r_d = pearson(sk, A["s"])
            row = dict(common, kind="init", init_seed=s, epochs=E, ft_seed=rs,
                       graft_status="ok", refit_converged=rinf["converged"], **m,
                       ref_test_auc=m_ref["test_auc"], test_auc_minus_ref=d,
                       delta_vs_ref_ci_lo=lo, delta_vs_ref_ci_hi=hi,
                       base_test_auc=m_base["test_auc"], ft_loss_before=finf["loss_before"],
                       ft_loss_after=finf["loss_after_written_back"],
                       contrib_r_vs_donor_orig=r_d,
                       contrib_r_vs_recipient_orig=pearson(sk, R["s"]),
                       contrib_r_vs_ref_ft=pearson(sk, s_ref),
                       logit_r_vs_ref_ft=pearson(logits(gk, cfg, Xtr), z_ref),
                       cos_median_vs_donor_orig=mt_d["median"],
                       cos_min_vs_donor_orig=mt_d["min"],
                       cos_median_vs_ref_ft=mt_r["median"],
                       equivalent_vs_donor_orig=bool(r_d >= EQUIV_R
                                                     and mt_d["median"] >= EQUIV_COS),
                       n_dead=len(dead), dead_share=len(dead) / max(len(units), 1),
                       dead_units=dead, n_groups=mt_d["n_groups"],
                       max_group_size=mt_d["max_group_size"],
                       n_nontrivial_groups=mt_d["n_nontrivial_groups"],
                       weights_file="reinit_weights/feature.json")
            row["cos_share_ge_0.9_vs_donor_orig"] = mt_d["share_ge"]
            row["cos_share_ge_0.9_vs_ref_ft"] = mt_r["share_ge"]
            wjson[cid]["inits"][str(s)] = {"conn": [[*k, w] for k, w in sorted(kc.items())],
                                           "bias": {str(k): v for k, v in kb.items()}}
            items.append((row, sk, kc, kb))
            rows.append(row)
            print(f"{cid} init {s}: test {m['test_auc']:.4f} (ref {m_ref['test_auc']:.4f}) "
                  f"r_donor {r_d:.3f} r_rec {row['contrib_r_vs_recipient_orig']:.3f} "
                  f"cos {mt_d['median']:.3f} dead {len(dead)}/{len(units)}", flush=True)
        pr, pc = [], []
        per_r = {i: [] for i in range(len(items))}
        per_c = {i: [] for i in range(len(items))}
        for i, j in combinations(range(len(items)), 2):
            r = pearson(items[i][1], items[j][1])
            c = match_units(units, items[i][2], items[i][3], items[j][2], items[j][3])["median"]
            pr.append(r)
            pc.append(c)
            for k in (i, j):
                per_r[k].append(r)
                per_c[k].append(c)
        for k, it in enumerate(items):
            it[0]["contrib_r_pairwise_median"] = _nanmed(per_r[k])
            it[0]["contrib_r_pairwise_min"] = float(np.nanmin(per_r[k])) if per_r[k] else ""
            it[0]["cos_median_pairwise_median"] = _nanmed(per_c[k])
        ir = [it[0] for it in items]
        ta = np.array([r["test_auc"] for r in ir])
        summaries.append(dict(
            common, kind="summary", epochs=E, n_inits=len(ir), ref_test_auc=m_ref["test_auc"],
            base_test_auc=m_base["test_auc"],
            init_test_auc_median=float(np.median(ta)), init_test_auc_min=float(ta.min()),
            init_test_auc_max=float(ta.max()),
            n_init_test_auc_ge_ref=int((ta >= m_ref["test_auc"]).sum()),
            median_contrib_r_vs_donor_orig=_nanmed([r["contrib_r_vs_donor_orig"] for r in ir]),
            median_contrib_r_vs_recipient_orig=_nanmed(
                [r["contrib_r_vs_recipient_orig"] for r in ir]),
            median_contrib_r_vs_ref_ft=_nanmed([r["contrib_r_vs_ref_ft"] for r in ir]),
            median_contrib_r_pairwise=_nanmed(pr),
            min_contrib_r_pairwise=float(np.nanmin(pr)) if pr else "",
            median_cos_vs_donor_orig=_nanmed([r["cos_median_vs_donor_orig"] for r in ir]),
            **{"median_cos_share_ge_0.9_vs_donor_orig": _nanmed(
                [r["cos_share_ge_0.9_vs_donor_orig"] for r in ir])},
            median_cos_vs_ref_ft=_nanmed([r["cos_median_vs_ref_ft"] for r in ir]),
            median_cos_pairwise=_nanmed(pc),
            n_equivalent_vs_donor_orig=int(sum(r["equivalent_vs_donor_orig"] for r in ir)),
            init_dead_share_pooled=sum(r["n_dead"] for r in ir) / max(len(units) * len(ir), 1),
            ref_dead_share=len(dead_ref) / max(len(units), 1),
            donor_orig_dead_share=len(dead_donor) / max(len(units), 1),
            ref_contrib_r_vs_donor_orig=ref_row["contrib_r_vs_donor_orig"],
            ref_contrib_r_vs_recipient_orig=ref_row["contrib_r_vs_recipient_orig"],
            ref_cos_median_vs_donor_orig=ref_row["cos_median_vs_donor_orig"],
            _pr=pr, _pc=pc))
        sm = summaries[-1]
        print(f"{cid}: base {m_base['test_auc']:.4f} ref {m_ref['test_auc']:.4f} init median "
              f"{sm['init_test_auc_median']:.4f} [{sm['init_test_auc_min']:.4f},"
              f"{sm['init_test_auc_max']:.4f}] r_donor {sm['median_contrib_r_vs_donor_orig']:.3f} "
              f"r_rec {sm['median_contrib_r_vs_recipient_orig']:.3f} r_pair "
              f"{sm['median_contrib_r_pairwise']:.3f} cos {sm['median_cos_vs_donor_orig']:.3f} "
              f"equiv {sm['n_equivalent_vs_donor_orig']}/{len(ir)} dead "
              f"{sm['init_dead_share_pooled']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    ir = [r for r in rows if r["kind"] == "init"]
    pooled = []
    for label, sel in (("all cases", lambda r: True),
                       ("D2 cases", lambda r: r["case_source"] == "D2")):
        rr_ = [r for r in ir if sel(r)]
        ss = [s for s in summaries if sel(s)]
        if not rr_:
            continue
        pooled.append({
            "kind": "pooled", "case": label, "epochs": E, "n_cases": len(ss), "n_inits": len(rr_),
            "init_test_auc_median": float(np.median([r["test_auc"] for r in rr_])),
            "test_auc_minus_ref": float(np.median([r["test_auc_minus_ref"] for r in rr_])),
            "n_init_test_auc_ge_ref": int(sum(s["n_init_test_auc_ge_ref"] for s in ss)),
            "median_contrib_r_vs_donor_orig": _nanmed([r["contrib_r_vs_donor_orig"] for r in rr_]),
            "median_contrib_r_vs_recipient_orig": _nanmed(
                [r["contrib_r_vs_recipient_orig"] for r in rr_]),
            "median_contrib_r_vs_ref_ft": _nanmed([r["contrib_r_vs_ref_ft"] for r in rr_]),
            "median_contrib_r_pairwise": _nanmed([x for s in ss for x in s["_pr"]]),
            "median_cos_vs_donor_orig": _nanmed([r["cos_median_vs_donor_orig"] for r in rr_]),
            "median_cos_share_ge_0.9_vs_donor_orig": _nanmed(
                [r["cos_share_ge_0.9_vs_donor_orig"] for r in rr_]),
            "median_cos_vs_ref_ft": _nanmed([r["cos_median_vs_ref_ft"] for r in rr_]),
            "median_cos_pairwise": _nanmed([x for s in ss for x in s["_pc"]]),
            "n_equivalent_vs_donor_orig": int(sum(r["equivalent_vs_donor_orig"] for r in rr_)),
            "init_dead_share_pooled": sum(r["n_dead"] for r in rr_) / max(
                sum(r["n_grafted_units"] for r in rr_), 1),
            "note": "medians over all inits / all within-case pairs; cases are not independent "
                    "(a recipient or donor feature can appear in several cases); "
                    "test_auc_minus_ref = median per-init delta"})
    for p in pooled:
        print(f"POOLED {p['case']}: n {p['n_cases']} init test median "
              f"{p['init_test_auc_median']:.4f} d_ref {p['test_auc_minus_ref']:+.4f} r_donor "
              f"{p['median_contrib_r_vs_donor_orig']:.3f} r_rec "
              f"{p['median_contrib_r_vs_recipient_orig']:.3f} r_pair "
              f"{p['median_contrib_r_pairwise']:.3f} cos {p['median_cos_vs_donor_orig']:.3f} "
              f"equiv {p['n_equivalent_vs_donor_orig']}/{p['n_inits']} dead "
              f"{p['init_dead_share_pooled']:.3f}")
    if dry_run:
        print("DRY RUN: nothing written")
        return
    (WEIGHTS_DIR / "feature.json").write_text(json.dumps(wjson, separators=(",", ":")))
    _write(FEAT_OUT, rows + summaries + pooled, FEAT_COLUMNS)


# ====================================================================== early stopping
#
# EARLY_STOPPING_PLAN.md (6f37967), study E, both levels. Strictly additive:
# the fixed-epoch runners above, their CSVs and weights files are untouched.
# Training: ``gs.finetune(..., X_fit, y_fit, seed, **ec.es_kwargs(X_val, y_val))``
# (masked, Adadelta lr 1.5, full batch BCE on the fit rows; validation BCE on the
# fixed val_split.json rows after every epoch; patience 50, min_delta 1e-5,
# max_epochs 5000; best-validation weights restored). Init draws are the fixed
# mode's (``reinit_all(seed)`` for E2, ``graft_into_knockout(sham_seed=seed)``
# for E1; seeds 1000-1009). Similarity metrics on the full train split and test
# AUC exactly as in the fixed mode; the test split is never used for training
# or stopping.

NET_OUT_ES = HERE / "reinit_network_es.csv"
FEAT_OUT_ES = HERE / "reinit_feature_es.csv"

ES_COLUMNS = [
    "early_stopping", "es_stop_epoch", "es_best_epoch", "es_best_val_bce", "es_stopped_by",
    "es_val_bce_before", "es_val_bce_after_written_back", "es_patience", "es_min_delta",
    "es_max_epochs", "es_torch_seed", "n_fit", "n_val", "fit_auc", "val_auc",
    "fit_loss_before", "fit_loss",
]
ES_SUMMARY_COLUMNS = [
    "median_es_stop_epoch", "median_es_best_epoch", "min_es_best_epoch", "max_es_best_epoch",
    "n_stopped_by_max_epochs",
]
NET_COLUMNS_ES = NET_COLUMNS[:NET_COLUMNS.index("weights_file") + 1] + [
    "evolved_esft_test_auc", "test_auc_minus_evolved_esft", "delta_vs_evolved_esft_ci_lo",
    "delta_vs_evolved_esft_ci_hi", "logit_r_vs_evolved_esft", "cos_median_vs_evolved_esft",
    "cos_share_ge_0.9_vs_evolved_esft", "equivalent_vs_evolved_esft",
] + ES_COLUMNS + NET_COLUMNS[NET_COLUMNS.index("weights_file") + 1:-2] + [
    "n_init_test_auc_ge_evolved_esft", "median_logit_r_vs_evolved_esft",
    "median_cos_vs_evolved_esft", "n_equivalent_vs_evolved_esft",
] + ES_SUMMARY_COLUMNS + ["n_topologies", "note"]
FEAT_COLUMNS_ES = (["kind", "case", "d_case"] + FEAT_COLUMNS[2:FEAT_COLUMNS.index("weights_file") + 1]
                   + ES_COLUMNS + ["refit_n_rows"]
                   + FEAT_COLUMNS[FEAT_COLUMNS.index("weights_file") + 1:-2]
                   + ["ref_es_stop_epoch", "ref_es_best_epoch"] + ES_SUMMARY_COLUMNS
                   + ["n_cases", "note"])


def load_es_split(ec, data):
    """{"X_fit", "y_fit", "X_val", "y_val"} from the committed val_split.json."""
    Xf, yf, Xv, yv, _ = ec.load_val_split(data)
    return {"X_fit": Xf, "y_fit": yf, "X_val": Xv, "y_val": yv}


def train_es(genome, config, es, seed: int, es_kwargs=None):
    """Early-stopped masked fine-tune of ``genome`` on the fit rows (torch seed
    ``seed``); returns ``gs.finetune``'s (genome, info)."""
    if es_kwargs is None:
        import ext_common as ec
        es_kwargs = ec.es_kwargs(es["X_val"], es["y_val"])
    return gs.finetune(genome, config, es["X_fit"], es["y_fit"], seed=seed, **es_kwargs)


def e2_init_es(g0, config, seed: int, es, es_kwargs=None):
    """E2 with early stopping for one init: the fixed mode's draw
    ``reinit_all(g0, seed)``, then ``train_es`` with torch seed = ``seed`` (the
    fixed mode's ``train_masked`` seed). Returns (initial genome, trained, info)."""
    gi = reinit_all(g0, seed)
    gE, info = train_es(gi, config, es, seed, es_kwargs)
    return gi, gE, info


def e1_pipeline(ko, cfg, donor_genome, donor_sub, ko_keys, sham_seed, rs, X_refit, y_refit,
                train):
    """E1's pipeline as in ``run_feature``: ``graft_into_knockout(sham_seed=...)``
    (None = donor weights), study C's readout refit on (X_refit, y_refit), then
    ``train(refit genome, rs)`` -> (genome, info). Returns
    (trained genome, graft record, refit info, train info, grafted genome before refit)."""
    from run_rescue import refit_graft
    gg, rec = gs.graft_into_knockout(ko, cfg, donor_genome, donor_sub,
                                     knocked_out_keys=ko_keys, sham_seed=sham_seed)
    refit, rinf = refit_graft(gg, rec, cfg, X_refit, y_refit)
    gft, finf = train(refit, rs)
    return gft, rec, rinf, finf, gg


def _es_fields(info, genome, cfg, es, seed):
    from sklearn.metrics import roc_auc_score
    return {"early_stopping": True, "es_stop_epoch": info["stop_epoch"],
            "es_best_epoch": info["best_epoch"], "es_best_val_bce": info["best_val_bce"],
            "es_stopped_by": info["stopped_by"], "es_val_bce_before": info["val_bce_before"],
            "es_val_bce_after_written_back": info["val_bce_after_written_back"],
            "es_patience": info["patience"], "es_min_delta": info["min_delta"],
            "es_max_epochs": info["max_epochs"], "es_torch_seed": seed,
            "n_fit": info["n_fit"], "n_val": info["n_val"],
            "fit_auc": roc_auc_score(es["y_fit"], gs.predict_proba(genome, cfg, es["X_fit"])),
            "val_auc": roc_auc_score(es["y_val"], gs.predict_proba(genome, cfg, es["X_val"])),
            "fit_loss_before": info["loss_before"], "fit_loss": info["loss_after_written_back"]}


def _es_summary(rows):
    st = [r["es_stop_epoch"] for r in rows]
    be = [r["es_best_epoch"] for r in rows]
    return {"median_es_stop_epoch": float(np.median(st)), "median_es_best_epoch": float(np.median(be)),
            "min_es_best_epoch": int(min(be)), "max_es_best_epoch": int(max(be)),
            "n_stopped_by_max_epochs": int(sum(r["es_stopped_by"] == "max_epochs" for r in rows))}


def _val_auc_plain(genome, cfg, es):
    from sklearn.metrics import roc_auc_score
    return {"fit_auc": roc_auc_score(es["y_fit"], gs.predict_proba(genome, cfg, es["X_fit"])),
            "val_auc": roc_auc_score(es["y_val"], gs.predict_proba(genome, cfg, es["X_val"])),
            "n_fit": len(es["y_fit"]), "n_val": len(es["y_val"])}


def run_network_es(dry_run=False, only_seed=None):
    """E2 with early stopping -> reinit_network_es.csv, reinit_weights/network_s<ts>_es.json.

    Per topology: an ``evolved`` row (the evolved weights as is, the reference
    for every ``*_vs_evolved`` metric, as in the fixed mode), an
    ``evolved_esft`` row (the evolved weights early-stop fine-tuned on the fit
    rows, torch seed = topology seed; compared with the evolved weights and the
    second reference for the ``*_vs_evolved_esft`` columns), 10 ``init`` rows,
    one ``summary`` row; then a ``pooled`` row. ``epochs`` = "es"."""
    import time

    import ext_common as ec
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(1)
    from explaneat.db.base import db
    db.init_db()
    data = ec.load_data()
    es = load_es_split(ec, data)
    eskw = ec.es_kwargs(es["X_val"], es["y_val"])
    if dry_run:
        eskw.update(max_epochs=8, patience=3)
    runs = ec.read_runs()
    boot = ec.bootstrap_indices(len(data["y_test"]))
    Xtr, Xte, yte = data["X_train"], data["X_test"], data["y_test"]
    seeds = INIT_SEEDS[:2] if dry_run else INIT_SEEDS
    topo = sorted(runs) if only_seed is None else [only_seed]
    if dry_run:
        topo = topo[:1]
    print(f"E2-ES: {len(topo)} topologies x {len(seeds)} inits; fit {len(es['y_fit'])} / val "
          f"{len(es['y_val'])}; patience {eskw['patience']} min_delta {eskw['min_delta']} "
          f"max {eskw['max_epochs']}", flush=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    rows, summaries = [], []
    for ts in topo:
        t0 = time.time()
        g0, cfg, info = ec.load_genome(runs[ts]["genome_id"])
        units, conns0, bias0 = network_params(g0, cfg)
        p_ev_te = gs.predict_proba(g0, cfg, Xte)
        ev_test = float(roc_auc_score(yte, p_ev_te))
        assert abs(ev_test - float(runs[ts]["test_auc"])) < 1e-9, (ev_test, runs[ts]["test_auc"])
        z_ev = logits(g0, cfg, Xtr)
        m0, _ = ec.evaluate(g0, cfg, data, boot)
        dead0 = dead_units(hidden_activations(g0, cfg, Xtr), units)
        grp = match_units(units, conns0, bias0, conns0, bias0)
        common = {"topology_seed": ts, "genome_id": info["genome_id"], "n_hidden": len(units),
                  "evolved_test_auc": ev_test, "weights_file": f"reinit_weights/network_s{ts}_es.json"}
        ref = dict(common, kind="evolved", init_seed="", epochs="",
                   n_network_connections=len(conns0), **m0, test_auc_minus_evolved=0.0,
                   logit_r_vs_evolved=1.0, cos_median_vs_evolved=1.0,
                   n_dead=len(dead0), dead_share=len(dead0) / max(len(units), 1),
                   dead_units=dead0, n_groups=grp["n_groups"],
                   max_group_size=grp["max_group_size"],
                   n_nontrivial_groups=grp["n_nontrivial_groups"], early_stopping=False,
                   **_val_auc_plain(g0, cfg, es), note="evolved weights as is (reference)")
        ref["cos_share_ge_0.9_vs_evolved"] = 1.0
        rows.append(ref)

        def compare(gE, finf, kind, seed, p_ref2=None, z_ref2=None, c_ref2=None):
            u, cE, bE = network_params(gE, cfg)
            assert u == units and set(cE) == set(conns0)
            m, p_te = ec.evaluate(gE, cfg, data, boot)
            d, lo, hi = ec.paired_delta(data, p_te, p_ev_te, boot)
            z = logits(gE, cfg, Xtr)
            mt = match_units(units, conns0, bias0, cE, bE)
            dead = dead_units(hidden_activations(gE, cfg, Xtr), units)
            r_ev = pearson(z, z_ev)
            row = dict(common, kind=kind, init_seed=seed, epochs="es",
                       n_network_connections=len(cE), **m, test_auc_minus_evolved=d,
                       delta_vs_evolved_ci_lo=lo, delta_vs_evolved_ci_hi=hi,
                       train_loss_before="", train_loss="", logit_r_vs_evolved=r_ev,
                       cos_median_vs_evolved=mt["median"], cos_min_vs_evolved=mt["min"],
                       equivalent=bool(r_ev >= EQUIV_R and mt["median"] >= EQUIV_COS),
                       n_dead=len(dead), dead_share=len(dead) / max(len(units), 1),
                       dead_units=dead, n_groups=mt["n_groups"],
                       max_group_size=mt["max_group_size"],
                       n_nontrivial_groups=mt["n_nontrivial_groups"],
                       **_es_fields(finf, gE, cfg, es, finf["seed"]))
            row["cos_share_ge_0.9_vs_evolved"] = mt["share_ge"]
            if p_ref2 is not None:
                d2, lo2, hi2 = ec.paired_delta(data, p_te, p_ref2, boot)
                m2 = match_units(units, c_ref2[0], c_ref2[1], cE, bE)
                r2 = pearson(z, z_ref2)
                row.update(test_auc_minus_evolved_esft=d2, delta_vs_evolved_esft_ci_lo=lo2,
                           delta_vs_evolved_esft_ci_hi=hi2, logit_r_vs_evolved_esft=r2,
                           cos_median_vs_evolved_esft=m2["median"],
                           equivalent_vs_evolved_esft=bool(r2 >= EQUIV_R
                                                           and m2["median"] >= EQUIV_COS))
                row["cos_share_ge_0.9_vs_evolved_esft"] = m2["share_ge"]
            return row, p_te, z, cE, bE

        g_eft, f_eft = train_es(g0, cfg, es, ts, eskw)
        eft, p_eft_te, z_eft, c_eft, b_eft = compare(g_eft, f_eft, "evolved_esft", "")
        eft["note"] = (f"evolved weights + early-stopped masked fine-tune on fit rows "
                       f"(torch seed {ts} = topology seed)")
        eft.update(evolved_esft_test_auc=eft["test_auc"], test_auc_minus_evolved_esft=0.0,
                   logit_r_vs_evolved_esft=1.0, cos_median_vs_evolved_esft=1.0)
        rows.append(eft)
        print(f"s{ts} evolved {ev_test:.4f} -> esft {eft['test_auc']:.4f} (stop "
              f"{eft['es_stop_epoch']} best {eft['es_best_epoch']}) r {eft['logit_r_vs_evolved']:.3f} "
              f"cos {eft['cos_median_vs_evolved']:.3f}", flush=True)
        wjson = {"topology_seed": ts, "genome_id": info["genome_id"],
                 "plan": "EARLY_STOPPING_PLAN.md (6f37967), study E2",
                 "init_convention": "numpy default_rng(seed).normal(0,1): biases of all "
                                    "genome nodes ascending, then enabled conns ascending "
                                    "(same draw as reinit_network.csv)",
                 "evolved": genome_weights_json(g0),
                 "evolved_esft": dict(genome_weights_json(g_eft),
                                      stop_epoch=f_eft["stop_epoch"],
                                      best_epoch=f_eft["best_epoch"]),
                 "inits": {}}
        items = []
        for s in seeds:
            gi, gE, finf = e2_init_es(g0, cfg, s, es, eskw)
            row, _, z, cE, bE = compare(gE, finf, "init", s, p_eft_te, z_eft, (c_eft, b_eft))
            row["evolved_esft_test_auc"] = eft["test_auc"]
            wjson["inits"][str(s)] = {"initial": genome_weights_json(gi),
                                      "es": dict(genome_weights_json(gE),
                                                 stop_epoch=finf["stop_epoch"],
                                                 best_epoch=finf["best_epoch"])}
            items.append((row, z, cE, bE))
            rows.append(row)
            print(f"s{ts} init {s}: test {row['test_auc']:.4f} r {row['logit_r_vs_evolved']:.3f} "
                  f"cos {row['cos_median_vs_evolved']:.3f} dead {row['n_dead']} stop "
                  f"{row['es_stop_epoch']} best {row['es_best_epoch']} ({row['es_stopped_by']})",
                  flush=True)
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
        summaries.append(dict(
            common, kind="summary", epochs="es", n_network_connections=len(conns0),
            n_inits=len(ir), init_test_auc_median=float(np.median(ta)),
            init_test_auc_min=float(ta.min()), init_test_auc_max=float(ta.max()),
            n_init_test_auc_ge_evolved=int((ta >= ev_test).sum()),
            n_init_test_auc_ge_evolved_ci_lo=int((ta >= ref["test_auc_ci_lo"]).sum()),
            median_logit_r_vs_evolved=_nanmed([r["logit_r_vs_evolved"] for r in ir]),
            median_logit_r_pairwise=_nanmed(pr),
            min_logit_r_pairwise=float(np.nanmin(pr)) if pr else "",
            median_cos_vs_evolved=_nanmed([r["cos_median_vs_evolved"] for r in ir]),
            median_cos_pairwise=_nanmed(pc),
            n_equivalent=int(sum(r["equivalent"] for r in ir)),
            init_dead_share_pooled=sum(r["n_dead"] for r in ir) / max(len(units) * len(ir), 1),
            evolved_dead_share=ref["dead_share"], max_group_size=ref["max_group_size"],
            n_nontrivial_groups=ref["n_nontrivial_groups"],
            evolved_esft_test_auc=eft["test_auc"],
            n_init_test_auc_ge_evolved_esft=int((ta >= eft["test_auc"]).sum()),
            median_logit_r_vs_evolved_esft=_nanmed([r["logit_r_vs_evolved_esft"] for r in ir]),
            median_cos_vs_evolved_esft=_nanmed([r["cos_median_vs_evolved_esft"] for r in ir]),
            n_equivalent_vs_evolved_esft=int(sum(r["equivalent_vs_evolved_esft"] for r in ir)),
            **_es_summary(ir), _pairs_r=pr, _pairs_c=pc,
            note=f"evolved_esft stop {eft['es_stop_epoch']} best {eft['es_best_epoch']}"))
        summaries[-1]["median_cos_share_ge_0.9_vs_evolved"] = _nanmed(
            [r["cos_share_ge_0.9_vs_evolved"] for r in ir])
        if not dry_run:
            (WEIGHTS_DIR / f"network_s{ts}_es.json").write_text(
                json.dumps(wjson, separators=(",", ":")))
        sm = summaries[-1]
        print(f"s{ts} ES: evolved {ev_test:.4f} esft {eft['test_auc']:.4f} init median "
              f"{sm['init_test_auc_median']:.4f} [{sm['init_test_auc_min']:.4f},"
              f"{sm['init_test_auc_max']:.4f}] r_ev {sm['median_logit_r_vs_evolved']:.3f} "
              f"r_pair {sm['median_logit_r_pairwise']:.3f} cos {sm['median_cos_vs_evolved']:.3f} "
              f"equiv {sm['n_equivalent']}/{sm['n_inits']} dead {sm['init_dead_share_pooled']:.3f} "
              f"stop {sm['median_es_stop_epoch']} best {sm['median_es_best_epoch']} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    ir = [r for r in rows if r["kind"] == "init"]
    ev = [r for r in rows if r["kind"] == "evolved"]
    efts = [r for r in rows if r["kind"] == "evolved_esft"]
    d = np.array([r["test_auc_minus_evolved"] for r in ir])
    pooled = {
        "kind": "pooled", "epochs": "es", "n_topologies": len(summaries), "n_inits": len(ir),
        "init_test_auc_median": float(np.median([r["test_auc"] for r in ir])),
        "init_test_auc_min": float(np.min([r["test_auc"] for r in ir])),
        "init_test_auc_max": float(np.max([r["test_auc"] for r in ir])),
        "test_auc_minus_evolved": float(np.median(d)),
        "test_auc_minus_evolved_esft": float(np.median(
            [r["test_auc_minus_evolved_esft"] for r in ir])),
        "n_init_test_auc_ge_evolved": int(sum(s["n_init_test_auc_ge_evolved"] for s in summaries)),
        "n_init_test_auc_ge_evolved_ci_lo": int(sum(s["n_init_test_auc_ge_evolved_ci_lo"]
                                                    for s in summaries)),
        "n_init_test_auc_ge_evolved_esft": int(sum(s["n_init_test_auc_ge_evolved_esft"]
                                                   for s in summaries)),
        "median_logit_r_vs_evolved": _nanmed([r["logit_r_vs_evolved"] for r in ir]),
        "median_logit_r_pairwise": _nanmed([x for s in summaries for x in s["_pairs_r"]]),
        "min_logit_r_pairwise": float(np.nanmin([x for s in summaries for x in s["_pairs_r"]])),
        "median_cos_vs_evolved": _nanmed([r["cos_median_vs_evolved"] for r in ir]),
        "median_cos_share_ge_0.9_vs_evolved": _nanmed(
            [r["cos_share_ge_0.9_vs_evolved"] for r in ir]),
        "median_cos_pairwise": _nanmed([x for s in summaries for x in s["_pairs_c"]]),
        "n_equivalent": int(sum(r["equivalent"] for r in ir)),
        "median_logit_r_vs_evolved_esft": _nanmed([r["logit_r_vs_evolved_esft"] for r in ir]),
        "median_cos_vs_evolved_esft": _nanmed([r["cos_median_vs_evolved_esft"] for r in ir]),
        "n_equivalent_vs_evolved_esft": int(sum(r["equivalent_vs_evolved_esft"] for r in ir)),
        "init_dead_share_pooled": sum(r["n_dead"] for r in ir) / max(
            sum(r["n_hidden"] for r in ir), 1),
        "evolved_dead_share": sum(r["n_dead"] for r in ev) / max(sum(r["n_hidden"] for r in ev), 1),
        **_es_summary(ir),
        "note": "medians over all inits / all within-topology pairs; test_auc_minus_evolved(_esft) "
                "= median per-init delta; evolved_esft over topologies: median test "
                f"{np.median([r['test_auc'] for r in efts]):.4f}, median delta vs evolved "
                f"{np.median([r['test_auc_minus_evolved'] for r in efts]):+.4f}, median logit r "
                f"{_nanmed([r['logit_r_vs_evolved'] for r in efts]):.4f}, median stop "
                f"{np.median([r['es_stop_epoch'] for r in efts])}, best "
                f"{np.median([r['es_best_epoch'] for r in efts])}"}
    p = pooled
    print(f"POOLED ES: init test median {p['init_test_auc_median']:.4f} delta median "
          f"{p['test_auc_minus_evolved']:+.4f} >=evolved {p['n_init_test_auc_ge_evolved']}/"
          f"{p['n_inits']} r_ev {p['median_logit_r_vs_evolved']:.3f} r_pair "
          f"{p['median_logit_r_pairwise']:.3f} cos {p['median_cos_vs_evolved']:.3f} cos_pair "
          f"{p['median_cos_pairwise']:.3f} equiv {p['n_equivalent']} dead "
          f"{p['init_dead_share_pooled']:.3f} (evolved {p['evolved_dead_share']:.3f}) stop "
          f"{p['median_es_stop_epoch']} best {p['median_es_best_epoch']}; {p['note']}")
    if dry_run:
        print("DRY RUN: nothing written")
        return
    _write(NET_OUT_ES, rows + summaries + [pooled], NET_COLUMNS_ES)


def run_feature_es(dry_run=False, only_case=None):
    """E1 with early stopping -> reinit_feature_es.csv, reinit_weights/feature_es.json.

    The fixed mode's cases and draws; the readout refit on the fit rows only;
    the masked fine-tune early-stopped (torch seed = recipient seed, as the fixed
    mode). Reference: alias with donor weights + refit on fit rows + ES ft."""
    import time

    import ext_common as ec

    torch.set_num_threads(1)
    from explaneat.db.base import db
    db.init_db()
    data = ec.load_data()
    es = load_es_split(ec, data)
    eskw = ec.es_kwargs(es["X_val"], es["y_val"])
    runs = ec.read_runs()
    boot = ec.bootstrap_indices(len(data["y_test"]))
    Xtr = data["X_train"]
    if dry_run:
        cases = ([c for c in e1_cases() if c[0] == only_case] if only_case
                 else [("B1", "D dry run", *D_CASES["B1"])])
        seeds = INIT_SEEDS[:2]
        eskw.update(max_epochs=8, patience=3)
    else:
        cases = e1_cases()
        seeds = INIT_SEEDS
    d_of = {key: cid for cid, key in D_CASES.items()}
    print(f"E1-ES: {len(cases)} cases, {len(seeds)} inits each; fit {len(es['y_fit'])} / val "
          f"{len(es['y_val'])}; patience {eskw['patience']} max {eskw['max_epochs']}", flush=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    def train(g, rs):
        return train_es(g, cfg_holder[0], es, rs, eskw)

    cfg_holder = [None]
    cache, rows, summaries, wjson = {}, [], [], {}
    for cid, source, rs, rfeat, ds, dfeat in cases:
        t0 = time.time()
        R = load_leaf(ec, runs, rs, rfeat, Xtr, cache)
        A = load_leaf(ec, runs, ds, dfeat, Xtr, cache)
        cfg = R["config"]
        cfg_holder[0] = cfg
        units = list(A["sub"].hidden_nodes)
        common = {"case": cid, "d_case": d_of.get((rs, rfeat, ds, dfeat), ""),
                  "case_source": source, "recipient_seed": rs,
                  "recipient_genome_id": R["info"]["genome_id"], "recipient_feature": rfeat,
                  "donor_seed": ds, "donor_genome_id": A["info"]["genome_id"],
                  "donor_feature": dfeat, "n_grafted_units": len(units),
                  "n_grafted_connections": len(A["sub"].all_connections),
                  "epochs": "es", "ft_seed": rs, "refit_n_rows": len(es["y_fit"])}
        ko, korec = gs.feature_knockout(R["genome"], cfg, R["sub"])
        ko_keys = korec["disabled_connections"]
        m_base, _ = ec.evaluate(R["genome"], cfg, data, boot)

        def pipeline(sham_seed):
            return e1_pipeline(ko, cfg, A["genome"], A["sub"], ko_keys, sham_seed, rs,
                               es["X_fit"], es["y_fit"], train)

        try:
            g_ref, rec_ref, rinf_ref, finf_ref, _ = pipeline(None)
        except gs.GraftCollisionError as e:
            rows.append(dict(common, kind="reference", graft_status=f"FAILED: {e}"))
            print(f"{cid}: graft failed: {e}")
            continue
        idm_ref = {int(k): int(v) for k, v in rec_ref["id_map"].items()}
        acts_ref = hidden_activations(g_ref, cfg, Xtr)
        excluded = sorted(d for d in units if idm_ref[d] not in acts_ref)
        units = [d for d in units if d not in excluded]
        keep = set(units) | set(cfg.genome_config.input_keys) | set(
            cfg.genome_config.output_keys)

        def restrict(cb):
            c_, b_ = cb
            return ({k: w for k, w in c_.items() if k[0] in keep and k[1] in keep},
                    {n: v for n, v in b_.items() if n in keep})

        common["excluded_units"] = excluded
        common["n_grafted_units"] = len(units)
        dc, db_ = restrict(donor_params(A["genome"], A["sub"]))
        rc, rb = restrict(graft_params(g_ref, A["sub"], rec_ref))
        m_ref, p_ref = ec.evaluate(g_ref, cfg, data, boot)
        s_ref = grafted_contribution(g_ref, cfg, Xtr, rec_ref)
        z_ref = logits(g_ref, cfg, Xtr)
        mt_ref = match_units(units, dc, db_, rc, rb)
        dead_ref = sorted(d for d in units if not np.any(acts_ref[idm_ref[d]] > 0))
        dead_donor = dead_units(hidden_activations(A["genome"], cfg, Xtr), units)
        ref_row = dict(common, kind="reference", init_seed="donor weights",
                       graft_status="ok", refit_converged=rinf_ref["converged"],
                       **m_ref, ref_test_auc=m_ref["test_auc"], test_auc_minus_ref=0.0,
                       base_test_auc=m_base["test_auc"],
                       ft_loss_before=finf_ref["loss_before"],
                       ft_loss_after=finf_ref["loss_after_written_back"],
                       contrib_r_vs_donor_orig=pearson(s_ref, A["s"]),
                       contrib_r_vs_recipient_orig=pearson(s_ref, R["s"]),
                       contrib_r_vs_ref_ft=1.0, logit_r_vs_ref_ft=1.0,
                       cos_median_vs_donor_orig=mt_ref["median"],
                       cos_min_vs_donor_orig=mt_ref["min"],
                       n_dead=len(dead_ref), dead_share=len(dead_ref) / max(len(units), 1),
                       dead_units=dead_ref, n_groups=mt_ref["n_groups"],
                       max_group_size=mt_ref["max_group_size"],
                       n_nontrivial_groups=mt_ref["n_nontrivial_groups"],
                       weights_file="reinit_weights/feature_es.json",
                       **_es_fields(finf_ref, g_ref, cfg, es, rs),
                       note="alias with donor weights, refit on fit rows + early-stopped ft; "
                            f"donor-original dead units {dead_donor}")
        ref_row["cos_share_ge_0.9_vs_donor_orig"] = mt_ref["share_ge"]
        rows.append(ref_row)
        wjson[cid] = {"donor_orig": {"conn": [[*k, w] for k, w in sorted(dc.items())],
                                     "bias": {str(k): v for k, v in db_.items()}},
                      "reference_ft": {"conn": [[*k, w] for k, w in sorted(rc.items())],
                                       "bias": {str(k): v for k, v in rb.items()},
                                       "stop_epoch": finf_ref["stop_epoch"],
                                       "best_epoch": finf_ref["best_epoch"]},
                      "inits": {}, "ids": "donor genome ids (graft id_map in reference)",
                      "id_map": {str(k): v for k, v in idm_ref.items()}}
        items = []
        for s in seeds:
            gk, rec, rinf, finf, _ = pipeline(s)
            kc, kb = restrict(graft_params(gk, A["sub"], rec))
            m, p_te = ec.evaluate(gk, cfg, data, boot)
            d, lo, hi = ec.paired_delta(data, p_te, p_ref, boot)
            sk = grafted_contribution(gk, cfg, Xtr, rec)
            mt_d = match_units(units, dc, db_, kc, kb)
            mt_r = match_units(units, rc, rb, kc, kb)
            idm = {int(k): int(v) for k, v in rec["id_map"].items()}
            acts = hidden_activations(gk, cfg, Xtr)
            dead = sorted(dd for dd in units if not np.any(acts[idm[dd]] > 0))
            r_d = pearson(sk, A["s"])
            row = dict(common, kind="init", init_seed=s, graft_status="ok",
                       refit_converged=rinf["converged"], **m,
                       ref_test_auc=m_ref["test_auc"], test_auc_minus_ref=d,
                       delta_vs_ref_ci_lo=lo, delta_vs_ref_ci_hi=hi,
                       base_test_auc=m_base["test_auc"], ft_loss_before=finf["loss_before"],
                       ft_loss_after=finf["loss_after_written_back"],
                       contrib_r_vs_donor_orig=r_d,
                       contrib_r_vs_recipient_orig=pearson(sk, R["s"]),
                       contrib_r_vs_ref_ft=pearson(sk, s_ref),
                       logit_r_vs_ref_ft=pearson(logits(gk, cfg, Xtr), z_ref),
                       cos_median_vs_donor_orig=mt_d["median"],
                       cos_min_vs_donor_orig=mt_d["min"],
                       cos_median_vs_ref_ft=mt_r["median"],
                       equivalent_vs_donor_orig=bool(r_d >= EQUIV_R
                                                     and mt_d["median"] >= EQUIV_COS),
                       n_dead=len(dead), dead_share=len(dead) / max(len(units), 1),
                       dead_units=dead, n_groups=mt_d["n_groups"],
                       max_group_size=mt_d["max_group_size"],
                       n_nontrivial_groups=mt_d["n_nontrivial_groups"],
                       weights_file="reinit_weights/feature_es.json",
                       **_es_fields(finf, gk, cfg, es, rs))
            row["cos_share_ge_0.9_vs_donor_orig"] = mt_d["share_ge"]
            row["cos_share_ge_0.9_vs_ref_ft"] = mt_r["share_ge"]
            wjson[cid]["inits"][str(s)] = {"conn": [[*k, w] for k, w in sorted(kc.items())],
                                           "bias": {str(k): v for k, v in kb.items()},
                                           "stop_epoch": finf["stop_epoch"],
                                           "best_epoch": finf["best_epoch"]}
            items.append((row, sk, kc, kb))
            rows.append(row)
            print(f"{cid} init {s}: test {m['test_auc']:.4f} (ref {m_ref['test_auc']:.4f}) "
                  f"r_donor {r_d:.3f} r_rec {row['contrib_r_vs_recipient_orig']:.3f} "
                  f"cos {mt_d['median']:.3f} dead {len(dead)}/{len(units)} stop "
                  f"{finf['stop_epoch']} best {finf['best_epoch']}", flush=True)
        pr, pc = [], []
        per_r = {i: [] for i in range(len(items))}
        per_c = {i: [] for i in range(len(items))}
        for i, j in combinations(range(len(items)), 2):
            r = pearson(items[i][1], items[j][1])
            c = match_units(units, items[i][2], items[i][3], items[j][2], items[j][3])["median"]
            pr.append(r)
            pc.append(c)
            for k in (i, j):
                per_r[k].append(r)
                per_c[k].append(c)
        for k, it in enumerate(items):
            it[0]["contrib_r_pairwise_median"] = _nanmed(per_r[k])
            it[0]["contrib_r_pairwise_min"] = float(np.nanmin(per_r[k])) if per_r[k] else ""
            it[0]["cos_median_pairwise_median"] = _nanmed(per_c[k])
        ir = [it[0] for it in items]
        ta = np.array([r["test_auc"] for r in ir])
        summaries.append(dict(
            common, kind="summary", n_inits=len(ir), ref_test_auc=m_ref["test_auc"],
            base_test_auc=m_base["test_auc"],
            init_test_auc_median=float(np.median(ta)), init_test_auc_min=float(ta.min()),
            init_test_auc_max=float(ta.max()),
            n_init_test_auc_ge_ref=int((ta >= m_ref["test_auc"]).sum()),
            median_contrib_r_vs_donor_orig=_nanmed([r["contrib_r_vs_donor_orig"] for r in ir]),
            median_contrib_r_vs_recipient_orig=_nanmed(
                [r["contrib_r_vs_recipient_orig"] for r in ir]),
            median_contrib_r_vs_ref_ft=_nanmed([r["contrib_r_vs_ref_ft"] for r in ir]),
            median_contrib_r_pairwise=_nanmed(pr),
            min_contrib_r_pairwise=float(np.nanmin(pr)) if pr else "",
            median_cos_vs_donor_orig=_nanmed([r["cos_median_vs_donor_orig"] for r in ir]),
            **{"median_cos_share_ge_0.9_vs_donor_orig": _nanmed(
                [r["cos_share_ge_0.9_vs_donor_orig"] for r in ir])},
            median_cos_vs_ref_ft=_nanmed([r["cos_median_vs_ref_ft"] for r in ir]),
            median_cos_pairwise=_nanmed(pc),
            n_equivalent_vs_donor_orig=int(sum(r["equivalent_vs_donor_orig"] for r in ir)),
            init_dead_share_pooled=sum(r["n_dead"] for r in ir) / max(len(units) * len(ir), 1),
            ref_dead_share=len(dead_ref) / max(len(units), 1),
            donor_orig_dead_share=len(dead_donor) / max(len(units), 1),
            ref_contrib_r_vs_donor_orig=ref_row["contrib_r_vs_donor_orig"],
            ref_contrib_r_vs_recipient_orig=ref_row["contrib_r_vs_recipient_orig"],
            ref_cos_median_vs_donor_orig=ref_row["cos_median_vs_donor_orig"],
            ref_es_stop_epoch=ref_row["es_stop_epoch"], ref_es_best_epoch=ref_row["es_best_epoch"],
            **_es_summary(ir), _pr=pr, _pc=pc))
        sm = summaries[-1]
        print(f"{cid}: base {m_base['test_auc']:.4f} ref {m_ref['test_auc']:.4f} (stop "
              f"{ref_row['es_stop_epoch']} best {ref_row['es_best_epoch']}) init median "
              f"{sm['init_test_auc_median']:.4f} [{sm['init_test_auc_min']:.4f},"
              f"{sm['init_test_auc_max']:.4f}] r_donor {sm['median_contrib_r_vs_donor_orig']:.3f} "
              f"r_rec {sm['median_contrib_r_vs_recipient_orig']:.3f} r_pair "
              f"{sm['median_contrib_r_pairwise']:.3f} cos {sm['median_cos_vs_donor_orig']:.3f} "
              f"equiv {sm['n_equivalent_vs_donor_orig']}/{len(ir)} dead "
              f"{sm['init_dead_share_pooled']:.3f} stop {sm['median_es_stop_epoch']} best "
              f"{sm['median_es_best_epoch']} [{time.time() - t0:.0f}s]", flush=True)
    ir = [r for r in rows if r["kind"] == "init"]
    refs = [r for r in rows if r["kind"] == "reference" and r.get("graft_status") == "ok"]
    pooled = []
    for label, sel in (("all cases", lambda r: True),
                       ("D2 cases", lambda r: r["case_source"] == "D2"),
                       ("D cases (B1 B2 C1 C2)", lambda r: r.get("d_case", "") != "")):
        rr_ = [r for r in ir if sel(r)]
        ss = [s for s in summaries if sel(s)]
        rf = [r for r in refs if sel(r)]
        if not rr_:
            continue
        pooled.append({
            "kind": "pooled", "case": label, "epochs": "es", "n_cases": len(ss),
            "n_inits": len(rr_),
            "init_test_auc_median": float(np.median([r["test_auc"] for r in rr_])),
            "test_auc_minus_ref": float(np.median([r["test_auc_minus_ref"] for r in rr_])),
            "n_init_test_auc_ge_ref": int(sum(s["n_init_test_auc_ge_ref"] for s in ss)),
            "median_contrib_r_vs_donor_orig": _nanmed([r["contrib_r_vs_donor_orig"] for r in rr_]),
            "median_contrib_r_vs_recipient_orig": _nanmed(
                [r["contrib_r_vs_recipient_orig"] for r in rr_]),
            "median_contrib_r_vs_ref_ft": _nanmed([r["contrib_r_vs_ref_ft"] for r in rr_]),
            "median_contrib_r_pairwise": _nanmed([x for s in ss for x in s["_pr"]]),
            "median_cos_vs_donor_orig": _nanmed([r["cos_median_vs_donor_orig"] for r in rr_]),
            "median_cos_share_ge_0.9_vs_donor_orig": _nanmed(
                [r["cos_share_ge_0.9_vs_donor_orig"] for r in rr_]),
            "median_cos_vs_ref_ft": _nanmed([r["cos_median_vs_ref_ft"] for r in rr_]),
            "median_cos_pairwise": _nanmed([x for s in ss for x in s["_pc"]]),
            "n_equivalent_vs_donor_orig": int(sum(r["equivalent_vs_donor_orig"] for r in rr_)),
            "init_dead_share_pooled": sum(r["n_dead"] for r in rr_) / max(
                sum(r["n_grafted_units"] for r in rr_), 1),
            "ref_es_stop_epoch": float(np.median([r["es_stop_epoch"] for r in rf])),
            "ref_es_best_epoch": float(np.median([r["es_best_epoch"] for r in rf])),
            **_es_summary(rr_),
            "note": "medians over all inits / all within-case pairs; cases are not independent "
                    "(a recipient or donor feature can appear in several cases); "
                    "test_auc_minus_ref = median per-init delta; ref_es_* = median over "
                    "references"})
    for p in pooled:
        print(f"POOLED {p['case']}: n {p['n_cases']} init test median "
              f"{p['init_test_auc_median']:.4f} d_ref {p['test_auc_minus_ref']:+.4f} r_donor "
              f"{p['median_contrib_r_vs_donor_orig']:.3f} r_rec "
              f"{p['median_contrib_r_vs_recipient_orig']:.3f} r_pair "
              f"{p['median_contrib_r_pairwise']:.3f} cos {p['median_cos_vs_donor_orig']:.3f} "
              f"equiv {p['n_equivalent_vs_donor_orig']}/{p['n_inits']} dead "
              f"{p['init_dead_share_pooled']:.3f} stop {p['median_es_stop_epoch']} best "
              f"{p['median_es_best_epoch']}")
    if dry_run:
        print("DRY RUN: nothing written")
        return
    (WEIGHTS_DIR / "feature_es.json").write_text(json.dumps(wjson, separators=(",", ":")))
    _write(FEAT_OUT_ES, rows + summaries + pooled, FEAT_COLUMNS_ES)


if __name__ == "__main__":
    main()
