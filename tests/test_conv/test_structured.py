"""Structure-first explanation: mechanism concepts, interventions, grouping, splits, exactness."""

import numpy as np
import torch

from explaneat.conv.explain.engine import ExplainNet
from explaneat.conv.explain.mechanism import MECH, edit, mechanism_concepts, thin
from explaneat.conv.explain.parts import _quotient_acyclic
from explaneat.conv.explain.structure import attribution_flow, group_structures, split_mixed_roles
from explaneat.conv.explain.structured import explain_network_structured
from explaneat.conv.genome import ConvNeatConfig, InnovationTracker, minimal_genome
from explaneat.conv.mutation import mutate_add_conn, mutate_add_node

torch.set_num_threads(1)


def _ring(H=16):
    x = np.zeros((H, H), np.float32)
    x[3, 4:12] = 1; x[12, 4:12] = 1; x[3:13, 4] = 1; x[3:13, 11] = 1
    return x


def _bar(H=16):
    x = np.zeros((H, H), np.float32)
    x[2:14, 7] = 1
    return x


def test_holes_and_stem():
    X = np.stack([_ring(), _bar()])[:, None]
    M = mechanism_concepts(X)
    j = {c: i for i, c in enumerate(MECH)}
    assert M["C"][0, j["hole_any"]] == 1 and M["C"][1, j["hole_any"]] == 0
    assert M["C"][1, j["stem"]] == 1 and M["counts"]["holes"][0] == 1


def test_thinning_keeps_a_line():
    ink = np.zeros((1, 12, 12), bool)
    ink[0, 2:10, 4:7] = True
    sk = thin(ink)
    assert sk.sum() >= 5 and (sk[0].sum(1) <= 2).all()


def test_fill_removes_the_hole():
    X = np.stack([_ring()] * 3)[:, None]
    T, C, ok = edit(X, "fill", np.random.default_rng(0))
    assert ok.all()
    M = mechanism_concepts(T)
    assert M["counts"]["holes"].sum() == 0


def _genome(seed, k=10):
    rng = np.random.default_rng(seed)
    cfg = ConvNeatConfig(in_channels=1, image_hw=(16, 16), n_classes=k, max_level=2)
    tr = InnovationTracker(cfg.n_out_maps)
    g = minimal_genome(0, cfg, tr, rng)
    for _ in range(6):
        mutate_add_node(g, cfg, tr, rng)
        mutate_add_conn(g, cfg, tr, rng)
    for c in g.conns.values():
        c.kernel = (rng.standard_normal(c.kernel.shape) * 0.4).astype(np.float32)
    g.head_w = (np.eye(k) + rng.standard_normal((k, k)) * 0.1).astype(np.float32)
    return g, cfg, rng


def test_grouping_is_a_dag_and_splits_are_exact():
    g, cfg, rng = _genome(1)
    X = rng.random((40, 1, 16, 16))
    net = ExplainNet(g, cfg)
    fl = attribution_flow(net, X, 10, batch=20)
    f = net.run(X)
    pooled = {n: f.maps[n].mean(dim=(1, 2, 3)).numpy() for n in g.hidden_keys()}
    groups, _ = group_structures(g, fl["edge_abs"], pooled, set(), k_min=2, k_max=3)
    gid = {n: i for i, p in enumerate(groups) for n in p}
    assert _quotient_acyclic(g, gid)
    g2, groups2, split_of = split_mixed_roles(g, groups)
    out1, out2 = ExplainNet(g, cfg).run(X).logits, ExplainNet(g2, cfg).run(X).logits
    assert float((out1 - out2).abs().max()) < 1e-10
    gid2 = {n: i for i, p in enumerate(groups2) for n in p}
    for n in gid2:  # no member has both internal and external consumers after the split
        cons = {gid2.get(c.dst) for c in g2.enabled_conns() if c.src == n}
        assert not (gid2[n] in cons and len(cons) > 1)


def test_structured_explanation_is_exact():
    g, cfg, rng = _genome(2)
    y = np.repeat(np.arange(10), 12)
    X = np.stack([_ring() if d in (0, 6, 8, 9) else _bar() for d in y])[:, None]
    X = np.clip(X + rng.random(X.shape).astype(np.float32) * 0.3, 0, 1)
    data = {"X_test": X, "y_test": y, "X_val": X, "y_val": y}
    E = explain_network_structured(g, cfg, data, n_explain=120, n_imp=120, n_walk=3, n_flow=60)
    assert E["check"]["max_abs_logit_diff"] < 1e-9 and E["check"]["coverage"] == 1.0
    assert E["check"]["split_max_abs_logit_diff"] < 1e-9 and E["check"]["max_abs_term_sum_err"] < 1e-9
    for w in E["walk"]:
        assert np.allclose(np.sum(list(w["net_terms"].values()), axis=0), w["logits"], atol=1e-9)
    assert all(p["analysis"]["status"] in ("proven", "supported", "contested", "correlate", "nuisance", "unknown", "inactive")
               for p in E["parts"])
