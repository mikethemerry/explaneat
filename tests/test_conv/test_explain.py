"""ConvNEAT explanation techniques (explaneat/conv/explain)."""

import numpy as np
import pytest
import torch

from explaneat.conv.data import make_synthetic
from explaneat.conv.explain import formula as FM
from explaneat.conv.explain import importance as IM
from explaneat.conv.explain import parts as PT
from explaneat.conv.explain import receptive as RF
from explaneat.conv.explain.engine import ExplainNet, functional_nodes
from explaneat.conv.explain.pipeline import explain_network
from explaneat.conv.genome import ConnGene, ConvGenome, ConvNeatConfig, InnovationTracker, NodeGene, minimal_genome
from explaneat.conv.mutation import delta_kernel, mutate_add_node, random_genome_by_mutation
from explaneat.conv.network import NaiveConvNet

torch.set_num_threads(1)

CONFIGS = [
    ConvNeatConfig(in_channels=1, image_hw=(16, 16), n_classes=1, max_level=2),
    ConvNeatConfig(in_channels=1, image_hw=(8, 8), n_classes=1, max_level=2),
]


def _random_genomes(n, seed, cfg):
    """Random genomes; every other one keeps zero biases and its add-node delta
    kernels, so the simplification rules fire inside parts."""
    tr = InnovationTracker(cfg.n_out_maps)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = random_genome_by_mutation(i, cfg, tr, rng, n_steps=int(rng.integers(5, 40)))
        if i % 2 == 0:
            for nd in g.nodes.values():
                if nd.kind != "input":
                    nd.bias = float(rng.normal() * 0.2)
        out.append(g)
    return out


@pytest.mark.parametrize("ci", [0, 1])
def test_engine_matches_naive(ci):
    cfg = CONFIGS[ci]
    x = torch.rand(5, 1, *cfg.image_hw, dtype=torch.float64)
    for g in _random_genomes(30, 100 + ci, cfg):
        ref, maps = NaiveConvNet(g, cfg, dtype=torch.float64)(x, return_maps=True)
        f = ExplainNet(g, cfg).forward(x, keep_contrib=True)
        assert (f.logits - ref).abs().max() < 1e-12
        for k in maps:
            assert (f.maps[k] - maps[k]).abs().max() < 1e-12


@pytest.mark.parametrize("ci", [0, 1])
def test_node_formulas_reproduce_engine(ci):
    """Every node's simplified operator formula, evaluated with the actual kernels,
    reproduces the engine's map to 1e-6 (random genomes, mixed levels and kernels)."""
    cfg = CONFIGS[ci]
    x = torch.rand(6, 1, *cfg.image_hw, dtype=torch.float64)
    n_checked = 0
    for g in _random_genomes(60, 200 + ci, cfg):
        f = ExplainNet(g, cfg).forward(x)
        ctx = FM.FormulaContext(g, cfg)
        for k, nd in g.nodes.items():
            if nd.kind == "input":
                continue
            e, _ = FM.node_formula(ctx, k)
            v = FM.evaluate(e, f.maps, cfg)
            assert (v - f.maps[k]).abs().max() < 1e-6
            assert FM.to_latex(e) and FM.to_text(e)
            n_checked += 1
    assert n_checked > 100


@pytest.mark.parametrize("ci", [0, 1])
def test_parts_compose_to_the_network(ci):
    """Automatic parts cover every reachable hidden node, their quotient graph is
    acyclic, and evaluating the hierarchy as formulas (leaf systems with inlining
    and simplification, then the root) reproduces the network's logits to 1e-6."""
    cfg = CONFIGS[ci]
    rng = np.random.default_rng(7 + ci)
    X = np.random.default_rng(ci).random((8, 1, *cfg.image_hw))
    n_multi = n_rules = 0
    for g in _random_genomes(50, 300 + ci, cfg):
        fn = functional_nodes(g)
        labels = {k: (frozenset([rng.choice(["a", "b"])]) if rng.random() < 0.7 else None)
                  for k in fn if g.nodes[k].kind == "hidden"}
        score = {c.innov: float(rng.random()) for c in g.enabled_conns()}
        parts = PT.propose_parts(g, labels, score)
        group_of = {n: i for i, p in enumerate(parts) for n in p}
        assert PT._quotient_acyclic(g, group_of)
        res = PT.compose_and_check(g, cfg, parts, X)
        assert res["coverage"] == 1.0 and not res["uncovered"]
        assert res["max_abs_logit_diff"] < 1e-6
        assert res["max_abs_exit_map_diff"] < 1e-6
        n_multi += any(len(p) > 1 for p in parts)
        n_rules += any(any(r != "nested sum flattened" for r in s.rules) for s in res["systems"])
    assert n_multi >= 5
    assert n_rules >= 3  # composed / delta / ReLU / resampling rules exercised


def _chain_genome(cfg, levels, kernels, biases=None):
    """input -> n1 -> n2 -> ... -> output 0 (levels / kernels per hop)."""
    g = ConvGenome(key=0)
    g.nodes[-1] = NodeGene(-1, "input", 0, 0.0)
    g.nodes[0] = NodeGene(0, "output", 0, 0.0)
    prev = -1
    for i, lv in enumerate(levels):
        g.nodes[i + 1] = NodeGene(i + 1, "hidden", lv, 0.0 if biases is None else biases[i])
        g.conns[i] = ConnGene(i, prev, i + 1, np.asarray(kernels[i], np.float32))
        prev = i + 1
    g.conns[len(levels)] = ConnGene(len(levels), prev, 0, np.asarray(kernels[len(levels)], np.float32))
    g.head_w = np.ones((1, 1), np.float32)
    g.head_b = np.zeros(1, np.float32)
    return g


def test_delta_nodes_drop_out_after_add_node():
    """A fresh add-node split (delta kernel, zero bias, non-negative source) simplifies
    back to the original edge: ReLU of a non-negative argument and the unit delta vanish."""
    cfg = CONFIGS[0]
    rng = np.random.default_rng(0)
    tr = InnovationTracker(1)
    g = minimal_genome(0, cfg, tr, rng)
    k_old = next(iter(g.conns.values())).kernel.copy()
    mutate_add_node(g, cfg, tr, rng, coarse=False)
    nid = g.hidden_keys()[0]
    g.nodes[nid].level = 0
    ctx = FM.FormulaContext(g, cfg)
    sysm = FM.part_system(ctx, {nid, 0})
    (node, e), = sysm.equations
    assert node == 0
    assert FM.to_text(e) == f"K[{nid}→0]⋆x"
    assert "unit delta kernel dropped" in sysm.rules and "ReLU of a non-negative argument removed" in sysm.rules
    np.testing.assert_allclose(FM.operators(e)[0].eff_kernel(), k_old, atol=1e-7)
    x = torch.rand(4, 1, 16, 16, dtype=torch.float64)
    ref = ExplainNet(g, cfg).forward(x).maps[0]
    assert (FM.evaluate(e, {-1: x}, cfg) - ref).abs().max() < 1e-6


def test_pools_compose_and_pool_after_upsample_cancels():
    cfg = CONFIGS[0]
    d = delta_kernel(1)
    g = _chain_genome(cfg, [1, 2], [d, d, np.full((3, 3), 0.1)])
    ctx = FM.FormulaContext(g, cfg)
    sysm = FM.part_system(ctx, {1, 2, 0})
    txt = FM.to_text(sysm.equations[-1][1])
    assert txt == "K[2→0]⋆U↑4(P↓4(x))"  # pool-then-upsample is not an identity: kept
    x = torch.rand(3, 1, 16, 16, dtype=torch.float64)
    assert (FM.evaluate(sysm.equations[-1][1], {-1: x}, cfg) - ExplainNet(g, cfg).forward(x).maps[0]).abs().max() < 1e-9
    # upsample then pool: exact identity
    g2 = _chain_genome(cfg, [1, 0], [d, d, np.full((3, 3), 0.1)])
    g2.nodes[1].level = 0
    g2.nodes[2].level = 1
    e = FM.part_system(FM.FormulaContext(g2, cfg), {1, 2, 0}).equations[-1][1]
    assert FM.to_text(e) == "K[2→0]⋆U↑2(P↓2(x))"


def test_kernels_compose_and_sum_exactly():
    """A -> N (non-negative kernel, zero bias) -> B and A -> B: N is linear, so
    B = ReLU(b + K~ * h_A) with K~ = K_AB + K_NB * K_AN (sum of chains). The chain
    form is exact; the effective kernel equals it away from the borders."""
    cfg = CONFIGS[0]
    rng = np.random.default_rng(3)
    g = ConvGenome(key=0)
    g.nodes[-1] = NodeGene(-1, "input", 0, 0.0)
    g.nodes[0] = NodeGene(0, "output", 0, 0.1)
    for k, b in ((1, -0.1), (2, 0.0), (3, 0.05)):
        g.nodes[k] = NodeGene(k, "hidden", 0, b)
    K = lambda s: rng.standard_normal((s, s)).astype(np.float32)  # noqa: E731
    g.conns[0] = ConnGene(0, -1, 1, K(3))
    g.conns[1] = ConnGene(1, 1, 2, np.abs(K(3)))
    g.conns[2] = ConnGene(2, 2, 3, K(5))
    g.conns[3] = ConnGene(3, 1, 3, K(3))
    g.conns[4] = ConnGene(4, 3, 0, K(1))
    g.head_w = np.ones((1, 1), np.float32)
    g.head_b = np.zeros(1, np.float32)
    ctx = FM.FormulaContext(g, cfg)
    sysm = FM.part_system(ctx, {2, 3})
    (node, e), = sysm.equations
    assert node == 3
    ops = FM.operators(e)
    assert len(ops) == 1 and not ops[0].simple and len(ops[0].chains) == 2
    assert "kernels composed across a linear node" in sysm.rules
    assert "kernels on the same input summed" in sysm.rules
    x = torch.rand(5, 1, 16, 16, dtype=torch.float64)
    f = ExplainNet(g, cfg).forward(x)
    assert (FM.evaluate(e, {1: f.maps[1]}, cfg) - f.maps[3]).abs().max() < 1e-9
    eff = ops[0].eff_kernel()
    assert eff.shape == (7, 7)
    direct = torch.nn.functional.conv2d(f.maps[1], torch.as_tensor(eff).view(1, 1, 7, 7), padding=3)
    pre = f.pre[3] - 0.05
    r = 3 + 1  # composed radius: borders differ
    assert (direct - pre)[..., r:-r, r:-r].abs().max() < 1e-9


def test_receptive_fields_and_exact_kernel():
    cfg = CONFIGS[0]
    rng = np.random.default_rng(1)
    k3, k5 = rng.standard_normal((3, 3)), rng.standard_normal((5, 5))
    g = _chain_genome(cfg, [0], [k3, k5])
    assert RF.receptive_box(g, cfg, 1)["h"] == 3
    assert RF.receptive_box(g, cfg, 0)["h"] == 7
    g2 = _chain_genome(cfg, [1], [k3, k5])  # x -> P2 -> K3 at 8x8, then U2 -> K5
    assert RF.receptive_box(g2, cfg, 1)["h"] == 6
    X = np.random.default_rng(0).random((10, 1, 16, 16))
    f = ExplainNet(g, cfg).forward(torch.as_tensor(X))
    lin = RF.linear_on(f.pre, g)
    box = RF.receptive_box(g, cfg, 1)
    eff = RF.effective_input_kernel(g, cfg, 1, lin, X, box)
    assert eff["exact"]
    np.testing.assert_allclose(eff["kernel"], k3.astype(np.float32), atol=1e-7)
    # node 0 depends on the ReLU node 1: exact only if 1 is always active
    lin[1] = False
    eff0 = RF.effective_input_kernel(g, cfg, 0, lin, X, RF.receptive_box(g, cfg, 0))
    assert not eff0["exact"] and "approximate" in eff0["method"]


def test_ablation_of_a_dead_edge_is_marginal():
    cfg = CONFIGS[0]
    g = _chain_genome(cfg, [0], [np.ones((3, 3)) * 0.2, np.ones((3, 3)) * 0.3])
    g.conns[9] = ConnGene(9, -1, 0, np.zeros((3, 3), np.float32))
    d = make_synthetic(200, 0)
    imp = IM.ablation_importance(ExplainNet(g, cfg), d["X"], d["y"], 1, [1], [9, 0])
    assert imp["edges"][9]["class"] == "marginal" and abs(imp["edges"][9]["d_logloss"]) < 1e-12
    assert IM.classify(0.01, 0.0) == "signal" and IM.classify(0.0, 0.0099) == "marginal"


def test_pipeline_end_to_end_on_a_random_genome():
    cfg = CONFIGS[0]
    tr = InnovationTracker(1)
    g = random_genome_by_mutation(0, cfg, tr, np.random.default_rng(5), n_steps=25)
    tr_d, va, te = make_synthetic(50, 1), make_synthetic(120, 2), make_synthetic(150, 3)
    data = {"X_train": tr_d["X"], "y_train": tr_d["y"], "X_val": va["X"], "y_val": va["y"],
            "X_test": te["X"], "y_test": te["y"], "c_test": te["concepts"]}
    r = explain_network(g, cfg, data, "synthetic", ["hbar", "blob", "vbar", "diag"], n_walk=6)
    assert r["check"]["max_abs_logit_diff"] < 1e-6 and r["check"]["coverage"] == 1.0
    assert all(w["sum_check"] < 1e-9 for w in r["walk"])
    for k, n in r["nodes"].items():
        if n["kind"] != "input":
            assert n["formula_err"] < 1e-6
