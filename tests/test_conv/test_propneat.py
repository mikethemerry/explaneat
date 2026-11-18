"""PropNEAT conv engine vs the naive reference engine."""

import numpy as np
import pytest
import torch

from explaneat.conv.genome import ConvNeatConfig, InnovationTracker
from explaneat.conv.mutation import random_genome_by_mutation
from explaneat.conv.network import NaiveConvNet, loss_fn
from explaneat.conv.propneat import PropNEATConvNet, flop_report, traverse

torch.set_num_threads(1)


def _genomes(n, seed, cfg):
    tr = InnovationTracker(cfg.n_out_maps)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = random_genome_by_mutation(i, cfg, tr, rng, n_steps=int(rng.integers(5, 45)))
        for nd in g.nodes.values():
            if nd.kind != "input":
                nd.bias = float(rng.normal() * 0.2)
        out.append(g)
    return out


CONFIGS = [
    ConvNeatConfig(in_channels=1, image_hw=(16, 16), n_classes=1, max_level=2),
    ConvNeatConfig(in_channels=2, image_hw=(8, 8), n_classes=3, max_level=2),
]


@pytest.mark.parametrize("mixed_k", ["embed", "split"])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-9), (torch.float32, 1e-5)])
def test_engine_parity_outputs_and_gradients(mixed_k, dtype, tol):
    """200+ random genomes: outputs and per-gene kernel gradients agree."""
    stats = {"skip": 0, "mixed_k": 0, "mixed_level": 0, "dropped": 0, "const": 0, "n": 0}
    for ci, cfg in enumerate(CONFIGS):
        torch.manual_seed(ci)
        x = torch.rand(6, cfg.in_channels, *cfg.image_hw, dtype=dtype)
        for g in _genomes(110, 10 + ci, cfg):
            naive = NaiveConvNet(g, cfg, dtype=dtype)
            prop = PropNEATConvNet(g, cfg, dtype=dtype, mixed_k=mixed_k)
            a, b = naive(x), prop(x)
            assert (a - b).abs().max().item() < tol
            w = torch.randn_like(a)
            (a * w).sum().backward()
            (b * w).sum().backward()
            gn, gp = naive.kernel_grads(), prop.kernel_grads()
            for c in g.enabled_conns():
                if c.innov in gp:
                    ref = gn.get(c.innov, np.zeros_like(gp[c.innov]))
                    scale = max(1.0, float(np.abs(ref).max()))
                    assert np.abs(ref - gp[c.innov]).max() < tol * scale, c.innov
                else:  # gene into a dropped node: cannot affect the output
                    assert c.innov not in gn or not gn[c.innov].any()
            # bookkeeping: make sure the interesting cases occur
            fwd, kept, depth = traverse(g)
            stats["n"] += 1
            stats["dropped"] += len(kept) < len(g.nodes)
            stats["const"] += any(n not in fwd and g.nodes[n].kind != "input" for n in kept)
            stats["mixed_level"] += len({g.nodes[n].level for n in kept}) > 1
            ks = {}
            for c in g.enabled_conns():
                if c.dst in kept:
                    ks.setdefault(prop.node_loc[c.dst][0], set()).add(c.k)
            stats["mixed_k"] += any(len(v) > 1 for v in ks.values())
            stats["skip"] += any(len(v) > 1 for v in prop.sources.values())
    assert stats["n"] >= 200
    for k in ("skip", "mixed_k", "mixed_level", "dropped", "const"):
        assert stats[k] >= 5, stats


@pytest.mark.parametrize("mixed_k", ["embed", "split"])
def test_mapping_round_trip(mixed_k):
    cfg = CONFIGS[0]
    for g in _genomes(60, 3, cfg):
        before_k = {i: c.kernel.copy() for i, c in g.conns.items()}
        before_b = {k: n.bias for k, n in g.nodes.items()}
        before_h = (g.head_w.copy(), g.head_b.copy())
        PropNEATConvNet(g, cfg, mixed_k=mixed_k).write_back()
        for i, c in g.conns.items():
            assert c.kernel.shape == before_k[i].shape
            assert np.array_equal(c.kernel, before_k[i])
        for k, n in g.nodes.items():
            assert np.float32(n.bias) == np.float32(before_b[k])
        assert np.array_equal(g.head_w, before_h[0]) and np.array_equal(g.head_b, before_h[1])


def test_masks_hold_after_training():
    cfg = CONFIGS[0]
    rng = np.random.default_rng(0)
    X = torch.rand(64, 1, 16, 16)
    y = (torch.rand(64) > 0.5).long()
    n_masked = 0
    for g in _genomes(20, 5, cfg):
        net = PropNEATConvNet(g, cfg)
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        for _ in range(15):
            loss = loss_fn(net(X), y, 1)
            opt.zero_grad(); loss.backward(); opt.step()
        net.assert_masks_hold()
        for t, ks in net.groups.items():
            for k_g in ks:
                W, M = net.weights[f"{t}_{k_g}"], net.mask(t, k_g)
                assert torch.all(W[M == 0] == 0)
                n_masked += int((M == 0).sum())
        net.write_back()
    assert n_masked > 0


def test_retrain_variant_lifts_masks_and_refuses_write_back():
    cfg = CONFIGS[0]
    g = _genomes(5, 9, cfg)[-1]
    net = PropNEATConvNet(g, cfg, retrain=True)
    for t, ks in net.groups.items():
        for k_g in ks:
            assert torch.all(net.mask(t, k_g) == 1)
    with pytest.raises(RuntimeError):
        net.write_back()


def test_flop_report_overhead_at_least_one():
    cfg = CONFIGS[0]
    for g in _genomes(20, 4, cfg):
        r = flop_report(g, cfg)
        if r["useful_macs"]:
            assert r["dense_macs"] >= r["useful_macs"]
