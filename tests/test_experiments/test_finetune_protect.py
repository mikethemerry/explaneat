"""Protection phase in genome_surgery.finetune (experiments/churn/GRAFT_F_G_PLAN.md, study G).

The protect_* kwargs are additive: with the defaults finetune must be
bit-identical to the version before the change (commit 1deebbf), both with
fixed epochs and with early stopping. With protection, the protected weights
and biases do not change during epochs 1..P while everything else trains, and
they train afterwards.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import genome_surgery as gs
from test_genome_surgery import _donor_ops

REPO = Path(__file__).resolve().parents[2]
REF_COMMIT = "1deebbf"


@pytest.fixture(scope="module")
def gs_ref(tmp_path_factory):
    try:
        src = subprocess.check_output(
            ["git", "show", f"{REF_COMMIT}:experiments/churn/genome_surgery.py"],
            cwd=REPO, text=True)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot read {REF_COMMIT} from git: {e}")
    path = tmp_path_factory.mktemp("ref_protect") / "genome_surgery_ref_protect.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("genome_surgery_ref_protect", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["genome_surgery_ref_protect"] = mod
    spec.loader.exec_module(mod)
    return mod


def _toy(n, seed=0, n_val=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n + n_val, 3))
    logit = 0.8 * X[:, 0] - 0.5 * X[:, 1] * (X[:, 2] > 0)
    y = (rng.uniform(size=len(X)) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    return X[:n], y[:n], X[n:], y[n:]


def _params(g):
    return ({k: (c.weight, c.enabled) for k, c in g.connections.items()},
            {k: n.bias for k, n in g.nodes.items()})


def _same_info(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k


@pytest.fixture
def grafted(small_config, recipient_genome, donor_genome):
    sub = gs.annotation_to_genome_subgraph(_donor_ops(), "feat", donor_genome, small_config)
    return gs.graft(recipient_genome, small_config, donor_genome, sub)


class TestDefaultsBitIdentical:
    @pytest.mark.parametrize("kw", [
        dict(epochs=40, seed=0),
        dict(epochs=40, seed=3, mask_phantom=True),
        dict(epochs=25, seed=1, frozen_conn_keys=[(-1, 1)], frozen_node_keys=[1]),
    ])
    @pytest.mark.parametrize("extra", [{}, dict(protect_epochs=0, protect_conn_keys=[(-1, 1)]),
                                       dict(protect_epochs=10)])
    def test_fixed_epochs(self, gs_ref, small_config, recipient_genome, kw, extra):
        X, y, _, _ = _toy(300, seed=5)
        new, inew = gs.finetune(recipient_genome, small_config, X, y, return_info=True,
                                **kw, **extra)
        old, iold = gs_ref.finetune(recipient_genome, small_config, X, y, return_info=True,
                                    **kw)
        assert _params(new) == _params(old)
        _same_info(inew, iold)

    @pytest.mark.parametrize("extra", [{}, dict(protect_epochs=0, protect_node_keys=[1])])
    def test_early_stopping(self, gs_ref, small_config, recipient_genome, extra):
        Xf, yf, Xv, yv = _toy(30, seed=0)
        kw = dict(epochs=None, seed=0, mask_phantom=True, return_info=True, val_X=Xv,
                  val_y=yv)
        new, inew = gs.finetune(recipient_genome, small_config, Xf, yf, **kw, **extra)
        old, iold = gs_ref.finetune(recipient_genome, small_config, Xf, yf, **kw)
        assert _params(new) == _params(old)
        _same_info(inew, iold)

    def test_grafted_genome_e500_like(self, gs_ref, small_config, grafted):
        g, rec = grafted
        X, y, _, _ = _toy(300, seed=2)
        new = gs.finetune(g, small_config, X, y, 60, 7, mask_phantom=True)
        old = gs_ref.finetune(g, small_config, X, y, 60, 7, mask_phantom=True)
        assert _params(new) == _params(old)


class TestProtectKeys:
    def test_keys_of_graft(self, grafted):
        g, rec = grafted
        conns, nodes = gs.graft_protect_keys(rec)
        assert nodes == sorted(rec["id_map"].values())
        assert all(k[1] in nodes for k in conns)
        readout = {tuple(k) for k in rec["readout_connections"]}
        assert not readout & set(conns)
        # every added edge into a grafted node is protected
        assert set(conns) == {tuple(k) for k in rec["added_connections"] if k[1] in nodes}


class TestProtection:
    P = 12

    def _run(self, cfg, g, rec, epochs, P, **kw):
        X, y, _, _ = _toy(300, seed=3)
        conns, nodes = gs.graft_protect_keys(rec)
        return gs.finetune(g, cfg, X, y, epochs, 0, mask_phantom=True, return_info=True,
                           protect_conn_keys=conns, protect_node_keys=nodes,
                           protect_epochs=P, **kw), (X, y)

    def test_protected_fixed_during_phase_others_train(self, small_config, grafted):
        g, rec = grafted
        (out, info), _ = self._run(small_config, g, rec, self.P, self.P)
        conns, nodes = gs.graft_protect_keys(rec)
        for k in conns:
            assert out.connections[k].weight == g.connections[k].weight
        for n in nodes:
            assert out.nodes[n].bias == g.nodes[n].bias
        others = [k for k in g.connections if k not in set(conns) and g.connections[k].enabled]
        assert any(out.connections[k].weight != g.connections[k].weight for k in others)
        readout = [tuple(k) for k in rec["readout_connections"]]
        assert all(out.connections[k].weight != g.connections[k].weight for k in readout)
        assert info["protect_epochs"] == self.P
        assert info["n_protected_connections"] == len(conns)
        assert info["n_protected_nodes"] == len(nodes)

    def test_protected_train_after_phase(self, small_config, grafted):
        g, rec = grafted
        (out, _), _ = self._run(small_config, g, rec, self.P + 10, self.P)
        conns, nodes = gs.graft_protect_keys(rec)
        assert any(out.connections[k].weight != g.connections[k].weight for k in conns)

    def test_differs_from_unprotected(self, small_config, grafted):
        g, rec = grafted
        (a, _), _ = self._run(small_config, g, rec, 30, self.P)
        (b, _), _ = self._run(small_config, g, rec, 30, 0)
        assert _params(a) != _params(b)

    def test_equals_frozen_run_for_phase(self, small_config, grafted):
        """Epochs 1..P of a protected run are exactly a run with those keys frozen."""
        g, rec = grafted
        (a, _), (X, y) = self._run(small_config, g, rec, self.P, self.P)
        conns, nodes = gs.graft_protect_keys(rec)
        b = gs.finetune(g, small_config, X, y, self.P, 0, mask_phantom=True,
                        frozen_conn_keys=conns, frozen_node_keys=nodes)
        assert _params(a) == _params(b)

    def test_early_stopping_patience_starts_after_phase(self, small_config, grafted):
        g, rec = grafted
        X, y, Xv, yv = _toy(30, seed=0)
        conns, nodes = gs.graft_protect_keys(rec)
        P = 40
        out, info = gs.finetune(g, small_config, X, y, None, 0, mask_phantom=True,
                                return_info=True, val_X=Xv, val_y=yv, patience=5,
                                min_delta=10.0, protect_conn_keys=conns,
                                protect_node_keys=nodes, protect_epochs=P)
        # min_delta 10: no improvement ever, best epoch 0; without the phase it would
        # stop at epoch 5, with it at P + patience
        assert info["best_epoch"] == 0
        assert info["stop_epoch"] == P + 5
        assert _params(out) == _params(g)  # best (epoch 0) restored

    def test_negative_phase_rejected(self, small_config, grafted):
        g, rec = grafted
        X, y, _, _ = _toy(30)
        with pytest.raises(ValueError, match="protect_epochs"):
            gs.finetune(g, small_config, X, y, 5, 0, protect_epochs=-1,
                        protect_node_keys=[1])
