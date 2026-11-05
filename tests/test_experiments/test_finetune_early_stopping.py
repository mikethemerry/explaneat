"""Early stopping in genome_surgery.finetune (experiments/churn/EARLY_STOPPING_PLAN.md).

The early-stopping kwargs are additive: with the defaults (val_X=None) finetune
must be bit-identical to the version before the amendment (commit 6f37967),
which study E (reinit.py) still calls.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import genome_surgery as gs

REPO = Path(__file__).resolve().parents[2]
REF_COMMIT = "6f37967"


@pytest.fixture(scope="module")
def gs_ref(tmp_path_factory):
    """genome_surgery.py as of the commit before the amendment."""
    try:
        src = subprocess.check_output(
            ["git", "show", f"{REF_COMMIT}:experiments/churn/genome_surgery.py"],
            cwd=REPO, text=True)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot read {REF_COMMIT} from git: {e}")
    path = tmp_path_factory.mktemp("ref") / "genome_surgery_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("genome_surgery_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["genome_surgery_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def _toy(n, seed=0, n_val=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n + n_val, 3))
    logit = 0.8 * X[:, 0]
    y = (rng.uniform(size=len(X)) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    return X[:n], y[:n], X[n:], y[n:]


def _params(g):
    return ({k: (c.weight, c.enabled) for k, c in g.connections.items()},
            {k: n.bias for k, n in g.nodes.items()})


def _assert_info_identical(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k  # NaN == NaN


class TestDefaultsBitIdentical:
    @pytest.mark.parametrize("kw", [
        dict(epochs=40, seed=0),
        dict(epochs=40, seed=3, mask_phantom=True),
        dict(epochs=25, seed=1, frozen_conn_keys=[(-1, 1)], frozen_node_keys=[1]),
        dict(epochs=25, seed=2, frozen_conn_keys=[(1, 0)], mask_phantom=True),
    ])
    @pytest.mark.parametrize("return_info", [False, True])
    def test_matches_pre_amendment_finetune(self, gs_ref, small_config, recipient_genome,
                                            kw, return_info):
        X, y, _, _ = _toy(300, seed=5)
        new = gs.finetune(recipient_genome, small_config, X, y, return_info=return_info, **kw)
        old = gs_ref.finetune(recipient_genome, small_config, X, y, return_info=return_info,
                              **kw)
        if return_info:
            (new, inew), (old, iold) = new, old
            _assert_info_identical(inew, iold)
            assert "early_stopping" not in inew
        assert _params(new) == _params(old)

    def test_es_kwargs_ignored_without_val(self, small_config, recipient_genome):
        X, y, _, _ = _toy(300, seed=5)
        a, ia = gs.finetune(recipient_genome, small_config, X, y, 30, 0, mask_phantom=True,
                            return_info=True)
        b, ib = gs.finetune(recipient_genome, small_config, X, y, 30, 0, mask_phantom=True,
                            return_info=True, patience=3, min_delta=0.5, max_epochs=7,
                            restore_best=False)
        assert _params(a) == _params(b)
        _assert_info_identical(ia, ib)


class TestEarlyStopping:
    def _run(self, small_config, genome, **kw):
        Xf, yf, Xv, yv = _toy(30, seed=0)
        return gs.finetune(genome, small_config, Xf, yf, None, 0, mask_phantom=True,
                           return_info=True, val_X=Xv, val_y=yv, **kw), (Xf, yf, Xv, yv)

    def test_triggers_on_overfitting_and_restores_best(self, small_config, recipient_genome):
        (g, info), (Xf, yf, Xv, yv) = self._run(small_config, recipient_genome)
        assert info["early_stopping"] is True
        assert info["stopped_by"] == "patience"
        assert 0 < info["best_epoch"] < info["stop_epoch"] < info["max_epochs"]
        assert info["stop_epoch"] == info["best_epoch"] + info["patience"]
        assert info["epochs"] == info["stop_epoch"] == len(info["losses"]) \
            == len(info["val_losses"])
        # the toy problem overfits: train BCE keeps falling, val BCE rises after best
        assert info["losses"][-1] < info["losses"][info["best_epoch"] - 1]
        assert info["val_losses"][-1] > info["best_val_bce"] + 1e-5
        assert info["best_val_bce"] == min([info["val_bce_before"]] + info["val_losses"])
        assert info["best_val_bce"] == info["val_losses"][info["best_epoch"] - 1]
        # restored weights = the fixed-epoch fine-tune stopped at the best epoch
        ref = gs.finetune(recipient_genome, small_config, Xf, yf, info["best_epoch"], 0,
                          mask_phantom=True)
        assert _params(g) == _params(ref)
        assert info["val_bce_after_written_back"] == pytest.approx(info["best_val_bce"],
                                                                    abs=1e-12)

    def test_no_restore_keeps_last_epoch(self, small_config, recipient_genome):
        (g, info), (Xf, yf, _, _) = self._run(small_config, recipient_genome,
                                              restore_best=False)
        ref = gs.finetune(recipient_genome, small_config, Xf, yf, info["stop_epoch"], 0,
                          mask_phantom=True)
        assert _params(g) == _params(ref)

    def test_max_epochs_bound(self, small_config, recipient_genome):
        (g, info), _ = self._run(small_config, recipient_genome, patience=10_000,
                                 max_epochs=17)
        assert info["stopped_by"] == "max_epochs" and info["stop_epoch"] == 17
        assert len(info["losses"]) == 17

    def test_min_delta(self, small_config, recipient_genome):
        """A min_delta larger than any improvement: best stays at epoch 0, the
        starting weights are restored."""
        (g, info), _ = self._run(small_config, recipient_genome, min_delta=10.0, patience=5)
        assert info["best_epoch"] == 0 and info["stop_epoch"] == 5
        assert _params(g) == _params(recipient_genome)

    def test_deterministic(self, small_config, recipient_genome):
        (a, ia), _ = self._run(small_config, recipient_genome)
        (b, ib), _ = self._run(small_config, recipient_genome)
        assert _params(a) == _params(b)
        _assert_info_identical(ia, ib)

    def test_frozen_parameters_stay_fixed(self, small_config, recipient_genome):
        (g, info), _ = self._run(small_config, recipient_genome,
                                 frozen_conn_keys=[(-1, 1)], frozen_node_keys=[1])
        assert g.connections[(-1, 1)].weight == recipient_genome.connections[(-1, 1)].weight
        assert g.nodes[1].bias == recipient_genome.nodes[1].bias

    @pytest.mark.parametrize("kw, msg", [
        (dict(val_y=None), "val_y"),
        (dict(epochs=10), "epochs=None"),
        (dict(mask_phantom=False), "masked"),
    ])
    def test_bad_arguments(self, small_config, recipient_genome, kw, msg):
        Xf, yf, Xv, yv = _toy(30)
        args = dict(epochs=None, mask_phantom=True, val_X=Xv, val_y=yv)
        args.update(kw)
        with pytest.raises(ValueError, match=msg):
            gs.finetune(recipient_genome, small_config, Xf, yf, seed=0, **args)

    def test_epochs_required_without_val(self, small_config, recipient_genome):
        Xf, yf, _, _ = _toy(30)
        with pytest.raises(ValueError, match="epochs is required"):
            gs.finetune(recipient_genome, small_config, Xf, yf, None, 0)


class TestRefitOnFitRows:
    def test_refit_uses_only_the_rows_given(self, small_config, recipient_genome):
        Xf, yf, Xv, yv = _toy(200, seed=1)
        keys = [(1, 0), (-3, 0)]
        g_fit, i_fit = gs.refit_readout(recipient_genome, small_config, Xf, yf, keys)
        assert i_fit["n_rows"] == len(yf)
        # adding rows changes the fit, so the validation rows were not used
        g_all, _ = gs.refit_readout(recipient_genome, small_config, np.vstack([Xf, Xv]),
                                    np.concatenate([yf, yv]), keys)
        assert _params(g_fit) != _params(g_all)
