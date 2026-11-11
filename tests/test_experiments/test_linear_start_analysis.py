"""Analysis must read legacy CLI-created weighted genomes without altering artifacts."""
import gzip
import pickle
import sys
from pathlib import Path

MNIST = Path(__file__).resolve().parents[2] / "experiments" / "mnist"
sys.path.insert(0, str(MNIST))
import linear_start as LS
import linear_start_analyze as analysis


def test_load_run_reads_main_module_weighted_genome(tmp_path, monkeypatch):
    import __main__

    cls = LS.WeightedAddNodeGenome
    original = cls.__module__
    monkeypatch.setattr(__main__, "WeightedAddNodeGenome", cls, raising=False)
    cls.__module__ = "__main__"
    run = tmp_path / "saved"
    run.mkdir()
    try:
        config_path = LS.write_config(tmp_path, [])
        genome = cls(17)
        with gzip.open(run / "best_genome.pkl.gz", "wb") as f:
            pickle.dump(dict(genome=genome, meta={"seed": 0},
                             config_text=config_path.read_text()), f)
    finally:
        cls.__module__ = original
    monkeypatch.delattr(__main__, "WeightedAddNodeGenome")
    monkeypatch.setattr(analysis, "RUNS_DIR", tmp_path)
    loaded, config, meta = analysis.load_run("saved")
    assert type(loaded) is cls
    assert loaded.key == 17
    assert config.genome_config.num_outputs == 10
    assert meta == {"seed": 0}


def test_linear_control_resets_optimizer_at_requested_generation_length():
    import numpy as np
    import torch
    import torch.nn.functional as F

    X = np.array([[0., 1.], [1., 0.], [0.5, 0.5]], dtype=np.float64)
    y = np.array([0, 1, 0])
    d = {f"{kind}_{split}": value for split in ("train", "val", "test")
         for kind, value in (("X", X), ("y", y))}
    W = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = torch.optim.Adadelta([W, b], lr=1.5)
    for _ in range(10):
        opt.zero_grad()
        loss = F.cross_entropy(torch.as_tensor(X) @ W + b, torch.as_tensor(y))
        loss.backward()
        opt.step()
    expected = analysis.ce_acc(torch.as_tensor(X) @ W + b, y)
    actual = analysis.linear_trajectory(np.zeros((2, 2)), np.zeros(2), d, 10, {10},
                                        reset_every=25)
    assert abs(actual[10]["train"]["ce"] - expected["ce"]) < 1e-14
