"""Tasks for the modular-crossover study (PLAN.md "Tasks").

A task is a list of stages. Each stage has its own train rows (what NEAT
trains and is selected on while the stage is active); the test rows used
for the outcomes are the final stage's. ``probe_X`` is a fixed input set
(the final stage's train rows) on which module contributions are compared
for alias alignment, so a module found in stage 1 and one found in stage 2
are compared on the same inputs.

- ``monk2``: the experiments/10seed MONK-2 split (exported once from the DB to
  data/monk2.npz by export_monk2.py), one stage, 17 standardised inputs.
- ``mux11``: an 11-input multiplexer, 3 address bits a0..a2 and 8 data bits
  d0..d7, all encoded -1/+1. y = d_k with k = a0 + 2 a1 + 4 a2 (bit = 1 <=>
  input = +1). Every one of the 2^11 = 2048 inputs occurs once; a fixed
  stratified 50/50 split (numpy default_rng(20261106)) gives 1024 train and
  1024 test rows. Stage 1 = rows with a2 = 0 (the 6-multiplexer on a0, a1,
  d0..d3, a2 held at -1); stage 2 = all rows. Noise-free.
  Ground-truth features: g_k(x) = [address == k] * [d_k == 1] for k = 0..7
  (y = sum_k g_k exactly), and the address decoders D_k(x) = [address == k].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"

MUX_SPLIT_SEED = 20261106


@dataclass
class Stage:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class Task:
    name: str
    config_path: Path
    stages: List[Stage]
    feature_names: List[str]
    # ground-truth features on probe_X (name -> 0/1 vector); empty for MONK-2
    truth: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def final(self) -> Stage:
        return self.stages[-1]

    @property
    def probe_X(self) -> np.ndarray:
        return self.final.X_train


def load_monk2() -> Task:
    d = np.load(HERE / "data" / "monk2.npz")
    st = Stage("monk2", d["X_train"].astype(np.float64), d["y_train"].astype(np.float64),
               d["X_test"].astype(np.float64), d["y_test"].astype(np.float64))
    return Task("monk2", CONFIGS / "monk2.ini", [st], [str(n) for n in d["feature_names"]])


MUX_NAMES = ["a0", "a1", "a2"] + [f"d{k}" for k in range(8)]


def mux_all_rows():
    """All 2048 inputs as -1/+1 (columns a0, a1, a2, d0..d7) and the labels."""
    bits = ((np.arange(2048)[:, None] >> np.arange(11)[None, :]) & 1).astype(np.int64)
    addr = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]
    y = bits[np.arange(2048), 3 + addr].astype(np.float64)
    return 2.0 * bits - 1.0, y, addr


def mux_truth(X) -> Dict[str, np.ndarray]:
    b = (np.asarray(X) > 0).astype(np.int64)
    addr = b[:, 0] + 2 * b[:, 1] + 4 * b[:, 2]
    out = {}
    for k in range(8):
        dec = (addr == k).astype(np.float64)
        out[f"D{k}"] = dec
        out[f"g{k}"] = dec * b[:, 3 + k]
    return out


def mux_split():
    """(train rows, test rows) indices into mux_all_rows(), stratified on y."""
    X, y, _ = mux_all_rows()
    rng = np.random.default_rng(MUX_SPLIT_SEED)
    tr, te = [], []
    for cls in (0.0, 1.0):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        half = len(idx) // 2
        tr.extend(idx[:half].tolist())
        te.extend(idx[half:].tolist())
    return np.array(sorted(tr)), np.array(sorted(te))


def load_mux11() -> Task:
    X, y, _ = mux_all_rows()
    tr, te = mux_split()
    s1_tr = tr[X[tr, 2] < 0]
    s1_te = te[X[te, 2] < 0]
    stages = [Stage("mux11-stage1", X[s1_tr], y[s1_tr], X[s1_te], y[s1_te]),
              Stage("mux11-stage2", X[tr], y[tr], X[te], y[te])]
    task = Task("mux11", CONFIGS / "mux11.ini", stages, list(MUX_NAMES))
    task.truth = mux_truth(task.probe_X)
    return task


TASKS = {"monk2": load_monk2, "mux11": load_mux11}


def load_task(name: str) -> Task:
    return TASKS[name]()
