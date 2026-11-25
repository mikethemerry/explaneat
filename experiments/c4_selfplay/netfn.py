"""Batched value function from a PropNEAT genome.

``ValueFn(genome, config)(X)`` maps encoded positions (``engine.encode``:
float64[42], +1 side to move, -1 opponent) to P(side to move wins), the
sigmoid output of the network. The torch module is built once; each call is
one forward pass under ``torch.no_grad()``. CPU only: MPS has no float64.
"""
import os

os.environ.setdefault("EXPLANEAT_DEVICE", "cpu")

from pathlib import Path  # noqa: E402

import neat  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explaneat.core.errors import GenomeNotValidError  # noqa: E402,F401
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STUDY_DIR / "configs" / "selfplay.ini"


def load_config(path=DEFAULT_CONFIG) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


def _as_batch(X) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    return X


class ValueFn:
    """P(side to move wins) for a batch of encoded positions.

    Raises ``GenomeNotValidError`` at construction for a genome with no
    input-output path. The genome is read, not modified.
    """

    def __init__(self, genome, config):
        self.net = NeuralNeat(genome, config, mask_phantom=True)
        self.net.eval()

    def __call__(self, X) -> np.ndarray:
        X = _as_batch(X)
        with torch.no_grad():
            out = self.net.forward(torch.from_numpy(X))
        return out.numpy().reshape(-1)


class ConstValue:
    """Constant value function (e.g. 0.5: an uninformed baseline)."""

    def __init__(self, v: float = 0.5):
        self.v = float(v)

    def __call__(self, X) -> np.ndarray:
        return np.full(_as_batch(X).shape[0], self.v)
