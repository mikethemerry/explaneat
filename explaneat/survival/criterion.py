"""Cox criterion with a torch-criterion call signature, for the backprop loop.

``BackpropPopulation(..., criterion=CoxCriterion(ties="efron"))`` trains every
genome full-batch on the exact partial likelihood.  The target is the [N, 2]
survival target (column 0 observed time, column 1 event indicator), which
``BackpropPopulation`` carries unchanged (it only unsqueezes 1-D targets).

The partial likelihood is NOT decomposable over rows: every event is coupled
to its whole risk set.  That is why the criterion takes the full target, and
why full-batch training (which ExplaNEAT does) is the regime where it is exact.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn

from .cox import CoxRiskSets, cox_ph_loss


def _fingerprint(t: np.ndarray, e: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(t, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(e, dtype=np.float64).tobytes())
    return h.hexdigest()


class CoxCriterion(nn.Module):
    """Negative mean Cox partial log-likelihood, ``criterion(pred, target)``.

    Caches the risk-set structure per distinct (time, event) content, so the
    sort happens once per dataset / fold rather than once per genome per epoch.
    The cache is keyed on a hash of the target's contents, so passing a
    different fold (e.g. validation rows) can never reuse the wrong risk sets.
    """

    #: recorded on experiments (``config_json["training"]["criterion"]``)
    name = "cox_ph"

    def __init__(self, ties: str = "efron"):
        super().__init__()
        if ties not in ("efron", "breslow"):
            raise ValueError(f"ties must be 'efron' or 'breslow', got {ties!r}")
        self.ties = ties
        self._cache_key = None
        self._risk_sets = None

    def describe(self) -> dict:
        return {"name": self.name, "ties": self.ties}

    def risk_sets(self, target: torch.Tensor) -> CoxRiskSets:
        if target.ndim != 2 or target.shape[1] != 2:
            raise ValueError(
                f"CoxCriterion needs an [N, 2] (time, event) target, got {tuple(target.shape)}")
        t = target[:, 0].detach().cpu().numpy()
        e = target[:, 1].detach().cpu().numpy()
        key = (t.shape[0], str(target.device), target.dtype, _fingerprint(t, e))
        if key != self._cache_key:
            self._risk_sets = CoxRiskSets(t, e, device=target.device, dtype=target.dtype)
            self._cache_key = key
        return self._risk_sets

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cox_ph_loss(pred, self.risk_sets(target), ties=self.ties)
