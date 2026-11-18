"""Vectorised paired bootstrap of AUC differences.

Same resamples as ``compute_runs.bootstrap_indices`` (10,000 resamples, RandomState(42)) and
the same percentile CI as ``explained_eval.bootstrap_delta_ci``; computed from per-resample
row counts instead of a Python loop (checked against the loop in ``selfcheck``).
"""
from __future__ import annotations

import numpy as np


class Boot:
    def __init__(self, idx, y):
        n = idx.shape[1]
        self.C = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.float64)
        self.y = np.asarray(y).ravel().astype(bool)
        pos = self.C[:, self.y].sum(axis=1)
        neg = self.C[:, ~self.y].sum(axis=1)
        self.valid = (pos > 0) & (neg > 0)
        self.den = pos * neg

    def aucs(self, s):
        s = np.asarray(s, float).ravel()
        order = np.argsort(s, kind="mergesort")
        ss = s[order]
        starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        Cs = self.C[:, order]
        yp = self.y[order]
        Wp = np.add.reduceat(Cs * yp, starts, axis=1)
        Wn = np.add.reduceat(Cs * ~yp, starts, axis=1)
        cum = np.cumsum(Wn, axis=1) - Wn
        num = np.sum(Wp * (cum + 0.5 * Wn), axis=1)
        out = np.full(len(num), np.nan)
        out[self.valid] = num[self.valid] / self.den[self.valid]
        return out

    def delta_ci(self, a, b):
        d = self.aucs(a) - self.aucs(b)
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(lo), float(hi)

    def auc_ci(self, a):
        d = self.aucs(a)
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(lo), float(hi)


def selfcheck(idx, y, a, b):
    import explained_eval as ee
    bt = Boot(idx, y)
    lo, hi = bt.delta_ci(a, b)
    lo2, hi2, _ = ee.bootstrap_delta_ci(y, a, b, idx)
    assert abs(lo - lo2) < 1e-9 and abs(hi - hi2) < 1e-9, (lo, lo2, hi, hi2)
    return lo, hi
