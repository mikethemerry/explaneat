"""One definition of "a network carries this signal" for the Experiment 2 signal tables.

Used by all three signal analyses (``experiments/adult_signals/catalogue.py``,
``experiments/churn_signals/catalogue.py``, ``experiments/heart100_audit/audit.py``) and the
completeness check (``experiments/signal_completeness/``); the tick tables
(``thesis/ch7/exp2/signal_ticks.py``) only read the states written by those analyses. Adult, Churn
and Heart-100 are judged by the same rule.

**Contribution.** A signal's *logit contribution* in one network is a per-row vector ``c`` on the
training rows, in logit units:

- a linear term (a slope, or a block of one-hot offsets): weight x input, where the weights are
  the network's best linear approximation, i.e. the least-squares projection of its logit on the
  linear columns (for Adult and Churn this is the linear block of the catalogue basis, whose
  non-linear columns are residualised on the linear ones, so the coefficients are the same);
- a non-linear signal: its block's component of that projection (Adult, Churn: bends and
  interactions defined on the data), or for Heart-100, whose non-linear rows are families of
  switching units, the change in the logit when those units are replaced by their least-squares
  affine fit on their own inputs (the units' non-linear part, as it reaches the output).

A network that reads none of a signal's input variables has weight zero on it, so its
contribution is zero whatever the projection says (``reads=False``): a projection can put weight
on an unread input that is correlated with a read one.

**Size.** ``contribution_sd(c)`` is the standard deviation of ``c`` after clipping it to its 1st and
99th percentiles (training rows). Clipping is the Adult rule: offsets of tens of logits on a few
dozen rows (single countries of birth, rare work classes, a rare ECG code) would otherwise make a
term that touches under 1% of the rows look large. It is applied to all three data sets.

**Presence** (design choices, the same for every row and every data set):

- ``used``: clipped SD >= 0.10 logit (``USED_SD``). This is "present" in every table.
- ``weak``: 0.03 <= clipped SD < 0.10 (``WEAK_SD``). Shown, never counted as present.
- ``rare_only``: raw SD >= 0.10 but not used (large values on under 1% of rows only).

**Direction** of a present linear term (``consensus_direction``): the sign of the correlation of the
network's contribution with the consensus, the mean contribution over the seeds where the term is
used; a correlation <= 0 is ``opp`` ("opposite direction to most seeds").
"""
from __future__ import annotations

import numpy as np

USED_SD = 0.10
WEAK_SD = 0.03
CLIP_Q = (0.01, 0.99)


def contribution_sd(c, clip: bool = True) -> float:
    """SD of a per-row logit contribution, after clipping to its 1st-99th percentiles."""
    c = np.asarray(c, float)
    if not clip:
        return float(np.std(c))
    lo, hi = np.quantile(c, CLIP_Q)
    return float(np.std(np.clip(c, lo, hi)))


def state_of(sd_clipped: float) -> str:
    """'used', 'weak' or '' from a clipped contribution SD."""
    if sd_clipped >= USED_SD:
        return "used"
    if sd_clipped >= WEAK_SD:
        return "weak"
    return ""


def presence(c, reads: bool = True) -> dict:
    """{sd_raw, sd, state, rare_only, reads} of one contribution vector (training rows)."""
    sd_raw = contribution_sd(c, clip=False)
    sd = contribution_sd(c)
    st = state_of(sd) if reads else ""
    return dict(sd_raw=sd_raw, sd=sd, state=st, rare_only=bool(reads and sd_raw >= USED_SD and st != "used"),
                reads=bool(reads))


def consensus(components):
    """Mean of the contributions of the seeds that use a signal (a list of arrays)."""
    return np.mean(np.stack(components), axis=0) if components else None


def consensus_direction(c, cons) -> str:
    """'ok' when a contribution correlates positively with the consensus, else 'opp'."""
    a = np.asarray(c, float)
    if cons is None or np.std(a) <= 1e-12 or np.std(cons) <= 1e-12:
        return "opp"
    return "ok" if float(np.corrcoef(a, cons)[0, 1]) > 0 else "opp"
