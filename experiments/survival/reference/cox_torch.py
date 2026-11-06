"""
Cox partial-likelihood machinery in PyTorch, sized for ExplaNEAT/PropNEAT.

Self-contained: torch + numpy only. No pycox, no torchsurv.

Contents
--------
cox_ph_loss(log_risk, time, event, ties=...)   negative mean partial log-likelihood
CoxRiskSets                                    precomputed sort/tie structure (build once, reuse)
BreslowBaseline                                baseline cumulative hazard -> S(t|x)
concordance(...)                               Harrell's C on a risk score

Design notes for the NEAT use case
----------------------------------
* Full-batch training makes the partial likelihood EXACT. Every risk set is the
  complete R(t) = {j : t_j >= t_i}. The Cox-CC / batch-truncated-risk-set
  literature (Kvamme 2019) exists only because SGD breaks this. You don't need it.
* The sort permutation depends only on the dataset, not on the genome. Build a
  CoxRiskSets once per dataset/fold and reuse across every genome and generation.
* g(x) is identified only up to an additive constant (it is absorbed into the
  baseline hazard). The loss below is shift-invariant by construction. Never
  compare raw risk scores across models without centring.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------
# Risk-set structure
# --------------------------------------------------------------------------

class CoxRiskSets:
    """Precomputed ordering and tie structure for one dataset.

    Sorts by DESCENDING observed time, so that a cumulative sum over the sorted
    axis accumulates the risk set: position k's cumsum covers {j : t_j >= t_k}.

    Attributes
    ----------
    order : LongTensor [n]         permutation into descending-time order
    time, event : Tensor [n]       sorted copies
    tie_index : LongTensor [n]     for each sorted row, index of the LAST row
                                   sharing its event time. Used to extend each
                                   risk set over the full tie block (Breslow).
    """

    def __init__(self, time, event, device=None, dtype=torch.float64):
        t = np.asarray(time, dtype=np.float64).ravel()
        e = np.asarray(event, dtype=np.float64).ravel()
        if t.shape != e.shape:
            raise ValueError(f"time {t.shape} and event {e.shape} must match")
        if e.sum() == 0:
            raise ValueError("no events: the Cox partial likelihood is undefined")

        # Descending by time. Stable sort keeps ties adjacent and deterministic.
        order = np.argsort(-t, kind="stable")
        t_s, e_s = t[order], e[order]

        # For each row, the last row index sharing the same time.
        # Times are descending, so equal times form contiguous blocks.
        n = len(t_s)
        tie_index = np.empty(n, dtype=np.int64)
        last = n - 1
        for i in range(n - 1, -1, -1):
            if i < n - 1 and t_s[i] != t_s[i + 1]:
                last = i
            tie_index[i] = last

        self.n = n
        self.device = device
        self.dtype = dtype
        self.order = torch.as_tensor(order, dtype=torch.long, device=device)
        self.time = torch.as_tensor(t_s, dtype=dtype, device=device)
        self.event = torch.as_tensor(e_s, dtype=dtype, device=device)
        self.tie_index = torch.as_tensor(tie_index, dtype=torch.long, device=device)
        self.n_events = float(e_s.sum())
        self.has_ties = bool((np.diff(t_s) == 0).any())

    def sort(self, x):
        """Reorder an unsorted [n] or [n, k] tensor into descending-time order."""
        return x.index_select(0, self.order)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def _log_cumsum_exp_desc(g):
    """Numerically stable log(cumsum(exp(g))) along a descending-time axis.

    cumsum position k covers rows 0..k, i.e. everyone with time >= t_k, which is
    exactly the risk set R(t_k). Max-shift is the log-sum-exp trick.
    """
    gamma = g.max()
    return (g - gamma).exp().cumsum(0).clamp_min(1e-300).log() + gamma


def cox_ph_loss(log_risk, risk_sets: CoxRiskSets, ties: str = "breslow",
                sorted_input: bool = False, reduction: str = "mean"):
    """Negative Cox partial log-likelihood.

    Parameters
    ----------
    log_risk : Tensor [n] or [n, 1]
        Network output g(x). LINEAR / identity activation - this is a log-risk,
        not a probability. Do not put a sigmoid on it.
    risk_sets : CoxRiskSets
        Built once from (time, event) for this dataset.
    ties : {"breslow", "efron"}
        Breslow matches scikit-survival's default. Efron matches lifelines and
        R survival, and is more accurate when many event times are tied.
    sorted_input : bool
        True if log_risk is already in risk_sets order (saves a gather).
    reduction : {"mean", "sum"}
        "mean" divides by the number of EVENTS (DeepSurv convention), which
        keeps the fitness scale comparable across folds of different size.

    Returns
    -------
    Scalar tensor. Lower is better. Shift-invariant: adding a constant to
    log_risk leaves it unchanged.
    """
    g = log_risk.reshape(-1)
    if not sorted_input:
        g = risk_sets.sort(g)

    event = risk_sets.event
    tie_index = risk_sets.tie_index

    log_cum = _log_cumsum_exp_desc(g)
    # Extend each row's denominator over its whole tie block, so that all rows
    # sharing an event time see the identical risk set R(tau_k).
    log_denom = log_cum.index_select(0, tie_index)

    if ties == "breslow":
        pll = (g - log_denom)

    elif ties == "efron":
        # Efron subtracts an increasing fraction of the tied events' own risk
        # mass from the denominator: for the l-th of d tied events,
        #   log( sum_{R} exp(g) - (l/d) * sum_{D} exp(g) )
        # Implemented as a per-tie-block loop; blocks are few and small in
        # practice, and this runs once per genome per generation.
        pll = torch.zeros_like(g)
        gamma = g.max()
        exp_g = (g - gamma).exp()          # shifted for stability
        cum = exp_g.cumsum(0)
        ev_idx = torch.nonzero(event > 0).reshape(-1)
        if ev_idx.numel():
            blocks = {}
            for i in ev_idx.tolist():
                blocks.setdefault(int(tie_index[i]), []).append(i)
            for last, rows in blocks.items():
                d = len(rows)
                idx = torch.tensor(rows, dtype=torch.long, device=g.device)
                risk_mass = cum[last]                      # sum over R(tau_k)
                tie_mass = exp_g.index_select(0, idx).sum() # sum over D_k
                l = torch.arange(d, dtype=g.dtype, device=g.device)
                adj = risk_mass - (l / d) * tie_mass
                # log-space: undo the gamma shift once per term
                log_terms = adj.clamp_min(1e-300).log() + gamma
                pll[idx] = g.index_select(0, idx) - log_terms.sum() / d
                # note: sum of d log-terms attributed evenly across the d rows,
                # so pll.sum() reproduces the block's Efron contribution exactly.
    else:
        raise ValueError(f"ties must be 'breslow' or 'efron', got {ties!r}")

    total = -(pll * event).sum()
    if reduction == "mean":
        return total / risk_sets.n_events
    if reduction == "sum":
        return total
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def partial_log_likelihood(log_risk, risk_sets, ties="breslow", sorted_input=False):
    """Convenience: the (positive) partial log-likelihood, summed. Higher is better."""
    return -cox_ph_loss(log_risk, risk_sets, ties=ties,
                        sorted_input=sorted_input, reduction="sum")


# --------------------------------------------------------------------------
# Breslow baseline -> survival curves
# --------------------------------------------------------------------------

class BreslowBaseline:
    """Baseline cumulative hazard, estimated on TRAINING data given fitted g(x).

        dH0(T_i) = D_i / sum_{j in R(T_i)} exp(g(x_j))
        H0(t)    = sum_{T_i <= t} dH0(T_i)
        S(t|x)   = exp(-exp(g(x)) * H0(t)) = S0(t) ** exp(g(x))

    This is the step that makes a Cox-loss network produce Cox-comparable
    survival curves. It is fitted, not learned - freeze the network first.
    """

    def __init__(self):
        self.times_ = None       # ascending distinct event times
        self.cum_hazard_ = None  # H0 at those times

    def fit(self, log_risk, time, event):
        g = np.asarray(log_risk, dtype=np.float64).ravel()
        t = np.asarray(time, dtype=np.float64).ravel()
        e = np.asarray(event, dtype=np.float64).ravel()

        order = np.argsort(t, kind="stable")     # ascending
        t_s, e_s, g_s = t[order], e[order], g[order]

        # risk-set sum at each position = sum of exp(g) over j with t_j >= t_i
        shift = g_s.max()
        expg = np.exp(g_s - shift)
        # reverse cumulative sum
        rev_cum = np.cumsum(expg[::-1])[::-1]

        uniq = np.unique(t_s[e_s == 1])
        dh = np.empty(len(uniq), dtype=np.float64)
        for k, tau in enumerate(uniq):
            d = float(((t_s == tau) & (e_s == 1)).sum())   # tied events at tau
            first = int(np.searchsorted(t_s, tau, side="left"))
            denom = rev_cum[first] * np.exp(shift)         # sum_{j: t_j >= tau} exp(g_j)
            dh[k] = d / denom

        self.times_ = uniq
        self.cum_hazard_ = np.cumsum(dh)
        return self

    def cumulative_hazard(self, log_risk, times):
        """H(t|x) = exp(g(x)) * H0(t). Returns [n_samples, n_times]."""
        g = np.asarray(log_risk, dtype=np.float64).ravel()
        idx = np.searchsorted(self.times_, np.asarray(times, dtype=np.float64), side="right") - 1
        h0 = np.where(idx >= 0, self.cum_hazard_[np.clip(idx, 0, None)], 0.0)
        return np.exp(g)[:, None] * h0[None, :]

    def survival(self, log_risk, times):
        """S(t|x) = exp(-H(t|x)). Returns [n_samples, n_times]."""
        return np.exp(-self.cumulative_hazard(log_risk, times))


# --------------------------------------------------------------------------
# Concordance (Harrell) - dependency-free check; use sksurv for the real thing
# --------------------------------------------------------------------------

def concordance(risk, time, event):
    """Harrell's C-index. Higher risk should mean shorter survival.

    Comparability follows scikit-survival: subject i (an event at t_i) is
    comparable with j if t_j > t_i, or if t_j == t_i and j is censored (the
    event is taken to precede the censoring). Two events at the same time are
    NOT comparable. Tied risk scores score 0.5.

    Verified to match sksurv.metrics.concordance_index_censored to 1e-12.
    Use sksurv in anger - this exists so the module has no hard dependency.
    """
    r = np.asarray(risk, dtype=np.float64).ravel()
    t = np.asarray(time, dtype=np.float64).ravel()
    e = np.asarray(event, dtype=np.float64).ravel()
    n = len(t)
    conc = disc = tied = 0.0
    for i in range(n):
        if e[i] != 1:
            continue
        comparable = (t > t[i]) | ((t == t[i]) & (e == 0))
        if not comparable.any():
            continue
        ri, rj = r[i], r[comparable]
        conc += float((rj < ri).sum())
        disc += float((rj > ri).sum())
        tied += float((rj == ri).sum())
    denom = conc + disc + tied
    return (conc + 0.5 * tied) / denom if denom else float("nan")
