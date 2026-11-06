"""Cox partial-likelihood machinery in PyTorch, sized for ExplaNEAT / PropNEAT.

Packaged from the verified reference implementation delivered with the
2026-08-20 survival brief (``docs/survival-in-explaneat.md``; reference copy
in ``experiments/survival/reference/cox_torch.py``).  torch + numpy only.

Contents
--------
CoxRiskSets                    precomputed sort / tie structure (build once per fold)
cox_ph_loss(g, risk_sets, ...) negative mean partial log-likelihood (Breslow or Efron)
partial_log_likelihood(...)    the positive, summed partial log-likelihood
null_cox_loss(risk_sets, ...)  the loss of a constant-output model (g = 0): the fitness floor

Design notes for the NEAT use case
----------------------------------
* Full-batch training makes the partial likelihood EXACT: every risk set is the
  complete R(t) = {j : t_j >= t}.  The batch-truncated risk-set literature
  (Kvamme et al. 2019) exists only because SGD breaks this.
* The sort permutation depends only on the data, never on the genome.  Build a
  ``CoxRiskSets`` once per dataset **fold** (from that fold's training rows only)
  and reuse it across every genome and generation.  Never build it on the full
  data and subset: that puts test rows into the training denominators.
* g(x) is identified only up to an additive constant (absorbed into the
  baseline hazard).  The loss is shift-invariant by construction.  Never compare
  raw risk scores across two fitted models without centring.

Change from the reference: the Efron correction is vectorised (one scatter-add
instead of a Python loop over tie blocks), because it is evaluated once per
genome per backprop epoch.  Per-row attribution inside a tie block differs from
the reference (which spread the block's log-terms evenly), but the summed loss
is identical; ``tests/test_survival`` checks it against lifelines and a naive
reference to 1e-8.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------
# Risk-set structure
# --------------------------------------------------------------------------


class CoxRiskSets:
    """Precomputed ordering and tie structure for one dataset (or fold).

    Rows are sorted by DESCENDING observed time, so a cumulative sum over the
    sorted axis accumulates the risk set: position k's cumsum covers
    {j : t_j >= t_k} once extended to the end of k's tie block.

    Attributes
    ----------
    order : LongTensor [n]       permutation into descending-time order
    time, event : Tensor [n]     sorted copies
    tie_index : LongTensor [n]   for each sorted row, the index of the LAST row
                                 sharing its time (extends the risk set over
                                 the whole tie block)
    event_pos : LongTensor [m]   sorted positions of the m events
    event_block : LongTensor [m] tie-block id of each event (0..n_blocks-1)
    event_rank : Tensor [m]      0..d-1 rank of each event within its block
    event_d : Tensor [m]         number of tied events in the event's block
    """

    def __init__(self, time, event, device=None, dtype=torch.float64):
        t = np.asarray(time, dtype=np.float64).ravel()
        e = np.asarray(event, dtype=np.float64).ravel()
        if t.shape != e.shape:
            raise ValueError(f"time {t.shape} and event {e.shape} must match")
        if not np.all(np.isin(e, (0.0, 1.0))):
            raise ValueError("event must be 0 (censored) or 1 (event)")
        if e.sum() == 0:
            raise ValueError("no events: the Cox partial likelihood is undefined")
        if not np.all(np.isfinite(t)):
            raise ValueError("time contains non-finite values")

        # Descending by time. Stable sort keeps ties adjacent and deterministic.
        order = np.argsort(-t, kind="stable")
        t_s, e_s = t[order], e[order]

        # For each row, the last row index sharing the same time.
        n = len(t_s)
        tie_index = np.empty(n, dtype=np.int64)
        last = n - 1
        for i in range(n - 1, -1, -1):
            if i < n - 1 and t_s[i] != t_s[i + 1]:
                last = i
            tie_index[i] = last

        # Efron structure: per event, its block, its rank inside the block and
        # the block's event count.
        event_pos = np.flatnonzero(e_s > 0)
        blocks_of_events = tie_index[event_pos]
        uniq_blocks, block_id = np.unique(blocks_of_events, return_inverse=True)
        d_per_block = np.bincount(block_id)
        rank = np.zeros(len(event_pos), dtype=np.float64)
        seen = np.zeros(len(uniq_blocks), dtype=np.int64)
        for k, b in enumerate(block_id):
            rank[k] = seen[b]
            seen[b] += 1

        self.n = n
        self.device = device
        self.dtype = dtype
        self.order = torch.as_tensor(order, dtype=torch.long, device=device)
        self.time = torch.as_tensor(t_s, dtype=dtype, device=device)
        self.event = torch.as_tensor(e_s, dtype=dtype, device=device)
        self.tie_index = torch.as_tensor(tie_index, dtype=torch.long, device=device)
        self.event_pos = torch.as_tensor(event_pos, dtype=torch.long, device=device)
        self.event_block = torch.as_tensor(block_id, dtype=torch.long, device=device)
        self.event_block_last = torch.as_tensor(blocks_of_events, dtype=torch.long, device=device)
        self.event_rank = torch.as_tensor(rank, dtype=dtype, device=device)
        self.event_d = torch.as_tensor(d_per_block[block_id], dtype=dtype, device=device)
        self.n_blocks = int(len(uniq_blocks))
        self.n_events = float(e_s.sum())
        self.has_ties = bool((np.diff(t_s) == 0).any())

    def sort(self, x):
        """Reorder an unsorted [n] or [n, k] tensor into descending-time order."""
        return x.index_select(0, self.order)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


def _log_cumsum_exp_desc(g):
    """Numerically stable log(cumsum(exp(g))) along the descending-time axis."""
    gamma = g.max()
    return (g - gamma).exp().cumsum(0).clamp_min(1e-300).log() + gamma


def cox_ph_loss(log_risk, risk_sets: CoxRiskSets, ties: str = "efron",
                sorted_input: bool = False, reduction: str = "mean"):
    """Negative Cox partial log-likelihood.

    Parameters
    ----------
    log_risk : Tensor [n] or [n, 1]
        Network output g(x). IDENTITY output activation: this is a log-risk,
        not a probability.  Do not put a sigmoid on it.
    risk_sets : CoxRiskSets
        Built once from (time, event) for this dataset / fold.
    ties : {"efron", "breslow"}
        Efron matches lifelines and R ``survival`` and is the default here
        (and the brief's recommendation).  Breslow matches scikit-survival's
        default.  Use the same choice for the comparator Cox model.
    sorted_input : bool
        True if log_risk is already in risk_sets order.
    reduction : {"mean", "sum"}
        "mean" divides by the number of EVENTS (DeepSurv convention), which
        keeps the fitness scale comparable across folds of different size.

    Returns a scalar tensor; lower is better; shift-invariant.
    """
    g = log_risk.reshape(-1)
    if g.shape[0] != risk_sets.n:
        raise ValueError(f"log_risk has {g.shape[0]} rows, risk sets have {risk_sets.n}")
    if not sorted_input:
        g = risk_sets.sort(g)

    if ties == "breslow":
        log_cum = _log_cumsum_exp_desc(g)
        log_denom = log_cum.index_select(0, risk_sets.tie_index)
        total = -((g - log_denom) * risk_sets.event).sum()
    elif ties == "efron":
        # For the l-th of d tied events at tau:
        #   log( sum_{R(tau)} exp(g) - (l/d) * sum_{D(tau)} exp(g) )
        gamma = g.max()
        exp_g = (g - gamma).exp()
        cum = exp_g.cumsum(0)
        pos = risk_sets.event_pos
        risk_mass = cum.index_select(0, risk_sets.event_block_last)
        ev_exp = exp_g.index_select(0, pos)
        tie_mass_block = torch.zeros(risk_sets.n_blocks, dtype=g.dtype, device=g.device)
        tie_mass_block = tie_mass_block.index_add(0, risk_sets.event_block, ev_exp)
        tie_mass = tie_mass_block.index_select(0, risk_sets.event_block)
        adj = risk_mass - (risk_sets.event_rank / risk_sets.event_d) * tie_mass
        log_terms = adj.clamp_min(1e-300).log() + gamma
        total = -(g.index_select(0, pos) - log_terms).sum()
    else:
        raise ValueError(f"ties must be 'breslow' or 'efron', got {ties!r}")

    if reduction == "mean":
        return total / risk_sets.n_events
    if reduction == "sum":
        return total
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def partial_log_likelihood(log_risk, risk_sets, ties="efron", sorted_input=False):
    """The (positive) partial log-likelihood, summed.  Higher is better."""
    return -cox_ph_loss(log_risk, risk_sets, ties=ties,
                        sorted_input=sorted_input, reduction="sum")


def null_cox_loss(risk_sets: CoxRiskSets, ties: str = "efron") -> float:
    """Loss of a constant-output model (g = 0): the "no signal found" floor.

    With Breslow ties this is exactly the mean over events of log|R(t_i)|.
    A genome with fitness (= -loss) at or below ``-null_cox_loss`` has learned
    nothing.  Use it to set ``fitness_threshold`` relative to the null model.
    """
    g = torch.zeros(risk_sets.n, dtype=risk_sets.dtype, device=risk_sets.device)
    return float(cox_ph_loss(g, risk_sets, ties=ties, sorted_input=True))
