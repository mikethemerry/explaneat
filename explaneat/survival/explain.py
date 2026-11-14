"""Read an explained survival network in hazard-ratio terms.

With an identity output, the network output IS the log-risk:

    g(x) = b0 + sum_k v_k * a_k(x),        h(t | x) = h0(t) * exp(g(x))

where a_k are the activations feeding the output node (inputs on direct edges,
hidden ReLU units, or the exits of annotations) and v_k the output weights.
So every additive term is a log-hazard-ratio contribution, and
exp(v_k * (a_k(x1) - a_k(x2))) is the hazard ratio it contributes between two
patients, constant over follow-up (the proportional-hazards assumption is
inherited from the Cox loss, not tested by it).  b0 is unidentifiable (it is
absorbed into the baseline hazard): only differences in g are meaningful.

Helpers:

* ``output_terms(structure)``: the additive decomposition of g at the output.
* ``direct_edge_reading``: log-HR per SD / per raw unit of a direct input edge.
* ``hinge_reading``: a ReLU unit relu(sum w_i x_i + b) with output weight v
  adds log-HR v * (w.x + b) where w.x > -b and nothing elsewhere; for a single
  input this is a threshold on that input, reported in raw units.
* ``contribution_summary``: the spread of one term over a sample, as hazard
  ratios against the sample median.

``scaler`` (optional) is ``{"mean": [...], "scale": [...]}`` from the split, so
thresholds and per-unit effects can be quoted in the original feature units;
input node ``-k`` is feature column ``k - 1``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


def _input_col(node_id: str) -> Optional[int]:
    try:
        k = int(str(node_id).split("_")[0])
    except ValueError:
        return None
    return -k - 1 if k < 0 else None


def _is_identity_relay(node) -> bool:
    """An identity node inserted by ``add_identity_node`` (bias 0, identity activation): it
    passes its weighted sum through, so its inputs are terms of g with the weights multiplied."""
    return (node is not None and node.type.value != "input" and node.type.value != "output"
            and str(node.activation or "").lower() == "identity" and float(node.bias or 0.0) == 0.0)


def output_terms(structure) -> Dict[str, object]:
    """g(x) = bias + sum(weight * activation(from_node)) at the (single) output node.

    Identity relay nodes between the terms and the output (an explained model state after
    ``add_identity_node`` on the output) are expanded, so the terms are the real readout units
    with their effective weights; ``via`` names the relay(s) a term passed through.
    """
    out_id = structure.output_node_ids[0]
    out = structure.get_node_by_id(out_id)
    terms: List[Dict[str, object]] = []

    def expand(to_id: str, scale: float, via: List[str]) -> None:
        for c in structure.connections:
            if not (c.enabled and c.to_node == to_id):
                continue
            node = structure.get_node_by_id(c.from_node)
            if _is_identity_relay(node):
                expand(c.from_node, scale * float(c.weight), via + [c.from_node])
            else:
                t = {"from": c.from_node, "weight": scale * float(c.weight)}
                if via:
                    t["via"] = list(via)
                terms.append(t)

    expand(out_id, 1.0, [])
    return {"output": out_id, "activation": out.activation, "bias": float(out.bias or 0.0),
            "terms": sorted(terms, key=lambda t: -abs(t["weight"]))}


def direct_edge_reading(node_id: str, weight: float, scaler=None, feature_names=None) -> Dict[str, object]:
    """A direct input -> output edge: log-HR = weight per (standardised) unit."""
    col = _input_col(node_id)
    out = {"node": node_id, "log_hr_per_sd": weight, "hr_per_sd": math.exp(weight)}
    if col is not None and feature_names:
        out["feature"] = feature_names[col]
    if col is not None and scaler:
        sd = float(scaler["scale"][col])
        out["log_hr_per_unit"] = weight / sd
        out["hr_per_unit"] = math.exp(weight / sd)
    return out


def hinge_reading(structure, hidden_id: str, out_weight: float, scaler=None,
                  feature_names=None) -> Dict[str, object]:
    """A ReLU unit h = relu(sum_i w_i x_i + b) feeding the output with weight v."""
    node = structure.get_node_by_id(hidden_id)
    b = float(node.bias or 0.0)
    ins = [(c.from_node, float(c.weight)) for c in structure.connections
           if c.enabled and c.to_node == hidden_id]
    out = {"node": hidden_id, "bias": b, "inputs": [{"from": f, "weight": w} for f, w in ins],
           "output_weight": out_weight,
           "reading": f"adds log-HR {out_weight:+.3g} * ({' + '.join(f'{w:.3g}*[{f}]' for f, w in ins)} "
                      f"{b:+.3g}) where that sum is positive, 0 elsewhere"}
    if len(ins) == 1:
        f, w = ins[0]
        thr_std = -b / w
        slope = out_weight * w  # log-HR per SD of the input, on the active side
        side = "above" if w > 0 else "below"
        out.update(single_input=f, threshold_sd=thr_std, active_side=side,
                   log_hr_per_sd_active=slope, hr_per_sd_active=math.exp(slope))
        col = _input_col(f)
        if col is not None and feature_names:
            out["feature"] = feature_names[col]
        if col is not None and scaler:
            mu, sd = float(scaler["mean"][col]), float(scaler["scale"][col])
            out["threshold_raw"] = mu + sd * thr_std
            out["log_hr_per_unit_active"] = slope / sd
            out["reading"] = (f"adds log-HR {slope / sd:+.4g} per unit of "
                              f"{out.get('feature', f)} {side} {mu + sd * thr_std:.4g}, nothing {('below' if side == 'above' else 'above')}")
    return out


def contribution_summary(values, weight: float = 1.0) -> Dict[str, float]:
    """Spread of one additive log-risk term c = weight * a(x) over a sample, as HRs vs its median."""
    c = weight * np.asarray(values, dtype=np.float64).ravel()
    med = float(np.median(c))
    q = {p: float(np.percentile(c, p)) for p in (5, 25, 75, 95)}
    return {
        "log_hr_sd": float(c.std()),
        "fraction_nonzero": float(np.mean(np.abs(c) > 1e-12)),
        "hr_p5_vs_median": math.exp(q[5] - med), "hr_p25_vs_median": math.exp(q[25] - med),
        "hr_p75_vs_median": math.exp(q[75] - med), "hr_p95_vs_median": math.exp(q[95] - med),
        "hr_p95_vs_p5": math.exp(q[95] - q[5]),
    }


def explain_output(structure, X_std: np.ndarray, scaler=None, feature_names=None) -> List[Dict[str, object]]:
    """Hazard-ratio reading of every term at the output, with its spread on ``X_std``."""
    import torch

    from explaneat.core.structure_network import StructureNetwork

    sn = StructureNetwork(structure)
    sn.override_output_activation(sn.model_output_activation)
    sn.override_hidden_activation(sn.model_hidden_activation)
    sn.forward(torch.as_tensor(X_std, dtype=torch.float64))
    info = output_terms(structure)
    readings = []
    for t in info["terms"]:
        node = structure.get_node_by_id(t["from"])
        acts = sn.get_node_activation(t["from"])
        if node.type.value == "input":
            r = direct_edge_reading(t["from"], t["weight"], scaler, feature_names)
        else:
            r = hinge_reading(structure, t["from"], t["weight"], scaler, feature_names)
        r["spread"] = contribution_summary(acts, t["weight"])
        readings.append(r)
    return readings
