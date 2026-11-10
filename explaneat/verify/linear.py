"""How much of the model's performance is non-linear: switching units replaced by affine fits.

* ``level="parts"`` (default): each of the root's children (or each leaf, without a single
  root) is replaced by the least-squares affine fit of its exits on its raw entries, alone and
  all together; the restore gain of a part = AUC with only that part's non-linearity restored
  onto the all-linear model, minus the all-linear AUC (``explanation_tools.performance_sources``).
* ``level="nodes"``: every hidden ReLU node that switches on the data (its pre-activation takes
  both signs) is replaced by the affine fit of its value on its inputs' values.
* ``core=True``: also the **linear core alone**: every part of the root fixed at its training
  mean (only the root's direct edges vary).

The fits are made on the training rows; the AUCs are on ``split`` (train by default).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _split(vm, split: str):
    m = vm.m
    if split == "test":
        if m.Xt is None:
            raise ValueError("no test split")
        return m.Xt, m.yt
    return m.X, m.y


def switching_nodes(vm) -> List[str]:
    from ..core.genome_network import NodeType
    m = vm.m
    pre = m.ev.pre_activations(m.X)
    out = []
    for n in vm.structure.nodes:
        if n.type != NodeType.HIDDEN or n.activation == "identity" or n.id not in pre:
            continue
        z = pre[n.id]
        if np.min(z) < -1e-12 and np.max(z) > 1e-12:
            out.append(n.id)
    return out


def linearise(vm, level: str = "parts", core: bool = False, split: str = "train") -> Dict[str, Any]:
    from ..analysis.composition import annotation_dicts_from_operations
    from ..analysis.explanation_tools import Linearisation, as_unit, default_auc
    m = vm.m
    X, y = _split(vm, split)
    ev = m.ev
    if level == "nodes":
        units = {}
        for nid in switching_nodes(vm):
            preds = [p for p, _ in ev.preds[nid]]
            units[nid] = {"entry_nodes": preds, "exit_nodes": [nid]}
        scope = "hidden ReLU nodes that switch on the training data"
    elif level == "parts":
        anns = annotation_dicts_from_operations(vm.operations)
        root = next((p for p in vm.parts if p.is_root), None)
        names = list(root.children) if root is not None and root.children else \
            [p.name for p in vm.parts if p.kind == "leaf"]
        by_name = {a["name"]: a for a in anns}
        units = {n: as_unit(by_name[n], anns, vm.structure) for n in names if n in by_name}
        scope = "children of the root" if root is not None and root.children else "leaf annotations"
    else:
        raise ValueError("--level parts | nodes")
    lin = Linearisation(ev, units, m.X)          # fitted on the training rows
    nl = lin.nonlinear_units

    def auc(overrides=None):
        return default_auc(y, ev.logit(X, overrides))

    full = auc()
    all_lin = auc(lin.overrides(nl)) if nl else full
    table = {}
    for name in units:
        f = lin.fits[name]
        alone = auc(lin.overrides([name]))
        restore = auc(lin.overrides([n for n in nl if n != name])) if name in nl else all_lin
        table[name] = {"switches": f.nonlinear, "min_r2_fit": min(f.r2) if f.r2 else 1.0,
                       "max_abs_residual": f.max_abs_residual,
                       "auc_linearised_alone": alone, "drop_linearised_alone": full - alone,
                       "auc_restored_alone": restore, "restore_gain": (restore - all_lin) if name in nl else 0.0}
    res = {"genome_id": vm.genome_id, "split": split, "level": level, "scope": scope,
           "n_rows": int(X.shape[0]), "auc_full": full, "auc_all_affine": all_lin,
           "nonlinear_gain": full - all_lin, "switching_units": nl, "units": table}
    if core:
        vals = m._vals_train()
        root = next((p for p in vm.parts if p.is_root), None)
        kids = [p for p in vm.parts if root is not None and p.name in (root.children or [])]
        fix = {x: float(np.mean(vals[x])) for p in kids for x in p.exits}
        n = X.shape[0]
        ov = {x: ((), (lambda v, k=k, n=n: np.full(n, k))) for x, k in fix.items()}
        res["auc_linear_core_alone"] = auc(ov) if ov else full
        res["linear_core_note"] = ("every part of the root fixed at its training mean: only the inputs that "
                                   "feed the score directly vary")
    return res
