"""Experiment 2 linearity claims with the binary one-step rule (read-only on the database).

For each of the 30 Experiment 2 networks (adult, churn100, heart100; seeds 0-9) this rebuilds the
plain-language model exactly as ``plain-language extract`` does (``load_model``, ``build_parts``)
and computes the linearity claims twice: without the binary one-step kind (``before``: must
reproduce the frozen v4 layer's linearity claims) and with it (``after``). Nothing is written to the
database or to any v4 file; the output is ``experiments/10seed/linearity_v2/<label>.json`` and
``summary.json``.

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 \\
    uv run python experiments/10seed/linearity_v2.py [labels...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "linearity_v2"
V4 = HERE / "plain_language_v4"
STUDIES = ("adult", "churn100", "heart100")
CLASSES = ("exact_linear", "support_linear", "mixed", "nonlinear")


def _count(d):
    return {k: sum(1 for v in d.values() if v == k) for k in CLASSES}


def _model_stmt(claims):
    return next(c["statement"] for c in claims if c["type"] == "linearity" and c["statement"]["kind"] == "model")


def _var_classes(claims):
    return {c["statement"]["variable"]: c["statement"]["class"] for c in claims
            if c["type"] == "linearity" and c["statement"]["kind"] == "variable"}


def run(label: str) -> dict:
    from explaneat.analysis.plain_language import build_parts, load_model
    from explaneat.analysis.plain_language_linearity import linearity_claims
    from explaneat.export.plain_cli import genome_for_label
    v4 = json.loads((V4 / f"{label}.json").read_text())
    gid = genome_for_label(label)
    assert gid == v4["genome_id"], (label, gid, v4["genome_id"])
    m, meta = load_model(gid)
    assert meta["explanation_id"] == v4["explanation_id"], label
    parts = build_parts(m)
    names = {p.id: p.name for p in parts}
    cb, gb = linearity_claims(m, parts, binary_step=False)
    ca, ga = linearity_claims(m, parts, binary_step=True)
    v4lin = v4["graph"]["linearity"]
    v4m = _model_stmt(v4["claims"])
    bm, am = _model_stmt(cb), _model_stmt(ca)
    reproduces = {
        "variables": _var_classes(cb) == _var_classes(v4["claims"]),
        "parts": gb["parts"] == v4lin["parts"],
        "units": gb["units"] == v4lin["units"],
        "variance_share_linear": abs(float(bm["variance_share_linear"]) - float(v4m["variance_share_linear"])) < 1e-5,
    }
    model_claim_after = next(c for c in ca if c["type"] == "linearity" and c["statement"]["kind"] == "model")
    reclassified = sorted(n for n, k in ga["units"].items() if k == "binary_step")
    became_linear = sorted(p for p, k in ga["parts"].items() if gb["parts"].get(p) == "nonlinear" and k != "nonlinear")
    var_changes = {b: [gb["variables"].get(b), k] for b, k in ga["variables"].items() if gb["variables"].get(b) != k}
    reach = m.reaching_output()
    other_affine = _other_affine(m, ga["units"], set(ga.get("one_group_exact", {})), reach)
    rec = {
        "label": label, "genome_id": gid, "explanation_id": meta["explanation_id"],
        "n_train": int(m.X.shape[0]),
        "before_reproduces_v4": reproduces,
        "units_before": _count_units(gb["units"], reach),
        "units_after": _count_units(ga["units"], reach),
        "units_reclassified": reclassified,
        "binary_step": ga.get("binary_step", {}),
        "binary_step_check_failed": [n for n, d in (ga.get("binary_step") or {}).items() if not d["passed"]],
        "one_group_exact_not_reclassified": ga.get("one_group_exact", {}),
        "other_exact_affine_not_reclassified": other_affine,
        "variables": {"v4": _var_classes(v4["claims"]), "before": gb["variables"], "after": ga["variables"]},
        "variables_count": {"v4": _count(_var_classes(v4["claims"])), "before": _count(gb["variables"]),
                            "after": _count(ga["variables"])},
        "variable_changes": var_changes,
        "parts": {p: {"name": names.get(p), "v4": v4lin["parts"].get(p), "before": gb["parts"].get(p),
                      "after": ga["parts"].get(p)} for p in ga["parts"]},
        "parts_count": {"v4": _count(v4lin["parts"]), "before": _count(gb["parts"]), "after": _count(ga["parts"])},
        "parts_became_linear_on_data": [{"id": p, "name": names.get(p)} for p in became_linear],
        "model": {"v4": v4m, "before": bm, "after": am},
        "model_evidence_after": model_claim_after["evidence"],
        "model_checks_after": model_claim_after["checks"],
        "linearity_claims_after": ca,
    }
    return rec


def _other_affine(m, units, skip, reach):
    """Switching units (after the rule), other than the one-group ones, whose activation is still
    exactly affine (max error < 1e-9) in the data columns that reach them on the training rows:
    e.g. a unit downstream of a unit fed by the same yes/no input. For review only."""
    import numpy as np
    from explaneat.analysis.explanation_tools import input_column
    from explaneat.core.genome_network import NodeType
    vals = m.ev.forward(m.X)
    anc = {}
    for n in m.ev._order({}):
        if m.nodes[n].type == NodeType.INPUT:
            anc[n] = {input_column(n)}
        else:
            anc[n] = set().union(*[anc[p] for p, w in m.ev.preds[n] if w != 0] or [set()])
    out = {}
    for n, k in units.items():
        if k != "switching" or n in skip or n not in reach:
            continue
        cols = sorted(anc[n])
        A = np.column_stack([np.ones(m.X.shape[0])] + [m.X[:, c] for c in cols])
        coef, *_ = np.linalg.lstsq(A, vals[n], rcond=None)
        err = float(np.max(np.abs(A @ coef - vals[n])))
        if err < 1e-9:
            out[n] = {"columns": [m.feature_names[c] for c in cols], "max_affine_error": err,
                      "direct_inputs": [p for p, w in m.ev.preds[n] if w != 0]}
    return out


def _count_units(units, reach):
    out = {}
    for n, k in units.items():
        if n in reach:
            out[k] = out.get(k, 0) + 1
    return out


def main(argv):
    if "explaneat_v4" not in os.environ.get("DATABASE_URL", ""):
        sys.exit("set DATABASE_URL=postgresql://localhost/explaneat_v4 (read-only use)")
    if not os.environ.get("EXPLANEAT_RECORDS_DIR"):
        sys.exit("set EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4")
    labels = argv or [f"{s}-s{i}" for s in STUDIES for i in range(10)]
    OUT.mkdir(exist_ok=True)
    summary = {}
    for lab in labels:
        rec = run(lab)
        (OUT / f"{lab}.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + "\n")
        st_b, st_a = rec["model"]["before"], rec["model"]["after"]
        summary[lab] = {k: rec[k] for k in ("before_reproduces_v4", "variables_count", "parts_count",
                                            "parts_became_linear_on_data", "variable_changes",
                                            "units_before", "units_after", "binary_step_check_failed")}
        summary[lab].update({
            "units_reclassified": len(rec["units_reclassified"]),
            "one_group_exact_not_reclassified": len(rec["one_group_exact_not_reclassified"]),
            "other_exact_affine_not_reclassified": len(rec["other_exact_affine_not_reclassified"]),
            "variance_share_linear": [st_b["variance_share_linear"], st_a["variance_share_linear"]],
            "auc_linearised_train": [st_b["auc_linearised_train"], st_a["auc_linearised_train"]],
            "auc_linearised_test": [st_b.get("auc_linearised_test"), st_a.get("auc_linearised_test")],
            "binary_step_linearisation_max_diff": rec["model_evidence_after"].get("binary_step_linearisation_max_diff"),
            "binary_step_max_affine_error": rec["model_evidence_after"].get("binary_step_max_affine_error"),
        })
        print(lab, summary[lab]["units_reclassified"], summary[lab]["variance_share_linear"],
              all(rec["before_reproduces_v4"].values()), flush=True)
    if not argv:
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
