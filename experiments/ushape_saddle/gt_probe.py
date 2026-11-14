"""Ground-truth probe of the explained parts (read-only): for each explained genome, the training-row
contribution of a set of parts (logit with the parts switched off, subtracted from the full logit)
against the true nonlinear terms and ad-hoc concepts. R^2 of an affine fit a*concept + b.

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/ushape_saddle/gt_probe.py
Writes gt_probe.json.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

def r2(c, y, extra=()):
    X = np.column_stack([c, *extra, np.ones_like(c)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(1 - res.var() / y.var()) if y.var() > 0 else None, beta.tolist()

def main():
    from explaneat.verify.model import load
    from explaneat.verify.probe import to_z
    out = {}
    for lab in ("shape-A", "shape-B", "shape-C"):
        rec = json.loads((HERE / "explanations" / f"{lab}.json").read_text())
        vm = load(rec["genome_id"])
        names = list(vm.base.feature_names)
        raw = vm.m.raw_train
        col = {n: i for i, n in enumerate(names)}
        Z = to_z(vm, raw)
        full = vm.m.logit(Z)
        hr, sbp, na = raw[:, col["heart_rate"]], raw[:, col["sbp"]], raw[:, col["sodium"]]
        zh, zp, zn = (hr - 76) / 13, (sbp - 132) / 18, (na - 139) / 4
        concepts = {"heart_rate * sbp": (hr * sbp, ()), "-(zh*zp) (the true saddle term)": (-zh * zp, ()),
                    "zh*zp with zh, zp linear terms": (-zh * zp, (zh, zp)),
                    "(sodium - 137)^2 (the true U term)": ((zn + 0.5) ** 2, ()),
                    "abs(sodium - 137)": (np.abs(na - 137), ())}
        parts = {p.name: p for p in vm.parts if not p.is_root and p.exits}
        res = {"parts": {}}
        def contrib(ps):
            ab = {}
            for p in ps:
                ab.update(p.baselines)
            return full - vm.m.logit(Z, ablate=ab)
        groups = {n: [p] for n, p in parts.items()}
        hrsbp = [p for n, p in parts.items() if n in {"pressure_rate_load", "pressure_over_rate", "rate_bmi_amplifier",
                 "tachycardia_low_sbp_gate", "hypertension_low_hr_gate", "high_sbp_and_hr_penalty", "sbp_hinge"}]
        if lab == "shape-B":
            groups["saddle parts (pressure_rate_load + pressure_over_rate)"] = [parts["pressure_rate_load"], parts["pressure_over_rate"]]
            groups["all heart-rate/sbp hinges (+ rate_bmi_amplifier)"] = hrsbp
        if lab == "shape-C":
            groups["the three gates (bp_hr_saddle without sbp_hinge)"] = [parts[n] for n in ("tachycardia_low_sbp_gate", "hypertension_low_hr_gate", "high_sbp_and_hr_penalty")]
        for g, ps in groups.items():
            y = contrib(ps)
            if y.var() < 1e-12:
                continue
            res["parts"][g] = {k: round(r2(c, y, e)[0], 3) for k, (c, e) in concepts.items()}
        out[lab] = res
    (HERE / "gt_probe.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
