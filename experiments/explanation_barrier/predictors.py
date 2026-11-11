"""Which complexity measure tracks the outcome? (descriptive; 15 networks, no inference)

    uv run python experiments/explanation_barrier/predictors.py   # after summarize.py

For each measure: Spearman rho with the verifier's leaf meaning score, with granularity and with
explaining minutes, over the attempted networks; and whether a single threshold on the measure
separates the "covered, not decomposed" networks from the explained ones. Writes predictors.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
rows = [r for r in json.loads((HERE / "ladder.json").read_text()) if r.get("attempted")]
widths = {r["label"]: int(r["arch"].split("x")[0]) for r in rows}
depth = {r["label"]: len(r["arch"].split("x")) for r in rows}
measures = {
    "n_params": lambda r: r["n_params"],
    "hidden_units": lambda r: r["hidden"],
    "first_layer_width (input fan-out)": lambda r: widths[r["label"]],
    "hidden_layers": lambda r: depth[r["label"]],
    "activation_patterns": lambda r: r["patterns"],
    "interaction_strength (1 - R2 additive)": lambda r: r["interaction_strength"],
    "order3plus_residual (1 - R2 pairwise)": lambda r: r["order3plus"],
    "1 - R2 linear": lambda r: 1 - r["r2_linear"],
    "linear_auc_gap": lambda r: r["linear_auc_gap"],
}
fail = np.array([r["outcome"] != "explained" for r in rows])
out = {}
for name, f in measures.items():
    x = np.array([f(r) for r in rows], float)
    res = {}
    for tgt in ("meaning_score_verifier", "granularity", "minutes"):
        y = np.array([r.get(tgt) if r.get(tgt) is not None else np.nan for r in rows], float)
        ok = ~np.isnan(y)
        rho, p = spearmanr(x[ok], y[ok])
        res[f"rho_{tgt}"] = round(float(rho), 2)
    res["separates_failures"] = bool(x[fail].min() > x[~fail].max())
    res["failures_min"], res["explained_max"] = float(x[fail].min()), float(x[~fail].max())
    out[name] = res
(HERE / "predictors.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"{'measure':42s} rho(meaning) rho(granularity) rho(minutes) separates  fail_min  expl_max")
for k, v in out.items():
    print(f"{k:42s} {v['rho_meaning_score_verifier']:>8} {v['rho_granularity']:>12} {v['rho_minutes']:>12}"
          f"  {str(v['separates_failures']):>9} {v['failures_min']:>9g} {v['explained_max']:>9g}")
