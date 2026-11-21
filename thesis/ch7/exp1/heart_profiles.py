"""Heart, original network: candidate patient profiles for the cholesterol figure (PROPOSAL data; read-only DB).

    uv run python thesis/ch7/exp1/heart_profiles.py        # -> data/heart_profiles.json

Feeds thesis/ch7/exp1/HEART_CHOLESTEROL_PROPOSAL.md (the definitions put to Mike before the figure is built).
No figure is drawn here.

The figure would show cholesterol's effect on the original Heart network's logit for a benign-profile and a
high-risk patient, through the two mechanisms of the explanation (experiments/10seed/plain_language/
heart-paper.json, parts benign_profile_gate and risk_aggregation_gate): the direct linear cholesterol terms
(the edge to the output and, through restbp_chol_interaction's always-on unit 2379, a second linear path) and
the risk-aggregation gate (unit 534, read out with a negative weight) that switches off as cholesterol rises.

Profiles (design choices, from the inputs the explanation names, not fitted):
  * subgroups of the training patients by the benign-profile gate's two main inputs (sex, exercise-induced
    angina) and the 534 gate's largest other input (number of vessels, ca):
      benign:    female, no exercise-induced angina, ca = 0
      high risk: male, exercise-induced angina, ca >= 1
  * each profile is either
      (A) the subgroup's median patient: every numeric input at the subgroup's training median, every
          category at its most common level in the subgroup (ties: the lower code), or
      (B) the typical patient of the plain-language layer (median / most common over all training patients)
          with only sex, exercise-induced angina and ca set to the subgroup's values (ca: the subgroup median);
  * cholesterol swept over the training range on a grid of N_GRID points (raw mg/dL), the other inputs fixed.
For each profile: n training patients in the subgroup, their disease rate, the share with unit 534 active at
their own values, the profile's values, the logit along the grid split into the direct cholesterol terms, the
gate's term (-w * a534) and the rest, and where the gate switches off.

Read-only: compute_runs.ReadOnlySession and PGOPTIONS default_transaction_read_only=on. Run it twice: the
output is byte-identical.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E = REPO / "experiments"
for p in (str(REPO), str(E / "10seed")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_dev")
os.environ["PGOPTIONS"] = (os.environ.get("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()

LAYER = E / "10seed/plain_language/heart-paper.json"
OUT = HERE / "data" / "heart_profiles.json"
N_GRID = 301          # design choice: cholesterol grid points over the training range
GATE, OUTPUT = "534", "0"
SUBGROUPS = {          # design choice: column -> required raw value (ca: a range)
    "benign": {"sex:0": 1.0, "exang:0": 1.0, "ca": (0.0, 0.0)},
    "high_risk": {"sex:1": 1.0, "exang:1": 1.0, "ca": (1.0, 3.0)},
}


def load():
    import compute_runs as cr
    from explaneat.db.models import Dataset, DatasetSplit
    cr.db.init_db(os.environ["DATABASE_URL"])
    layer = json.loads(LAYER.read_text())
    with cr.ReadOnlySession() as s:
        g = s.get(cr.Genome, uuid.UUID(layer["genome_id"]))
        exp = g.population.experiment
        cfg = cr.load_neat_config(exp.neat_config_text or "", exp.config_json)
        genome = g.to_neat_genome(cfg)
        split = s.get(DatasetSplit, exp.split_id)
        ds = s.get(Dataset, split.dataset_id)
        X, y = ds.get_data()
        names = list(ds.feature_names)
        mean = np.asarray(split.scaler_params["mean"], float)
        scale = np.asarray(split.scaler_params["scale"], float)
        tr = list(split.train_indices)
    return layer, genome, cfg, names, np.asarray(X, float)[tr], np.asarray(y).ravel()[tr], mean, scale


def network(genome, cfg):
    import torch
    from explaneat.core.explaneat import ExplaNEAT
    from explaneat.core.structure_network import StructureNetwork
    sn = StructureNetwork(ExplaNEAT(genome, cfg).get_phenotype_network())
    # the logit: the output unit's weighted sum before its activation (the model's own hidden activation)
    sn.override_output_activation("identity")
    sn.override_hidden_activation(sn.model_hidden_activation)

    def run(Z):
        out = sn.forward(torch.tensor(Z, dtype=torch.float64)).detach().numpy().ravel()
        acts = {k: sn.get_node_activation(k).ravel().copy() for k in ("155", GATE, "3040", "2379")}
        return out, acts
    return sn, run


def main():
    import torch
    torch.set_num_threads(1)
    layer, genome, cfg, names, Xr, y, mean, scale = load()
    col = {n: i for i, n in enumerate(names)}
    _, run = network(genome, cfg)
    w = {(e["from"], e["to"]): e["weight"] for e in layer["graph"]["edges"]}
    w_gate = w[(GATE, OUTPUT)]
    # the network's own weights (genome), not the layer's rounded copy
    gw = {(str(c.key[0]), str(c.key[1])): c.weight for c in genome.connections.values() if c.enabled}
    chol_c = col["chol"]
    w_direct = gw[(str(-chol_c - 1), OUTPUT)]               # chol -> output (input key = -(column + 1))
    w_2379 = gw[("2379", OUTPUT)] * gw[(str(-chol_c - 1), "2379")]   # chol -> 2379 -> output (2379 always on)
    _, acts_tr = run((Xr - mean) / scale)
    ref = np.asarray(layer["reference_patient"]["raw"], float)
    assert len(ref) == len(names)
    grid = np.linspace(Xr[:, chol_c].min(), Xr[:, chol_c].max(), N_GRID)
    out = {"source": {"layer": str(LAYER.relative_to(REPO)), "genome": layer["genome_id"],
                      "explanation_id": layer["explanation_id"], "n_train": int(len(y))},
           "weights": {"gate_to_output": w_gate, "chol_to_output": w_direct, "chol_via_2379": w_2379,
                       "chol_scale_mg_dl": float(scale[chol_c])},
           "grid_mg_dl": grid, "profiles": {}}
    for key, rule in SUBGROUPS.items():
        m = np.ones(len(y), bool)
        for c, v in rule.items():
            lo, hi = v if isinstance(v, tuple) else (v, v)
            m &= (Xr[:, col[c]] >= lo) & (Xr[:, col[c]] <= hi)
        sub = Xr[m]
        for variant in ("A_subgroup_median", "B_typical_patient"):
            if variant.startswith("A"):
                prof = np.median(sub, axis=0)
                for base in sorted({n.split(":")[0] for n in names if ":" in n}):
                    cols = [col[n] for n in names if n.startswith(base + ":")]
                    counts = sub[:, cols].sum(axis=0)
                    prof[cols] = 0.0
                    prof[cols[int(np.argmax(counts))]] = 1.0     # most common level (argmax: the lower code)
            else:
                prof = ref.copy()
                for c, v in rule.items():
                    if c == "ca":
                        prof[col["ca"]] = float(np.median(sub[:, col["ca"]]))
                    else:
                        base = c.split(":")[0]
                        for n in names:
                            if n.startswith(base + ":"):
                                prof[col[n]] = 0.0
                        prof[col[c]] = v
            P = np.repeat(prof[None, :], N_GRID, axis=0)
            P[:, chol_c] = grid
            logit, acts = run((P - mean) / scale)
            z = (grid - mean[chol_c]) / scale[chol_c]
            gate_term = w_gate * acts[GATE]
            direct = (w_direct + w_2379) * z
            on = acts[GATE] > 0
            off_at = None
            if on.any() and not on.all():
                k = int(np.argmax(~on)) if on[0] else None
                off_at = float(grid[k]) if k is not None else None
            out["profiles"][f"{key}/{variant}"] = {
                "rule": {c: (list(v) if isinstance(v, tuple) else v) for c, v in rule.items()},
                "n_train": int(m.sum()), "disease_rate": float(y[m].mean()),
                "share_gate_active_own_values": float(np.mean(acts_tr[GATE][m] > 0)),
                "chol_own_median": float(np.median(sub[:, chol_c])),
                "chol_own_p10_p90": [float(np.percentile(sub[:, chol_c], 10)), float(np.percentile(sub[:, chol_c], 90))],
                "values": {n: float(prof[i]) for i, n in enumerate(names)},
                "gate_on_at_lowest": bool(on[0]), "gate_on_anywhere": bool(on.any()),
                "gate_switches_off_at_mg_dl": off_at,
                "logit": logit, "gate_term": gate_term, "direct_chol_terms": direct,
                "rest": logit - gate_term - direct,
                "benign_unit_155": acts["155"][0], "threshold_unit_3040_on_from_mg_dl":
                    float(grid[int(np.argmax(acts["3040"] > 0))]) if (acts["3040"] > 0).any() else None,
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(out), indent=1, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(REPO))


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 10)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == "__main__":
    main()
