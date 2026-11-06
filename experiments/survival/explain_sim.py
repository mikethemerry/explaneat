"""Hazard-ratio reading of an evolved network on the non-linear simulation.

The truth is known: g_true(x) = log(5) * exp(-(x0^2 + x1^2) / (2 * 0.5^2)), a
bump in (x0, x1) with zero linear main effect, every other input irrelevant.
This reads the evolved phenotype with explaneat.survival.explain (analysis
classes directly, no DB): the additive log-HR terms at the output, each
hidden ReLU unit as a hinge, and its spread over the training rows, then
checks the reading against the truth (which inputs matter, the shape along
x0 and x1).

    uv run python experiments/survival/explain_sim.py sim/sim-nonlinear-s0
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from explaneat.analysis.annotation_function import AnnotationFunction  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.structure_network import StructureNetwork  # noqa: E402
from explaneat.survival.data import load_split_indices  # noqa: E402
from explaneat.survival.explain import explain_output, output_terms  # noqa: E402
from explaneat.survival.simulate import simulate  # noqa: E402

from run_seed import make_config  # noqa: E402


def main(run):
    rec = json.loads((HERE / f"{run}.json").read_text())
    b = pickle.loads((HERE / f"{run}.genome.pkl").read_bytes())
    src = rec["source"]
    X, t, e, g_true = simulate(n=src["n"], d=src["d"], mode=src["mode"], seed=src["sim_seed"])
    sp = load_split_indices(HERE / Path(run).parent / b["splits_file"])
    mu, sd = np.array(b["scaler"]["mean"]), np.array(b["scaler"]["scale"])
    Z = (X - mu) / sd
    tr = sp["train"]
    config = make_config(HERE / "configs/survival.ini", X.shape[1])
    ph = ExplaNEAT(b["genome"], config).get_phenotype_network()
    names = b["feature_names"]

    readings = explain_output(ph, Z[tr], b["scaler"], names)
    terms = output_terms(ph)
    used_inputs = sorted({c.from_node for c in ph.connections if c.enabled and c.from_node.startswith("-")},
                         key=lambda s: -int(s))

    # closed form of the whole log-risk (one annotation over the phenotype)
    ann = {"entry_nodes": list(ph.input_node_ids), "exit_nodes": list(ph.output_node_ids),
           "subgraph_nodes": [n.id for n in ph.nodes], "subgraph_connections": []}
    fn = AnnotationFunction.from_structure(ann, ph)
    latex = fn.to_latex(force=True)

    # shape along x0 and x1 (others at 0 = their mean) vs the truth
    grid = np.linspace(-1, 1, 41)
    sn = StructureNetwork(ph)
    sn.override_output_activation(sn.model_output_activation)
    sn.override_hidden_activation("relu")

    def g_at(raw):
        z = (raw - mu) / sd
        return sn.forward(torch.as_tensor(z, dtype=torch.float64)).detach().numpy().ravel()

    profiles = {}
    for j in (0, 1, 2):
        raw = np.tile(mu, (len(grid), 1))
        raw[:, j] = grid
        gm = g_at(raw)
        # truth along axis j with every other input at its mean (about 0)
        others = np.delete(mu[:2], j).sum() ** 2 if j < 2 else 0.0
        truth = (np.log(5) * np.exp(-(grid ** 2 + others) / 0.5) if j < 2 else np.zeros_like(grid))
        profiles[names[j]] = {"grid": grid.tolist(), "model_log_hr_vs_centre": (gm - gm[20]).tolist(),
                              "true_log_hr_vs_centre": (truth - truth[20]).tolist(),
                              "model_range": float(gm.max() - gm.min()),
                              "true_range": float(truth.max() - truth.min())}
    g_model = g_at(X[sp["test"]])
    out = {
        "run": run, "terms_at_output": terms, "inputs_used": [names[-int(n) - 1] for n in used_inputs],
        "readings": readings, "log_risk_latex": latex, "profiles_other_inputs_at_mean": profiles,
        "test_spearman_true_g": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
            g_model, g_true[sp["test"]]).statistic),
    }
    path = HERE / "explain" / f"{Path(run).name}.reading.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("inputs used:", out["inputs_used"])
    for r in readings:
        print(" ", r["node"], "|", r.get("reading", r.get("feature")), "| HR p95/p5 %.2f" % r["spread"]["hr_p95_vs_p5"],
              "| active %.0f%%" % (100 * r["spread"]["fraction_nonzero"]))
    for k, p in profiles.items():
        print(f"  profile {k}: model range {p['model_range']:.3f}  true range {p['true_range']:.3f}")
    print("  test spearman with true g:", round(out["test_spearman_true_g"], 3))
    print(latex)


if __name__ == "__main__":
    main(sys.argv[1])
