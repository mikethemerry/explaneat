"""Task 3 (THESIS_FINAL_WORK.md): recompute the AAAI paper's whole-model
evidence and SHAP numbers with the current code.

Two fixes landed after the paper:
  * output-activation parity: StructureNetwork used to apply the output gene's
    ReLU at the output, where the trained model (NeuralNeat) applies a sigmoid.
    Current code: ``override_output_activation(model_output_activation)``.
  * SHAP ``l1_reg``: KernelExplainer's default ``num_features(10)`` kept only
    the top 10 features per row. Current code: ``l1_reg=False``.

For each comparison the "old" column re-creates the pre-fix behaviour
(ReLU output; SHAP with shap's default l1_reg) and the "new" column is the
current code path, so the change attributable to each fix is visible.

Read-only: genomes and explanations are loaded from the DB; ablations disable
connections on in-memory copies only.

Usage (PYTHONHASHSEED=0, one torch thread):
  uv run python scripts/thesis/paper_recompute.py core          # performance, ablations, activations, MONK-2 exhaustive, capgain
  uv run python scripts/thesis/paper_recompute.py shap heart    # SHAP old+new (cheap)
  uv run python scripts/thesis/paper_recompute.py shap monk2
  uv run python scripts/thesis/paper_recompute.py shap adult    # SHAP new only (slow: run once, in background)
  uv run python scripts/thesis/paper_recompute.py figures       # SHAP bar charts from the saved JSON
Outputs under results/thesis/paper_recompute/.
"""
import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.model_state import ModelStateEngine  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from explaneat.core.structure_network import StructureNetwork  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Dataset, DatasetSplit, Explanation, Genome  # noqa: E402
from mcp_server.helpers import load_split_data  # noqa: E402

OUT = ROOT / "results" / "thesis" / "paper_recompute"
SEED = 42
N_BOOT = 10_000
SHAP_MAX_SAMPLES = 200  # MCP compute_shap default, used for the paper's SHAP

PAPER = {
    "heart": {"genome_id": "9022a9eb-ecf9-41ba-b138-28ed4d131c9d",
              "explanation_id": "a03ebc86-a14d-4860-9eb6-29f040d16633"},
    "adult": {"genome_id": "4bc8fa07-bbf5-48e4-96a8-77be8b577ba3",
              "explanation_id": "7e3761fe-ba64-4bbd-8d36-ca75213b968a"},
    "monk2": {"genome_id": "23dde0fa-a635-4974-91aa-c05266ea0747",
              "explanation_id": "d6ff7ca0-6806-45b6-b397-592a2518230e"},
}

# Ablation edge sets, as in scripts/aaai27/run_b_ablations.py
ADULT_EDGES = [(-73, 655), (-81, 655), (-82, 655), (-89, 2129), (-94, 2129)]
HEART_CONFIGS = [
    ("linear core only", [(155, 0), (534, 0), (2379, 0)], 0.860),
    ("- benign-profile gate", [(155, 0)], 0.869),
    ("- resting-BP interaction", [(2379, 0)], 0.885),
    ("- risk-aggregation gate", [(534, 0)], 0.899),
]


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load(session, label):
    spec = PAPER[label]
    g = session.get(Genome, uuid.UUID(spec["genome_id"]))
    exp = g.population.experiment
    config = load_neat_config(exp.neat_config_text or "", exp.config_json)
    genome = g.to_neat_genome(config)
    phenotype = ExplaNEAT(genome, config).get_phenotype_network()
    e = session.get(Explanation, uuid.UUID(spec["explanation_id"]))
    eng = ModelStateEngine(copy.deepcopy(phenotype))
    eng.load_operations({"operations": copy.deepcopy(list(e.operations))})
    state = eng.current_state
    split_id = str(exp.split_id)
    split = session.get(DatasetSplit, exp.split_id)
    ds = session.get(Dataset, split.dataset_id)
    data = {}
    for part in ("train", "test"):
        X, y, fnames, cnames, ncls = load_split_data(session, split_id, part, 1.0, 10 ** 9)
        data[part] = (X, y.ravel())
    # X columns follow the dataset's feature order (= the phenotype's input
    # node order -1, -2, ...), not the model state's split input copies.
    assert len(fnames) == len(phenotype.input_node_ids)
    feat = list(fnames)
    raw_X, raw_y = ds.get_data()
    return dict(genome=genome, config=config, phenotype=phenotype, state=state,
                engine=eng, split_id=split_id, data=data, feature_names=feat,
                dataset_feature_names=fnames, class_names=cnames, num_classes=ncls,
                raw_X=raw_X, train_idx=list(split.train_indices),
                test_idx=list(split.test_indices))


def sn_predict(structure, X, output_activation):
    sn = StructureNetwork(structure)
    sn.override_output_activation(output_activation)
    out = sn.forward(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    return out.ravel(), sn


def nn_predict(genome, config, X):
    net = NeuralNeat(genome, config)
    return net.forward(torch.as_tensor(X, dtype=torch.float64)).detach().numpy().ravel()


def disabled(structure, edges):
    s = copy.deepcopy(structure)
    want = {(str(a), str(b)) for a, b in edges}
    hit = set()
    for c in s.connections:
        if (str(c.from_node), str(c.to_node)) in want:
            c.enabled = False
            hit.add((str(c.from_node), str(c.to_node)))
    missing = want - hit
    if missing:
        raise KeyError(f"edges not in structure: {missing}")
    return s


def metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "acc": float(accuracy_score(y, (p > 0.5).astype(int)))}


def paired_boot(y, a, b, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), size=len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.array(d)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ----------------------------------------------------------------------------
# core: performance, ablations, activations, MONK-2 exhaustive, capgain
# ----------------------------------------------------------------------------

def core():
    res = {}
    with db.session_scope() as s:
        M = {k: load(s, k) for k in PAPER}

    for label, m in M.items():
        r = {"performance": {}}
        for part, (X, y) in m["data"].items():
            p_nn = nn_predict(m["genome"], m["config"], X)
            p_new, _ = sn_predict(m["state"], X, "sigmoid")
            p_old, _ = sn_predict(m["state"], X, "relu")
            r["performance"][part] = {
                "n": int(len(y)),
                "neuralneat": metrics(y, p_nn),
                "structure_new_sigmoid": metrics(y, p_new),
                "structure_old_relu": metrics(y, p_old),
                "max_abs_new_minus_neuralneat": float(np.max(np.abs(p_new - p_nn))),
                "old_relu_share_rows_clipped_to_0": float(np.mean(p_old == 0.0)),
            }
        res[label] = r

    # ---- Adult native-country ablation ----
    m = M["adult"]
    Xte, yte = m["data"]["test"]
    ab = {}
    for tag, act in (("new_sigmoid", "sigmoid"), ("old_relu", "relu")):
        full, _ = sn_predict(m["phenotype"], Xte, act)
        abl, _ = sn_predict(disabled(m["phenotype"], ADULT_EDGES), Xte, act)
        ab[tag] = {"full": metrics(yte, full), "ablated": metrics(yte, abl),
                   "delta_auc": float(roc_auc_score(yte, abl) - roc_auc_score(yte, full)),
                   "delta_auc_ci95": paired_boot(yte, full, abl)}
    res["adult"]["ablation_native_country"] = ab

    # ---- Heart ablation ladder ----
    m = M["heart"]
    Xte, yte = m["data"]["test"]
    Xtr, ytr = m["data"]["train"]
    ladder = {}
    for tag, act in (("new_sigmoid", "sigmoid"), ("old_relu", "relu")):
        full_te, _ = sn_predict(m["phenotype"], Xte, act)
        full_tr, _ = sn_predict(m["phenotype"], Xtr, act)
        rows = []
        for name, edges, paper in HEART_CONFIGS:
            st = disabled(m["phenotype"], edges)
            pte, _ = sn_predict(st, Xte, act)
            ptr, _ = sn_predict(st, Xtr, act)
            rows.append({"config": name, "paper_test_auc": paper,
                         "test_auc": float(roc_auc_score(yte, pte)),
                         "delta": float(roc_auc_score(yte, pte) - roc_auc_score(yte, full_te)),
                         "ci95": paired_boot(yte, full_te, pte),
                         "train_auc": float(roc_auc_score(ytr, ptr))})
        ladder[tag] = {"full_test_auc": float(roc_auc_score(yte, full_te)),
                       "full_train_auc": float(roc_auc_score(ytr, full_tr)), "rows": rows}
    res["heart"]["ablation_ladder"] = ladder

    # ---- Heart activation fractions (test split; hidden ReLUs, unaffected by the fix) ----
    _, sn = sn_predict(m["state"], Xte, "sigmoid")
    acts = {}
    for nid in ("155_a", "155_b", "3040", "534", "3240", "2379"):
        a = sn.get_node_activation(nid).ravel()
        acts[nid] = {"active_fraction": float(np.mean(a > 0)), "n": int(len(a))}
    res["heart"]["activation_fractions_test"] = acts

    # ---- MONK-2: exhaustive check on all 432 configurations ----
    m = M["monk2"]
    X = np.vstack([m["data"]["train"][0], m["data"]["test"][0]])
    y = np.concatenate([m["data"]["train"][1], m["data"]["test"][1]])
    p_new, sn_new = sn_predict(m["state"], X, "sigmoid")
    p_old, _ = sn_predict(m["state"], X, "relu")
    p_nn = nn_predict(m["genome"], m["config"], X)
    # output pre-activation z: sigmoid^-1 of the new prediction
    z = np.log(p_new) - np.log1p(-p_new)
    # rule from the raw one-hot indicators: exactly two of the six first-value columns
    raw_X = m["raw_X"]
    idx = m["train_idx"] + m["test_idx"]
    first = [0, 3, 6, 8, 11, 15]  # a1:1 a2:1 a3:1 a4:1 a5:1 a6:1 (inputs -1 -4 -7 -9 -12 -16)
    names = m["dataset_feature_names"]
    count = (raw_X[idx][:, first] > 0.5).sum(axis=1)
    rule = (count == 2).astype(int)
    res["monk2"]["exhaustive"] = {
        "n_configurations": int(len(y)),
        "first_value_columns": [names[i] for i in first] if names else first,
        "labels_equal_rule": int(np.sum(rule == y)),
        "new_sigmoid_pred_eq_rule": int(np.sum((p_new > 0.5).astype(int) == rule)),
        "old_relu_pred_eq_rule": int(np.sum((p_old > 0.5).astype(int) == rule)),
        "neuralneat_pred_eq_rule": int(np.sum((p_nn > 0.5).astype(int) == rule)),
        "auc_new": float(roc_auc_score(y, p_new)), "auc_old": float(roc_auc_score(y, p_old)),
        "min_preact_positive": float(z[rule == 1].min()),
        "max_preact_negative": float(z[rule == 0].max()),
        "max_preact_count3plus": float(z[count >= 3].max()),
        "max_preact_count01": float(z[count <= 1].max()),
        "old_relu_min_output_positive": float(p_old[rule == 1].min()),
    }

    # ---- capgain pathway: knots (pre-activation contribution, unaffected) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure
    import importlib
    capmod = importlib.import_module("aaai27.run_d_capgain_figure")
    orig = matplotlib.figure.Figure.savefig
    target = OUT / "capgain-response_recomputed.png"

    def _redirect(self, fname, *a, **k):
        return orig(self, str(target), *a, **k)

    matplotlib.figure.Figure.savefig = _redirect
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        capmod.main()
    matplotlib.figure.Figure.savefig = orig
    res["adult"]["capgain_script_output"] = buf.getvalue()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "core.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:20000])


# ----------------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------------

def run_shap(label, variants):
    import shap
    with db.session_scope() as s:
        m = load(s, label)
        X, y, *_ = load_split_data(s, m["split_id"], "test", 1.0, SHAP_MAX_SAMPLES)
    feat = m["feature_names"]
    out = {"label": label, "n_rows": int(len(X)), "background": int(min(len(X), SHAP_MAX_SAMPLES)),
           "split": "test", "max_samples": SHAP_MAX_SAMPLES, "seed": SEED,
           "feature_names": feat, "dataset_feature_names": m["dataset_feature_names"],
           "shap_version": shap.__version__, "variants": {}}
    for tag, act, l1 in variants:
        sn = StructureNetwork(m["state"])
        sn.override_output_activation(act)

        def f(x, sn=sn):
            o = sn.forward(torch.as_tensor(x, dtype=torch.float64)).detach().numpy()
            return o.ravel() if o.ndim == 2 and o.shape[1] == 1 else o

        np.random.seed(SEED)
        t0 = time.time()
        bg = X if len(X) <= SHAP_MAX_SAMPLES else shap.sample(X, SHAP_MAX_SAMPLES)
        ex = shap.KernelExplainer(f, bg)
        kw = {} if l1 == "default" else {"l1_reg": l1}
        sv = np.asarray(ex.shap_values(X, silent=True, **kw))
        mabs = np.mean(np.abs(sv), axis=0).ravel()
        order = np.argsort(-mabs)
        out["variants"][tag] = {
            "output_activation": act, "l1_reg": str(l1), "seconds": time.time() - t0,
            "base_value": float(np.ravel(ex.expected_value)[0]),
            "mean_abs_shap": mabs.tolist(),
            "ranking": [[feat[i], float(mabs[i])] for i in order],
            "n_exact_zero_features": int(np.sum(mabs == 0.0)),
        }
        print(f"[{label} {tag}] {time.time()-t0:.0f}s top10:",
              [(feat[i], round(float(mabs[i]), 4)) for i in order[:10]], flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"shap_{label}.json").write_text(json.dumps(out, indent=2))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for label in PAPER:
        fp = OUT / f"shap_{label}.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        vs = d["variants"]
        tags = [t for t in ("old_relu_l1default", "new_sigmoid_l1false") if t in vs]
        fig, axes = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 5.2), squeeze=False)
        for ax, t in zip(axes[0], tags):
            top = vs[t]["ranking"][:15][::-1]
            ax.barh([n for n, _ in top], [v for _, v in top],
                    color="#4c72b0" if t.startswith("new") else "#999999")
            ttl = ("current code: sigmoid output, l1_reg=False" if t.startswith("new")
                   else "pre-fix: ReLU output, l1_reg=num_features(10)")
            ax.set_title(f"{label}: {ttl}", fontsize=9)
            ax.set_xlabel("mean |SHAP| (test split, %d rows)" % d["n_rows"])
            ax.tick_params(labelsize=8)
            ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"shap_{label}.png", dpi=200)
        plt.close(fig)
        print("wrote", OUT / f"shap_{label}.png")


if __name__ == "__main__":
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
    cmd = sys.argv[1]
    if cmd == "core":
        core()
    elif cmd == "shap":
        label = sys.argv[2]
        # Both variants for every genome. (Adult was expected to take 3-36 min
        # per call; with this network the new variant took 29 s, so the
        # pre-fix variant is run too, for the comparison.)
        variants = [("old_relu_l1default", "relu", "default"),
                    ("new_sigmoid_l1false", "sigmoid", False)]
        run_shap(label, variants)
    elif cmd == "figures":
        figures()
