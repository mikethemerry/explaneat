"""Cross-seed comparison of the 10 Heart best genomes (read-only, training split).

Decomposes each network's output pre-activation z (p = sigmoid(z)) into
    z = bias + L(x) + N(x)
where L is the linear core (direct input->output edges) and N is the sum of the
hidden-node readouts into the output. Then compares across seeds:
  - linear-core weights per feature (presence, sign agreement, magnitude);
  - which input features feed the nonlinear readouts, weighted by readout SD;
  - functional similarity of z and N across seeds (correlations), and how linear N is.

    uv run python experiments/10seed/compare_heart.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from uuid import UUID

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from mcp_server.helpers import build_engine, load_split_data  # noqa: E402
from mcp_server.server import get_db  # noqa: E402
from explaneat.analysis.activation_extractor import ActivationExtractor  # noqa: E402
from explaneat.db.models import Dataset  # noqa: E402

SPLIT = "165cebbd-6609-46ca-867e-c09e17b2d6ec"
DATASET = "eea247c2-bdb7-41af-8ec1-9e26dcb5050d"
GENOMES = {
    "s0": "e7221bd4-a46a-4c5b-ad27-ab01bbdeb290", "s1": "d3b1f42b-8347-408c-951e-db9111eaaf72",
    "s2": "0057df7a-62a2-43c3-b107-9afe3735a7c3", "s3": "c426b094-0843-44e9-ab9a-7d9553d52601",
    "s4": "15783a91-457e-49a9-9209-cc3aa77df0cb", "s5": "1f2cf7ea-5c30-452a-9277-0b4043e888d9",
    "s6": "9b644a62-ebe7-48cc-82cf-c53f7da36073", "s7": "7c355ee7-77ad-426c-8a6d-9df81d1c65b3",
    "s8": "4809878b-8b08-4508-91f1-6f5c7207e787", "s9": "b19b9536-00dd-4e95-9f5a-def6dd8d6889",
    "paper": "9022a9eb-ecf9-41ba-b138-28ed4d131c9d",
}


def decompose(genome_id, X):
    db = get_db()
    with db.session_scope() as s:
        ph = build_engine(s, genome_id).original_phenotype
    out = str(ph.output_node_ids[0])
    inputs = {str(n) for n in ph.input_node_ids}
    conns = [c for c in ph.connections if c.enabled]
    nodes = {str(n.id): n for n in ph.nodes}
    srcs = sorted({str(c.from_node) for c in conns if str(c.to_node) == out})
    ext = ActivationExtractor.from_structure(ph)
    acts, _ = ext.extract(X, {"entry_nodes": srcs, "exit_nodes": [out]})
    acts = dict(zip(srcs, acts.T))
    bias = nodes[out].bias or 0.0
    linear_w, readouts = {}, {}
    for c in conns:
        if str(c.to_node) != out:
            continue
        f = str(c.from_node)
        if f in inputs:
            linear_w[f] = linear_w.get(f, 0.0) + c.weight
        else:
            readouts[f] = c.weight * acts[f]
    L = sum(w * acts[f] for f, w in linear_w.items())
    N = sum(readouts.values()) if readouts else np.zeros(len(X))
    # upstream input features of each hidden readout
    preds = defaultdict(set)
    for c in conns:
        preds[str(c.to_node)].add(str(c.from_node))

    def upstream(n, seen=None):
        seen = seen or set()
        for p in preds[n]:
            if p in seen:
                continue
            seen.add(p)
            if p not in inputs:
                upstream(p, seen)
        return seen
    hidden_feats = {h: sorted(i for i in upstream(h) if i in inputs) for h in readouts}
    hidden = [n for n in nodes if n not in inputs and n != out]
    return dict(z=bias + L + N, L=L, N=N, bias=bias, linear_w=linear_w,
                readouts=readouts, hidden_feats=hidden_feats, n_hidden=len(hidden))


def main():
    db = get_db()
    with db.session_scope() as s:
        X, y, *_ = load_split_data(s, SPLIT, "train", sample_fraction=1.0, max_samples=10**9)
        names = list(s.get(Dataset, UUID(DATASET)).feature_names)
    fname = lambda nid: names[-int(nid) - 1]
    D = {k: decompose(g, X) for k, g in GENOMES.items()}
    seeds = [k for k in GENOMES if k != "paper"]
    res = {}

    # 1. linear core
    W = np.array([[D[k]["linear_w"].get(str(-(i + 1)), 0.0) for i in range(len(names))] for k in seeds])
    lin = []
    for i, n in enumerate(names):
        col = W[:, i]
        present = int((col != 0).sum())
        nz = col[col != 0]
        lin.append(dict(feature=n, present=present,
                        mean=round(float(col.mean()), 3),
                        sign_agree=(int(max((nz > 0).sum(), (nz < 0).sum())) if present else 0)))
    res["linear_core"] = sorted(lin, key=lambda r: -abs(r["mean"]))
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    cos = Wn @ Wn.T
    res["linear_weight_cosine_offdiag_median"] = float(np.median(cos[~np.eye(len(seeds), dtype=bool)]))

    # 2. nonlinear readouts: features weighted by readout SD
    feat_weight = defaultdict(float)
    feat_seeds = defaultdict(set)
    per_seed = {}
    for k in seeds + ["paper"]:
        items = []
        for h, contrib in D[k]["readouts"].items():
            sd = float(np.std(contrib))
            fs = [fname(f) for f in D[k]["hidden_feats"][h]]
            items.append(dict(node=h, sd=round(sd, 3), features=fs))
            if k != "paper":
                for f in fs:
                    feat_weight[f] += sd / max(len(fs), 1)
                    feat_seeds[f].add(k)
        per_seed[k] = sorted(items, key=lambda r: -r["sd"])
    res["nonlinear_readouts"] = per_seed
    res["nonlinear_feature_frequency"] = sorted(
        [dict(feature=f, n_seeds=len(feat_seeds[f]), sd_share=round(feat_weight[f], 3)) for f in feat_seeds],
        key=lambda r: (-r["n_seeds"], -r["sd_share"]))

    # 3. functional similarity
    Z = np.array([D[k]["z"] for k in seeds])
    Nn = np.array([D[k]["N"] for k in seeds])
    off = ~np.eye(len(seeds), dtype=bool)
    rz = np.corrcoef(Z)
    rn = np.corrcoef(Nn)
    agree = np.array([[np.mean((Z[i] > 0) == (Z[j] > 0)) for j in range(len(seeds))] for i in range(len(seeds))])
    res["z_corr_median_min"] = [float(np.median(rz[off])), float(rz[off].min())]
    res["N_corr_median_min"] = [float(np.nanmedian(rn[off])), float(np.nanmin(rn[off]))]
    res["decision_agreement_median_min"] = [float(np.median(agree[off])), float(agree[off].min())]
    res["paper_vs_seeds_z_corr"] = [round(float(np.corrcoef(D["paper"]["z"], D[k]["z"])[0, 1]), 3) for k in seeds]

    # 4. per-seed: share of z variance from N; how linear N and z are in x; AUC with/without N
    rows = []
    for k in seeds + ["paper"]:
        d = D[k]
        r2_N = LinearRegression().fit(X, d["N"]).score(X, d["N"]) if np.std(d["N"]) > 0 else float("nan")
        r2_z = LinearRegression().fit(X, d["z"]).score(X, d["z"])
        rows.append(dict(seed=k, n_hidden=d["n_hidden"], n_direct=len(d["linear_w"]), n_readouts=len(d["readouts"]),
                         sd_L=round(float(np.std(d["L"])), 2), sd_N=round(float(np.std(d["N"])), 2),
                         r2_N_on_x=round(r2_N, 3), r2_z_on_x=round(r2_z, 3),
                         auc=round(roc_auc_score(y, d["z"]), 4),
                         auc_N_at_mean=round(roc_auc_score(y, d["L"] + d["N"].mean()), 4)))
    res["per_seed"] = rows
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
