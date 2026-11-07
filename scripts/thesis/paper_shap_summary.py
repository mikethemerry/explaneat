"""Summaries of the recomputed SHAP JSON for the paper-claim table (task 3).

Reads results/thesis/paper_recompute/shap_{heart,adult,monk2}.json (written by
paper_recompute.py) and the stored explanations (read-only) for the group
memberships the paper's claims refer to; writes shap_summary.json.

Run: PYTHONHASHSEED=0 uv run python scripts/thesis/paper_shap_summary.py
"""
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "thesis" / "paper_recompute"

from explaneat.db import db  # noqa: E402
from explaneat.db.models import Explanation  # noqa: E402

EXPL = {"adult": "7e3761fe-ba64-4bbd-8d36-ca75213b968a",
        "heart": "a03ebc86-a14d-4860-9eb6-29f040d16633",
        "monk2": "d6ff7ca0-6806-45b6-b397-592a2518230e"}


def entries(label, name):
    with db.session_scope() as s:
        e = s.get(Explanation, uuid.UUID(EXPL[label]))
        for op in e.operations:
            if op["type"] == "annotate" and op["params"]["name"] == name:
                return [str(x) for x in op["params"]["entry_nodes"]]
    raise KeyError(name)


def col(node_id):
    """Input node '-k' (or split copy '-k_a') -> X column k-1."""
    return int(str(node_id).split("_")[0].lstrip("-")) - 1


def main():
    out = {}
    for label in ("heart", "adult", "monk2"):
        d = json.loads((OUT / f"shap_{label}.json").read_text())
        feat = d["feature_names"]
        r = {"n_rows": d["n_rows"]}
        V = {t: np.array(v["mean_abs_shap"]) for t, v in d["variants"].items()}
        for t, v in V.items():
            order = np.argsort(-v)
            r[t] = {"top10": [[feat[i], round(float(v[i]), 4), int(k + 1)] for k, i in enumerate(order[:10])],
                    "total": float(v.sum()),
                    "n_exact_zero": int((v == 0).sum()),
                    "rank": {feat[i]: int(k + 1) for k, i in enumerate(order)}}
        if len(V) == 2:
            a, b = V["old_relu_l1default"], V["new_sigmoid_l1false"]
            r["spearman_old_vs_new"] = float(spearmanr(a, b).correlation)
            r["top5_overlap"] = len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5]))
        if label == "adult":
            backbone = sorted({col(n) for n in entries("adult", "linear_backbone")})
            unused = sorted({col(n) for n in entries("adult", "unused_inputs")})
            countries = [col(n) for n in ("-73", "-81", "-82", "-89", "-94")]
            cg_core = sorted({col(n) for n in entries("adult", "capital_gain_core")})
            for t, v in V.items():
                tot = v.sum()
                r[t]["backbone_share"] = float(v[backbone].sum() / tot)
                r[t]["capital_gain_core_inputs_share"] = float(v[cg_core].sum() / tot)
                r[t]["unused_inputs_max"] = float(v[unused].max())
                r[t]["unused_inputs_share"] = float(v[unused].sum() / tot)
                r[t]["native_country_5"] = {feat[c]: [round(float(v[c]), 5), r[t]["rank"][feat[c]]] for c in countries}
                bb_order = [feat[i] for i in np.argsort(-v) if i in set(backbone)]
                r[t]["backbone_top5"] = bb_order[:5]
            r["n_backbone"] = len(backbone)
            r["n_unused"] = len(unused)
        if label == "heart":
            root_entries = entries("heart", "heart_disease_risk_score")
            for t, v in V.items():
                r[t]["top3"] = [x[0] for x in r[t]["top10"][:3]]
        out[label] = r
    (OUT / "shap_summary.json").write_text(json.dumps(out, indent=2))
    for label, r in out.items():
        print("==", label)
        for t in ("old_relu_l1default", "new_sigmoid_l1false"):
            if t in r:
                x = {k: v for k, v in r[t].items() if k != "rank"}
                print(" ", t, json.dumps(x)[:1500])
        print("  spearman", r.get("spearman_old_vs_new"), "top5 overlap", r.get("top5_overlap"))


if __name__ == "__main__":
    main()
