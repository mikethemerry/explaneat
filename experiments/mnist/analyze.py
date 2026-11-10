"""Post-import checks and structure statistics for an imported MNIST run.

1. Parity: the imported best genome, loaded from the DB the way the UI / MCP
   tools load it (``load_genome_and_config`` -> ``ExplaNEAT`` phenotype ->
   ``StructureNetwork``), on the DB dataset/split, must reproduce the in-memory
   train/test metrics (``logs/<name>.json``, computed with NeuralNeat).
2. Tools: the MCP tools get_phenotype, get_model_state, classify_nodes and
   get_coverage (no annotations) run on it.
3. Structure (no explanation attempted): hidden nodes, depth, fan-in, inputs
   used, and the share of hidden ReLU units that are switching / always-on /
   dead on the training rows.

    DATABASE_URL=postgresql://localhost/explaneat_dev \\
        uv run python experiments/mnist/analyze.py mnist38-s0
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.genome_network import NodeType  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from explaneat.core.structure_network import StructureNetwork  # noqa: E402
from explaneat.db.models import Dataset, DatasetSplit  # noqa: E402
from mcp_server.helpers import load_genome_and_config  # noqa: E402
from mcp_server.server import get_db  # noqa: E402
from run_seed import RUNS_DIR, load_config  # noqa: E402

TOL = 0.01  # "nearly" dead / always-on: active on <= 1% / >= 99% of rows


def metrics(out: np.ndarray, y: np.ndarray, multiclass: bool) -> dict:
    if multiclass:
        t = torch.as_tensor(out)
        yt = torch.as_tensor(y, dtype=torch.long)
        return {"ce": float(F.cross_entropy(t, yt)), "acc": float((t.argmax(1) == yt).double().mean())}
    p = out.ravel()
    eps = 1e-12
    bce = float(-np.mean(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))))
    return {"auc": float(roc_auc_score(y, p)), "acc": float(((p >= 0.5) == (y == 1)).mean()), "bce": bce}


def tool(fn, *a, **k):
    t = time.time()
    try:
        res = json.loads(fn(*a, **k))
        err = res.get("error") if isinstance(res, dict) else None
        return dict(ok=err is None, error=err, seconds=round(time.time() - t, 2)), res
    except Exception as e:  # report, don't hide
        return dict(ok=False, error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t, 2)), None


def structure_stats(pheno, sn, X_train, multiclass):
    ids_in = set(pheno.input_node_ids)
    ids_out = set(pheno.output_node_ids)
    enabled = [c for c in pheno.connections if c.enabled]
    valid = set(sn.node_info)  # nodes in the layered network
    conns = [c for c in enabled if c.from_node in valid and c.to_node in valid]
    hidden = [n for n in valid if n not in ids_in and n not in ids_out]
    fan_in = Counter(c.to_node for c in conns)
    fan_out = Counter(c.from_node for c in conns)
    hidden_fi = [fan_in[h] for h in hidden]
    out_fi = [fan_in[o] for o in ids_out]
    inputs_used = sorted({c.from_node for c in conns if c.from_node in ids_in})
    depth = max(info["depth"] for info in sn.node_info.values())
    direct = sum(1 for c in conns if c.from_node in ids_in and c.to_node in ids_out)
    in_to_hidden = sum(1 for c in conns if c.from_node in ids_in and c.to_node not in ids_out)
    hid_to_hid = sum(1 for c in conns if c.from_node not in ids_in and c.to_node not in ids_out)
    hid_to_out = sum(1 for c in conns if c.from_node not in ids_in and c.to_node in ids_out)

    sn.forward(torch.as_tensor(X_train, dtype=torch.float64))
    frac = {h: float((sn.get_node_activation(h) > 0).mean()) for h in hidden}
    strict = Counter("dead" if f == 0 else "always_on" if f == 1 else "switching" for f in frac.values())
    loose = Counter("dead" if f <= TOL else "always_on" if f >= 1 - TOL else "switching"
                    for f in frac.values())
    n_h = max(len(hidden), 1)
    depth_hist = Counter(sn.node_info[h]["depth"] for h in hidden)

    def dist(v):
        return dict(mean=float(np.mean(v)), median=float(np.median(v)), max=int(np.max(v)),
                    min=int(np.min(v))) if v else None

    return dict(
        hidden_nodes=len(hidden), depth_layers=int(depth),
        hidden_per_depth={int(k): v for k, v in sorted(depth_hist.items())},
        connections=len(conns), direct_input_output=direct, input_to_hidden=in_to_hidden,
        hidden_to_hidden=hid_to_hid, hidden_to_output=hid_to_out,
        inputs_used=len(inputs_used), inputs_total=len(ids_in),
        fan_in_hidden=dist(hidden_fi), fan_in_output=dist(out_fi),
        fan_out_hidden=dist([fan_out[h] for h in hidden]),
        hidden_activity_strict={k: strict.get(k, 0) for k in ("switching", "always_on", "dead")},
        hidden_activity_strict_share={k: strict.get(k, 0) / n_h for k in ("switching", "always_on", "dead")},
        hidden_activity_1pct={k: loose.get(k, 0) for k in ("switching", "always_on", "dead")},
        hidden_activity_1pct_share={k: loose.get(k, 0) / n_h for k in ("switching", "always_on", "dead")},
        hidden_active_fraction=dict(sorted(frac.items(), key=lambda kv: kv[1])),
        inputs_used_ids=inputs_used,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    args = ap.parse_args()
    imp = json.loads((HERE / "logs" / f"{args.name}.import.json").read_text())
    summ = json.loads((HERE / "logs" / f"{args.name}.json").read_text())
    multiclass = summ["task"] == "mnist10"
    gid = imp["best_genome_id"]
    with gzip.open(RUNS_DIR / args.name / "best_genome.pkl.gz", "rb") as f:
        mem_genome = pickle.load(f)["genome"]

    db = get_db()
    db.init_db()
    report = dict(name=args.name, genome_id=gid, experiment_id=imp["experiment_id"])
    with db.session_scope() as session:
        neat_genome, config, _ = load_genome_and_config(session, gid)
        split = session.get(DatasetSplit, __import__("uuid").UUID(imp["split_id"]))
        X, y = session.get(Dataset, split.dataset_id).get_data()
        tr, te = list(split.train_indices), list(split.test_indices)
    pheno = ExplaNEAT(neat_genome, config).get_phenotype_network()
    sn = StructureNetwork(pheno)
    sn.override_output_activation(sn.model_output_activation)
    parity = {}
    for nm, idx in (("train", tr), ("test", te)):
        out = sn.forward(torch.as_tensor(X[idx], dtype=torch.float64)).detach().numpy()
        m = metrics(out, y[idx].astype(np.float64) if not multiclass else y[idx], multiclass)
        parity[nm] = dict(structure_network=m, in_memory=summ[nm],
                          max_abs_diff=max(abs(m[k] - summ[nm][k]) for k in m))
    # DB genome == in-memory genome, and NeuralNeat outputs bit-equal
    mem_cfg = load_config(summ["task"])
    xt = torch.as_tensor(X[te], dtype=torch.float64)
    with torch.no_grad():
        a = NeuralNeat(neat_genome, config).forward(xt).numpy()
        b = NeuralNeat(mem_genome, mem_cfg).forward(xt).numpy()
        s = sn.forward(xt).detach().numpy()
    parity["neuralneat_db_vs_memory_bit_equal"] = bool(np.array_equal(a, b))
    parity["structure_vs_neuralneat_max_abs_output_diff"] = float(np.max(np.abs(s - b)))
    parity["output_activation"] = sn.model_output_activation
    parity["ok"] = all(parity[k]["max_abs_diff"] < 1e-9 for k in ("train", "test")) and \
        parity["neuralneat_db_vs_memory_bit_equal"]
    report["parity"] = parity

    from mcp_server.tools.coverage import classify_nodes, get_coverage
    from mcp_server.tools.models import get_model_state, get_phenotype
    tools = {}
    tools["get_phenotype"], ph = tool(get_phenotype, gid)
    tools["get_model_state"], _ = tool(get_model_state, gid)
    hidden = [n.id for n in pheno.nodes if n.type == NodeType.HIDDEN]
    sample = hidden[: min(5, len(hidden))] or [pheno.output_node_ids[0]]
    tools["classify_nodes"], cl = tool(classify_nodes, gid, json.dumps(sample))
    tools["get_coverage"], cov = tool(get_coverage, gid)
    from mcp_server.tools.evidence import compute_performance
    tools["compute_performance"], perf = tool(compute_performance, gid, imp["split_id"],
                                              split="test", max_samples=100000)
    if perf is not None:
        tools["compute_performance"]["result"] = {k: v for k, v in perf.items()
                                                  if isinstance(v, (int, float, str))}
    if ph is not None:
        tools["get_phenotype"]["n_nodes"] = len(ph.get("nodes", []))
        tools["get_phenotype"]["n_connections"] = len(ph.get("connections", []))
    if cov is not None:
        tools["get_coverage"]["result_keys"] = sorted(cov)[:12]
        for k in ("structural_coverage", "compositional_coverage"):
            if k in cov:
                tools["get_coverage"][k] = cov[k]
    # The UI path: the REST routes the React explorer calls for a genome.
    from fastapi.testclient import TestClient
    from explaneat.api.app import create_app
    client = TestClient(create_app())
    for route in ("", "/phenotype", "/model"):
        t = time.time()
        r = client.get(f"/api/genomes/{gid}{route}")
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        tools[f"api GET /api/genomes/{{id}}{route}"] = dict(
            ok=r.status_code == 200, error=None if r.status_code == 200 else str(body)[:300],
            seconds=round(time.time() - t, 2),
            n_nodes=len(body.get("nodes", [])) if isinstance(body, dict) else None)
    report["tools"] = tools
    report["structure"] = structure_stats(pheno, sn, X[tr], multiclass)
    out_path = HERE / "logs" / f"{args.name}.analysis.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    brief = {k: v for k, v in report["structure"].items()
             if k not in ("hidden_active_fraction", "inputs_used_ids")}
    print(json.dumps(dict(parity=parity, tools=tools, structure=brief), indent=2, default=str))


if __name__ == "__main__":
    main()
