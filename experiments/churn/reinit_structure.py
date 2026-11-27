"""Study E, amendment E-S (RESCUE2_REINIT_PLAN.md): explanatory-structure
equivalence of the Kaiming re-inits.

Does a re-initialised, retrained network break down into the same named features
as the evolved one? A re-init keeps the evolved topology and node ids, so the
evolved genome's explanation operations replay onto it unchanged
(``ext_common.explained_state``, checked against NeuralNeat to 1e-9 on train).
Then, for every annotation, on train:

- feature r: Pearson r between the re-init's and the evolved network's
  activation at each exit node; the annotation's score is the minimum over its
  exits (r, not |r|: invariant to positive rescaling and shifts, the ReLU
  symmetry, not to sign);
- readout-role r (leaves with exit -> output edges): Pearson r of the direct logit
  contribution, sum over those edges of w * a;
- lost: an exit constant on train in the re-init but not in the evolved network
  (a dead feature).

An exit constant in the evolved network itself carries no function and is left
out of the score; an annotation with no such live exit is flagged ``excluded``
and left out of the summary counts (plan clarification, committed before the
Kaiming analysis).

Leaves are primary; compositions get feature r only. Recovered = feature r >= 0.9
(D2's alias threshold); role recovered = readout-role r >= 0.9; an init's
explanation is preserved when every leaf is recovered.

Inputs: the evolved genomes and explanations (DB, read only) and the trained
weights in ``reinit_weights/network_s<ts>_kaiming[_es].json`` (``reinit.py
--stage network --init kaiming [--early-stopping]``). Output:
``reinit_structure_kaiming.csv`` (``kind``: leaf, composition, leaf_robustness,
summary, pooled).

    uv run python experiments/churn/reinit_structure.py
    uv run python experiments/churn/reinit_structure.py --dry-run   # 1 topology, 2 inits, no output
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

import ext_common as ec
import explained_eval as ee
import reinit as ri

HERE = Path(__file__).resolve().parent
R_MIN = 0.9
# (regime label, weights file name for reinit.wfile, key inside inits[seed])
REGIMES = [("E = 500", "network_s{ts}", "E500"), ("E = 2000", "network_s{ts}", "E2000"),
           ("early stopping", "network_s{ts}_es", "es")]

COLUMNS = [
    "kind", "init", "regime", "topology_seed", "genome_id", "init_seed", "annotation",
    "n_exits", "excluded", "n_exits_with_readout", "feature_r_min", "feature_r_per_exit", "role_r",
    "recovered", "role_recovered", "lost", "lost_exits", "dead_in_evolved_exits",
    "replay_max_abs_err",
    # robustness / summary / pooled
    "n_inits", "n_inits_recovered", "n_inits_role_recovered", "n_inits_lost",
    "n_leaves", "n_leaf_inits", "n_recovered", "n_lost", "n_alive_different",
    "share_recovered", "share_lost", "share_alive_different", "n_role_eligible",
    "n_role_recovered", "share_role_recovered", "median_share_recovered",
    "median_share_role_recovered", "median_share_lost", "n_preserved",
    "median_feature_r", "n_compositions", "n_composition_recovered",
    "share_composition_recovered", "n_topologies", "note",
]


def genome_from_weights(g0, wj):
    """Deep copy of ``g0`` with the biases and enabled weights of a
    ``reinit.genome_weights_json`` record."""
    g = copy.deepcopy(g0)
    for n, b in wj["bias"].items():
        g.nodes[int(n)].bias = float(b)
    for a, b, w in wj["conn"]:
        g.connections[(int(a), int(b))].weight = float(w)
    return g


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def readout_edges(state, exits, outputs):
    return [(c.from_node, c.weight) for c in state.connections
            if c.enabled and c.from_node in exits and c.to_node in outputs]


def contribution(edges, acts):
    s = 0.0
    for x, w in edges:
        s = s + w * acts[x]
    return s


def compare(ann, acts0, acts, edges0, edges, leaf):
    exits = [str(x) for x in ann["exit_nodes"]]
    rs, lost, dead0 = [], [], []
    for x in exits:
        c0, c1 = np.std(acts0[x]) == 0, np.std(acts[x]) == 0
        if c0:
            dead0.append(x)
        if c1 and not c0:
            lost.append(x)
        rs.append(pearson(acts[x], acts0[x]))
    # exits constant in the evolved network carry no function: left out of the score
    live = [r for x, r in zip(exits, rs) if x not in dead0]
    r_min = float(min(live)) if live and all(np.isfinite(r) for r in live) else float("nan")
    row = {"annotation": ann["name"], "n_exits": len(exits), "feature_r_min": r_min,
           "excluded": not live,
           "feature_r_per_exit": [f"{x}:{r!r}" for x, r in zip(exits, rs)],
           "recovered": bool(np.isfinite(r_min) and r_min >= R_MIN), "lost": bool(lost),
           "lost_exits": lost, "dead_in_evolved_exits": dead0}
    if leaf:
        row["n_exits_with_readout"] = len({x for x, _ in edges0})
        if edges0:
            rr = pearson(contribution(edges, acts), contribution(edges0, acts0))
            row["role_r"] = rr
            row["role_recovered"] = bool(np.isfinite(rr) and rr >= R_MIN)
    return row


def run(dry_run=False, init="kaiming"):
    from explaneat.db.base import db
    db.init_db()
    data = ec.load_data()
    runs = ec.read_runs()
    Xtr = data["X_train"]
    topo = sorted(runs)[:1] if dry_run else sorted(runs)
    rows = []
    for ts in topo:
        g0, cfg, info = ec.load_genome(runs[ts]["genome_id"])
        ops = info["operations"]
        state0, anns, ev0, _ = ec.explained_state(g0, cfg, ops, Xtr)
        acts0 = ev0.forward(Xtr)
        outputs = {str(o) for o in cfg.genome_config.output_keys}
        leaf_ids = {a["id"] for a in ee.leaf_annotations(anns)}
        base = {"init": init, "topology_seed": ts, "genome_id": info["genome_id"]}
        for regime, fname, key in REGIMES:
            wj = json.loads((HERE / ri.wfile(fname.format(ts=ts), init)).read_text())
            seeds = sorted(wj["inits"], key=int)[:2] if dry_run else sorted(wj["inits"], key=int)
            for s in seeds:
                g = genome_from_weights(g0, wj["inits"][s][key])
                state, anns_r, ev, err = ec.explained_state(g, cfg, ops, Xtr)  # raises > 1e-9
                assert [a["id"] for a in anns_r] == [a["id"] for a in anns]
                acts = ev.forward(Xtr)
                for a in anns:
                    leaf = a["id"] in leaf_ids
                    exits = {str(x) for x in a["exit_nodes"]}
                    r = compare(a, acts0, acts, readout_edges(state0, exits, outputs),
                                readout_edges(state, exits, outputs), leaf)
                    rows.append(dict(base, kind="leaf" if leaf else "composition",
                                     regime=regime, init_seed=int(s),
                                     replay_max_abs_err=err, **r))
            lr = [r for r in rows if r["kind"] == "leaf" and r["topology_seed"] == ts
                  and r["regime"] == regime]
            print(f"s{ts} {regime}: leaves recovered {sum(r['recovered'] for r in lr)}/{len(lr)} "
                  f"lost {sum(r['lost'] for r in lr)} role recovered "
                  f"{sum(r.get('role_recovered', False) for r in lr)}/"
                  f"{sum('role_r' in r for r in lr)}", flush=True)
    out = rows + robustness(rows) + summaries(rows) + pooled(rows)
    if dry_run:
        for p in pooled(rows):
            print({k: p[k] for k in ("regime", "share_recovered", "share_lost",
                                     "share_role_recovered", "n_preserved", "median_feature_r")})
        print("DRY RUN: nothing written")
        return
    path = ri.out_path(HERE / "reinit_structure.csv", init)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow({k: ri._fmt(r.get(k, "")) for k in COLUMNS})
    print(f"wrote {path.name} ({len(out)} rows)")


def _groups(rows, kind, keys):
    g = {}
    for r in rows:
        if r["kind"] == kind and not r["excluded"]:
            g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


def robustness(rows):
    out = []
    for (init, regime, ts, name), rs in _groups(
            rows, "leaf", ("init", "regime", "topology_seed", "annotation")).items():
        out.append({"kind": "leaf_robustness", "init": init, "regime": regime,
                    "topology_seed": ts, "genome_id": rs[0]["genome_id"], "annotation": name,
                    "n_exits": rs[0]["n_exits"], "n_inits": len(rs),
                    "n_inits_recovered": sum(r["recovered"] for r in rs),
                    "n_inits_role_recovered": (sum(r.get("role_recovered", False) for r in rs)
                                               if "role_r" in rs[0] else ""),
                    "n_inits_lost": sum(r["lost"] for r in rs)})
    return out


def _counts(leaf_rows):
    n = len(leaf_rows)
    rec = sum(r["recovered"] for r in leaf_rows)
    lost = sum(r["lost"] for r in leaf_rows)
    other = n - rec - sum(r["lost"] and not r["recovered"] for r in leaf_rows)
    elig = [r for r in leaf_rows if "role_r" in r]
    rr = sum(r["role_recovered"] for r in elig)
    return {"n_leaf_inits": n, "n_recovered": rec, "n_lost": lost, "n_alive_different": other,
            "share_recovered": rec / n, "share_lost": lost / n, "share_alive_different": other / n,
            "n_role_eligible": len(elig), "n_role_recovered": rr,
            "share_role_recovered": rr / len(elig) if elig else float("nan")}


def _per_init(leaf_rows):
    """{init seed: (share recovered, share role recovered, share lost, preserved)}."""
    by = {}
    for r in leaf_rows:
        by.setdefault(r["init_seed"], []).append(r)
    out = {}
    for s, rs in sorted(by.items()):
        elig = [r for r in rs if "role_r" in r]
        out[s] = (np.mean([r["recovered"] for r in rs]),
                  np.mean([r["role_recovered"] for r in elig]) if elig else float("nan"),
                  np.mean([r["lost"] for r in rs]), all(r["recovered"] for r in rs))
    return out


def summaries(rows):
    out = []
    for (init, regime, ts), rs in _groups(rows, "leaf", ("init", "regime", "topology_seed")).items():
        pi = _per_init(rs)
        comp = [r for r in rows if r["kind"] == "composition" and not r["excluded"]
                and r["init"] == init and r["regime"] == regime and r["topology_seed"] == ts]
        out.append({"kind": "summary", "init": init, "regime": regime, "topology_seed": ts,
                    "genome_id": rs[0]["genome_id"], "n_inits": len(pi),
                    "n_leaves": len({r["annotation"] for r in rs}), **_counts(rs),
                    "median_share_recovered": float(np.median([v[0] for v in pi.values()])),
                    "median_share_role_recovered": ri._nanmed([v[1] for v in pi.values()]),
                    "median_share_lost": float(np.median([v[2] for v in pi.values()])),
                    "n_preserved": sum(v[3] for v in pi.values()),
                    "median_feature_r": ri._nanmed([r["feature_r_min"] for r in rs]),
                    "n_compositions": len({r["annotation"] for r in comp}),
                    "n_composition_recovered": sum(r["recovered"] for r in comp),
                    "share_composition_recovered": (np.mean([r["recovered"] for r in comp])
                                                    if comp else "")})
    return out


def pooled(rows):
    out = []
    for (init, regime), rs in _groups(rows, "leaf", ("init", "regime")).items():
        pis = [v for ts in sorted({r["topology_seed"] for r in rs})
               for v in _per_init([r for r in rs if r["topology_seed"] == ts]).values()]
        comp = [r for r in rows if r["kind"] == "composition" and not r["excluded"]
                and r["init"] == init and r["regime"] == regime]
        out.append({"kind": "pooled", "init": init, "regime": regime,
                    "n_topologies": len({r["topology_seed"] for r in rs}),
                    "n_leaves": len({(r["topology_seed"], r["annotation"]) for r in rs}),
                    "n_inits": len(pis), **_counts(rs),
                    "median_share_recovered": float(np.median([v[0] for v in pis])),
                    "median_share_role_recovered": ri._nanmed([v[1] for v in pis]),
                    "median_share_lost": float(np.median([v[2] for v in pis])),
                    "n_preserved": sum(v[3] for v in pis),
                    "median_feature_r": ri._nanmed([r["feature_r_min"] for r in rs]),
                    "n_compositions": len({(r["topology_seed"], r["annotation"]) for r in comp}),
                    "n_composition_recovered": sum(r["recovered"] for r in comp),
                    "share_composition_recovered": (np.mean([r["recovered"] for r in comp])
                                                    if comp else ""),
                    "note": "counts over leaf x init pairs; n_preserved = inits (topology x seed) "
                            "with every leaf recovered; n_alive_different = neither recovered "
                            "nor lost"})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Study E amendment E-S: explanatory structure")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--init", choices=ri.INITS, default="kaiming")
    args = ap.parse_args()
    run(dry_run=args.dry_run, init=args.init)


if __name__ == "__main__":
    main()
