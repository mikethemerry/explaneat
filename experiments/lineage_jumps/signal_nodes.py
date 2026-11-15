"""Jumps as signal discovery (RESULTS.md section 11), stage 1: per-lineage computation (read-only).

For one lineage this writes ``<scratch>/signal/<run>.json`` with
- the champion's hidden nodes (phenotype) and, where explained, its named parts: each mean-ablated
  at its training-mean activation (``explaneat.verify.probe.Ablation(to="mean")``: the bias the
  unit passes on is kept), with the drop in training AUC, the rise in training log-loss and the
  drop in each ground-truth signal r; the generation its node gene entered the lineage
  (meaning.py's rule: the last lineage step that added it);
- for every lineage ancestor (one per generation): training AUC and the signal r values.

Signals:
- shape runs (score_g300.py's scoring): U r = Pearson r of the model logit over sodium 128..150
  mmol/L (others typical) with the true U term; saddle r = Pearson r of the mean-removed logit on a
  41 x 41 heart rate x SBP grid (training 1st-99th percentiles, others typical) with the true
  -zh*zp surface. shape-A has no saddle in its truth, so only U r.
- MONK-2 (true rule: exactly two of the six attributes at their first value; k = that count):
  lower step r = Pearson r of the logit with [k = 2] on the training rows with k <= 2 (the rule's
  "at least two" half); upper step r = the same on the rows with k >= 2 (the "at most two" half).
  Each node is also probed (concept_probe.score_concept) against the six "a_i is first value"
  indicators and k.
- churn200: none (AUC only).

Sources: DB genomes (shape runs, explaneat_dev) or the replay dumps (monk2, churn200), as build.py.
Nothing is written to any database.

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/lineage_jumps/signal_nodes.py shape-C-s1-g300
"""
import copy
import dataclasses
import gzip
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments" / "ushape_saddle")):
    if p not in sys.path:
        sys.path.insert(0, p)

from threadpoolctl import threadpool_limits  # noqa: E402

threadpool_limits(1)
import torch  # noqa: E402

torch.set_num_threads(1)

from sqlalchemy import text  # noqa: E402

import data as shape_data  # noqa: E402  (experiments/ushape_saddle/data.py)
from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.genome_network import NodeType  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.serialization import deserialize_genome  # noqa: E402
from explaneat.verify.model import VerifyModel, load_base  # noqa: E402
from explaneat.verify.probe import Ablation, hold_row, to_z  # noqa: E402

SCRATCH = Path(os.environ.get(
    "JUMPS_SCRATCH",
    "/private/tmp/claude-501/-Users-mike-dev-explaneat/49dbb6fc-1545-4ef1-86df-c44223c670ba/scratchpad"))
OUT = SCRATCH / "signal"

LINEAGES = {
    "shape-C-s1-g300": dict(kind="both"),
    "shape-C-s2-g300": dict(kind="both"),
    "shape-A-s0": dict(kind="u", explained=("explaneat_dev", None)),
    "monk2-s7": dict(kind="monk2", explained=("explaneat_v4", None), dump="monk2-s7"),
    "churn200-s7": dict(kind="none", explained=("explaneat_v4", None), dump="churn200-s7"),
}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def logloss(y, eta):
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class Signals:
    """Grids and targets for one dataset, built once from the champion's model."""

    def __init__(self, vm, kind):
        self.kind = kind
        m = vm.m
        if kind in ("u", "both"):
            names = list(vm.base.feature_names)
            assert names == shape_data.FEATURES, names
            col = {n: i for i, n in enumerate(names)}
            base = hold_row(vm, "typical")
            xs = np.arange(128.0, 151.0)
            R = np.repeat(base[None, :], len(xs), axis=0)
            R[:, col["sodium"]] = xs
            self.Zu = to_z(vm, R)
            self.tu = next(iter(shape_data.u_term(R).values()))
            if kind == "both":
                lo_h, hi_h = np.percentile(m.raw_train[:, col["heart_rate"]], [1, 99])
                lo_p, hi_p = np.percentile(m.raw_train[:, col["sbp"]], [1, 99])
                H, P = np.meshgrid(np.linspace(lo_h, hi_h, 41), np.linspace(lo_p, hi_p, 41))
                R = np.repeat(base[None, :], H.size, axis=0)
                R[:, col["heart_rate"]] = H.ravel()
                R[:, col["sbp"]] = P.ravel()
                self.Zs = to_z(vm, R)
                T = next(iter(shape_data.saddle_term(R).values()))
                self.ts = T - T.mean()
        if kind == "monk2":
            names = list(vm.base.feature_names)
            first = [names.index(f"a{i}:1") for i in range(1, 7)]
            raw = m.raw_train
            self.ind = (raw[:, first] > 0.5).astype(float)
            self.k = self.ind.sum(axis=1)
            self.target = (self.k == 2).astype(float)
            assert np.array_equal(self.target, m.y), "MONK-2 rule does not reproduce the labels"
            self.lo = self.k <= 2
            self.hi = self.k >= 2

    def names(self):
        return {"both": ["u_r", "saddle_r"], "u": ["u_r"], "monk2": ["lower_r", "upper_r"],
                "none": []}[self.kind]

    def score(self, f, eta_train):
        """f: Z -> logit. eta_train: logit on the training rows."""
        out = {}
        if self.kind in ("u", "both"):
            out["u_r"] = corr(f(self.Zu), self.tu)
        if self.kind == "both":
            s = f(self.Zs)
            out["saddle_r"] = corr(s - s.mean(), self.ts)
        if self.kind == "monk2":
            out["lower_r"] = corr(eta_train[self.lo], self.target[self.lo])
            out["upper_r"] = corr(eta_train[self.hi], self.target[self.hi])
        return out


def full_scores(vm, sig, f=None):
    f = f or (lambda Z: vm.m.logit(Z))
    eta = f(vm.m.X)
    out = dict(train_auc=auc(vm.m.y, eta), train_logloss=logloss(vm.m.y, eta))
    out.update(sig.score(f, eta))
    return out


def psql_json(dbname, sql):
    out = subprocess.run(["psql", dbname, "-At", "-c", sql], capture_output=True, text=True,
                         check=True).stdout.strip()
    return json.loads(out) if out else None


def ancestor_vm(base, config, gd, gid):
    g = deserialize_genome(gd, config)
    ph = ExplaNEAT(g, config).get_phenotype_network()
    b = dataclasses.replace(base, phenotype=ph, genome_id=str(gid))
    return VerifyModel(b, operations=[])


def monk_probe(vm, sig, nid, act):
    from explaneat.analysis.concept_probe import score_concept
    from explaneat.analysis.concepts import _concept
    res = {}
    for i in range(6):
        c = _concept(f"a{i + 1}_is_first", f"a{i + 1} = 1", sig.ind[:, i], "rule")
        res[c.name] = score_concept(c, act)["score"]
    res["k_first_count"] = score_concept(_concept("k_first_count", "count", sig.k, "rule"), act)["score"]
    res["k_eq_2"] = score_concept(_concept("k_eq_2", "rule", sig.target, "rule"), act)["score"]
    best = max(res, key=lambda k: -1 if res[k] is None else res[k])
    return dict(scores=res, best=best, best_score=res[best])


def main(label):
    spec = LINEAGES[label]
    run = json.loads((SCRATCH / "runs" / f"{label}.json").read_text())
    champ_gid = run["champion"]["genome_id"]
    lin = run["lineage"]
    assert lin[-1]["key"] == run["champion"]["key"]
    t0 = time.time()
    base = load_base(champ_gid, cache=False)
    with db.session_scope() as s:
        s.execute(text("SET TRANSACTION READ ONLY"))
        e = s.execute(text("select neat_config_text, config_json from experiments where id = :i"),
                      {"i": run["experiment_id"]}).one()
        config = load_neat_config(e[0] or "", e[1])
        gdata = {}
        if "dump" not in spec:
            ids = [r["genome_id"] for r in lin]
            for rid, d in s.execute(text("select id, genome_data from genomes where id = any(:ids)"),
                                    {"ids": [uuid.UUID(x) for x in ids]}):
                gdata[str(rid)] = d
    if "dump" in spec:
        need = {(r["generation"], r["key"]): i for i, r in enumerate(lin)}
        byi = {}
        with gzip.open(SCRATCH / "dump" / f"{spec['dump']}.genomes.jsonl.gz", "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if (r["g"], r["k"]) in need:
                    byi[need[(r["g"], r["k"])]] = r["d"]
        get = lambda i: byi[i]  # noqa: E731
    else:
        get = lambda i: gdata[lin[i]["genome_id"]]  # noqa: E731

    vm = VerifyModel(base, operations=[])
    sig = Signals(vm, spec["kind"])
    champ_scores = full_scores(vm, sig)
    print(f"{label}: champion {champ_gid} gen {run['champion']['generation']}", champ_scores, flush=True)

    # entry generation of every champion node gene along the lineage (meaning.py's rule)
    entry = {}
    for r in lin[1:]:
        for n in r.get("added_nodes", []):
            entry[str(n)] = r["generation"]
    centry = {}
    for r in lin[1:]:
        for c in r.get("added_conns", []) + r.get("enabled_conns", []):
            centry[c] = r["generation"]
    vals = vm.m._vals_train()
    hidden = [n for n in vm.structure.nodes if n.type == NodeType.HIDDEN]
    geno_hidden = [k for k, v in json.loads(json.dumps(get(len(lin) - 1)))["nodes"].items()
                   if int(k) >= 0 and int(k) not in config.genome_config.output_keys]
    nodes = []
    for n in hidden:
        ab = Ablation(vm, nodes=[n.id], to="mean")
        sc = full_scores(vm, sig, ab.logit)
        act = np.asarray(vals[n.id], float)
        inc = [f"{c.from_node}_{c.to_node}" for c in vm.structure.connections
               if c.enabled and str(c.to_node) == n.id]
        row = dict(node=n.id, activation=n.activation, entry_generation=entry.get(n.id, 0),
                   in_conn_entries={c: centry.get(c, 0) for c in inc},
                   last_rewired=max([entry.get(n.id, 0)] + [centry.get(c, 0) for c in inc]),
                   mean_activation=float(act.mean()), active_share=float(np.mean(act > 1e-9)),
                   n_in=sum(1 for c in vm.structure.connections if c.enabled and str(c.to_node) == n.id),
                   n_out=sum(1 for c in vm.structure.connections if c.enabled and str(c.from_node) == n.id),
                   ablated=sc,
                   drop={k: champ_scores[k] - sc[k] for k in champ_scores if k != "train_logloss"},
                   logloss_rise=sc["train_logloss"] - champ_scores["train_logloss"])
        if spec["kind"] == "monk2":
            row["probe"] = monk_probe(vm, sig, n.id, act)
        nodes.append(row)
        print(f"  node {n.id} born {row['entry_generation']}: "
              + " ".join(f"{k} {v:+.3f}" for k, v in row["drop"].items()), flush=True)

    conns = []
    for c in vm.structure.connections:
        if not c.enabled:
            continue
        a, b = str(c.from_node), str(c.to_node)
        ab = Ablation(vm, edges=[f"{a}->{b}"], to="mean")
        sc = full_scores(vm, sig, ab.logit)
        conns.append(dict(gene=f"{a}_{b}", from_node=a, to_node=b, weight=float(c.weight),
                          entry_generation=centry.get(f"{a}_{b}", 0),
                          drop={k: champ_scores[k] - sc[k] for k in champ_scores if k != "train_logloss"},
                          logloss_rise=sc["train_logloss"] - champ_scores["train_logloss"]))
    print(f"  {len(conns)} connections ablated", flush=True)

    parts = []
    if "explained" in spec:
        dbname = spec["explained"][0]
        ops = psql_json(dbname, f"select operations from explanations where genome_id = '{champ_gid}'")
        if ops:
            vme = VerifyModel(base, operations=ops)
            ce = full_scores(vme, sig)
            for p in vme.parts:
                if p.is_root:
                    continue
                ab = Ablation(vme, parts=[p.id], to="mean")
                sc = full_scores(vme, sig, ab.logit)
                parts.append(dict(id=p.id, name=p.name, exits=list(p.exits), is_leaf=not p.children,
                                  children=list(p.children),
                                  nodes=sorted({str(x) for x in (p.ann.get("subgraph_nodes") or [])}),
                                  drop={k: ce[k] - sc[k] for k in ce if k != "train_logloss"},
                                  logloss_rise=sc["train_logloss"] - ce["train_logloss"]))
                print(f"  part {p.name}: " + " ".join(f"{k} {v:+.3f}" for k, v in parts[-1]["drop"].items()),
                      flush=True)

    anc = []
    for i, r in enumerate(lin):
        try:
            v = ancestor_vm(base, config, get(i), r.get("genome_id") or f"{label}:{r['generation']}:{r['key']}")
            sc = full_scores(v, sig)
            sc["n_hidden"] = sum(1 for n in v.structure.nodes if n.type == NodeType.HIDDEN)
        except Exception as ex:  # noqa: BLE001
            sc = dict(error=f"{type(ex).__name__}: {ex}")
        sc.update(generation=r["generation"], key=r["key"], fitness=r["fitness"])
        anc.append(sc)
        if i % 25 == 0:
            print(f"  ancestor {r['generation']}: " + " ".join(
                f"{k} {v:.3f}" for k, v in sc.items() if isinstance(v, float)), flush=True)
    err = max(abs(a["train_auc"] - a["fitness"]) for a in anc if "train_auc" in a)
    out = dict(run=label, kind=spec["kind"], signals=sig.names(), champion=run["champion"],
               champion_scores=champ_scores, n_genotype_hidden=len(geno_hidden), nodes=nodes, connections=conns,
               parts=parts, ancestors=anc, ancestor_auc_vs_fitness_max_abs_err=err,
               seconds=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{label}.json").write_text(json.dumps(out, default=float))
    print(f"{label}: done in {time.time() - t0:.0f} s; ancestor AUC vs fitness max err {err:.2e}", flush=True)


if __name__ == "__main__":
    db.init_db()
    for lab in sys.argv[1:]:
        main(lab)
