"""Import a trained ladder MLP into Postgres ``explaneat_dev`` as a NEAT genome, and check parity.

    DATABASE_URL=postgresql://localhost/explaneat_dev \
        uv run python experiments/explanation_barrier/import_mlp.py <task> <arch> [--dry-run]

The MLP (models/<task>-<arch>.pt) becomes a genome with input nodes -1..-d (feature order), hidden
nodes 1..H (layer by layer; ReLU, sum aggregation, response 1), output node 0 (sigmoid, as every
engine applies to the output) and one enabled connection per weight. Writes only new rows:

  * Dataset + DatasetSplit for a synthetic task (get-or-create by its opaque name; the description
    does not give the generating function). Connect-4 reuses ``connect4_win_ordinal`` and its split
    (the same rows and scaler as experiments/complex_real), read only;
  * one Experiment ``ladder-<label>`` (refuses if it exists), one Population, one Genome.

Parity (asserted, max |diff| < 1e-6 on the predicted probability, train and test): the genome as the
MCP tools load it (DB -> load_genome_and_config -> ExplaNEAT phenotype -> StructureNetwork), and
NeuralNeat on the DB genome, against the PyTorch MLP on the same rows. Writes logs/<label>.import.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import neat
import numpy as np
import torch
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import data  # noqa: E402
from train import MLP, parse  # noqa: E402

CONFIG_TEMPLATE = REPO / "experiments" / "10seed" / "configs" / "heart.ini"
LOGS = HERE / "logs"


def label_of(task, arch):
    """Opaque label: the explaining agent sees this name, not the generating function."""
    return "lad-" + hashlib.sha1(f"{task}/{arch}".encode()).hexdigest()[:5]


def dataset_name(task):
    return "connect4_win_ordinal" if task == "c4" else "ladder_syn_" + hashlib.sha1(task.encode()).hexdigest()[:5]


def config_text(d):
    s = CONFIG_TEMPLATE.read_text()
    s = s.replace("num_inputs = 26", f"num_inputs = {d}")
    for k in ("bias_max_value", "weight_max_value", "response_max_value"):
        s = s.replace(f"{k} = 30.0", f"{k} = 1000.0")
    for k in ("bias_min_value", "weight_min_value", "response_min_value"):
        s = s.replace(f"{k} = -30.0", f"{k} = -1000.0")
    assert f"num_inputs = {d}" in s
    return s


def load_config(text):
    p = LOGS / "_tmp_config.ini"
    p.parent.mkdir(exist_ok=True)
    p.write_text(text)
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(p))


def mlp_to_genome(net: MLP, d: int, key=1):
    g = neat.DefaultGenome(key)
    prev = [-(i + 1) for i in range(d)]
    nid = 1
    for layer in net.hidden:
        W, b = layer.weight.detach().numpy(), layer.bias.detach().numpy()
        cur = []
        for j in range(W.shape[0]):
            n = DefaultNodeGene(nid)
            n.bias, n.response, n.activation, n.aggregation = float(b[j]), 1.0, "relu", "sum"
            g.nodes[nid] = n
            for i, src in enumerate(prev):
                c = DefaultConnectionGene((src, nid))
                c.weight, c.enabled = float(W[j, i]), True
                g.connections[(src, nid)] = c
            cur.append(nid)
            nid += 1
        prev = cur
    W, b = net.out.weight.detach().numpy(), net.out.bias.detach().numpy()
    o = DefaultNodeGene(0)
    o.bias, o.response, o.activation, o.aggregation = float(b[0]), 1.0, "sigmoid", "sum"
    g.nodes[0] = o
    for i, src in enumerate(prev):
        c = DefaultConnectionGene((src, 0))
        c.weight, c.enabled = float(W[0, i]), True
        g.connections[(src, 0)] = c
    return g


def get_or_create_dataset(session, task, d):
    from explaneat.db.models import Dataset, DatasetSplit
    name = dataset_name(task)
    ds = session.query(Dataset).filter_by(name=name).first()
    if ds is not None:
        split = session.query(DatasetSplit).filter_by(dataset_id=ds.id).first()
        X_db, y_db = ds.get_data()
        assert np.array_equal(X_db, d["X_raw"]) and np.array_equal(np.asarray(y_db).ravel(), d["y_all"])
        assert list(split.train_indices) == [int(i) for i in d["train_rows"]]
        assert np.allclose(split.scaler_params["mean"], d["mean"]) and np.allclose(split.scaler_params["scale"], d["scale"])
        return ds, split, False
    assert task != "c4", "the Connect-4 dataset must already exist (experiments/complex_real)"
    names = d["names"]
    ds = Dataset(name=name, version="v1", source="synthetic",
                 description="Synthetic tabular data with 10 inputs for the explanation complexity ladder "
                             "(experiments/explanation_barrier). Exploratory. The generating function is "
                             "withheld from the explaining agent.",
                 num_samples=len(d["y_all"]), num_features=len(names), num_classes=2, feature_names=names,
                 feature_types=d["types"], target_name="outcome", target_description="1 = outcome, 0 = no outcome",
                 class_names=["0", "1"],
                 additional_metadata=dict(task_type="classification", experiment="experiments/explanation_barrier",
                                          n_rows=d["meta"]["n_rows"], n_train=d["meta"]["n_train"],
                                          n_test=d["meta"]["n_test"], scaler=d["meta"]["scaler"]))
    ds.set_data(d["X_raw"].astype(np.float64), d["y_all"].astype(np.int64))
    session.add(ds)
    session.flush()
    tr, te = d["train_rows"], d["test_rows"]
    split = DatasetSplit(dataset_id=ds.id, name="stratified 80/20 seed 42", split_type="train_test",
                         test_size=0.2, random_state=42, shuffle=True, stratify=True,
                         train_indices=[int(i) for i in tr], test_indices=[int(i) for i in te],
                         scaler_type="StandardScaler",
                         scaler_params={"mean": [float(v) for v in d["mean"]], "scale": [float(v) for v in d["scale"]]},
                         preprocessing_steps=["raw units", "StandardScaler (train rows)"],
                         train_size=len(tr), test_size_actual=len(te))
    session.add(split)
    session.flush()
    return ds, split, True


def parity(gid, split_id, net):
    from explaneat.core.explaneat import ExplaNEAT
    from explaneat.core.neuralneat import NeuralNeat
    from explaneat.core.structure_network import StructureNetwork
    from mcp_server.helpers import load_genome_and_config, load_split_data
    from mcp_server.server import get_db
    db = get_db()
    with db.session_scope() as s:
        genome, config, _ = load_genome_and_config(s, gid)
        out = {}
        for nm in ("train", "test"):
            X, y, *_ = load_split_data(s, split_id, nm, 1.0, 10**9)
            Xt = torch.as_tensor(np.asarray(X, dtype=np.float64))
            with torch.no_grad():
                ref = net(Xt).numpy().ravel()
                nn_ = NeuralNeat(genome, config).forward(Xt).numpy().ravel()
            sn = StructureNetwork(ExplaNEAT(genome, config).get_phenotype_network())
            sp = sn.forward(Xt).detach().numpy().ravel()
            out[nm] = dict(n=int(len(y)), max_abs_diff_structure_network=float(np.max(np.abs(sp - ref))),
                           max_abs_diff_neuralneat=float(np.max(np.abs(nn_ - ref))),
                           auc_torch=float(roc_auc_score(np.asarray(y).ravel(), ref)),
                           auc_structure_network=float(roc_auc_score(np.asarray(y).ravel(), sp)))
    worst = max(max(v["max_abs_diff_structure_network"], v["max_abs_diff_neuralneat"]) for v in out.values())
    out["max_abs_diff"] = worst
    out["ok"] = bool(worst < 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("arch")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from explaneat.db.base import db
    from explaneat.db.models import Experiment, Genome, Population
    from explaneat.db.serialization import serialize_population_config

    label = label_of(a.task, a.arch)
    d = data.prepare(a.task)
    nd = d["X_train"].shape[1]
    net = MLP(nd, parse(a.arch))
    net.load_state_dict(torch.load(HERE / "models" / f"{a.task}-{a.arch}.pt"))
    net.eval()
    metrics = json.loads((HERE / "models" / f"{a.task}-{a.arch}.json").read_text())
    text = config_text(nd)
    config = load_config(text)
    genome = mlp_to_genome(net, nd)
    genome.fitness = metrics["train_auc"]
    exp_name = f"ladder-{label}"
    db.init_db()
    with db.session_scope() as session:
        if session.query(Experiment).filter_by(name=exp_name).first():
            sys.exit(f"experiment {exp_name} already exists; not touching it")
        ds, split, created = get_or_create_dataset(session, a.task, d)
        cj = serialize_population_config(config)
        cj["training"] = dict(fitness_function="auc", genome_type="DefaultGenome", plain_mlp=True,
                              optimiser="Adam, minibatch BCE, early stopping (experiments/explanation_barrier/train.py)")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True).strip()
        exp = Experiment(experiment_sha=commit[:40], name=exp_name,
                         description=f"Explanation complexity ladder network {label}: a plain dense ReLU MLP "
                                     f"(hidden layers {a.arch}) trained by backprop and imported as a genome. "
                                     "Exploratory, not pre-registered (experiments/explanation_barrier).",
                         dataset_name=ds.name, dataset_version=ds.version, dataset_id=ds.id, split_id=split.id,
                         config_json=cj, neat_config_text=text, start_time=datetime.utcnow(),
                         end_time=datetime.utcnow(), status="completed", git_commit_sha=commit[:40],
                         git_branch=branch, random_seed=0,
                         hardware_info=dict(device="cpu", run_params=dict(label=label, hidden=a.arch, in_memory=True)))
        session.add(exp)
        session.flush()
        pop = Population(experiment_id=exp.id, generation=0, population_size=1, num_species=0,
                         best_fitness=genome.fitness, mean_fitness=genome.fitness, stdev_fitness=0.0, config_json=cj)
        session.add(pop)
        session.flush()
        row = Genome.from_neat_genome(genome, pop.id)
        row.mutation_history = dict(note="plain MLP imported by experiments/explanation_barrier/import_mlp.py")
        session.add(row)
        session.flush()
        out = dict(label=label, experiment=exp_name, experiment_id=str(exp.id), genome_id=str(row.id),
                   dataset=ds.name, dataset_id=str(ds.id), split_id=str(split.id), dataset_created=created,
                   hidden=a.arch, n_params=metrics["n_params"])
        if a.dry_run:
            session.rollback()
    if a.dry_run:
        print(json.dumps(out, indent=2))
        print("dry run: rolled back")
        return
    out["parity"] = parity(out["genome_id"], out["split_id"], net)
    LOGS.mkdir(exist_ok=True)
    (LOGS / f"{label}.import.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    assert out["parity"]["ok"], "parity check failed"


if __name__ == "__main__":
    main()
