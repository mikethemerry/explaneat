"""Export a self-contained reproducibility bundle for the paper's EXPLANATIONS.

Reads the three frozen paper genomes + their explanations from Postgres and
writes static files under reproducibility/ so a reader needs NO database:
genome / phenotype / operations / annotations / evidence / split / performance
per model, plus a complete NEAT config (INI) and the exact preprocessed split
arrays (data.npz) so reproducibility/load.py runs offline.

Anonymity: the database URL / credentials are read from the environment (via
explaneat.db) and never written out; serialised objects carry no host/user/path
fields; copied scripts are redacted of author-identifying strings; a final grep
reports any residual hits.

Run: uv run python scripts/aaai27/export_repro_bundle.py
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

from explaneat.db import db
from explaneat.db.models import Experiment, Genome, Explanation, Dataset, DatasetSplit
from explaneat.core.config_utils import build_neat_config_text, load_neat_config
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.neuralneat import NeuralNeat

ROOT = "reproducibility"
EXP3_BRANCH = "analysis/aaai27-exp3-candidates"

RUNS = [
    dict(key="adult", label="UCI Adult / Census Income",
         experiment="dbab685c-e9c8-4118-bb4e-0fbdfab4c4c9",
         genome="4bc8fa07-bbf5-48e4-96a8-77be8b577ba3", table1_auc=0.905),
    dict(key="heart", label="UCI Cleveland Heart Disease",
         experiment="c8f3805e-bbb7-4eb9-aa07-cff73176d98c",
         genome="9022a9eb-ecf9-41ba-b138-28ed4d131c9d", table1_auc=0.897),
    dict(key="monk2", label="UCI MONK's Problem 2",
         experiment="45e998d7-dac1-4e6b-953e-3f1d32cbc59e",
         genome="23dde0fa-a635-4974-91aa-c05266ea0747", table1_auc=1.000),
]

# Author-identifying redactions applied to copied scripts / any emitted text.
REDACTIONS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"(/Users/|/home/)[^\s'\"]*"), "<path>"),
    (re.compile(r"\bMike Merry\b"), "[AUTHOR]"),
    (re.compile(r"\bMerry\b"), "[AUTHOR]"),
    (re.compile(r"\bMIKE\b"), "[AUTHOR]"),
    (re.compile(r"\bMike\b"), "[AUTHOR]"),
    (re.compile(r"\bAuckland\b"), "[INSTITUTION]"),
]


def redact(text):
    for pat, repl in REDACTIONS:
        text = pat.sub(repl, text)
    return text


def node_type(node_id, input_keys, output_keys):
    if node_id in output_keys:
        return "output"
    if node_id in input_keys or node_id < 0:
        return "input"
    return "hidden"


def genome_json(genome, config):
    ik = set(config.genome_config.input_keys)
    ok = set(config.genome_config.output_keys)
    nodes = [dict(id=nid, type=node_type(nid, ik, ok), bias=n.bias,
                  activation=n.activation, aggregation=n.aggregation,
                  response=n.response)
             for nid, n in sorted(genome.nodes.items())]
    conns = [dict(**{"from": k[0], "to": k[1]}, weight=c.weight, enabled=c.enabled)
             for k, c in sorted(genome.connections.items())]
    return dict(key=genome.key, input_node_ids=sorted(ik), output_node_ids=sorted(ok),
                num_hidden=sum(1 for n in nodes if n["type"] == "hidden"),
                nodes=nodes, connections=conns)


def phenotype_json(genome, config):
    pheno = ExplaNEAT(genome, config).get_phenotype_network()
    in_ids = set(pheno.input_node_ids)
    out_ids = set(pheno.output_node_ids)
    nodes = []
    for n in pheno.nodes:
        t = getattr(n.type, "value", n.type)
        nodes.append(dict(id=n.id, type=t, bias=getattr(n, "bias", None),
                          activation=getattr(n, "activation", None),
                          aggregation=getattr(n, "aggregation", None),
                          response=getattr(n, "response", None)))
    conns = [dict(**{"from": c.from_node, "to": c.to_node}, weight=c.weight)
             for c in pheno.connections if c.enabled]
    return dict(input_node_ids=list(pheno.input_node_ids),
                output_node_ids=list(pheno.output_node_ids),
                nodes=nodes, connections=conns)


def annotations_json(operations):
    anns = []
    by_id = {}
    for op in operations:
        if op.get("type") != "annotate":
            continue
        p = op.get("params", {})
        res = op.get("result", {})
        aid = res.get("annotation_id") or f"ann_{op.get('seq', 0)}"
        a = dict(id=aid, name=p.get("name"),
                 entry_nodes=p.get("entry_nodes", []),
                 exit_nodes=p.get("exit_nodes", []),
                 subgraph_nodes=p.get("subgraph_nodes", []),
                 hypothesis=p.get("hypothesis") or p.get("description"),
                 parent=p.get("parent_annotation_id"),
                 children=[])
        anns.append(a)
        by_id[aid] = a
    for a in anns:                      # derive children from parent links
        if a["parent"] and a["parent"] in by_id:
            by_id[a["parent"]]["children"].append(a["id"])
    return anns


def evidence_json(operations):
    out = []
    for op in operations:
        if op.get("type") != "annotate":
            continue
        p = op.get("params", {})
        ev = p.get("evidence") or {}
        records = ev.get("records", [])
        # legacy category entries, if any
        legacy = []
        for cat, items in ev.items():
            if cat in ("records", "_legacy") or not isinstance(items, list):
                continue
            legacy.extend(dict(category=cat, **it) for it in items)
        if records or legacy:
            out.append(dict(annotation=p.get("name"),
                            records=records, legacy_entries=legacy))
    return out


def export_run(run, session):
    d = os.path.join(ROOT, run["key"])
    os.makedirs(d, exist_ok=True)
    exp = session.query(Experiment).filter_by(id=run["experiment"]).first()
    g_row = session.query(Genome).filter_by(id=run["genome"]).first()
    expl = session.query(Explanation).filter_by(genome_id=run["genome"]).first()
    split = session.query(DatasetSplit).filter_by(id=exp.split_id).first()
    dataset = session.query(Dataset).filter_by(id=split.dataset_id).first()

    # Build config the SAME way the app does; capture the complete INI so the
    # bundle is DB-free. (neat_config_text is empty; config comes from config_json.)
    config = load_neat_config(exp.neat_config_text or "", exp.config_json)
    ini_text = build_neat_config_text(exp.neat_config_text or "", exp.config_json)
    genome = g_row.to_neat_genome(config)
    ops = expl.operations or [] if expl else []

    # preprocessed split arrays (z-normalised, as trained) for offline load.py
    X, y = dataset.get_data()
    mean = np.array(split.scaler_params["mean"])
    scale = np.where(np.array(split.scaler_params["scale"]) == 0, 1.0,
                     np.array(split.scaler_params["scale"]))
    Xs = ((X - mean) / scale).astype(np.float32)
    tr, te = split.train_indices, split.test_indices

    # performance (train/test AUC + accuracy) via the app's forward engine
    net = NeuralNeat(genome, config)
    def scores(idx):
        return net.forward(torch.tensor(Xs[idx], dtype=torch.float64)).detach().numpy().ravel()
    perf = {}
    for name, idx in (("train", tr), ("test", te)):
        pr = scores(idx)
        perf[name] = dict(n=len(idx), auc=round(float(roc_auc_score(y[idx], pr)), 4),
                          accuracy=round(float(accuracy_score(y[idx], (pr > 0.5))), 4))
    perf["reported_test_auc_table1"] = run["table1_auc"]

    files = {
        "genome.json": genome_json(genome, config),
        "phenotype.json": phenotype_json(genome, config),
        "operations.json": ops,
        "annotations.json": annotations_json(ops),
        "evidence.json": evidence_json(ops),
        "split.json": dict(
            dataset_name=dataset.name, encoding="one-hot",
            test_size=split.test_size, random_state=split.random_state,
            stratify=bool(split.stratify),
            train_size=len(tr), test_size_actual=len(te),
            n_features=int(Xs.shape[1]),
            scaler=dict(type=split.scaler_type,
                        mean=list(map(float, split.scaler_params["mean"])),
                        scale=list(map(float, split.scaler_params["scale"]))),
            note="Public UCI benchmark; 80:20 split at random_state=42. "
                 "data.npz holds the exact z-normalised arrays used here."),
        "performance.json": perf,
    }
    for fname, obj in files.items():
        # scrub any stray home paths from serialised content (verified clean; belt+braces)
        text = redact(json.dumps(obj, indent=2, default=str))
        with open(os.path.join(d, fname), "w") as f:
            f.write(text + "\n")

    with open(os.path.join(d, "neat_config.ini"), "w") as f:
        f.write(ini_text)
    np.savez_compressed(os.path.join(d, "data.npz"),
                        X_train=Xs[tr], y_train=y[tr].astype(np.int8),
                        X_test=Xs[te], y_test=y[te].astype(np.int8))
    return perf


def copy_scripts():
    dst = os.path.join(ROOT, "scripts")
    os.makedirs(dst, exist_ok=True)
    copied = []
    # scripts on the current checkout (exclude this bundle-builder itself: it
    # needs a DB, and its redaction/grep regexes literally contain the strings
    # we scrub for, which would self-flag)
    self_name = os.path.basename(__file__)
    for fn in sorted(os.listdir("scripts/aaai27")):
        if not fn.endswith(".py") or fn == self_name:
            continue
        with open(os.path.join("scripts/aaai27", fn)) as f:
            src = f.read()
        with open(os.path.join(dst, fn), "w") as f:
            f.write(redact(src))
        copied.append(fn)
    # MONK-2 dedup script from the unmerged branch (read via git, no checkout)
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{EXP3_BRANCH}:scripts/aaai27/ingest_monk2.py"],
            text=True)
        with open(os.path.join(dst, "ingest_monk2.py"), "w") as f:
            f.write(redact(raw))
        copied.append("ingest_monk2.py (from branch " + EXP3_BRANCH + ")")
    except subprocess.CalledProcessError as e:
        copied.append(f"ingest_monk2.py [NOT FOUND on {EXP3_BRANCH}: {e}]")
    return copied


def write_readme():
    txt = f"""# ExplaNEAT — reproducibility bundle (paper explanations)

Static export of the three explained models. **No database is required**: the
genomes and explanations are static JSON, and each model directory ships the
exact preprocessed split so `load.py` runs offline.

## Datasets

The three datasets are public UCI benchmarks — Adult / Census Income, Cleveland
Heart Disease, and MONK's Problem 2 — reconstructed with the given split
(`random_state = 42`, `test_size = 0.2`; see each `split.json`). MONK-2 is the
full 432-configuration input-space enumeration (deduplicated from the raw PMLB
file; see `scripts/ingest_monk2.py`). The exact z-normalised arrays used for
evaluation are included as `data.npz` so no download is needed to verify.

## Per-model files (`adult/`, `heart/`, `monk2/`)

- `genome.json` — full evolved genome: nodes (id, type, bias, activation,
  aggregation, response) and connections (from, to, weight, enabled). Same
  fields the app serialises to load a genome.
- `phenotype.json` — the active pruned phenotype (reachable nodes + enabled
  connections only).
- `operations.json` — the complete ordered event stream (node splits, identity
  inserts, renames, annotations) with parameters; replaying it reconstructs the
  annotated model state.
- `annotations.json` — resolved annotation hierarchy (name, entry/exit/subgraph
  nodes, hypothesis, parent/children).
- `evidence.json` — evidence records per annotation (closed-form formulas, SHAP,
  ablation studies) with payloads and narratives.
- `split.json` — dataset name, test_size, random_state, train/test sizes,
  stratify flag, and the StandardScaler params (mean/scale).
- `performance.json` — train/test AUC and accuracy (Table 1 test AUC recorded).
- `neat_config.ini` — complete NEAT config (so a `neat.Config` can be built
  offline).
- `data.npz` — z-normalised `X_train/y_train/X_test/y_test` for `load.py`.

## Reproduce the numbers (no DB)

```bash
uv run python reproducibility/load.py adult   # or heart / monk2
```

`load.py` rebuilds the network with the repo's phenotype/NeuralNeat builder from
`genome.json` + `neat_config.ini`, runs a forward pass on `data.npz`, and prints
train/test AUC + accuracy — which should match `performance.json`.

- **Coverage**: replay `operations.json` on the phenotype (repo
  `ModelStateEngine`) to obtain the annotated model state, then count covered
  nodes against the annotation subgraphs (repo `CoverageComputer`).
- **AUC**: as above via `load.py`.
- **Ablation** (e.g. Adult native-country edges): load `genome.json`, disable
  the documented connections on an in-memory copy, and re-run the forward pass
  on the test arrays — the delta reproduces the `evidence.json` ablation record.

## `scripts/`

The analysis scripts used to produce the paper's numbers (baselines, ablation
bootstrap CIs, densified twin, path-length checks, MONK-2 ingest). Author-
identifying strings have been redacted for anonymous review.
"""
    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.write(redact(txt))


LOAD_PY = '''"""Offline loader: rebuild a paper genome and reproduce its AUC — no database.

Usage: uv run python reproducibility/load.py <adult|heart|monk2>

Requires the `explaneat` package (the repo) importable, but NO database.
"""
import json
import os
import sys

import neat
import numpy as np
import torch
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from sklearn.metrics import roc_auc_score, accuracy_score

from explaneat.core.neuralneat import NeuralNeat

HERE = os.path.dirname(os.path.abspath(__file__))


def build_genome(gj, config):
    """Rebuild a neat genome from genome.json (mirrors deserialize_genome)."""
    g = neat.DefaultGenome(gj["key"])
    for n in gj["nodes"]:
        node = DefaultNodeGene(n["id"])
        node.bias = n["bias"]; node.response = n["response"]
        node.activation = n["activation"]; node.aggregation = n["aggregation"]
        g.nodes[n["id"]] = node
    for c in gj["connections"]:
        key = (c["from"], c["to"])
        conn = DefaultConnectionGene(key)
        conn.weight = c["weight"]; conn.enabled = c["enabled"]
        g.connections[key] = conn
    return g


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("adult", "heart", "monk2"):
        sys.exit("usage: load.py <adult|heart|monk2>")
    d = os.path.join(HERE, sys.argv[1])
    gj = json.load(open(os.path.join(d, "genome.json")))
    perf = json.load(open(os.path.join(d, "performance.json")))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(d, "neat_config.ini"))
    genome = build_genome(gj, config)
    net = NeuralNeat(genome, config)

    data = np.load(os.path.join(d, "data.npz"))
    print(f"{sys.argv[1]}: rebuilt genome ({len(genome.nodes)} nodes, "
          f"{len(genome.connections)} connections)")
    for split in ("train", "test"):
        X = torch.tensor(data[f"X_{split}"].astype(np.float64), dtype=torch.float64)
        y = data[f"y_{split}"]
        p = net.forward(X).detach().numpy().ravel()
        auc = roc_auc_score(y, p); acc = accuracy_score(y, (p > 0.5))
        exp = perf[split]
        ok = abs(auc - exp["auc"]) < 1e-3 and abs(acc - exp["accuracy"]) < 1e-3
        print(f"  {split}: AUC={auc:.4f} (reported {exp['auc']})  "
              f"acc={acc:.4f} (reported {exp['accuracy']})  "
              f"{'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
'''


def anonymity_grep():
    patterns = {
        "author name": re.compile(r"\bMike\b|\bMerry\b", re.I),
        "Auckland": re.compile(r"Auckland", re.I),
        "email (@word.tld)": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "raw @": re.compile(r"@"),
        "/Users/": re.compile(r"/Users/"),
        "/home/": re.compile(r"/home/"),
    }
    hits = {k: [] for k in patterns}
    for base, _, fnames in os.walk(ROOT):
        for fn in fnames:
            if fn.endswith(".npz"):
                continue
            path = os.path.join(base, fn)
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            for k, pat in patterns.items():
                n = len(pat.findall(text))
                if n:
                    hits[k].append((path, n))
    return hits


def main():
    os.makedirs(ROOT, exist_ok=True)
    perfs = {}
    with db.session_scope() as s:
        for run in RUNS:
            perfs[run["key"]] = export_run(run, s)
            print(f"exported {run['key']}")
    copied = copy_scripts()
    write_readme()
    with open(os.path.join(ROOT, "load.py"), "w") as f:
        f.write(LOAD_PY)
    print("copied scripts:", copied)

    print("\n=== performance vs Table 1 ===")
    for run in RUNS:
        p = perfs[run["key"]]
        got = p["test"]["auc"]; want = run["table1_auc"]
        print(f"  {run['key']:6}: test AUC {got:.3f} (Table 1 {want:.3f})  "
              f"{'MATCH' if abs(got - want) <= 0.001 else 'CHECK'}")

    print("\n=== anonymity grep (reproducibility/) ===")
    hits = anonymity_grep()
    for k, lst in hits.items():
        if lst:
            print(f"  {k}: {sum(n for _, n in lst)} hit(s) in {len(lst)} file(s)")
            for path, n in lst:
                print(f"      {path} ({n})")
        else:
            print(f"  {k}: none")


if __name__ == "__main__":
    main()
