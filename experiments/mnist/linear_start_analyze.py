"""Analysis for the MNIST-10 linear-start series (exploratory, not pre-registered).

    uv run python experiments/mnist/linear_start_analyze.py controls      # linear controls
    uv run python experiments/mnist/linear_start_analyze.py runs          # per-run table
    uv run python experiments/mnist/linear_start_analyze.py hidden <name> # hidden units of one run

``controls``: a 784 -> 10 linear model trained exactly like a PropNEAT champion
lineage with no structural change (fresh Adadelta(lr=1.5) every 5 epochs, softmax CE;
LINEAR_CHECK.md shows this equals the PropNEAT step bit for bit), from N(0, 1) (seeds
0-2) and from the LR coefficients, for 500 and 2,500 epochs.  This is "linear with the
same training budget" against which "past linear" is judged, besides LR itself.

``runs``: one row per run from logs/linear_start/<name>.{json,csv} plus hidden-unit
activity of the best genome (dead / always-on / switching on the train rows).

``hidden``: what the hidden units of one run add over its linear backbone: per-unit
input pixels, active fraction, output weights, and the change in accuracy / CE
(overall and per class) when the hidden units are mean-ablated (replaced by their
train-mean output) or removed (with their connections), on train, validation and test.

Writes experiments/mnist/logs/linear_start/{controls,runs,hidden_<name>}.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import linear_start as LS  # noqa: E402
from run_seed import RUNS_DIR  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402

OUT = LS.LOG_DIR
torch.set_num_threads(1)


def ce_acc(logits, y):
    lt = torch.as_tensor(logits)
    yt = torch.as_tensor(y, dtype=torch.long)
    return dict(ce=float(F.cross_entropy(lt, yt)), acc=float((lt.argmax(1) == yt).double().mean()))


def per_class_acc(logits, y):
    pred = np.asarray(logits).argmax(1)
    return [float((pred[y == c] == c).mean()) for c in range(10)]


# ---------------------------------------------------------------- controls
def linear_trajectory(W, b, d, epochs, checkpoints):
    W = torch.nn.Parameter(torch.tensor(W.copy()))
    b = torch.nn.Parameter(torch.tensor(b.copy()))
    X = torch.as_tensor(d["X_train"])
    y = torch.as_tensor(d["y_train"], dtype=torch.long)
    out = {}
    for e in range(epochs):
        if e % 5 == 0:
            opt = torch.optim.Adadelta([W, b], lr=1.5)
        loss = F.cross_entropy(X @ W + b, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e + 1 in checkpoints:
            with torch.no_grad():
                out[e + 1] = {s: ce_acc(torch.as_tensor(d[f"X_{s}"]) @ W + b, d[f"y_{s}"])
                              for s in ("train", "val", "test")}
    return out


def controls():
    d = LS.load_data()
    coef, intercept = LS.fit_lr(d)
    res = {"lr_sklearn": {s: ce_acc(d[f"X_{s}"] @ coef.T + intercept, d[f"y_{s}"])
                          for s in ("train", "val", "test")}}
    cps = {500, 2500}
    for seed in range(3):
        rng = torch.Generator().manual_seed(seed)
        W = torch.randn(784, 10, generator=rng, dtype=torch.float64).numpy()
        b = torch.randn(10, generator=rng, dtype=torch.float64).numpy()
        res[f"normal_init_s{seed}"] = linear_trajectory(W, b, d, 2500, cps)
        print(seed, json.dumps(res[f"normal_init_s{seed}"][500]["test"]), flush=True)
    res["lr_init"] = linear_trajectory(coef.T, intercept, d, 2500, cps)
    (OUT / "controls.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["lr_init"], indent=1))


# ---------------------------------------------------------------- genome forward
def load_run(name):
    with gzip.open(RUNS_DIR / name / "best_genome.pkl.gz", "rb") as f:
        b = pickle.load(f)
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as t:
        t.write(b["config_text"])
    config = LS.load_config(Path(t.name), False)
    return b["genome"], config, b["meta"]


class Forward:
    """Numpy forward pass of a genome with per-node activations (ReLU hidden, identity out).

    ``ablate``: {hidden node: constant} replaces that unit's output by a constant;
    ``remove``: hidden nodes dropped together with all their connections.
    """

    def __init__(self, genome, config):
        from neat.graphs import feed_forward_layers

        gc = config.genome_config
        self.inputs, self.outputs = list(gc.input_keys), list(gc.output_keys)
        self.genome = genome
        conns = [k for k, c in genome.connections.items() if c.enabled]
        self.layers = feed_forward_layers(self.inputs, self.outputs, conns)
        self.incoming = {}
        for (a, o), c in genome.connections.items():
            if c.enabled:
                self.incoming.setdefault(o, []).append((a, c.weight))
        self.hidden = sorted(n for layer in self.layers for n in layer if n not in self.outputs)

    def __call__(self, X, ablate=None, remove=()):
        ablate, remove = ablate or {}, set(remove)
        v = {k: X[:, i] for i, k in enumerate(self.inputs)}
        for layer in self.layers:
            for n in layer:
                if n in remove:
                    continue
                s = np.full(X.shape[0], self.genome.nodes[n].bias)
                for a, w in self.incoming.get(n, []):
                    if a in v:
                        s = s + w * v[a]
                if n in self.outputs:
                    v[n] = s
                else:
                    v[n] = np.full(X.shape[0], ablate[n]) if n in ablate else np.maximum(s, 0.0)
        logits = np.stack([v.get(o, np.full(X.shape[0], self.genome.nodes[o].bias))
                           for o in self.outputs], axis=1)
        return logits, v


def hidden_activity(genome, config, X):
    fw = Forward(genome, config)
    logits, v = fw(X)
    frac = {h: float((v[h] > 0).mean()) for h in fw.hidden}
    kinds = {"switching": 0, "always_on": 0, "dead": 0}
    for f in frac.values():
        kinds["dead" if f == 0 else "always_on" if f == 1 else "switching"] += 1
    return fw, logits, frac, kinds


def parity(genome, config, X, logits):
    with torch.no_grad():
        nn_out = NeuralNeat(genome, config).forward(torch.as_tensor(X)).numpy()
    return float(np.abs(nn_out - logits).max())


# ---------------------------------------------------------------- runs table
def runs():
    d = LS.load_data()
    rows = []
    for js in sorted(OUT.glob("ls-*.json")):
        s = json.loads(js.read_text())
        name = s["name"]
        g, config, _ = load_run(name)
        fw, logits, frac, kinds = hidden_activity(g, config, d["X_train"])
        par = parity(g, config, d["X_train"][:2000], logits[:2000])
        gens = list(csv.DictReader(open(OUT / f"{name}.csv")))
        ns = [int(r["n_species"]) for r in gens]
        rows.append(dict(name=name, init=s["init"], epochs=s["n_epochs_backprop"],
                         generations=s["n_generations"], overrides=s["overrides"],
                         controller=s["controller"], penalty=s["hidden_penalty"],
                         weighted=s["addnode_weighted"], best_generation=s["best_generation"],
                         train=s["train"], val=s["val"], test=s["test"],
                         hidden=len(fw.hidden), hidden_kinds=kinds, hidden_active_fraction=frac,
                         conns=s["best_size"]["conns"], backbone=s["best_backbone"],
                         inputs_used=s["best_size"]["inputs_used"], depth=s["best_size"]["depth"],
                         species=dict(start=ns[0], end=ns[-1], max=max(ns),
                                      median=float(np.median(ns)),
                                      every10=[ns[i] for i in range(0, len(ns), 10)]),
                         pop_with_hidden_end=int(gens[-1]["pop_with_hidden"]),
                         pop_hidden_max_end=int(gens[-1]["pop_hidden_max"]),
                         wall_min=round(s["wall_clock_s"] / 60, 1), parity_max_abs=par))
        print(name, rows[-1]["train"]["acc"], rows[-1]["val"]["acc"], rows[-1]["test"]["acc"],
              len(fw.hidden), kinds, f"parity {par:.1e}", flush=True)
    (OUT / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")


# ---------------------------------------------------------------- hidden units of one run
def hidden(name):
    d = LS.load_data()
    g, config, meta = load_run(name)
    fw, logits_tr, frac, kinds = hidden_activity(g, config, d["X_train"])
    par = parity(g, config, d["X_train"][:2000], logits_tr[:2000])
    ins, outs = set(fw.inputs), set(fw.outputs)
    _, v_tr = fw(d["X_train"])
    means = {h: float(v_tr[h].mean()) for h in fw.hidden}

    def pix(k):
        i = fw.inputs.index(k)
        return f"r{i // 28:02d}c{i % 28:02d}"

    units = []
    for h in fw.hidden:
        fan_in = [(pix(a) if a in ins else f"h{a}", round(w, 3)) for a, w in fw.incoming.get(h, [])]
        fan_out = [(o, round(c.weight, 3)) for (a, o), c in g.connections.items()
                   if c.enabled and a == h]
        units.append(dict(node=h, bias=round(g.nodes[h].bias, 3), active_fraction=frac[h],
                          train_mean=means[h], fan_in=fan_in, fan_out=fan_out))

    res = dict(name=name, meta=meta, parity_max_abs=par, hidden_kinds=kinds, units=units)
    variants = {"full": {}, "mean_ablate_all": dict(ablate=means),
                "remove_all": dict(remove=fw.hidden)}
    for h in fw.hidden:
        variants[f"mean_ablate_{h}"] = dict(ablate={h: means[h]})
    for split in ("train", "val", "test"):
        X, y = d[f"X_{split}"], d[f"y_{split}"]
        res[split] = {}
        for vname, kw in variants.items():
            lg, _ = fw(X, **kw)
            res[split][vname] = dict(ce_acc(lg, y), per_class=per_class_acc(lg, y))
        # rows whose prediction changes when all hidden units are mean-ablated
        full, abl = fw(X)[0].argmax(1), fw(X, ablate=means)[0].argmax(1)
        ch = full != abl
        res[split]["changed_rows"] = dict(
            n=int(ch.sum()), fixed=int(((full == y) & ch).sum()), broken=int(((abl == y) & ch).sum()),
            by_true_class=[int(ch[y == c].sum()) for c in range(10)])
    (OUT / f"hidden_{name}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    for split in ("train", "val", "test"):
        print(split, {k: (round(v["acc"], 4), round(v["ce"], 4)) for k, v in res[split].items()
                      if k in ("full", "mean_ablate_all", "remove_all")}, res[split]["changed_rows"])
    print("units", len(units), kinds, f"parity {par:.1e}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "controls":
        controls()
    elif cmd == "runs":
        runs()
    elif cmd == "hidden":
        hidden(sys.argv[2])
