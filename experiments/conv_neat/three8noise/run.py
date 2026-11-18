"""ExConNEAT on 3 vs 8 vs noise (data.py), the current winning configuration.

  uv run python experiments/conv_neat/three8noise/run.py                  # MPS, out results/run_s0
  uv run python experiments/conv_neat/three8noise/run.py --resume         # continue from checkpoint.pkl
  touch experiments/conv_neat/three8noise/results/run_s0/STOP             # checkpoint and exit after this generation

Protocol (as run_long.py; recorded in config.json):
* population 50, structural mutation rates x4, lambda = 0.001 (fitness = training accuracy -
  lambda * (hidden maps + enabled kernels)), ReLU, global-average-pooling head, 3 outputs.
* successive halving (schedule.SuccessiveHalvingPopulation): every genome 2 epochs, the top 20% by
  fitness 8 more, species leaders brought to 20 (mean budget ~ the 5 epochs per genome of the
  plain runs). All 11,400 training rows every generation.
* validation (3,000 rows) only monitors: each generation's best-by-fitness genome's validation
  accuracy. Stop at a plateau (no gain >= 0.001 for 20 generations), at 180 min of wall time, or
  at the generation cap. Selected genome = best-by-validation among the monitored genomes.
* test (2,976 rows) is read once, at the end, for the selected genome only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)

from data import CLASS_NAMES, NOISE_TYPES, load  # noqa: E402
from explaneat.conv.explain.engine import depth_map, functional_nodes  # noqa: E402
from explaneat.conv.genome import ConvNeatConfig  # noqa: E402
from explaneat.conv.network import build_engine, pick_device, predict  # noqa: E402
from explaneat.conv.population import (config_to_dict, genome_to_dict, load_checkpoint,  # noqa: E402
                                       save_checkpoint)
from explaneat.conv.schedule import SuccessiveHalvingPopulation  # noqa: E402


def size_of(g) -> dict:
    fn = functional_nodes(g)
    dm = depth_map(g, fn)
    hid = [n for n in g.hidden_keys()]
    fe = [c for c in g.enabled_conns() if c.src in fn and c.dst in fn]
    return {"hidden_maps": len(hid), "functional_hidden_maps": len([n for n in hid if n in fn]),
            "enabled_kernels": len(g.enabled_conns()), "functional_kernels": len(fe),
            "kernel_sizes": {str(k): sum(1 for c in fe if c.k == k) for k in sorted({c.k for c in fe})},
            "n_params": int(g.n_params()), "complexity": int(g.complexity()),
            "depth": int(max(dm.values())) if dm else 0,
            "levels_used": sorted({int(g.nodes[n].level) for n in fn})}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--penalty", type=float, default=0.001)
    p.add_argument("--mut-scale", type=float, default=4.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--max-gens", type=int, default=400)
    p.add_argument("--max-minutes", type=float, default=180.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=2)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)

    torch.set_num_threads(a.threads)
    dev = pick_device(a.device)
    out = a.out or os.path.join(HERE, "results", f"run_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    ckpt, logp, stopfile = (os.path.join(out, f) for f in ("checkpoint.pkl", "log.jsonl", "STOP"))
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    data = load(a.seed)
    halving = dict(base=2, top_frac=0.2, top_extra=8, leader_total=20)

    if a.resume:
        pop, extra = load_checkpoint(ckpt, data, dev)
        pop.__class__ = SuccessiveHalvingPopulation
        pop.base, pop.top_frac, pop.top_extra, pop.leader_total = (halving[k] for k in
                                                                   ("base", "top_frac", "top_extra", "leader_total"))
        pop.extra_epochs = 0
        elapsed0 = extra.get("elapsed_s", 0.0)
        rows = [json.loads(l) for l in open(logp)] if os.path.exists(logp) else []
        with open(logp, "w") as f:
            for r in rows:
                if r["generation"] < pop.generation:
                    f.write(json.dumps(r) + "\n")
        print(f"resumed at generation {pop.generation}", flush=True)
    else:
        if os.path.exists(ckpt):
            sys.exit(f"{ckpt} exists: use --resume or another --out")
        cfg = ConvNeatConfig(in_channels=1, image_hw=(28, 28), n_classes=3, max_level=2, head="gap",
                             pop_size=a.pop, generations=a.max_gens, epochs_per_gen=halving["base"],
                             batch_size=256, complexity_penalty=a.penalty)
        cfg.p_add_node *= a.mut_scale
        cfg.p_add_conn *= a.mut_scale
        cfg.p_kernel_size *= a.mut_scale
        pop = SuccessiveHalvingPopulation(cfg, data, a.seed, dev, fitness_split="train", **halving)
        elapsed0 = 0.0
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({"args": vars(a), "config": config_to_dict(cfg), "halving": halving, "device": str(dev),
                       "torch": torch.__version__, "host": platform.node(), "classes": CLASS_NAMES,
                       "sizes": {s: np.bincount(data[f"y_{s}"], minlength=3).tolist() for s in ("train", "val", "test")}},
                      f, indent=1)
        open(logp, "w").close()
    cfg = pop.cfg

    t_start = time.time()
    stop_reason = None
    while True:
        gen = pop.generation
        row = pop.step(gen, verbose=False)
        row["elapsed_s"] = round(elapsed0 + time.time() - t_start, 1)
        row["device"] = str(dev)
        row.update(pop.schedule_log(gen))
        if row["since_improvement"] >= a.patience:
            stop_reason = (f"plateau: no validation gain >= {a.min_delta} for {a.patience} generations "
                           f"(reference {pop.monitor_ref:.4f} at generation {pop.monitor_ref_gen})")
        elif gen + 1 >= a.max_gens:
            stop_reason = f"generation cap ({a.max_gens})"
        elif row["elapsed_s"] / 60 >= a.max_minutes:
            stop_reason = f"wall budget ({a.max_minutes:g} min)"
        elif os.path.exists(stopfile):
            stop_reason = "paused"
        with open(logp, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("generation", "monitor_val_acc", "selected_val_acc", "mean_val_acc",
                                               "mean_complexity", "selected_complexity", "since_improvement",
                                               "seconds", "elapsed_s")}), flush=True)
        if stop_reason:
            break
        pop.reproduce()
        if pop.generation % a.checkpoint_every == 0:
            save_checkpoint(pop, ckpt, {"elapsed_s": elapsed0 + time.time() - t_start})
            with open(os.path.join(out, "selected_so_far.json"), "w") as f:
                json.dump(genome_to_dict(pop.monitor_best), f, default=float)
    elapsed = elapsed0 + time.time() - t_start
    if stop_reason == "paused":
        pop.reproduce()
        save_checkpoint(pop, ckpt, {"elapsed_s": elapsed})
        os.remove(stopfile)
        print("paused; resume with --resume", flush=True)
        return
    save_checkpoint(pop, ckpt, {"elapsed_s": elapsed, "finished": True})

    sel = pop.monitor_best
    with open(os.path.join(out, "selected_genome.json"), "w") as f:
        json.dump(genome_to_dict(sel), f, indent=1, default=float)
    # the single test read
    net = build_engine(copy.deepcopy(sel), cfg)
    net.eval()
    probs = predict(net, torch.tensor(data["X_test"]))
    pred = probs.argmax(1) if probs.ndim == 2 else probs
    y, t = data["y_test"], data["type_test"]
    from sklearn.metrics import confusion_matrix
    res = {"stop_reason": stop_reason, "generations_run": pop.generation, "wall_clock_s": round(elapsed, 1),
           "device": str(dev),
           "selected": {"key": sel.key, "generation": sel.metrics["generation"], "val_acc": sel.metrics["val_acc"],
                        "train_acc": sel.metrics.get("train_acc"), "fitness": sel.fitness, **size_of(sel)},
           "test": {"acc": float((pred == y).mean()),
                    "noise_vs_digit_acc": float(((pred == 2) == (y == 2)).mean()),
                    "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
                    "noise_recall_by_type": {nt: float((pred[t == i] == 2).mean()) for i, nt in enumerate(NOISE_TYPES)},
                    "noise_by_type_predicted_as": {nt: np.bincount(pred[t == i], minlength=3).tolist()
                                                   for i, nt in enumerate(NOISE_TYPES)}},
           "event_counts": dict(pop.event_counts), "n_trainings": pop.n_trainings}
    np.save(os.path.join(out, "test_pred.npy"), pred)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main()
