"""Long ConvNEAT run on MNIST 10-class with a plateau stopping rule, resumable from checkpoints.

  uv run python experiments/conv_neat/run_long.py --seed 0                      # local (MPS) defaults
  uv run python experiments/conv_neat/run_long.py --seed 0 --resume             # continue from out/checkpoint.pkl
  python experiments/conv_neat/run_long.py --seed 0 --device cuda --workers 6   # rig (see rig/README.md)

Protocol (recorded in results.json / config.json):
* data: MNIST (cached ~/.explaneat/data/mnist_784.npz). Validation = 10,000 images carved from the
  60,000 standard training images (stratified, seeded); training pool = the other 50,000 (or
  --n-train of them); test = the standard 10,000, used once at the end.
* each generation every genome trains --epochs epochs (Adam, --batch) on that generation's rows:
  with --train-chunk C, the next C rows of a seeded permutation of the pool (rotating); else all rows.
* fitness (--fitness train, default) = accuracy on the generation's training rows - lambda * (hidden
  nodes + enabled edges). Validation is only monitored: each generation's best-by-train genome's
  validation accuracy. Stop when it has not improved by >= --min-delta for --patience generations,
  or at --max-gens. The selected genome is the best-by-validation one among the monitored genomes.
* --fitness val reproduces the earlier runs' rule (validation in fitness).
* checkpoint (full population state, numpy/pickle, portable across machines) every
  --checkpoint-every generations and at the end; --resume continues from it (may change --workers,
  --device; the experiment settings come from the checkpoint).
* a file named STOP in the output directory makes the run checkpoint and exit after the current
  generation (resume later).
* --warm-start CKPT starts a new run directory from another run's population (resume onto a new
  training set: e.g. the converged 10k-subset population continued on all 50k rows). The validation
  split is the same rows for the same seed and --n-val whatever --n-train is; the plateau monitor and
  the selected genome start afresh. Label such runs as warm-start continuations, not clean runs.
* --schedule halving: successive halving (schedule.SuccessiveHalvingPopulation; every genome
  --halving-base epochs, the top --halving-top fraction --halving-extra more, species leaders brought
  to --halving-leader). --schedule scarcity: lambda x --scarcity-penalty-scale and deletion mutations
  only (schedule.py); it stops when the mean genome size has not fallen for 2 generations, after
  --scarcity-max-gens generations, or when the generation's best validation accuracy falls more than
  --val-guard below the starting genome's (the validation guard). Both are meant for --warm-start.
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
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from explaneat.conv.data import mnist_splits  # noqa: E402
from explaneat.conv.genome import ConvNeatConfig  # noqa: E402
from explaneat.conv.network import accuracy, build_engine, loss_fn, pick_device, predict  # noqa: E402
from explaneat.conv.population import (ConvNeatPopulation, config_to_dict, genome_to_dict,  # noqa: E402
                                       load_checkpoint, save_checkpoint)
from explaneat.conv.schedule import (OTHER_STRUCT, STRUCT, Phase, PhaseSchedule,  # noqa: E402
                                     ScheduledPopulation, SuccessiveHalvingPopulation)


def adopt_schedule(pop, a, base_cfg_dict=None):
    """Turn a loaded population into the --schedule variant (class swap; the state is the same)."""
    if a.schedule == "none":
        return pop
    defaults = ConvNeatConfig()
    if a.schedule == "halving":
        pop.__class__ = SuccessiveHalvingPopulation
        pop.base, pop.top_frac, pop.top_extra, pop.leader_total = (a.halving_base, a.halving_top,
                                                                   a.halving_extra, a.halving_leader)
        pop.extra_epochs = 0
        for f in STRUCT:  # back to the run's own rates (a scarcity phase may have zeroed them)
            setattr(pop.cfg, f, getattr(defaults, f) * a.mut_scale)
        for f in OTHER_STRUCT + ("p_weight",):
            setattr(pop.cfg, f, getattr(defaults, f))
        pop.cfg.complexity_penalty = a.penalty
        pop.cfg.epochs_per_gen = a.halving_base
    elif a.schedule == "scarcity":
        base = ConvNeatConfig(**base_cfg_dict) if base_cfg_dict else pop.cfg
        ph = Phase("scarcity", 1, struct_scale=0.0, other_struct=False, weight_scale=0.0, epochs=a.epochs,
                   penalty_scale=a.scarcity_penalty_scale, p_delete_conn=1.0, p_delete_node=0.5)
        pop.__class__ = ScheduledPopulation
        pop.schedule = PhaseSchedule([ph])
        pop.base_cfg = copy.deepcopy(base)
        pop.base_cfg.complexity_penalty = a.penalty
    return pop


def _cfg_from_dict(d):
    d = dict(d)
    d["image_hw"] = tuple(d["image_hw"])
    d["kernel_sizes"] = tuple(d["kernel_sizes"])
    return d


def device_of(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    return pick_device(name)


def finetune(g, cfg, data, epochs, seed, device, batch):
    """Extra backprop on the selected genome (structure fixed) on the whole training pool;
    keep the best-validation epoch. Test is read for that epoch only."""
    g = copy.deepcopy(g)
    rng = np.random.default_rng(seed + 999)
    net = build_engine(g, cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    X = {s: torch.tensor(data[f"X_{s}"], device=device) for s in ("train", "val", "test")}
    Y = {s: torch.tensor(data[f"y_{s}"], device=device) for s in ("train", "val", "test")}
    best_state = copy.deepcopy(net.state_dict())
    best = (accuracy(predict(net, X["val"]), data["y_val"]), -1)
    n = len(X["train"])
    for ep in range(epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(n), device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(net(X["train"][idx]), Y["train"][idx], cfg.n_classes)
            opt.zero_grad(); loss.backward(); opt.step()
        va = accuracy(predict(net, X["val"]), data["y_val"])
        if va > best[0]:
            best = (va, ep)
            best_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.assert_masks_hold()
    return {"val_acc": best[0], "test_acc": accuracy(predict(net, X["test"]), data["y_test"]),
            "best_epoch": best[1], "epochs": epochs}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n-train", type=int, default=None, help="training pool size (default: all 50,000)")
    p.add_argument("--n-val", type=int, default=10000)
    p.add_argument("--train-chunk", type=int, default=5000, help="rows per generation (0 = whole pool)")
    p.add_argument("--penalty", type=float, default=0.001)
    p.add_argument("--fitness", default="train", choices=["train", "val"])
    p.add_argument("--head", default="gap", choices=["gap", "grid2"])
    p.add_argument("--max-level", type=int, default=2)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--max-gens", type=int, default=400)
    p.add_argument("--finetune", type=int, default=20)
    p.add_argument("--mut-scale", type=float, default=1.0,
                   help="multiply the structural mutation rates (add node, add connection, kernel size); "
                        "a rate above 1 is an expected count per offspring")
    p.add_argument("--max-minutes", type=float, default=None, help="wall-time budget (stop reason 'wall budget')")
    p.add_argument("--no-final", action="store_true", help="skip the test read-out and fine-tuning (sweeps)")
    p.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, default=0, help="train genomes in N processes (0 = in-process)")
    p.add_argument("--worker-threads", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--warm-start", default=None, help="checkpoint of another run: start from its population")
    p.add_argument("--finalize-only", default=None, metavar="STOP_REASON",
                   help="with --resume: no more generations; evaluate the selected genome on test once, fine-tune, "
                        "write results.json (the checkpoint is left as it is)")
    p.add_argument("--schedule", default="none", choices=["none", "halving", "scarcity"])
    p.add_argument("--halving-base", type=int, default=2)
    p.add_argument("--halving-top", type=float, default=0.2)
    p.add_argument("--halving-extra", type=int, default=8)
    p.add_argument("--halving-leader", type=int, default=20)
    p.add_argument("--scarcity-penalty-scale", type=float, default=5.0)
    p.add_argument("--scarcity-max-gens", type=int, default=10)
    p.add_argument("--val-guard", type=float, default=0.005)
    a = p.parse_args(argv)

    torch.set_num_threads(a.threads)
    dev = device_of(a.device)
    out = a.out or os.path.join(HERE, "results", f"mnist10long_{a.head}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "checkpoint.pkl")
    logp = os.path.join(out, "log.jsonl")
    stopfile = os.path.join(out, "STOP")

    saved_conf = None
    if a.resume:
        with open(os.path.join(out, "config.json")) as f:
            saved_conf = json.load(f)
        saved = saved_conf["args"]
        for k in ("seed", "n_train", "n_val", "schedule", "halving_base", "halving_top", "halving_extra",
                  "halving_leader", "scarcity_penalty_scale", "scarcity_max_gens", "val_guard", "penalty",
                  "mut_scale", "epochs", "warm_start"):
            if k in saved:
                setattr(a, k, saved[k])
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    data = mnist_splits(a.seed, classes=tuple(range(10)), n_train=a.n_train, n_val=a.n_val)
    wdev = str(dev)

    start_info = {}
    if a.resume:
        pop, extra = load_checkpoint(ckpt, data, dev, a.workers, wdev, a.worker_threads)
        pop = adopt_schedule(pop, a, (saved_conf.get("base_config") if saved_conf else None))
        start_info = saved_conf.get("warm_start_info", {}) if saved_conf else {}
        cfg = pop.cfg
        elapsed0 = extra.get("elapsed_s", 0.0)
        # drop log rows written after the checkpoint (they will be recomputed)
        rows = [json.loads(l) for l in open(logp)] if os.path.exists(logp) else []
        with open(logp, "w") as f:
            for r in rows:
                if r["generation"] < pop.generation:
                    f.write(json.dumps(r) + "\n")
        print(f"resumed at generation {pop.generation} from {ckpt}", flush=True)
    elif a.warm_start:
        if os.path.exists(ckpt):
            sys.exit(f"{ckpt} exists: use --resume, or choose another --out")
        src = os.path.dirname(os.path.abspath(a.warm_start))
        src_conf = json.load(open(os.path.join(src, "config.json")))
        assert src_conf["args"]["seed"] == a.seed and src_conf["args"]["n_val"] == a.n_val, \
            "warm start keeps the seed and --n-val, so the validation rows are the same"
        pop, extra = load_checkpoint(a.warm_start, data, dev, a.workers, wdev, a.worker_threads)
        base_cfg = config_to_dict(pop.cfg)
        sel0 = pop.monitor_best
        wsp = os.path.abspath(a.warm_start)
        start_info = {"from": os.path.relpath(wsp, HERE) if wsp.startswith(HERE) else wsp, "source_generation": pop.generation,
                      "source_finished": bool(extra.get("finished")),
                      "source_selected_val_acc": sel0.metrics["val_acc"] if sel0 else None,
                      "source_selected_complexity": sel0.complexity() if sel0 else None,
                      "source_mean_complexity": float(np.mean([g.complexity() for g in pop.population])),
                      "source_n_train": src_conf["args"]["n_train"], "label": "warm-start continuation, not a clean run"}
        pop = adopt_schedule(pop, a, base_cfg)
        pop.fitness_split = a.fitness
        pop.train_chunk = a.train_chunk or None
        pop.monitor_best, pop.monitor_ref, pop.monitor_ref_gen = None, -1.0, -1
        pop.best = None
        if extra.get("finished"):  # the source's last generation was evaluated but not reproduced
            pop.reproduce()
        cfg = pop.cfg
        elapsed0 = 0.0
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({"args": vars(a), "config": config_to_dict(cfg), "base_config": base_cfg,
                       "warm_start_info": start_info, "device": str(dev),
                       "torch": torch.__version__, "host": platform.node(),
                       "sizes": {s_: int(len(data[f"y_{s_}"])) for s_ in ("train", "val", "test")},
                       "rows": {s_: [int(r) for r in v] for s_, v in data["rows"].items()}}, f)
        open(logp, "w").close()
        print("warm start from", a.warm_start, json.dumps(start_info), flush=True)
    else:
        if os.path.exists(ckpt):
            sys.exit(f"{ckpt} exists: use --resume, or choose another --out")
        cfg = ConvNeatConfig(in_channels=1, image_hw=(28, 28), n_classes=10, max_level=a.max_level,
                             head=a.head, pop_size=a.pop, generations=a.max_gens, epochs_per_gen=a.epochs,
                             batch_size=a.batch, complexity_penalty=a.penalty)
        if a.mut_scale != 1.0:
            cfg.p_add_node *= a.mut_scale
            cfg.p_add_conn *= a.mut_scale
            cfg.p_kernel_size *= a.mut_scale
        pop = ConvNeatPopulation(cfg, data, a.seed, dev, fitness_split=a.fitness,
                                 train_chunk=a.train_chunk or None, workers=a.workers,
                                 worker_device=wdev, worker_threads=a.worker_threads)
        elapsed0 = 0.0
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({"args": vars(a), "config": config_to_dict(cfg), "device": str(dev),
                       "torch": torch.__version__, "host": platform.node(),
                       "sizes": {s: int(len(data[f"y_{s}"])) for s in ("train", "val", "test")},
                       "rows": {s: [int(r) for r in v] for s, v in data["rows"].items()}}, f)
        open(logp, "w").close()

    t_start = time.time()
    stop_reason = a.finalize_only
    if not a.resume and not a.warm_start:
        start_info = {}
    try:
        while not a.finalize_only:
            gen = pop.generation
            row = pop.step(gen, verbose=False)
            row["elapsed_s"] = round(elapsed0 + time.time() - t_start, 1)
            row["host"] = platform.node()
            row["device"] = str(dev)
            row["workers"] = a.workers
            if hasattr(pop, "schedule_log"):
                row.update(pop.schedule_log(gen))
            if a.schedule == "scarcity":
                rows_ = [json.loads(l) for l in open(logp)] + [row]
                mc = [r["mean_complexity"] for r in rows_]
                n_ph = len(rows_)
                v0 = start_info.get("source_selected_val_acc")
                if v0 is not None and row["best_val_acc"] < v0 - a.val_guard:
                    stop_reason = (f"validation guard: best validation {row['best_val_acc']:.4f} fell more than "
                                   f"{a.val_guard} below the start ({v0:.4f})")
                elif n_ph >= 3 and min(mc[-2:]) >= mc[-3] - 1e-9:
                    stop_reason = "mean genome size stopped falling (2 generations without a fall)"
                elif n_ph >= a.scarcity_max_gens:
                    stop_reason = f"scarcity phase length ({a.scarcity_max_gens} generations)"
            if stop_reason:
                pass
            elif pop.fitness_split == "train" and row["since_improvement"] >= a.patience:
                stop_reason = (f"plateau: best-by-train validation accuracy not improved by >= {a.min_delta} "
                               f"for {a.patience} generations (reference {pop.monitor_ref:.4f} at generation "
                               f"{pop.monitor_ref_gen})")
            elif gen + 1 >= a.max_gens:
                stop_reason = f"generation cap ({a.max_gens})"
            elif a.max_minutes and (elapsed0 + time.time() - t_start) / 60 >= a.max_minutes:
                stop_reason = f"wall budget ({a.max_minutes:g} min)"
            elif os.path.exists(stopfile):
                stop_reason = "paused"
            with open(logp, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("generation", "monitor_val_acc", "selected_val_acc",
                                                   "mean_val_acc", "mean_complexity", "since_improvement",
                                                   "seconds", "phase") if k in row}), flush=True)
            if stop_reason:
                break
            pop.reproduce()
            if pop.generation % a.checkpoint_every == 0:
                save_checkpoint(pop, ckpt, {"elapsed_s": elapsed0 + time.time() - t_start})
    finally:
        pop.close()
    elapsed = elapsed0 + time.time() - t_start
    if stop_reason == "paused":
        pop.reproduce()
        save_checkpoint(pop, ckpt, {"elapsed_s": elapsed})
        os.remove(stopfile)
        print("paused; resume with --resume", flush=True)
        return
    if a.finalize_only:
        elapsed = elapsed0
    else:
        save_checkpoint(pop, ckpt, {"elapsed_s": elapsed, "finished": True})

    sel = pop.monitor_best if pop.fitness_split == "train" else pop.best
    if a.no_final:
        res = {"stop_reason": stop_reason, "generations_run": pop.generation, "wall_clock_s": round(elapsed, 1),
               "selected": {"generation": sel.metrics["generation"], "val_acc": sel.metrics["val_acc"],
                            "complexity": sel.complexity(), "n_params": sel.n_params()}}
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(json.dumps(res), flush=True)
        return
    net = build_engine(copy.deepcopy(sel), cfg)
    res = {"stop_reason": stop_reason, "generations_run": pop.generation, "wall_clock_s": round(elapsed, 1),
           "warm_start_info": start_info or None,
           "selected": {"key": sel.key, "generation": sel.metrics["generation"], "val_acc": sel.metrics["val_acc"],
                        "train_acc": sel.metrics.get("train_acc"), "fitness": sel.fitness,
                        "test_acc": accuracy(predict(net, torch.tensor(data["X_test"])), data["y_test"]),
                        "complexity": sel.complexity(), "n_params": sel.n_params(),
                        "n_hidden": len(sel.hidden_keys()), "n_enabled_edges": len(sel.enabled_conns())},
           "event_counts": dict(pop.event_counts), "n_trainings": pop.n_trainings}
    print(json.dumps(res["selected"]), flush=True)
    res["finetuned"] = finetune(sel, cfg, data, a.finetune, a.seed, dev, cfg.batch_size)
    with open(os.path.join(out, "selected_genome.json"), "w") as f:
        json.dump(genome_to_dict(sel), f, indent=1, default=float)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main()
