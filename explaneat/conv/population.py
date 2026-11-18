"""ConvNEAT evolution loop: NEAT for topology, backprop for weights.

Mirrors the spirit of explaneat.core.backproppop.BackpropPopulation: every
generation each genome is trained by backprop for a few epochs and the trained
weights are written back into its genes (Lamarckian), then fitness is
evaluated and NEAT reproduction (speciation, fitness sharing, elitism,
crossover, mutation) makes the next generation.

Fitness = validation accuracy - complexity_penalty * (hidden nodes + enabled edges).
Training uses the PropNEAT conv engine (propneat.py); the masks are checked
after every training call. (The existing system trains full-batch with
Adadelta; here minibatch Adam, because the conv phenotypes are slower per
step. Recorded in RESULTS.md.)
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from .crossover import crossover, distance
from .genome import ConvGenome, ConvNeatConfig, InnovationTracker, minimal_genome
from .mutation import mutate
from .network import accuracy, build_engine, predict, train_genome

log = logging.getLogger("explaneat.conv")


class Species:
    def __init__(self, sid: int, rep: ConvGenome):
        self.sid = sid
        self.rep = rep
        self.members: List[ConvGenome] = []


class ConvNeatPopulation:
    def __init__(self, cfg: ConvNeatConfig, data: Dict[str, np.ndarray], seed: int,
                 device: torch.device, fitness_split: str = "val", train_chunk: Optional[int] = None,
                 workers: int = 0, worker_device: Optional[str] = None, worker_threads: int = 1):
        """``fitness_split``: "val" (fitness = validation accuracy - penalty; the original runs) or
        "train" (fitness = accuracy on the generation's training rows - penalty; validation then only
        monitors, as in experiments/long_run). ``train_chunk``: if set, each generation trains on the
        next ``train_chunk`` rows of a seeded permutation of the whole training pool (one pass over
        the pool every ``n_train // train_chunk`` generations); None = all training rows every
        generation. ``workers`` > 0 trains genomes in that many processes (see parallel.py)."""
        assert fitness_split in ("val", "train")
        self.fitness_split = fitness_split
        self.train_chunk = train_chunk
        self.workers = workers
        self.worker_device = worker_device or str(device)
        self.worker_threads = worker_threads
        self._pool = None
        self._data_np = {k: data[k] for k in ("X_train", "y_train", "X_val", "y_val")}
        # monitoring (fitness_split="train"): each generation's best-by-fitness genome's val accuracy
        self.monitor_best: Optional[ConvGenome] = None  # best-by-val among the monitored genomes
        self.monitor_ref = -1.0  # plateau reference (raised only by >= min_delta)
        self.monitor_ref_gen = -1
        self.generation = 0  # next generation to evaluate
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.device = device
        self.tracker = InnovationTracker(cfg.n_out_maps)
        self.X = {s: torch.tensor(data[f"X_{s}"], dtype=torch.float32, device=device)
                  for s in ("train", "val")}
        self.y = {s: torch.tensor(data[f"y_{s}"], device=device) for s in ("train", "val")}
        self.y_np = {s: data[f"y_{s}"] for s in ("train", "val")}
        self.next_key = 0
        self.population: List[ConvGenome] = []
        for _ in range(cfg.pop_size):
            self.population.append(minimal_genome(self._key(), cfg, self.tracker, self.rng))
        self.species: List[Species] = []
        self.next_sid = 0
        self.best: Optional[ConvGenome] = None
        self.generation_log: List[dict] = []
        self.event_counts: Counter = Counter()
        self.n_trainings = 0  # each one ends with PropNEATConvNet.assert_masks_hold()

    def _key(self) -> int:
        self.next_key += 1
        return self.next_key - 1

    # ------------------------------------------------------------------
    def chunk_rows(self, gen: int) -> Optional[np.ndarray]:
        """Training rows for generation ``gen`` (None = all). Stateless: pass p of the pool uses
        permutation default_rng([seed, 7, p]); chunks are consecutive slices of it."""
        if not self.train_chunk:
            return None
        n = len(self.y_np["train"])
        per_pass = max(1, n // self.train_chunk)
        p, k = divmod(gen, per_pass)
        perm = np.random.default_rng([self.seed, 7, p]).permutation(n)
        return np.sort(perm[k * self.train_chunk:(k + 1) * self.train_chunk])

    def evaluate(self, gen: int) -> None:
        cfg = self.cfg
        rows = self.chunk_rows(gen)
        seeds = [int(self.rng.integers(2 ** 62)) for _ in self.population]
        if self.workers > 0:
            from .parallel import evaluate_parallel
            results = evaluate_parallel(self, rows, seeds)
        else:
            if rows is None:
                Xtr, ytr, ytr_np = self.X["train"], self.y["train"], self.y_np["train"]
            else:
                ri = torch.as_tensor(rows, device=self.device)
                Xtr, ytr, ytr_np = self.X["train"][ri], self.y["train"][ri], self.y_np["train"][rows]
            results = [eval_genome(g, cfg, Xtr, ytr, ytr_np, self.X["val"], self.y_np["val"],
                                   self._epochs(g), s, self.device, self.fitness_split)
                       for g, s in zip(self.population, seeds)]
        new_pop = []
        for g, m in results:
            self.n_trainings += 1
            m.update({"complexity": g.complexity(), "generation": gen})
            g.metrics = m
            score = m["val_acc"] if self.fitness_split == "val" else m["train_acc"]
            g.fitness = score - cfg.complexity_penalty * g.complexity()
            new_pop.append(g)
            if self.best is None or g.fitness > self.best.fitness:
                self.best = copy.deepcopy(g)
        self.population = new_pop

    # ---- hooks for search schedules (explaneat/conv/schedule.py); the defaults are the plain loop
    def _epochs(self, g: ConvGenome) -> int:
        """Backprop epochs for genome ``g`` this generation (in-process evaluation only)."""
        return self.cfg.epochs_per_gen

    def _mutate(self, child: ConvGenome, p1: ConvGenome, p2: Optional[ConvGenome]) -> list:
        """Mutate a new offspring (``p2`` is None unless it came from crossover)."""
        return mutate(child, self.cfg, self.tracker, self.rng)

    def best_by_fitness(self) -> ConvGenome:
        """This generation's best-by-fitness genome (ties: lowest key)."""
        return min(self.population, key=lambda g: (-g.fitness, g.key))

    def monitor(self, gen: int, min_delta: float = 0.001) -> dict:
        """Plateau monitoring on validation accuracy of the generation's best-by-fitness genome
        (with fitness_split="train" this is the best-by-train genome, as in experiments/long_run).
        The plateau reference moves only on an improvement >= min_delta; ``monitor_best`` keeps the
        highest-validation monitored genome (the run's selected genome)."""
        g = self.best_by_fitness()
        v = g.metrics["val_acc"]
        if v >= self.monitor_ref + min_delta:
            self.monitor_ref, self.monitor_ref_gen = v, gen
        if self.monitor_best is None or v > self.monitor_best.metrics["val_acc"]:
            self.monitor_best = copy.deepcopy(g)
        return {"monitor_val_acc": v, "monitor_key": g.key, "monitor_complexity": g.complexity(),
                "plateau_ref": self.monitor_ref, "plateau_ref_gen": self.monitor_ref_gen,
                "since_improvement": gen - self.monitor_ref_gen}

    def speciate(self) -> None:
        cfg = self.cfg
        for s in self.species:
            s.members = []
        for g in self.population:
            for s in self.species:
                if distance(g, s.rep, cfg) < cfg.compat_threshold:
                    s.members.append(g)
                    break
            else:
                s = Species(self.next_sid, g)
                self.next_sid += 1
                s.members.append(g)
                self.species.append(s)
        self.species = [s for s in self.species if s.members]
        for s in self.species:
            s.rep = s.members[int(self.rng.integers(len(s.members)))]
        # nudge the threshold towards the target number of species
        if len(self.species) > cfg.target_species:
            cfg.compat_threshold *= 1.1
        elif len(self.species) < cfg.target_species:
            cfg.compat_threshold = max(0.05, cfg.compat_threshold / 1.1)

    def reproduce(self) -> None:
        cfg = self.cfg
        fits = np.array([g.fitness for g in self.population])
        lo = fits.min()
        adj = []
        for s in self.species:
            mf = np.mean([g.fitness - lo + 1e-3 for g in s.members])  # fitness sharing
            adj.append(mf)
        adj = np.array(adj)
        quota = adj / adj.sum() * cfg.pop_size
        counts = np.floor(quota).astype(int)
        rem = cfg.pop_size - counts.sum()
        for i in np.argsort(-(quota - counts))[:rem]:
            counts[i] += 1
        new: List[ConvGenome] = []
        for s, n in zip(self.species, counts):
            if n <= 0:
                continue
            ranked = sorted(s.members, key=lambda g: g.fitness, reverse=True)
            for e in ranked[:min(cfg.elitism, n)]:
                child = e.copy(self._key())
                child.history = e.history + ["elite"]
                new.append(child)
                n -= 1
            parents = ranked[:max(1, int(np.ceil(cfg.survival_threshold * len(ranked))))]
            for _ in range(n):
                p1 = parents[int(self.rng.integers(len(parents)))]
                p2 = None
                if len(parents) > 1 and self.rng.random() < cfg.p_crossover:
                    p2 = parents[int(self.rng.integers(len(parents)))]
                    child = crossover(p1, p2, self._key(), cfg, self.rng)
                    self.event_counts["recombination:crossover"] += 1
                    for e in child.disable_log:  # disables made by this crossover
                        if not e["detail"].startswith("inherited:"):
                            self.event_counts["disable:" + e["cause"]] += 1
                else:
                    child = p1.copy(self._key())
                    child.history = list(p1.history)
                    child.disable_log = list(p1.disable_log)
                n_dis = len(child.disable_log)
                for ev in self._mutate(child, p1, p2):
                    self.event_counts["mutation:" + ev] += 1
                for e in child.disable_log[n_dis:]:
                    self.event_counts["disable:" + e["cause"]] += 1
                new.append(child)
        self.population = new

    def step(self, gen: int, verbose: bool = True) -> dict:
        """Evaluate generation ``gen`` and speciate; returns its log row (no reproduction)."""
        t0 = time.time()
        self.evaluate(gen)
        self.speciate()
        fits = [g.fitness for g in self.population]
        accs = [g.metrics["val_acc"] for g in self.population]
        cx = [g.complexity() for g in self.population]
        row = {"generation": gen, "best_fitness": max(fits), "mean_fitness": float(np.mean(fits)),
               "best_val_acc": max(accs), "mean_val_acc": float(np.mean(accs)),
               "mean_complexity": float(np.mean(cx)),
               "max_complexity": int(max(cx)), "n_species": len(self.species),
               "compat_threshold": self.cfg.compat_threshold,
               "best_so_far_fitness": self.best.fitness,
               "best_so_far_val_acc": self.best.metrics["val_acc"],
               "best_so_far_complexity": self.best.complexity()}
        if self.fitness_split == "train":
            tr = [g.metrics["train_acc"] for g in self.population]
            row.update({"best_train_acc": max(tr), "mean_train_acc": float(np.mean(tr))})
            row.update(self.monitor(gen))
            row["selected_val_acc"] = self.monitor_best.metrics["val_acc"]
            row["selected_complexity"] = self.monitor_best.complexity()
            row["selected_n_params"] = self.monitor_best.n_params()
        row["seconds"] = round(time.time() - t0, 2)
        self.generation_log.append(row)
        self.generation = gen + 1
        if verbose:
            print(json.dumps(row), flush=True)
        return row

    def run(self, generations: Optional[int] = None, verbose: bool = True) -> ConvGenome:
        gens = generations if generations is not None else self.cfg.generations
        for gen in range(gens):
            self.step(gen, verbose)
            if gen < gens - 1:
                self.reproduce()
        self.close()
        return self.best

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def eval_genome(g: ConvGenome, cfg: ConvNeatConfig, Xtr, ytr, ytr_np, Xval, yval_np, epochs: int,
                seed: int, device, fitness_split: str):
    """Train one genome (Lamarckian, write-back) and score it. Runs in-process or in a worker."""
    loss = train_genome(g, cfg, Xtr, ytr, epochs, np.random.default_rng(seed), device)
    net = build_engine(g, cfg).to(device)
    m = {"train_loss": loss, "val_acc": accuracy(predict(net, Xval), yval_np)}
    if fitness_split == "train":
        m["train_acc"] = accuracy(predict(net, Xtr), ytr_np)
    return g, m


# ---- checkpoints -----------------------------------------------------------
# Everything is numpy / plain Python (no tensors), so a checkpoint written on one machine
# (macOS, torch 2.2, numpy 1.26) resumes on another (Linux CUDA, torch >= 2.7, numpy 2.x).
_STATE = ("cfg", "seed", "rng", "tracker", "next_key", "population", "species", "next_sid", "best",
          "generation_log", "event_counts", "n_trainings", "fitness_split", "train_chunk",
          "monitor_best", "monitor_ref", "monitor_ref_gen", "generation")


def save_checkpoint(pop: "ConvNeatPopulation", path: str, extra: Optional[dict] = None) -> None:
    import os
    import pickle
    state = {k: getattr(pop, k) for k in _STATE}
    state["rng"] = pop.rng.bit_generator.state
    state["extra"] = extra or {}
    state["format"] = "convneat-checkpoint-v1"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=4)
    os.replace(tmp, path)


def load_checkpoint(path: str, data: Dict[str, np.ndarray], device: torch.device, workers: int = 0,
                    worker_device: Optional[str] = None, worker_threads: int = 1):
    """Rebuild a population from ``save_checkpoint`` (data must be the same splits)."""
    import pickle
    with open(path, "rb") as f:
        st = pickle.load(f)
    assert st.get("format") == "convneat-checkpoint-v1", "not a ConvNEAT checkpoint"
    pop = ConvNeatPopulation.__new__(ConvNeatPopulation)
    ConvNeatPopulation.__init__(pop, st["cfg"], data, st["seed"], device, st["fitness_split"], st["train_chunk"],
                                workers, worker_device, worker_threads)
    for k in _STATE:
        if k != "rng":
            setattr(pop, k, st[k])
    pop.rng = np.random.default_rng()
    pop.rng.bit_generator.state = st["rng"]
    return pop, st.get("extra", {})


def genome_to_dict(g: ConvGenome) -> dict:
    return {
        "key": g.key,
        "fitness": g.fitness,
        "metrics": g.metrics,
        "nodes": [{"key": n.key, "kind": n.kind, "level": n.level, "bias": n.bias} for n in g.nodes.values()],
        "conns": [{"innov": c.innov, "src": c.src, "dst": c.dst, "k": c.k, "enabled": c.enabled,
                   "kernel": c.kernel.tolist()} for c in g.conns.values()],
        "head_w": g.head_w.tolist(), "head_b": g.head_b.tolist(),
        "disable_log": g.disable_log,
        "history": Counter(g.history),
    }


def config_to_dict(cfg: ConvNeatConfig) -> dict:
    d = asdict(cfg)
    d["image_hw"] = list(cfg.image_hw)
    d["kernel_sizes"] = list(cfg.kernel_sizes)
    return d


def genome_from_dict(d: dict) -> ConvGenome:
    from .genome import ConnGene, NodeGene
    g = ConvGenome(key=d["key"], fitness=d.get("fitness"), metrics=d.get("metrics", {}),
                   disable_log=d.get("disable_log", []))
    for n in d["nodes"]:
        g.nodes[n["key"]] = NodeGene(n["key"], n["kind"], n["level"], n["bias"])
    for c in d["conns"]:
        g.conns[c["innov"]] = ConnGene(c["innov"], c["src"], c["dst"],
                                       np.array(c["kernel"], dtype=np.float32), c["enabled"])
    g.head_w = np.array(d["head_w"], dtype=np.float32)
    g.head_b = np.array(d["head_b"], dtype=np.float32)
    return g
