"""Island runner for the modular-crossover study (PLAN.md).

``IslandPopulation`` subclasses BackpropPopulation (masked backprop, the
default) and splits its ``run`` loop body into ``evaluate_generation`` and
``reproduce_generation`` so several islands can step in lockstep with a
checkpoint hook between evaluation and reproduction. The loop body is the
same as BackpropPopulation.run without reporters or validation data.

``run_study(task, arm, seed, ...)`` runs one (task, arm, seed): 4 islands,
checkpoints every K generations, arm-specific sharing, in memory.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import neat
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from chunks import find_chunks, gs
from explaneat.core.backproppop import BackpropPopulation
from explaneat.evaluators.evaluators import auc_fitness
from module_genome import ModuleGenome
from operators import (OperatorNotApplicable, differing_modules, graft_chunk,
                       knockout_module, refit, add_module_keep_ids, valid_feedforward)
from registry import ModuleLibrary, pearson
from tasks import Task

ARMS = {
    "plain": None,  # arm 1: PropNEAT + whole-elite migration at checkpoints
    "modules": "explained",  # arm 2
    "sham": "sham",  # arm 3
    "topology": "topology",  # arm 4
}
OPS = ("insertion", "crossover", "replacement")
N_EPOCHS = 5
TORCH_THREADS = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)


class IslandPopulation(BackpropPopulation):
    def __init__(self, config, xs, ys, island: int):
        super().__init__(config, xs, ys, mask_phantom=True)
        self.island = island
        self.best_this_gen = None

    def set_data(self, xs, ys):
        self.xs = torch.tensor(np.asarray(xs), dtype=torch.float64)
        self.ys = torch.tensor(np.asarray(ys), dtype=torch.float64).reshape(-1, 1)

    def evaluate_generation(self, n_epochs=N_EPOCHS):
        self.backpropagate(self.xs, self.ys, nEpochs=n_epochs)
        auc_fitness(self.population, self.config, self.xs, self.ys, self.device)
        best = None
        for _, g in self.population.items():
            if best is None or g.fitness > best.fitness:
                best = g
        self.best_this_gen = best
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best
        return best

    def reproduce_generation(self, hook=None):
        old = self.population
        self.population = self.reproduction.reproduce(
            self.config, self.species, self.config.pop_size, self.generation)
        extinct = False
        if not self.species.species:
            extinct = True
            self.population = self.reproduction.create_new(
                self.config.genome_type, self.config.genome_config, self.config.pop_size)
        if hook is not None and not extinct:
            hook(self, old, self.population)
        self.species.speciate(self.config, self.population, self.generation)
        self.generation += 1
        return extinct


def make_config(task: Task, pop_size: int):
    """neat Config for ModuleGenome from the task's .ini (whose genome section
    is [DefaultGenome], as in experiments/10seed); pop_size overridden."""
    import tempfile
    text = Path(task.config_path).read_text().replace("[DefaultGenome]", "[ModuleGenome]")
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        cfg = neat.Config(ModuleGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                          neat.DefaultStagnation, path)
    finally:
        os.unlink(path)
    cfg.pop_size = pop_size
    return cfg


def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def genome_size(genome, config):
    """(live hidden nodes, enabled connections between live nodes)."""
    from chunks import live_structure
    try:
        _, live, edges = live_structure(genome, config)
    except Exception:
        return 0, 0
    return len(live), len(edges)


class Study:
    """One (task, arm, seed) run."""

    def __init__(self, task: Task, arm: str, seed: int, n_gen: int, pop_size: int,
                 n_islands: int = 4, K: int = 25, op_fraction: float = 0.2,
                 stage_switch: Optional[List[int]] = None, n_migrants: Optional[int] = None,
                 log=None):
        if arm not in ARMS:
            raise ValueError(arm)
        self.task, self.arm, self.seed = task, arm, seed
        self.weight_mode = ARMS[arm]
        self.n_gen, self.pop_size, self.n_islands, self.K = n_gen, pop_size, n_islands, K
        self.op_fraction = op_fraction
        self.stage_switch = list(stage_switch or [])  # generation at which stage i+1 starts
        assert len(self.stage_switch) == len(task.stages) - 1
        self.n_migrants = n_islands - 1 if n_migrants is None else n_migrants
        self.log = log or (lambda *a: None)
        self.library = ModuleLibrary()
        self.op_records: List[dict] = []
        self.gen_records: List[dict] = []
        self.checkpoint_records: List[dict] = []
        self.module_origin: Dict[int, int] = {}  # module id -> island of first variant

    def stage_at(self, gen):
        s = 0
        for i, g in enumerate(self.stage_switch):
            if gen >= g:
                s = i + 1
        return s

    # ------------------------------------------------------------------ run
    def run(self):
        seed_everything(self.seed)
        t0 = time.time()
        task = self.task
        self.config = make_config(task, self.pop_size)
        st = task.stages[0]
        self.islands = [IslandPopulation(self.config, st.X_train, st.y_train, i)
                        for i in range(self.n_islands)]
        cur_stage = 0
        self.solved_gen = None
        for gen in range(self.n_gen):
            s = self.stage_at(gen)
            if s != cur_stage:
                cur_stage = s
                for isl in self.islands:
                    isl.set_data(task.stages[s].X_train, task.stages[s].y_train)
            stage = task.stages[cur_stage]
            for isl in self.islands:
                isl.evaluate_generation()
            self._record_generation(gen, cur_stage)
            if gen == self.n_gen - 1:
                break
            checkpoint = (gen + 1) % self.K == 0
            hook = None
            if checkpoint:
                if self.arm == "plain":
                    migrants = [copy.deepcopy(isl.best_this_gen) for isl in self.islands]
                    hook = self._migration_hook(gen, migrants)
                else:
                    self._checkpoint(gen, stage)
                    hook = self._operator_hook(gen, stage)
            for isl in self.islands:
                extinct = isl.reproduce_generation(hook)
                if extinct:
                    self.log(f"gen {gen} island {isl.island}: complete extinction, reset")
                    self.gen_records[-1].setdefault("extinctions", []).append(isl.island)
        self.wall_s = time.time() - t0
        return self.final_summary()

    # ------------------------------------------------------------------ records
    def _record_generation(self, gen, stage_idx):
        stage = self.task.stages[stage_idx]
        isl_rows = []
        for isl in self.islands:
            fits = [g.fitness for g in isl.population.values()]
            carriers = {}
            foreign = 0
            for g in isl.population.values():
                mids = g.module_ids()
                for m in mids:
                    carriers[m] = carriers.get(m, 0) + 1
                if any(self.module_origin.get(m, isl.island) != isl.island for m in mids):
                    foreign += 1
            b = isl.best_this_gen
            isl_rows.append({"island": isl.island, "best_fitness": float(b.fitness),
                             "mean_fitness": float(np.mean(fits)),
                             "n_species": len(isl.species.species),
                             "best_key": b.key, "best_modules": sorted(b.module_ids()),
                             "n_carrying_any_module": sum(1 for g in isl.population.values()
                                                          if g.module_ids()),
                             "n_carrying_foreign_module": foreign,
                             "carriers": {str(k): v for k, v in sorted(carriers.items())}})
        best_isl = max(range(self.n_islands),
                       key=lambda i: (isl_rows[i]["best_fitness"], -i))
        final_stage = stage_idx == len(self.task.stages) - 1
        best_f = isl_rows[best_isl]["best_fitness"]
        if final_stage and best_f >= 1.0 and self.solved_gen is None:
            self.solved_gen = gen
        rec = {"generation": gen, "stage": stage_idx, "best_fitness": best_f,
               "best_island": best_isl, "islands": isl_rows}
        self.gen_records.append(rec)
        if gen % 10 == 0 or gen == self.n_gen - 1:
            self.log(f"gen {gen} stage {stage_idx} best {best_f:.4f} "
                     f"islands {[round(r['best_fitness'], 4) for r in isl_rows]} "
                     f"lib {len(self.library.modules)}")

    # ------------------------------------------------------------------ checkpoint
    def _checkpoint(self, gen, stage):
        rows = []
        for isl in self.islands:
            g = isl.best_this_gen
            try:
                chunks = find_chunks(g, self.config, stage.X_train, stage.y_train,
                                     probe_X=self.task.probe_X)
            except Exception as e:  # pragma: no cover - logged
                self.log(f"chunk finder failed gen {gen} island {isl.island}: {e!r}")
                chunks = []
            new_tags = {}
            old_tags = g.tags()
            for ch in chunks:
                v = self.library.add(ch, isl.island, gen, g.key)
                prior = [old_tags.get(n) for n in ch.hidden_nodes]
                # did the chunk's nodes already carry this module id (descended
                # from a graft / earlier tag) or is this an independent alias?
                self.library.history[-1].update(
                    n_nodes_already_tagged_same=sum(1 for t in prior if t == v.module_id),
                    n_nodes_tagged_other=sum(1 for t in prior
                                             if t is not None and t != v.module_id))
                self.module_origin.setdefault(v.module_id, isl.island)
                truth = {}
                for name, f in self.task.truth.items():
                    truth[name] = pearson(ch.s_probe, f)
                if truth:
                    tb = max(truth, key=lambda k: (np.nan_to_num(truth[k], nan=-2), k))
                    self.library.history[-1].update(truth_best=tb, truth_r=truth[tb])
                for n in ch.hidden_nodes:
                    new_tags.setdefault(n, v.module_id)
                rows.append(dict(self.library.history[-1]))
            # re-derived boundaries: the best genome's chunk nodes carry the
            # aligned module ids (open chunks)
            tags = dict(g.module_tags)
            tags.update(new_tags)
            g.module_tags = tags
        self.checkpoint_records.append({"generation": gen, "n_chunks": len(rows),
                                        "n_modules": len(self.library.modules),
                                        "chunks": rows})
        self.log(f"checkpoint gen {gen}: {len(rows)} chunks, "
                 f"{len(self.library.modules)} modules")

    # ------------------------------------------------------------------ arm 1
    def _migration_hook(self, gen, migrants):
        def hook(isl, old, new):
            nonelite = sorted(k for k in new if k not in old)
            rng = np.random.default_rng([self.seed, gen, isl.island, 1])
            others = [m for j, m in enumerate(migrants) if j != isl.island]
            slots = sorted(rng.choice(nonelite, size=min(len(others), len(nonelite)),
                                      replace=False).tolist())
            for slot, mig in zip(slots, others):
                g = copy.deepcopy(mig)
                del new[slot]
                g.key = next(isl.reproduction.genome_indexer)
                g.fitness = None
                new[g.key] = g
                self.op_records.append({"generation": gen, "island": isl.island,
                                        "op": "migration", "child_key": g.key,
                                        "applied": True})
        return hook

    # ------------------------------------------------------------------ arms 2-4
    def _operator_hook(self, gen, stage):
        def hook(isl, old, new):
            nonelite = sorted(k for k in new if k not in old)
            n_op = int(round(self.op_fraction * len(nonelite)))
            rng = np.random.default_rng([self.seed, gen, isl.island, 2])
            chosen = sorted(rng.choice(nonelite, size=n_op, replace=False).tolist())
            parents = sorted(old.values(), key=lambda g: (-g.fitness, g.key))
            top = parents[:max(2, int(np.ceil(0.2 * len(parents))))]
            for i, key in enumerate(chosen):
                op = OPS[i % len(OPS)]
                rec = {"generation": gen, "island": isl.island, "child_key": key,
                       "op_requested": op}
                child = new[key]
                out = None
                for attempt in [op] + (["insertion"] if op != "insertion" else []):
                    try:
                        out, info = self._apply(attempt, isl, child, top, parents, stage, rng)
                        rec.update(info)
                        rec["op"] = attempt
                        break
                    except OperatorNotApplicable as e:
                        rec.setdefault("not_applicable", []).append(f"{attempt}: {e}")
                    except Exception as e:  # treated as an invalid result (PLAN.md)
                        rec["error"] = f"{attempt}: {e!r}"
                        out = None
                        break
                if out is None:
                    rec["applied"] = False
                elif not valid_feedforward(out, self.config):
                    rec["applied"] = False
                    rec["invalid"] = True
                else:
                    out.key = key
                    out.fitness = None
                    new[key] = out
                    rec["applied"] = True
                self.op_records.append(rec)
        return hook

    def _foreign_modules(self, island, carried):
        out = []
        for mid in self.library.module_ids():
            if mid in carried:
                continue
            if self.library.best_variant(mid, exclude_island=island) is not None:
                out.append(mid)
        return out

    def _apply(self, op, isl, child, top, parents, stage, rng):
        wm = self.weight_mode
        X, y = stage.X_train, stage.y_train
        if op == "insertion":
            cands = self._foreign_modules(isl.island, child.module_ids())
            if not cands:
                raise OperatorNotApplicable("no module from another island that the child lacks")
            mid = int(rng.choice(cands))
            v = self.library.best_variant(mid, exclude_island=isl.island)
            g, ro, _ = graft_chunk(child, self.config, v.chunk, mid, wm, rng)
            info = {"module_id": mid, "variant_id": v.variant_id, "source_island": v.island}
            if wm != "topology":
                g, rinfo = refit(g, self.config, X, y, ro)
                info["refit"] = rinfo
            return g, info
        if op == "replacement":
            opts = []
            for mid in sorted(child.module_ids()):
                if mid in self.library.modules and \
                        self.library.best_variant(mid, exclude_island=isl.island) is not None:
                    opts.append(mid)
            if not opts:
                raise OperatorNotApplicable("child carries no module with another island's alias")
            mid = int(rng.choice(opts))
            v = self.library.best_variant(mid, exclude_island=isl.island)
            ko, removed = knockout_module(child, mid)
            g, ro, _ = graft_chunk(ko, self.config, v.chunk, mid, wm, rng)
            info = {"module_id": mid, "variant_id": v.variant_id, "source_island": v.island,
                    "n_removed_nodes": len(removed)}
            if wm != "topology":
                g, rinfo = refit(g, self.config, X, y, ro)
                info["refit"] = rinfo
            return g, info
        if op == "crossover":
            # parent a: random top genome; parent b: a top genome (else any
            # member) whose module content differs from a's
            pa = top[int(rng.integers(len(top)))]
            pool = [g for g in top if g is not pa and (differing_modules(pa, g)
                                                       or differing_modules(g, pa))]
            if not pool:
                pool = [g for g in parents if g is not pa and (differing_modules(pa, g)
                                                               or differing_modules(g, pa))]
            if not pool:
                raise OperatorNotApplicable("no parent pair with differing modules")
            pb = pool[int(rng.integers(len(pool)))]
            if pb.fitness > pa.fitness:
                pa, pb = pb, pa
            c = ModuleGenome(child.key)
            c.configure_crossover(pa, pb, self.config.genome_config)
            c.mutate(self.config.genome_config)
            # per module carried with different content by b: inherit b's
            # version with probability 1/2
            mods_b = differing_modules(pa, pb)
            taken, ro_all = [], []
            for mid in mods_b:
                if rng.random() < 0.5:
                    c, ro = add_module_keep_ids(c, self.config, pb, mid, wm, rng)
                    taken.append(mid)
                    ro_all.extend(ro)
            info = {"parent_a": pa.key, "parent_b": pb.key, "modules_from_b": taken,
                    "modules_b_differing": mods_b}
            if ro_all and wm != "topology":
                c, rinfo = refit(c, self.config, X, y, ro_all)
                info["refit"] = rinfo
            return c, info
        raise ValueError(op)

    # ------------------------------------------------------------------ final
    def final_summary(self):
        task, cfg = self.task, self.config
        fin = task.final
        per_island = []
        cands = []
        for isl in self.islands:
            for g in isl.population.values():
                cands.append((isl.island, g))
        # final best: highest train AUC (fitness), then lowest train BCE,
        # then lowest island, then lowest key
        scored = []
        for i, g in cands:
            p = gs.predict_proba(g, cfg, fin.X_train)
            scored.append((-g.fitness, bce(fin.y_train, p), i, g.key, g))
        scored.sort(key=lambda t: t[:4])
        best_island, best = scored[0][2], scored[0][4]
        self.final_best = best
        self.final_best_island = best_island
        island_best = {}
        for t in scored:
            island_best.setdefault(t[2], t[4])
        for i in range(self.n_islands):
            g = island_best[i]
            per_island.append(self._metrics(g) | {"island": i, "key": g.key})
        out = {"task": task.name, "arm": self.arm, "seed": self.seed,
               "n_gen": self.n_gen, "pop_size": self.pop_size, "n_islands": self.n_islands,
               "K": self.K, "op_fraction": self.op_fraction,
               "stage_switch": self.stage_switch,
               "wall_s": self.wall_s, "solved_gen": self.solved_gen,
               "best_island": best_island, "best_key": best.key,
               **{f"final_{k}": v for k, v in self._metrics(best).items()},
               "per_island": per_island,
               "n_library_modules": len(self.library.modules),
               "n_library_variants_added": len(self.library.history)}
        return out

    def _metrics(self, g):
        fin, cfg = self.task.final, self.config
        ptr = gs.predict_proba(g, cfg, fin.X_train)
        pte = gs.predict_proba(g, cfg, fin.X_test)
        nh, nc = genome_size(g, cfg)
        return {"train_auc": float(roc_auc_score(fin.y_train, ptr)),
                "test_auc": float(roc_auc_score(fin.y_test, pte)),
                "train_acc": float(accuracy_score(fin.y_train, ptr > 0.5)),
                "test_acc": float(accuracy_score(fin.y_test, pte > 0.5)),
                "train_bce": bce(fin.y_train, ptr), "test_bce": bce(fin.y_test, pte),
                "hidden_nodes": nh, "connections": nc,
                "modules": sorted(int(m) for m in g.module_ids()),
                "fitness": float(g.fitness)}

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.task.name}-{self.arm}-s{self.seed}"
        summary = self.final_summary()
        (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1, default=float))
        detail = {"generations": self.gen_records, "checkpoints": self.checkpoint_records,
                  "operators": self.op_records, "library_history": self.library.history,
                  "module_origin": self.module_origin,
                  "library": {str(m): [v.to_json() for v in self.library.variants(m)]
                              for m in self.library.module_ids()}}
        (out_dir / f"{stem}.detail.json").write_text(json.dumps(detail, default=_js))
        elites = {"final_best": self.final_best, "final_best_island": self.final_best_island,
                  "island_best": {isl.island: max(isl.population.values(),
                                                  key=lambda g: (g.fitness, -g.key))
                                  for isl in self.islands}}
        with open(out_dir / f"{stem}.elites.pkl", "wb") as f:
            pickle.dump(elites, f)
        return summary


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)
