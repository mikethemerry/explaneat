"""Study H, Test 4: XOR lineages with full ancestry capture (PLAN_H.md).

Re-runs Part 2 of ``experiments/xor/run_xor.py`` exactly: same config, seeds
0-29 in order in one process with one shared Config (node keys continue across
seeds, as in the original), and the same stopping rule (the first generation
whose best-ever genome classifies all four rows). It adds a read-only reporter
that snapshots every evaluated genome, and reads ``parent_log``. Check: every
seed's solved generation must equal ``xor_results.json``; a mismatch stops the
script.

For the solving genome's ``fitter_parent`` lineage it records, per lineage
generation:
- rows correct and BCE;
- each phenotype hidden unit's active set P (pre-activation > 0 on the rows
  00, 10, 01, 11) and the sign u of its direct output weight;
- whether (P, u) is a feasible ReLU realisation pair from Part 1's LP.
It also records the structure signature.

Outputs (in experiments/lineage/):
- ``h_xor_lineages.csv``: one row per seed x lineage generation;
- ``h_xor.csv``: one row per seed, with g_solve, g_F and the solving step's type.

    PYTHONHASHSEED=0 uv run python experiments/lineage/run_xor_lineage.py
"""
import copy
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_xor", REPO / "experiments/xor/run_xor.py")
rx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rx)

from explaneat.analysis.lineage import structure_signature  # noqa: E402
from explaneat.core.ancestry_reporter import AncestryReporter  # noqa: E402
from explaneat.db.serialization import serialize_genome  # noqa: E402


class Snapshot(AncestryReporter):
    """Read-only: deep-copies every genome after evaluation (no RNG use)."""

    def __init__(self):
        super().__init__()
        self.gens = {}

    def post_evaluate(self, config, population, species, best_genome):
        self.gens[len(self.gens)] = {k: copy.deepcopy(g) for k, g in population.items()}


def run_seed(seed, generations, config):
    """``run_xor.run_seed`` plus the snapshot reporter."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pop = rx.SafePopulation(config, rx.X, rx.Y, mask_phantom=True)
    snap = Snapshot()
    pop.reporters.add(snap)
    for gen in range(generations):
        pop.run(rx.bce_fitness, n=1, nEpochs=5)
        best = pop.best_genome
        p = rx.NeuralNeat(best, config).forward(pop.xs).detach().numpy().ravel()
        if np.all((p > 0.5) == (rx.Y > 0.5)):
            return gen, best, pop, snap
    return None, best, pop, snap


def lineage_keys(pop, snap, gen_end, key_end):
    """fitter_parent chain [(gen, key, followed)] ending at (gen_end, key_end)."""
    out = [(gen_end, key_end, None)]
    g, k = gen_end, key_end
    while g > 0:
        p1, p2, elite = pop.parent_log[g][k]
        if elite:
            nk, how = k, "elite"
        else:
            f1 = snap.gens[g - 1][p1].fitness
            f2 = snap.gens[g - 1][p2].fitness
            nk, how = (p2, "parent2") if f2 > f1 else (p1, "parent1")
        out[-1] = (g, k, how)
        g, k = g - 1, nk
        out.append((g, k, None))
    return out[::-1]


def units(g, config, feasible_pairs):
    E, hidden, ins, out = rx.phenotype(g, config)
    bias = {n: g.nodes[n].bias for n in g.nodes}
    rows = []
    if hidden:
        pre, z = rx.evaluate(E, hidden, ins, out, bias, rx.X)
    for h in hidden:
        P = "".join(str(int(v > 0)) for v in pre[h])
        w_out = [w for a, b, w in E if a == h and b == out]
        u = (1 if w_out[0] > 0 else -1) if w_out else 0
        nonlinear = 0 < P.count("1") < 4
        feas = nonlinear and (any((P, s) in feasible_pairs for s in (1, -1)) if u == 0
                              else (P, u) in feasible_pairs)
        rows.append(dict(node=h, P=P, active=rx.BOOL[tuple(int(c) for c in P)], u=u,
                         nonlinear=nonlinear, feasible=feas))
    return rows


def main():
    ref = json.load(open(REPO / "experiments/xor/xor_results.json"))
    ref_gen = {r["seed"]: r["solved_generation"] for r in ref["runs"]}
    rows_relu, _ = rx.enumerate_realisations("relu")
    feasible_pairs = {(r["P"], r["u"]) for r in rows_relu}
    for hidden in ("relu", "threshold"):  # main() of run_xor does this before evolving
        rx.enumerate_realisations(hidden)
    config = rx.load_config()
    bce = torch.nn.BCELoss()
    xs = torch.tensor(rx.X, dtype=torch.float64)
    ys = torch.tensor(rx.Y, dtype=torch.float64).unsqueeze(1)
    lin_rows, seed_rows = [], []
    for seed in range(30):
        gen, best, pop, snap = run_seed(seed, 150, config)
        if gen != ref_gen[seed]:
            sys.exit(f"seed {seed}: solved generation {gen} != reference {ref_gen[seed]}: STOP")
        # The solving genome: best-ever key, evaluated at the latest generation with that fitness.
        cands = [g for g in range(gen + 1) if best.key in snap.gens[g]
                 and snap.gens[g][best.key].fitness == best.fitness]
        g_end = max(cands)
        chain = lineage_keys(pop, snap, g_end, best.key)
        prev_sig, info = None, []
        for g, k, how in chain:
            gen_g = snap.gens[g][k]
            p = rx.NeuralNeat(copy.deepcopy(gen_g), config).forward(xs).detach()
            correct = int(np.sum((p.numpy().ravel() > 0.5) == (rx.Y > 0.5)))
            loss = float(bce(p, ys))
            sig = structure_signature(serialize_genome(gen_g))
            us = units(gen_g, config, feasible_pairs)
            r = dict(seed=seed, generation=g, genome_key=k, followed=how,
                     is_elite_copy=how == "elite", fitness=gen_g.fitness, rows_correct=correct,
                     bce=loss, structural_change=prev_sig is not None and sig != prev_sig,
                     n_hidden=len(us), n_nonlinear=sum(u["nonlinear"] for u in us),
                     n_feasible=sum(u["feasible"] for u in us),
                     feasible_units=";".join(f"{u['node']}:{u['active']}(u={u['u']:+d})"
                                             for u in us if u["feasible"]),
                     units=";".join(f"{u['node']}:{u['P']}:{u['u']:+d}" for u in us))
            info.append((r, {u["node"] for u in us if u["feasible"]}, set(gen_g.nodes)))
            lin_rows.append(r)
            prev_sig = sig
        g_solve = info[-1][0]["generation"]
        assert info[-1][0]["rows_correct"] == 4
        first_f = next((r for r, f, _ in info if r["n_feasible"] > 0), None)
        last_r, f_last, nodes_last = info[-1]
        prev_r, f_prev, nodes_prev = info[-2] if len(info) > 1 else (None, set(), set())
        new_feasible = f_last - f_prev
        seed_rows.append(dict(
            seed=seed, solved_generation=gen, g_solve=g_solve,
            lineage_length=len(chain),
            g_F=first_f["generation"] if first_f else None,
            g_solve_minus_g_F=(g_solve - first_f["generation"]) if first_f else None,
            coincide=bool(first_f and first_f["generation"] == g_solve),
            solve_step_structural=last_r["structural_change"],
            solve_step_elite=last_r["is_elite_copy"],
            rows_correct_before=prev_r["rows_correct"] if prev_r else None,
            feasible_units_at_solve=last_r["feasible_units"],
            new_feasible_at_solve=len(new_feasible) > 0,
            new_feasible_units_added_at_solve=len(new_feasible & (nodes_last - nodes_prev)) > 0,
            new_feasible_units_moved_in_at_solve=len(new_feasible & nodes_prev) > 0))
        print(json.dumps(seed_rows[-1]), flush=True)

    for path, rows in ((HERE / "h_xor_lineages.csv", lin_rows), (HERE / "h_xor.csv", seed_rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    n = len(seed_rows)
    k = sum(r["coincide"] for r in seed_rows)
    print(f"coincide {k}/{n}")


if __name__ == "__main__":
    main()
