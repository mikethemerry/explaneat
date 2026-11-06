"""XOR realisations: theory (LP enumeration) vs what PropNEAT actually evolves.

Part 1 enumerates every way the minimal topology (2 inputs, 1 ReLU hidden node,
skip connections input->output allowed) can compute XOR. A realisation is
characterised by the hidden node's *active set* P (the inputs on which its
pre-activation is > 0, i.e. which Boolean function it fires on) and the sign
u of the hidden->output weight. Positive scaling of the hidden node's incoming
weights is a ReLU symmetry, so |u| can be fixed to 1. Each (P, u) is a linear
feasibility problem in the remaining weights, solved with an LP and a margin.
Classes are then merged under the problem's symmetry x1 <-> x2. For comparison,
the same enumeration is done with a sigmoid/threshold hidden unit, which has the
extra symmetry sigma(-z) = 1 - sigma(z) and so merges P with its complement.

Part 2 runs PropNEAT (BackpropPopulation, ReLU hidden, sigmoid output, masked
backprop) for many seeds until the best genome classifies all four XOR rows
correctly. Each solution is pruned to its active phenotype and classified by
hidden-node active sets and output-weight signs, then mapped onto Part 1's
classes when it has the minimal topology.

Part 3 illustrates non-identifiability on the evolved solutions: for every pair,
the correlation of their output logits over a dense grid of inputs in [0, 1]^2
(function similarity) against the cosine between their weight vectors
(minimal-topology pairs only, after normalising the hidden unit's incoming
vector to unit length and scaling its outgoing weight by the inverse, which
removes the ReLU scaling symmetry).

    uv run python experiments/xor/run_xor.py [--seeds 30] [--generations 150]
"""
import argparse
import itertools
import json
import os
import random
import sys
import tempfile

import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")
import torch  # noqa: E402

torch.set_num_threads(1)
import neat  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 0], dtype=float)
ROWS = ["00", "10", "01", "11"]

# Name every Boolean function of two inputs by its truth table on ROWS order.
BOOL = {
    (0, 0, 0, 0): "FALSE", (1, 1, 1, 1): "TRUE",
    (0, 0, 0, 1): "AND", (0, 1, 1, 1): "OR", (1, 0, 0, 0): "NOR", (1, 1, 1, 0): "NAND",
    (0, 1, 1, 0): "XOR", (1, 0, 0, 1): "XNOR",
    (0, 1, 0, 1): "x1", (1, 0, 1, 0): "NOT x1", (0, 0, 1, 1): "x2", (1, 1, 0, 0): "NOT x2",
    (0, 1, 0, 0): "x1 AND NOT x2", (0, 0, 1, 0): "x2 AND NOT x1",
    (1, 1, 0, 1): "x1 OR NOT x2", (1, 0, 1, 1): "x2 OR NOT x1",
}
SWAP = {"00": "00", "10": "01", "01": "10", "11": "11"}


# ---------------------------------------------------------------- Part 1
def feasible(P, u, hidden="relu", margin=1.0):
    """LP feasibility of XOR with hidden active set P (tuple of 0/1 over ROWS)
    and hidden->output sign u.

    ReLU: variables [w1, w2, b, v1, v2, c]. h = a = w1 x1 + w2 x2 + b on P
    (a >= margin) and 0 off P (a <= 0). |u| = 1 without loss of generality,
    because positive scaling of (w1, w2, b) by k and u by 1/k leaves the
    network unchanged.
    Threshold: variables [w1, w2, b, v1, v2, c, U]. h = 1 on P (a >= margin),
    0 off P (a <= -margin). The output weight U has no scaling symmetry to
    absorb it, so it is a free variable with sign u (|U| >= margin)."""
    relu = hidden == "relu"
    n = 6 if relu else 7
    A, rhs = [], []
    for (x1, x2), p, y in zip(X, P, Y):
        a = [x1, x2, 1, 0, 0, 0] + ([] if relu else [0])
        if p:
            A.append([-t for t in a]); rhs.append(-margin)
        else:
            A.append(a); rhs.append(0.0 if relu else -margin)
        if relu:   # z = v1 x1 + v2 x2 + c + u * a * p
            z = [u * p * x1, u * p * x2, u * p, x1, x2, 1]
        else:      # z = v1 x1 + v2 x2 + c + U * p
            z = [0, 0, 0, x1, x2, 1, p]
        if y:
            A.append([-t for t in z]); rhs.append(-margin)
        else:
            A.append(z); rhs.append(-margin)
    bounds = [(-20, 20)] * 6
    if not relu:
        bounds.append((margin, 20) if u > 0 else (-20, -margin))
    res = linprog(np.zeros(n), A_ub=np.array(A), b_ub=np.array(rhs),
                  bounds=bounds, method="highs")
    return res.status == 0, (res.x.round(3).tolist() if res.status == 0 else None)


def enumerate_realisations(hidden):
    rows = []
    for P in itertools.product([0, 1], repeat=4):
        for u in (1, -1):
            ok, sol = feasible(P, u, hidden)
            if ok:
                rows.append(dict(active=BOOL[P], P="".join(map(str, P)), u=u, example=sol))
    # Two groupings. "swap": the input permutation x1 <-> x2 only. "full": also
    # XOR's own input-complement symmetry x -> 1 - x (XOR(1-a, 1-b) = XOR(a, b);
    # it maps rows 00<->11, 10<->01 and keeps every weight affine, so it maps a
    # realisation onto another realisation of the same signal). For threshold
    # units "full" also includes the unit's sign flip (complement P, negate U).
    def keys(r, full):
        P = dict(zip(ROWS, r["P"]))
        out = {(r["P"], r["u"]), ("".join(P[SWAP[k]] for k in ROWS), r["u"])}
        if full:
            comp_rows = {"00": "11", "11": "00", "10": "01", "01": "10"}
            out |= {("".join(dict(zip(ROWS, k[0]))[comp_rows[q]] for q in ROWS), k[1])
                    for k in list(out)}
            if hidden != "relu":
                flip = lambda t: "".join("1" if c == "0" else "0" for c in t)
                out |= {(flip(k[0]), -k[1]) for k in list(out)}
        return min(out)
    classes = {g: {} for g in ("swap", "full")}
    for r in rows:
        for g in classes:
            classes[g].setdefault(keys(r, g == "full"), []).append(r)
    return rows, classes


# ---------------------------------------------------------------- Part 2
CONFIG = """
[NEAT]
fitness_criterion = max
fitness_threshold = 1e9
pop_size = 150
reset_on_extinction = True

[DefaultGenome]
activation_default = relu
activation_mutate_rate = 0.0
activation_options = relu
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.0
bias_mutate_rate = 0.0
bias_replace_rate = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.5
conn_delete_prob = 0.3
enabled_default = True
enabled_mutate_rate = 0.1
feed_forward = True
initial_connection = full_direct
node_add_prob = 0.3
node_delete_prob = 0.2
num_hidden = 0
num_inputs = 2
num_outputs = 1
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.0
weight_mutate_rate = 0.0
weight_replace_rate = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
"""


def load_config():
    f = tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False)
    f.write(CONFIG); f.close()
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, f.name)


def bce_fitness(genomes, config, xs, ys, device):
    loss = torch.nn.BCELoss()
    for _, g in genomes.items():
        try:
            p = NeuralNeat(g, config).forward(xs)
            g.fitness = float(1.0 / loss(p, ys))
        except Exception:
            g.fitness = 0.0


def phenotype(g, config):
    """Active phenotype: enabled connections on some input->output path."""
    ins = config.genome_config.input_keys
    out = config.genome_config.output_keys[0]
    edges = [(a, b, c.weight) for (a, b), c in g.connections.items() if c.enabled]
    fwd, bwd = {}, {}
    for a, b, _ in edges:
        fwd.setdefault(a, set()).add(b); bwd.setdefault(b, set()).add(a)
    def reach(starts, adj):
        seen, st = set(starts), list(starts)
        while st:
            n = st.pop()
            for m in adj.get(n, ()):
                if m not in seen:
                    seen.add(m); st.append(m)
        return seen
    live = reach(ins, fwd) & reach([out], bwd)
    E = [(a, b, w) for a, b, w in edges if a in live and b in live]
    hidden = sorted(n for n in live if n not in ins and n != out)
    return E, hidden, ins, out


def evaluate(E, hidden, ins, out, bias, xs):
    """Manual forward pass: ReLU hidden, sigmoid output. Returns (h-preacts, z)."""
    order, done = [], set(ins)
    pending = list(hidden) + [out]
    while pending:
        for n in list(pending):
            srcs = [a for a, b, _ in E if b == n]
            if all(s in done for s in srcs):
                order.append(n); done.add(n); pending.remove(n)
    vals = {k: xs[:, i] for i, k in enumerate(ins)}
    pre = {}
    for n in order:
        z = np.full(len(xs), bias[n])
        for a, b, w in E:
            if b == n:
                z = z + w * vals[a]
        pre[n] = z
        vals[n] = z if n == out else np.maximum(z, 0)
    return pre, vals[out]


class SafePopulation(BackpropPopulation):
    """With only two inputs, mutation sometimes deletes every path to the
    output. NeuralNeat rejects such genomes, so they are held out of backprop
    (and score 0 in bce_fitness) instead of crashing the run."""

    def backpropagate(self, xs, ys, nEpochs=5):
        held = {}
        for k, g in list(self.population.items()):
            try:
                NeuralNeat(g, self.config)
            except Exception:
                held[k] = self.population.pop(k)
        try:
            return super().backpropagate(xs, ys, nEpochs=nEpochs)
        finally:
            self.population.update(held)


def run_seed(seed, generations, config):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pop = SafePopulation(config, X, Y, mask_phantom=True)
    for gen in range(generations):
        pop.run(bce_fitness, n=1, nEpochs=5)
        best = pop.best_genome  # run() reproduces after evaluating
        p = NeuralNeat(best, config).forward(pop.xs).detach().numpy().ravel()
        if np.all((p > 0.5) == (Y > 0.5)):
            return gen, best, p
    return None, best, p


def train(g, config, epochs=300):
    """Masked full-batch backprop exactly as BackpropPopulation does it
    (Adadelta lr 1.5, BCE), for a fixed number of epochs; returns the genome
    with trained weights written back and the predictions."""
    import copy
    net = NeuralNeat(copy.deepcopy(g), config, mask_phantom=True)
    xs = torch.tensor(X, dtype=torch.float64); ys = torch.tensor(Y, dtype=torch.float64).unsqueeze(1)
    opt = torch.optim.Adadelta(net.parameters(), lr=1.5)
    bce = torch.nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = bce(net.forward(xs), ys); loss.backward(); opt.step()
    net.update_genome_weights()
    return net.genome, net.forward(xs).detach().numpy().ravel()


def solves(p, margin=0.1):
    return bool(np.all(np.where(Y > 0.5, p > 0.5 + margin, p < 0.5 - margin)))


def prune(g, config, seed):
    """Greedy reduction to a minimal sub-network that still solves XOR:
    repeatedly try disabling each enabled connection (seeded order), retrain
    300 epochs, keep the deletion if XOR is still solved with margin 0.1.
    This is a function-level simplification (like identifying which parts of
    an explanation do the work), not part of evolution."""
    import copy
    rng = random.Random(seed)
    g, p = train(g, config, 0)
    changed = True
    while changed:
        changed = False
        keys = sorted(k for k, c in g.connections.items() if c.enabled)
        rng.shuffle(keys)
        for k in keys:
            g2 = copy.deepcopy(g)
            g2.connections[k].enabled = False
            torch.manual_seed(seed)
            try:
                g2t, p2 = train(g2, config)
            except Exception:
                continue
            if solves(p2):
                g, p, changed = g2t, p2, True
                break
    return g, p


def collapse_relays(g, config):
    """Fold every hidden unit that is active on all four XOR rows (a 'relay':
    ReLU is the identity there, so the unit is exactly linear on the data) into
    direct connections: for relay t with pre-activation sum_i w_i n_i + b and
    an edge t -> m of weight u, add edges n_i -> m with weight u * w_i and add
    u * b to m's bias, then delete t. Exact on the data. Repeats until no relay
    is left. Returns (genome, number collapsed)."""
    import copy
    g = copy.deepcopy(g)
    n_collapsed = 0
    while True:
        E, hidden, ins, out = phenotype(g, config)
        bias = {k: n.bias for k, n in g.nodes.items()}
        pre, _ = evaluate(E, hidden, ins, out, bias, X)
        relays = [h for h in hidden if np.all(pre[h] > 0)]
        if not relays:
            return g, n_collapsed
        t = relays[0]
        w_in = [(a, w) for a, b, w in E if b == t]
        w_out = [(b, w) for a, b, w in E if a == t]
        for m, u in w_out:
            g.nodes[m].bias += u * bias[t]
            for n, w in w_in:
                key = (n, m)
                if key in g.connections and g.connections[key].enabled:
                    g.connections[key].weight += u * w
                else:
                    c = g.create_connection(config.genome_config, n, m)
                    c.weight = u * w; c.enabled = True
                    g.connections[key] = c
        for key in [k for k in g.connections if t in k]:
            del g.connections[key]
        del g.nodes[t]
        n_collapsed += 1


def reduce(g, config, seed):
    """Alternate relay collapse (exact) and greedy pruning (retrain) to a fixed
    point. Returns (genome, predictions, relays collapsed)."""
    total = 0
    g, p = prune(g, config, seed)
    while True:
        g2, k = collapse_relays(g, config)
        if k == 0:
            return g, p, total
        total += k
        _, p2 = train(g2, config, 0)
        assert solves(p2, 0.0), "relay collapse must preserve the function on the data"
        g, p = prune(g2, config, seed)


def describe(best, config):
    E, hidden, ins, out = phenotype(best, config)
    bias = {k: n.bias for k, n in best.nodes.items()}
    pre, z = evaluate(E, hidden, ins, out, bias, X)
    nodes = []
    for h in hidden:
        P = tuple(int(v > 0) for v in pre[h])
        nodes.append(dict(node=h, active=BOOL[P], P="".join(map(str, P)),
                          w_in={str(a): round(w, 3) for a, b, w in E if b == h},
                          w_out={str(b): round(w, 3) for a, b, w in E if a == h},
                          bias=round(bias[h], 3)))
    skips = {str(a): round(w, 3) for a, b, w in E if b == out and a in ins}
    return dict(n_hidden=len(hidden), n_connections=len(E), hidden=nodes,
                skip_weights=skips, output_bias=round(bias[out], 3),
                logits=[round(float(v), 3) for v in z]), (E, hidden, ins, out, bias)


def minimal_vector(E, hidden, ins, out, bias):
    """[w1, w2, b, u, v1, v2, c] with the hidden incoming vector normalised to
    unit length and u scaled by the inverse norm (ReLU positive-scaling symmetry)."""
    (h,) = hidden
    w = {a: wt for a, b, wt in E if b == h}
    v = {a: wt for a, b, wt in E if b == out and a in ins}
    u = [wt for a, b, wt in E if a == h and b == out][0]
    inc = np.array([w.get(ins[0], 0.0), w.get(ins[1], 0.0), bias[h]])
    nrm = np.linalg.norm(inc)
    return np.concatenate([inc / nrm, [u * nrm, v.get(ins[0], 0.0), v.get(ins[1], 0.0), bias[out]]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--generations", type=int, default=150)
    args = ap.parse_args()

    theory = {}
    for hidden in ("relu", "threshold"):
        rows, classes = enumerate_realisations(hidden)
        theory[hidden] = dict(n_feasible=len(rows), **{
            g: [dict(key=list(k), members=[f"{r['active']} (u={r['u']:+d})" for r in v],
                     example=v[0]["example"]) for k, v in sorted(cl.items())]
            for g, cl in classes.items()})
        print(f"\n[{hidden} hidden] {len(rows)} feasible (active set, sign) pairs; "
              f"{len(classes['swap'])} classes under x1<->x2, "
              f"{len(classes['full'])} under the full symmetry group:")
        for c in theory[hidden]["full"]:
            print("  ", " | ".join(c["members"]), " e.g.", c["example"])

    config = load_config()
    grid = np.array([[a, b] for a in np.linspace(0, 1, 21) for b in np.linspace(0, 1, 21)])
    runs, minimal = [], []
    for seed in range(args.seeds):
        gen, best, p = run_seed(seed, args.generations, config)
        raw, _ = describe(best, config)
        if gen is not None:
            pruned_g, p, n_relays = reduce(best, config, seed)
        else:
            pruned_g, n_relays = best, 0
        d, parts = describe(pruned_g, config)
        d.update(seed=seed, solved_generation=gen, solved=gen is not None and solves(p, 0.0),
                 preds=[round(float(v), 3) for v in p], relays_collapsed=n_relays,
                 raw=dict(n_hidden=raw["n_hidden"], n_connections=raw["n_connections"],
                          hidden_active_sets=[h["active"] for h in raw["hidden"]],
                          n_nonlinear_hidden=sum(h["active"] not in ("TRUE", "FALSE") for h in raw["hidden"])))
        _, zg = evaluate(*parts[:4], parts[4], grid)
        d["_grid_logit"] = zg
        if gen is not None and d["n_hidden"] == 1:
            d["minimal_class"] = d["hidden"][0]["active"] + f" (u={'+' if list(d['hidden'][0]['w_out'].values())[0] > 0 else '-'})"
            minimal.append((seed, minimal_vector(*parts), d["minimal_class"]))
        runs.append(d)
        print(f"seed {seed:2d}: solved gen {gen}; raw hidden {raw['n_hidden']} "
              f"({d['raw']['n_nonlinear_hidden']} nonlinear), conns {raw['n_connections']} -> pruned "
              f"hidden {d['n_hidden']}, conns {d['n_connections']}, active sets "
              f"{[h['active'] for h in d['hidden']]}, out {[h['w_out'] for h in d['hidden']]}, "
              f"skips {d['skip_weights']}, class {d.get('minimal_class')}", flush=True)

    # Part 3: function similarity vs weight similarity
    solved = [r for r in runs if r["solved"]]
    fun_r = []
    for a, b in itertools.combinations(solved, 2):
        fun_r.append(float(np.corrcoef(a["_grid_logit"], b["_grid_logit"])[0, 1]))
    pairs = []
    for (sa, va, ca), (sb, vb, cb) in itertools.combinations(minimal, 2):
        ra = next(r for r in runs if r["seed"] == sa)["_grid_logit"]
        rb = next(r for r in runs if r["seed"] == sb)["_grid_logit"]
        pairs.append(dict(seeds=[sa, sb], same_class=ca == cb,
                          logit_r=float(np.corrcoef(ra, rb)[0, 1]),
                          weight_cos=float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb))))
    for r in runs:
        r.pop("_grid_logit")

    from collections import Counter
    summary = dict(
        n_seeds=args.seeds, n_solved=len(solved),
        solved_generation_median=float(np.median([r["solved_generation"] for r in solved])) if solved else None,
        raw_hidden_count=dict(Counter(r["raw"]["n_hidden"] for r in solved)),
        raw_nonlinear_hidden_count=dict(Counter(r["raw"]["n_nonlinear_hidden"] for r in solved)),
        pruned_hidden_count=dict(Counter(r["n_hidden"] for r in solved)),
        minimal_classes=dict(Counter(c for _, _, c in minimal)),
        hidden_active_sets=dict(Counter(h["active"] for r in solved for h in r["hidden"])),
        grid_logit_r_all_pairs_median=float(np.median(fun_r)) if fun_r else None,
        minimal_pairs_same_class=dict(
            n=sum(p["same_class"] for p in pairs),
            logit_r_median=float(np.median([p["logit_r"] for p in pairs if p["same_class"]] or [np.nan])),
            weight_cos_median=float(np.median([p["weight_cos"] for p in pairs if p["same_class"]] or [np.nan]))),
        minimal_pairs_diff_class=dict(
            n=sum(not p["same_class"] for p in pairs),
            logit_r_median=float(np.median([p["logit_r"] for p in pairs if not p["same_class"]] or [np.nan])),
            weight_cos_median=float(np.median([p["weight_cos"] for p in pairs if not p["same_class"]] or [np.nan]))),
    )
    print("\nSUMMARY", json.dumps(summary, indent=1))
    with open(os.path.join(HERE, "xor_results.json"), "w") as f:
        json.dump(dict(theory=theory, summary=summary, runs=runs, minimal_pairs=pairs), f, indent=1)


if __name__ == "__main__":
    main()
