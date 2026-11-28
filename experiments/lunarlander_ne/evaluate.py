"""Pick, test, read and render the champion of a run.

1. Candidates: the per-generation best genomes (``champions.pkl``) of the last --last generations.
   Each is scored on --val validation seeds (VAL_BASE + i); the best mean is the champion.
2. The champion is tested on --test held-out seeds (TEST_BASE + i): mean, sd, share >= 200.
3. Parity: NeuralNeat (ExplaNEAT's torch engine, sigmoid outputs) picks the same action as
   ``Policy`` on every state visited in 5 test episodes.
4. Structure: phenotype nodes/connections, every enabled connection with its weight, and the
   action each input pushes towards (sign of direct input->output weights), plus a per-step
   action table on the test episodes (which action, given which leg contacts).
5. Validation score of every 10th generation's best genome (a held-out learning curve).
6. A gif of the champion on the first test seed.

Writes champion.pkl, champion.json, test_returns.csv, val_curve.csv, rollout_seed<k>.gif into
the run folder. Deterministic (fixed seeds, numpy forward pass; torch with 1 thread for parity).
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("EXPLANEAT_DEVICE", "cpu")
import numpy as np  # noqa: E402

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
from policy import ACTION_NAMES, INPUT_NAMES, Policy, rollout  # noqa: E402
from run import TEST_BASE, VAL_BASE, load_config  # noqa: E402

_W = {}


def _init(config_path):
    import gymnasium as gym
    _W["config"] = load_config(config_path)
    _W["env"] = gym.make("LunarLander-v3")


def _score(task):
    key, genome, seeds = task
    pol = Policy(genome, _W["config"])
    return key, [rollout(pol, s, env=_W["env"])[0] for s in seeds]


def parity(genome, config, seeds):
    import torch
    torch.set_num_threads(1)
    from explaneat.core.neuralneat import NeuralNeat
    pol = Policy(genome, config)
    states = []
    for s in seeds:
        rec = []
        rollout(pol, s, record=rec)
        states += [o for o, _, _ in rec]
    X = np.array(states, dtype=np.float64)
    net = NeuralNeat(genome, config, mask_phantom=True)
    net.eval()
    with torch.no_grad():
        out = net.forward(torch.from_numpy(X)).numpy()
    a_nn = out.argmax(axis=1)
    a_pol = np.array([pol.act(x) for x in X])
    return int((a_nn == a_pol).sum()), len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--last", type=int, default=50)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=str(STUDY / "configs" / "lander.ini"))
    args = ap.parse_args()
    run = Path(args.run_dir)
    config = load_config(args.config)
    champs = pickle.loads((run / "champions.pkl").read_bytes())
    gens = sorted(champs)
    val_seeds = [VAL_BASE + i for i in range(args.val)]
    test_seeds = [TEST_BASE + i for i in range(args.test)]
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.config,))

    # 1. champion on validation
    cand = gens[-args.last:]
    val = dict(pool.map(_score, [(g, champs[g], val_seeds) for g in cand], chunksize=1))
    best_g = max(cand, key=lambda g: (np.mean(val[g]), -g))
    champ = champs[best_g]

    # 5. held-out learning curve (every 10th generation's best, and the last)
    curve_g = sorted(set(gens[::10]) | {gens[-1]})
    curve = dict(pool.map(_score, [(g, champs[g], val_seeds[:20]) for g in curve_g], chunksize=1))
    with open(run / "val_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "train_fitness", "val20_mean"])
        for g in curve_g:
            w.writerow([g, f"{champs[g].fitness:.4f}", f"{np.mean(curve[g]):.4f}"])

    # 2. test
    chunks = [test_seeds[i:i + 10] for i in range(0, len(test_seeds), 10)]
    parts = pool.map(_score, [(i, champ, c) for i, c in enumerate(chunks)], chunksize=1)
    test = [r for _, rs in sorted(parts) for r in rs]
    pool.close()
    pool.join()
    with open(run / "test_returns.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "return"])
        for s, r in zip(test_seeds, test):
            w.writerow([s, f"{r:.4f}"])

    # 3. parity
    agree, n = parity(champ, config, test_seeds[:5])

    # 4. structure
    pol = Policy(champ, config)
    gc = config.genome_config
    name = {k: INPUT_NAMES[i] for i, k in enumerate(gc.input_keys)}
    name.update({k: ACTION_NAMES[i] for i, k in enumerate(gc.output_keys)})
    for h in pol.hidden:
        name[h] = f"h{h}"
    edges = sorted(((name[a], name[b], round(wt, 3)) for b, lst in pol.incoming.items()
                    for a, wt in lst), key=lambda e: (e[1], -abs(e[2])))
    biases = {name[n]: round(pol.bias[n], 3) for n in pol.order}
    act_counts = {}
    for s in test_seeds[:20]:
        rec = []
        rollout(pol, s, record=rec)
        for o, a, _ in rec:
            legs = f"legs={int(o[6])}{int(o[7])}"
            act_counts.setdefault(legs, [0, 0, 0, 0])[a] += 1
    summary = dict(
        run=str(run), champion_generation=best_g, champion_key=champ.key,
        train_fitness=champ.fitness, val_mean=float(np.mean(val[best_g])), val_n=args.val,
        test_mean=float(np.mean(test)), test_sd=float(np.std(test)),
        test_median=float(np.median(test)), test_share_ge_200=float(np.mean(np.array(test) >= 200)),
        test_n=args.test, solved=bool(np.mean(test) >= 200),
        hidden_nodes=len(pol.hidden), connections=pol.n_conns,
        genome_nodes=len(champ.nodes), genome_connections_enabled=sum(
            c.enabled for c in champ.connections.values()),
        parity_neuralneat=f"{agree}/{n}", edges=edges, biases=biases,
        action_counts_by_legs={k: dict(zip(ACTION_NAMES, v)) for k, v in sorted(act_counts.items())},
    )
    (run / "champion.json").write_text(json.dumps(summary, indent=1))
    with open(run / "champion.pkl", "wb") as f:
        pickle.dump(champ, f)

    # 6. gif
    import gymnasium as gym
    import imageio
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=test_seeds[0])
    frames, total = [env.render()], 0.0
    for t in range(1000):
        obs, r, term, trunc, _ = env.step(pol.act(obs))
        total += r
        if t % 2 == 0:
            frames.append(env.render())
        if term or trunc:
            break
    env.close()
    imageio.mimsave(run / f"rollout_seed{test_seeds[0]}.gif", frames, duration=1 / 25, loop=0)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("edges",)}, indent=1))
    print("gif return", round(total, 2))


if __name__ == "__main__":
    main()
