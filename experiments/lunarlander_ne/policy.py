"""Genome -> LunarLander policy, and seeded rollouts.

``Policy(genome, config)`` is a minimal numpy feedforward evaluator of a neat-python genome with
PropNEAT's semantics: hidden node = relu(bias + sum w*x), output score = bias + sum w*x, action =
argmax over the 4 output scores (NeuralNeat puts a sigmoid on the outputs; argmax is unchanged,
checked by ``evaluate.py``'s parity test). Only enabled connections on an input->output path count
(``neat.graphs.feed_forward_layers``), so the phenotype here is what ExplaNEAT would see.

Inputs (LunarLander-v3 observation): x, y, vx, vy, angle, angular velocity, left leg, right leg.
Actions: 0 noop, 1 left engine (pushes right), 2 main engine, 3 right engine (pushes left).
"""
import numpy as np
from neat.graphs import feed_forward_layers

ENV_ID = "LunarLander-v3"
INPUT_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel", "leg_L", "leg_R"]
ACTION_NAMES = ["noop", "left_engine", "main_engine", "right_engine"]


class Policy:
    def __init__(self, genome, config):
        gc = config.genome_config
        self.inputs = list(gc.input_keys)
        self.outputs = list(gc.output_keys)
        conns = [cg.key for cg in genome.connections.values() if cg.enabled]
        layers = feed_forward_layers(self.inputs, self.outputs, conns)
        used = set(self.inputs)
        for layer in layers:
            used |= set(layer)
        self.order = [n for layer in layers for n in layer]
        incoming = {n: [] for n in self.order}
        for (a, b) in conns:
            if b in incoming and a in used:
                incoming[b].append((a, genome.connections[(a, b)].weight))
        self.incoming = incoming
        self.bias = {n: genome.nodes[n].bias for n in self.order}
        out_set = set(self.outputs)
        self.hidden = [n for n in self.order if n not in out_set]
        self.n_conns = sum(len(v) for v in incoming.values())
        # outputs with no path still give their bias as score
        for o in self.outputs:
            if o not in self.bias:
                self.bias[o] = genome.nodes[o].bias
                self.incoming[o] = []

    def scores(self, obs):
        v = {k: float(obs[i]) for i, k in enumerate(self.inputs)}
        out_set = set(self.outputs)
        for n in self.order:
            s = self.bias[n]
            for a, w in self.incoming[n]:
                s += w * v[a]
            v[n] = s if n in out_set else (s if s > 0.0 else 0.0)
        return np.array([v[o] if o in v else self.bias[o] for o in self.outputs])

    def act(self, obs):
        return int(np.argmax(self.scores(obs)))

    def size(self):
        """(hidden nodes, enabled connections) of the phenotype (input->output paths only)."""
        return len(self.hidden), self.n_conns


def rollout(policy, seed, env=None, record=None, max_steps=1000):
    """Return (total reward, steps). ``record``: list to append (obs, action, reward)."""
    import gymnasium as gym
    own = env is None
    if own:
        env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=int(seed))
    total, steps = 0.0, 0
    for _ in range(max_steps):
        a = policy.act(obs)
        nobs, r, term, trunc, _ = env.step(a)
        if record is not None:
            record.append((obs, a, r))
        total += float(r)
        steps += 1
        obs = nobs
        if term or trunc:
            break
    if own:
        env.close()
    return total, steps


def mean_return(genome, config, seeds, env=None):
    pol = Policy(genome, config)
    rets = [rollout(pol, s, env=env)[0] for s in seeds]
    return float(np.mean(rets)), rets
