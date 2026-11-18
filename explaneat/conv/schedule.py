"""Search schedules for ConvNEAT: exploration, exploitation and scarcity over time.

Mutation is exploration, backprop epochs are exploitation, and a stronger parsimony penalty with
deletion-only mutation is a period of scarcity. Instead of fixed settings for the whole run, the
population can follow a schedule (phases) or evolve its own settings (self-adaptive, PBT-style),
and training budget can be handed out by successive halving.

Everything here hooks into :class:`explaneat.conv.population.ConvNeatPopulation` through two
methods whose defaults are the plain loop (``_epochs(g)``: epochs for a genome; ``_mutate(child,
p1, p2)``: mutation of a new offspring), plus ``evaluate`` for successive halving. With no
schedule the plain class is unchanged bit-for-bit (tests/test_conv/test_schedule.py).

Generation convention: the phase of generation ``g`` sets the mutation that *creates* generation
``g`` (from generation ``g-1``) and the epochs and penalty used to *evaluate* generation ``g``.
Generation 0 (the minimal start) is not mutated.

Related ideas: SharpNEAT's phased search (complexify / simplify), Population Based Training
(exploit = copy the better member, explore = perturb its hyperparameters), self-adaptive mutation
rates in evolution strategies, successive halving / Hyperband, adaptive parsimony.

Deletion mutations (scarcity)
-----------------------------
* ``delete_conn``: removes an enabled connection gene, chosen uniformly among the 25 % with the
  smallest kernel norm (function-aware: small kernels first). It never removes the last enabled
  input of a node, so no map becomes a constant.
* ``delete_node``: removes a hidden node and every gene touching it, chosen among the 25 % of
  hidden nodes with the smallest outgoing kernel norm, and never if that would leave a node with no
  enabled input.
* after each deletion, ``prune_dead_ends`` drops hidden nodes with no enabled outgoing edge
  (iterated). That step is exact: such a node cannot reach an output.
Removing genes never creates a cycle, so validity holds (the totality test covers them).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .genome import ConvGenome, ConvNeatConfig
from .mutation import _n_times, mutate
from .population import ConvNeatPopulation, eval_genome
from .crossover import distance

STRUCT = ("p_add_node", "p_add_conn", "p_kernel_size")  # the rates --mut-scale scales (run_long.py)
OTHER_STRUCT = ("p_toggle_enable", "p_level")


# ---------------------------------------------------------------------------- deletion mutations
def kernel_norm(c) -> float:
    return float(np.sqrt(np.square(c.kernel.astype(np.float64)).sum()))


def _n_enabled_in(g: ConvGenome) -> Dict[int, int]:
    n = {k: 0 for k in g.nodes}
    for c in g.enabled_conns():
        n[c.dst] += 1
    return n


def prune_dead_ends(g: ConvGenome) -> int:
    """Remove hidden nodes with no enabled outgoing edge (repeatedly). Exact: they cannot reach
    an output. Returns the number of nodes removed."""
    removed = 0
    while True:
        has_out = {c.src for c in g.enabled_conns()}
        dead = [k for k in g.hidden_keys() if k not in has_out]
        if not dead:
            return removed
        for k in dead:
            _remove_node(g, k)
            removed += 1


def _remove_node(g: ConvGenome, k: int) -> None:
    for inn in [i for i, c in g.conns.items() if c.src == k or c.dst == k]:
        del g.conns[inn]
    del g.nodes[k]


def mutate_delete_conn(g: ConvGenome, cfg: ConvNeatConfig, rng: np.random.Generator,
                       frac: float = 0.25) -> Optional[str]:
    n_in = _n_enabled_in(g)
    cands = [c for c in g.enabled_conns() if n_in[c.dst] > 1]
    if not cands:
        return None
    cands.sort(key=lambda c: (kernel_norm(c), c.innov))
    m = max(1, int(np.ceil(frac * len(cands))))
    c = cands[int(rng.integers(m))]
    del g.conns[c.innov]
    prune_dead_ends(g)
    g.history.append("mutation:delete_conn")
    return "delete_conn"


def mutate_delete_node(g: ConvGenome, cfg: ConvNeatConfig, rng: np.random.Generator,
                       frac: float = 0.25) -> Optional[str]:
    n_in = _n_enabled_in(g)
    out = {k: [] for k in g.nodes}
    for c in g.enabled_conns():
        out[c.src].append(c)
    # removable: every enabled successor keeps at least one other enabled input
    cands = [k for k in g.hidden_keys() if all(n_in[c.dst] > 1 for c in out[k])]
    if not cands:
        return None
    cands.sort(key=lambda k: (sum(kernel_norm(c) for c in out[k]), k))
    m = max(1, int(np.ceil(frac * len(cands))))
    _remove_node(g, cands[int(rng.integers(m))])
    prune_dead_ends(g)
    g.history.append("mutation:delete_node")
    return "delete_node"


def mutate_deletions(g: ConvGenome, cfg: ConvNeatConfig, rng: np.random.Generator,
                     p_delete_conn: float, p_delete_node: float) -> list:
    ev = []
    for _ in range(_n_times(p_delete_node, rng)):
        ev.append(mutate_delete_node(g, cfg, rng))
    for _ in range(_n_times(p_delete_conn, rng)):
        ev.append(mutate_delete_conn(g, cfg, rng))
    return [e for e in ev if e]


# ---------------------------------------------------------------------------- phase schedules
@dataclass
class Phase:
    """One phase. Rates are multipliers of the population's *base* config (the config it was
    built with): ``struct_scale`` scales add-node / add-connection / kernel-size (as --mut-scale),
    ``other_struct`` keeps (True) or zeroes (False) toggle-enable and level, ``weight_scale``
    scales p_weight, ``penalty_scale`` scales lambda. ``epochs`` replaces epochs_per_gen.
    ``p_delete_*`` are expected deletion counts per offspring."""

    name: str
    generations: int
    struct_scale: float = 1.0
    other_struct: bool = True
    weight_scale: float = 1.0
    epochs: Optional[int] = None
    penalty_scale: float = 1.0
    p_delete_conn: float = 0.0
    p_delete_node: float = 0.0


class PhaseSchedule:
    """A cycle of phases repeated for the whole run."""

    def __init__(self, phases: Sequence[Phase]):
        assert phases and all(p.generations > 0 for p in phases)
        self.phases = list(phases)
        self.cycle = sum(p.generations for p in phases)

    def phase_at(self, gen: int) -> Phase:
        r = gen % self.cycle
        for p in self.phases:
            if r < p.generations:
                return p
            r -= p.generations
        raise AssertionError

    def switches(self, n_gens: int) -> List[tuple]:
        """[(generation, phase name)] where the phase changes, for the first n_gens generations."""
        out, prev = [], None
        for g in range(n_gens):
            name = self.phase_at(g).name
            if name != prev:
                out.append((g, name))
            prev = name
        return out

    def to_dict(self) -> dict:
        return {"cycle_generations": self.cycle, "phases": [asdict(p) for p in self.phases]}


#: Arm 2 of experiments/conv_neat/schedules (base config: default rates, 5 epochs, lambda 0.001).
PHASE_CYCLE = PhaseSchedule([
    Phase("explore", 5, struct_scale=4.0, epochs=2),
    Phase("exploit", 3, struct_scale=0.0, other_struct=False, weight_scale=0.5, epochs=20),
    Phase("scarcity", 2, struct_scale=0.0, other_struct=False, weight_scale=0.0, epochs=5,
          penalty_scale=5.0, p_delete_conn=1.0, p_delete_node=0.5),
])

#: Arm 1 as a one-phase schedule (equals the plain population with --mut-scale 4; tested).
FIXED_4X = PhaseSchedule([Phase("fixed", 1, struct_scale=4.0)])


def apply_phase(cfg: ConvNeatConfig, base: ConvNeatConfig, ph: Phase) -> None:
    for f in STRUCT:
        setattr(cfg, f, getattr(base, f) * ph.struct_scale)
    for f in OTHER_STRUCT:
        setattr(cfg, f, getattr(base, f) if ph.other_struct else 0.0)
    cfg.p_weight = base.p_weight * ph.weight_scale
    cfg.epochs_per_gen = base.epochs_per_gen if ph.epochs is None else ph.epochs
    cfg.complexity_penalty = base.complexity_penalty * ph.penalty_scale


class ScheduledPopulation(ConvNeatPopulation):
    """ConvNEAT following a :class:`PhaseSchedule`. The population's config object is rewritten
    in place each generation from the base config (the speciation threshold keeps adapting)."""

    def __init__(self, cfg: ConvNeatConfig, data, seed: int, device, schedule: PhaseSchedule, **kw):
        assert kw.get("workers", 0) == 0, "schedules run in-process"
        self.schedule = schedule
        self.base_cfg = copy.deepcopy(cfg)
        super().__init__(cfg, data, seed, device, **kw)

    def _set(self, gen: int) -> Phase:
        ph = self.schedule.phase_at(gen)
        apply_phase(self.cfg, self.base_cfg, ph)
        return ph

    def evaluate(self, gen: int) -> None:
        self._set(gen)
        super().evaluate(gen)

    def reproduce(self) -> None:
        self._set(self.generation)  # the phase of the generation being created
        super().reproduce()

    def _mutate(self, child, p1, p2) -> list:
        ev = mutate(child, self.cfg, self.tracker, self.rng)
        ph = self.schedule.phase_at(self.generation)
        if ph.p_delete_conn or ph.p_delete_node:
            ev += mutate_deletions(child, self.cfg, self.rng, ph.p_delete_conn, ph.p_delete_node)
        return ev

    def schedule_log(self, gen: int) -> dict:
        ph = self.schedule.phase_at(gen)
        return {"phase": ph.name, "epochs": self.cfg.epochs_per_gen, "penalty": self.cfg.complexity_penalty,
                "p_add_node": self.cfg.p_add_node, "p_weight": self.cfg.p_weight}


# ---------------------------------------------------------------------------- self-adaptive (PBT-style)
class SelfAdaptivePopulation(ConvNeatPopulation):
    """Each genome carries ``strategy = {"mut_scale", "epochs"}``. The structural rates of its
    offspring are the base rates x the offspring's mut_scale, and it is trained for round(epochs)
    epochs. An offspring inherits the fitter parent's strategy; each value is then multiplied by
    0.8 or 1.25 (50/50) with probability ``p_perturb`` (ES-style: strategy first, then the
    mutation it controls). Elites keep their strategy. Initial values: ``init x exp(N(0, spread))``.
    """

    def __init__(self, cfg: ConvNeatConfig, data, seed: int, device, init_mut: float = 4.0,
                 init_epochs: float = 5.0, spread: float = 0.25, p_perturb: float = 0.2,
                 mut_range=(0.25, 16.0), epochs_range=(1.0, 30.0), **kw):
        assert kw.get("workers", 0) == 0, "self-adaptive epochs run in-process"
        self.base_cfg = copy.deepcopy(cfg)
        self.p_perturb, self.mut_range, self.epochs_range = p_perturb, mut_range, epochs_range
        super().__init__(cfg, data, seed, device, **kw)
        srng = np.random.default_rng([seed, 11])
        for g in self.population:
            g.strategy = {"mut_scale": float(np.clip(init_mut * np.exp(srng.normal(0, spread)), *mut_range)),
                          "epochs": float(np.clip(init_epochs * np.exp(srng.normal(0, spread)), *epochs_range))}

    def _epochs(self, g) -> int:
        return int(round(g.strategy["epochs"]))

    def _mutate(self, child, p1, p2) -> list:
        par = p1
        f = lambda g: -np.inf if g.fitness is None else g.fitness  # noqa: E731
        if p2 is not None and f(p2) > f(p1):
            par = p2
        s = dict(par.strategy)
        for key, rng_ in (("mut_scale", self.mut_range), ("epochs", self.epochs_range)):
            if self.rng.random() < self.p_perturb:
                s[key] = float(np.clip(s[key] * (0.8 if self.rng.random() < 0.5 else 1.25), *rng_))
        child.strategy = s
        c = copy.copy(self.cfg)
        for f in STRUCT:
            setattr(c, f, getattr(self.base_cfg, f) * s["mut_scale"])
        return mutate(child, c, self.tracker, self.rng)

    def schedule_log(self, gen: int) -> dict:
        m = np.array([g.strategy["mut_scale"] for g in self.population])
        e = np.array([g.strategy["epochs"] for g in self.population])
        return {"phase": "self-adaptive",
                "mut_scale_mean": float(m.mean()), "mut_scale_sd": float(m.std()),
                "mut_scale_q": [float(q) for q in np.quantile(m, [0.1, 0.5, 0.9])],
                "epochs_mean": float(e.mean()), "epochs_sd": float(e.std()),
                "epochs_q": [float(q) for q in np.quantile(e, [0.1, 0.5, 0.9])],
                "epochs_total": int(sum(self._epochs(g) for g in self.population))}


# ---------------------------------------------------------------------------- successive halving
class SuccessiveHalvingPopulation(ConvNeatPopulation):
    """Training budget by successive halving, per generation: every genome gets ``base`` epochs;
    the top ``top_frac`` by fitness get ``top_extra`` more; species leaders (the best genome of
    each species, species assigned to last generation's representatives as speciation does) are
    brought to ``leader_total`` epochs. Each extra round is a fresh Adam run continuing from the
    written-back weights; fitness is re-scored after it."""

    def __init__(self, cfg, data, seed, device, base: int = 2, top_frac: float = 0.2, top_extra: int = 8,
                 leader_total: int = 20, **kw):
        assert kw.get("workers", 0) == 0 and not kw.get("train_chunk"), "in-process, all rows"
        self.base, self.top_frac, self.top_extra, self.leader_total = base, top_frac, top_extra, leader_total
        super().__init__(cfg, data, seed, device, **kw)
        self.extra_epochs = 0

    def _epochs(self, g) -> int:
        return self.base

    def _train_more(self, g, epochs: int, gen: int) -> None:
        if epochs <= 0:
            return
        g2, m = eval_genome(g, self.cfg, self.X["train"], self.y["train"], self.y_np["train"],
                            self.X["val"], self.y_np["val"], epochs, int(self.rng.integers(2 ** 62)),
                            self.device, self.fitness_split)
        self.n_trainings += 1
        self.extra_epochs += epochs
        m.update({"complexity": g.complexity(), "generation": gen})
        g.metrics = m
        score = m["val_acc"] if self.fitness_split == "val" else m["train_acc"]
        g.fitness = score - self.cfg.complexity_penalty * g.complexity()
        if g.fitness > self.best.fitness:
            self.best = copy.deepcopy(g)

    def evaluate(self, gen: int) -> None:
        self.extra_epochs = 0
        super().evaluate(gen)
        ranked = sorted(self.population, key=lambda g: (-g.fitness, g.key))
        top = ranked[:max(1, int(np.ceil(self.top_frac * len(ranked))))]
        for g in top:
            self._train_more(g, self.top_extra, gen)
        got = {g.key: self.base + self.top_extra for g in top}
        groups: Dict[int, List[ConvGenome]] = {}
        for g in self.population:
            sid = -1
            for s in self.species:
                if distance(g, s.rep, self.cfg) < self.cfg.compat_threshold:
                    sid = s.sid
                    break
            groups.setdefault(sid, []).append(g)
        leaders = [min(ms, key=lambda g: (-g.fitness, g.key)) for ms in groups.values()]
        for g in leaders:
            self._train_more(g, self.leader_total - got.get(g.key, self.base), gen)

    def schedule_log(self, gen: int) -> dict:
        return {"phase": "halving", "extra_epochs": self.extra_epochs,
                "epochs_total": self.base * len(self.population) + self.extra_epochs}
