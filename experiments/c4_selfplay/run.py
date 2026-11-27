"""Connect-4 self-play round loop (value-only, pure zero): self-play -> replay buffer ->
a continuous PropNEAT population trained on the buffer -> its champion plays the next round.

Round 0's player is ``ConstValue(0.5)`` + MCTS. The population is created once (from round
0's buffer) and continues across rounds: each round replaces ``pop.xs``/``pop.ys`` with the
current buffer, resets ``pop.best_genome`` and runs ``--gens`` generations of backprop
(``--epochs`` full-batch Adadelta epochs each, BCE on the soft labels) + ``neg_bce_fitness``
+ reproduction. The round's champion is the best-fitness genome of that round (fitness is not
comparable across rounds: the buffer changes). Generation numbers run on across rounds.

Stagnation is judged within a round: at the start of each round (before ``pop.run``, and so
after a resume) every species' ``fitness_history`` is emptied and its ``last_improved`` set to
the current generation, so ``max_stagnation`` never compares fitness on different buffers. A
round of fewer than ``max_stagnation`` generations therefore never removes a species as
stagnant.

    PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/run.py --name s0 --seed 0 \
        --rounds 20 --games 200 --sims 100 --gens 5 --epochs 20 --buffer 4

Outputs (``<run_dir>`` = ``$EXPLANEAT_HOME/c4_selfplay_runs/<name>``, default ``~/.explaneat``):
  <run_dir>/genomes_r<round:03d>.pkl.gz  {generation: {genome_key: genome}} for every
                                         generation of the round (trained, evaluated genomes)
  <run_dir>/champions.pkl.gz             {round: {"genome": champion genome, "generation": the
                                         generation it was evaluated in, "key": genome key}}
  <run_dir>/parents.csv.gz               pop.ancestry_rows() (whole run so far)
  <run_dir>/ckpt.pkl.gz                  checkpoint at the end of the last finished round
  <results>/<name>/rounds.csv            one row per round
  <results>/<name>/meta.json             args, commit, config text, versions, start time
  <results>/<name>/config.ini            the NEAT config used (copy of configs/selfplay.ini)

A run resumes from its checkpoint unless ``--fresh``; the other arguments must match
(``--rounds`` may grow, ``--workers`` may change). A resumed run gives the same results as an
uninterrupted one.

v2 options (all off by default; with all off, every output is identical to the v1 loop):
  --workers N        train the population in N worker processes (parallel_pop; bitwise identical
                     to serial, so it changes no output and does not by itself turn on v2 columns)
  --set S.key=value  NEAT config overrides on configs/selfplay.ini (unknown keys refused); the
                     derived config is results/<name>/config.ini. With num_hidden > 0 the initial
                     hidden units are renumbered to shared ids 1..H (homologous; RNG-neutral)
  --species-band LO,HI,STEP,FLOOR
                     after each generation's evaluation, move compatibility_threshold one STEP
                     toward LO <= species <= HI (never below FLOOR); the new value is used by the
                     speciation at the end of the same generation (neat speciates after
                     reproduction, after post_evaluate). Saved in / restored from the checkpoint
  --value-mix M      label = M * outcome + (1 - M) * MCTS root value (selfplay.play_games)
  --holdout-frac F   hold out F of the newest round's unique positions (rng from seed and round,
                     not the self-play rng) from that round's training; the champion is the
                     genome of the round's generations with the best -BCE on them (ties: latest
                     generation, then lowest key; invalid/non-finite genomes excluded). Held-out
                     positions still enter the buffer for later rounds
When any of --set / --species-band / --value-mix / --holdout-frac is on, rounds.csv gets the
columns V2_FIELDS appended and the result-changing options join the resume check.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import netfn  # noqa: E402  (first: forces EXPLANEAT_DEVICE=cpu)

import argparse  # noqa: E402
import configparser  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import count  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from neat.reporting import BaseReporter  # noqa: E402

import selfplay as SP  # noqa: E402
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.errors import GenomeNotValidError  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from fitness import bce, neg_bce_fitness  # noqa: E402
from parallel_pop import ParallelBackpropPopulation  # noqa: E402

TORCH_THREADS = 1  # the nets are tiny
DEFAULT_RESULTS = HERE / "results"
FIELDS = ["round", "gen_start", "gen_end", "buffer_raw", "buffer_unique", "champion_key",
          "champion_generation", "champion_fitness", "champion_nodes", "champion_conns", "selfplay_s", "train_s",
          "games", "first_win_share", "draw_share", "mean_len"]
WALL_FIELDS = ("selfplay_s", "train_s")
V2_FIELDS = ["n_species", "threshold", "champion_holdout_fitness", "value_mix"]
# arguments that must match between a checkpoint and the resuming call
RUN_ARGS = ("seed", "games", "sims", "gens", "epochs", "buffer", "temp_moves", "dirichlet")
# v2 options that change results, with their off value: they join the check only when on, so a
# v1 checkpoint (and a v1 resume) is unchanged. --workers is not here: it changes no result.
V2_ARGS = {"set": [], "species_band": None, "value_mix": 1.0, "holdout_frac": 0.0}
HOLDOUT_TAG = 0x484F4C44  # "HOLD": the holdout rng is default_rng([HOLDOUT_TAG, seed, round])


def runs_root() -> Path:
    home = os.environ.get("EXPLANEAT_HOME")
    return (Path(home) if home else Path.home() / ".explaneat") / "c4_selfplay_runs"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)


def git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                     cwd=REPO, text=True).splitlines()
    return commit, any(line.strip() and not line.rstrip().endswith(".pyc") for line in status)


def dump(obj, path: Path) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path.stat().st_size


def load(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def parse_dirichlet(s):
    if s is None:
        return None
    a, e = s.split(",")
    return float(a), float(e)


def genome_size(g):
    return len(g.nodes), sum(1 for c in g.connections.values() if c.enabled)


class GenerationRecorder(BaseReporter):
    """Keeps a deep copy of every generation's trained, evaluated genomes for the round (and
    its species count), and logs the generation's best fitness."""

    def __init__(self, pop):
        self.pop = pop
        self.round = None
        self.generations = {}
        self.n_species = {}

    def post_evaluate(self, config, population, species, best):
        gen = self.pop.generation
        self.generations[gen] = {k: copy.deepcopy(g) for k, g in population.items()}
        self.n_species[gen] = len(species.species)
        fits = [g.fitness for g in population.values()]
        logging.info("round %d gen %d  best %.5f (key %d)  mean %.5f  species %d", self.round, gen,
                     best.fitness, best.key, float(np.mean(fits)), len(species.species))


class ThresholdController:
    """Dynamic compatibility threshold: one step toward the target species band per generation
    (copy of experiments/complex_real/run_seed.py's)."""

    def __init__(self, config, lo, hi, step, floor):
        self.config, self.lo, self.hi, self.step, self.floor = config, lo, hi, step, floor

    @property
    def threshold(self):
        return self.config.species_set_config.compatibility_threshold

    def update(self, n_species):
        t = self.threshold
        if n_species < self.lo:
            t = max(self.floor, t - self.step)
        elif n_species > self.hi:
            t = t + self.step
        self.config.species_set_config.compatibility_threshold = t
        return t


class SpeciesBand(BaseReporter):
    """Runs the controller on each evaluated generation's species count. ``post_evaluate``
    comes before reproduction and speciation in ``BackpropPopulation.run``, so the new
    threshold is the one the same generation's ``speciate`` (forming the next generation's
    species) reads. The species set must read ``config.species_set_config`` (rebound after a
    resume: the unpickled species set holds its own copy)."""

    def __init__(self, pop, controller):
        self.pop, self.controller = pop, controller

    def post_evaluate(self, config, population, species, best):
        used = self.controller.threshold
        n = len(species.species)
        nxt = self.controller.update(n)
        logging.info("gen %d  species %d  threshold %.4f -> %.4f", self.pop.generation, n, used, nxt)


def parse_band(s):
    if s is None:
        return None
    lo, hi, step, floor = s.split(",")
    return int(lo), int(hi), float(step), float(floor)


def build_config(overrides, path):
    """configs/selfplay.ini with ``Section.key=value`` overrides (as c4_capacity's
    build_config; unknown keys refused), written to ``path``."""
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(netfn.DEFAULT_CONFIG)
    for o in overrides:
        key, val = o.split("=", 1)
        sec, k = key.split(".", 1)
        if not cp.has_option(sec, k):
            sys.exit(f"unknown option {key}")
        cp.set(sec, k, val)
    with open(path, "w") as f:
        cp.write(f)


def align_initial_hidden(pop, config):
    """Copy of experiments/c4_capacity/run_seed.py's. neat-python gives every new genome's
    initial hidden nodes fresh global ids, so with ``num_hidden > 0`` no two initial genomes
    share a hidden gene: every connection through a hidden node is disjoint, crossover is
    non-homologous and each genome starts as its own species. Renumber each initial genome's
    hidden nodes (in creation order) to the shared ids 1..H, so the initial population is H
    homologous hidden units with different random weights, then re-speciate. RNG-neutral."""
    H = config.genome_config.num_hidden
    outs = set(config.genome_config.output_keys)
    for g in pop.population.values():
        hid = sorted(k for k in g.nodes if k not in outs)
        m = {old: i + 1 for i, old in enumerate(hid)}
        rn = lambda k: m.get(k, k)  # noqa: E731
        nodes = {}
        for k, n in g.nodes.items():
            n.key = rn(k)
            nodes[n.key] = n
        g.nodes = nodes
        conns = {}
        for (a, b), c in g.connections.items():
            c.key = (rn(a), rn(b))
            conns[c.key] = c
        g.connections = conns
    config.genome_config.node_indexer = count(max(outs | {H}) + 1)
    pop.species = config.species_set_type(config.species_set_config, pop.reporters)
    pop.species.speciate(config, pop.population, pop.generation)


def make_population(config, X, y, workers, initial_state=None):
    kw = dict(mask_phantom=True, skip_invalid_genomes=True)
    if initial_state is not None:
        kw["initial_state"] = initial_state
    if workers > 1:
        return ParallelBackpropPopulation(config, X, y, workers=workers, **kw)
    return BackpropPopulation(config, X, y, **kw)


def holdout_split(X, y, frac, seed, r):
    """(held-out positions deduplicated with their mean label in the newest round, set of their
    int8 bytes). The rng depends only on (seed, round)."""
    hrng = np.random.default_rng([HOLDOUT_TAG, seed, r])
    _, _, Xh, yh = SP.split_holdout(X, y, frac, hrng)
    hb = SP.ReplayBuffer(1)
    hb.add(r, Xh, yh)
    Xh, yh = hb.data()
    keys = {row.tobytes() for row in np.ascontiguousarray(Xh, dtype=np.int8)}
    return Xh, yh, keys


def drop_positions(X, y, keys):
    if not keys:
        return X, y
    Xi = np.ascontiguousarray(X, dtype=np.int8)
    keep = np.fromiter((row.tobytes() not in keys for row in Xi), bool, count=len(Xi))
    return X[keep], y[keep]


def holdout_champion(generations, config, Xh, yh):
    """The genome of the round with the best -BCE on the holdout (ties: latest generation, then
    lowest key), invalid or non-finite genomes excluded. Returns (genome, generation, fitness)."""
    xs = torch.tensor(Xh, dtype=torch.float64)
    ys = torch.tensor(yh, dtype=torch.float64)
    best = None
    for gen in sorted(generations):
        for k in sorted(generations[gen]):
            g = generations[gen][k]
            try:
                net = NeuralNeat(g, config, mask_phantom=True)
            except GenomeNotValidError:
                continue
            net.eval()
            with torch.no_grad():
                pred = net.forward(xs)
            if not bool(torch.isfinite(pred).all()):
                continue
            f = -bce(pred, ys)
            if not math.isfinite(f):
                continue
            rank = (f, gen, -k)
            if best is None or rank > best[0]:
                best = (rank, g, gen, f)
    if best is None:
        raise RuntimeError("no valid genome in this round")
    return best[1], best[2], best[3]


def set_training_data(pop, X, y):
    pop.xs = torch.tensor(X, dtype=torch.float64).to(pop.device)
    pop.ys = torch.tensor(y, dtype=torch.float64).reshape(-1, 1).to(pop.device)


def pick_champion(pop, generations, config):
    """The round's best genome and its generation; if NeuralNeat rejects it, the best valid
    genome of the round. ``pop.best_genome`` is already a snapshot (BackpropPopulation deep
    copies it) and the recorder's genomes are deep copies, so neither is copied again."""
    champ = pop.best_genome
    try:
        netfn.ValueFn(champ, config)
        return champ, pop.best_generation, False
    except GenomeNotValidError:
        pass
    cands = sorted(((gen, g) for gen, pg in sorted(generations.items()) for g in pg.values()),
                   key=lambda t: t[1].fitness, reverse=True)
    for gen, g in cands:
        try:
            netfn.ValueFn(g, config)
        except GenomeNotValidError:
            continue
        logging.warning("champion %d is invalid; using best valid genome %d of generation %d "
                        "(fitness %.5f)", champ.key, g.key, gen, g.fitness)
        return g, gen, True
    raise RuntimeError("no valid genome in this round")


def reset_stagnation(pop):
    """Start every species' stagnation clock afresh: fitness on a new buffer is not comparable
    with fitness on the previous one."""
    for s in pop.species.species.values():
        s.fitness_history = []
        s.last_improved = pop.generation


def checkpoint_state(pop, config):
    """Population state, following experiments/c4_capacity/run_seed.py's resume code, plus the
    genome config's global node indexer (neat-python re-derives it from the first genome it
    mutates when it is None, which would differ after a resume)."""
    rep = pop.reproduction
    nxt = next(rep.genome_indexer)
    rep.genome_indexer = count(nxt)
    sp = pop.species
    sp_next = next(sp.indexer)
    sp.indexer = count(sp_next)
    gc = config.genome_config
    node_next = None
    if gc.node_indexer is not None:
        node_next = next(gc.node_indexer)
        gc.node_indexer = count(node_next)
    reporters, indexer = sp.reporters, sp.indexer
    sp.reporters = sp.indexer = None  # reporters would drag the run along; counts don't pickle in 3.14
    try:
        state = copy.deepcopy(dict(population=pop.population, species=sp, generation=pop.generation,
                                   genome_indexer_next=nxt, species_indexer_next=sp_next,
                                   node_indexer_next=node_next, ancestors=rep.ancestors,
                                   parent_log=pop.parent_log))
    finally:
        sp.reporters, sp.indexer = reporters, indexer
    return state


def restore_population(st, config, X, y, workers=1):
    if st["node_indexer_next"] is not None:
        config.genome_config.node_indexer = count(st["node_indexer_next"])
    pop = make_population(config, X, y, workers,
                          initial_state=(st["population"], st["species"], st["generation"]))
    pop.reproduction.genome_indexer = count(st["genome_indexer_next"])
    pop.reproduction.ancestors = st["ancestors"]
    pop.species.reporters = pop.reporters
    pop.species.indexer = count(st["species_indexer_next"])
    pop.parent_log = st["parent_log"]
    return pop


def write_parents(pop, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])
        for r in pop.ancestry_rows():
            w.writerow([r[0], r[1], "" if r[2] is None else r[2], "" if r[3] is None else r[3], int(bool(r[4]))])
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--sims", type=int, required=True)
    ap.add_argument("--gens", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--buffer", type=int, required=True, help="replay buffer size in rounds")
    ap.add_argument("--temp-moves", type=int, default=8)
    ap.add_argument("--dirichlet", default=None, help="ALPHA,EPS root noise (default none)")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--fresh", action="store_true")
    # v2 options (default off)
    ap.add_argument("--workers", type=int, default=1, help="backprop worker processes (identical results)")
    ap.add_argument("--set", action="append", default=[], metavar="Section.key=value",
                    help="NEAT config override on configs/selfplay.ini (repeatable)")
    ap.add_argument("--species-band", default=None, metavar="LO,HI,STEP,FLOOR",
                    help="species-band compatibility-threshold controller")
    ap.add_argument("--value-mix", type=float, default=1.0, help="label = M*outcome + (1-M)*search value")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="hold out this share of the newest round's unique positions; champion by holdout")
    args = ap.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")
    dirichlet = parse_dirichlet(args.dirichlet)
    band = parse_band(args.species_band)
    if not 0.0 <= args.holdout_frac < 1.0:
        sys.exit("--holdout-frac must be in [0, 1)")
    v2_on = {k: getattr(args, k) for k, off in V2_ARGS.items() if getattr(args, k) != off}
    fields = FIELDS + V2_FIELDS if v2_on else FIELDS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("experimenter.backproppop", "experimenter.evaluators", "experimenter.methodTimer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    name = args.name
    run_dir = runs_root() / name
    run_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(args.results_dir) / name
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path, cfg_path = res_dir / "rounds.csv", res_dir / "meta.json", res_dir / "config.ini"
    ckpt_path = run_dir / "ckpt.pkl.gz"

    seed_everything(args.seed)
    run_args = {k: getattr(args, k) for k in RUN_ARGS}
    run_args.update(v2_on)
    st = None
    if ckpt_path.exists() and not args.fresh:
        st = load(ckpt_path)
        if st["args"] != run_args:
            sys.exit(f"checkpoint arguments {st['args']} differ from {run_args}; use --fresh")
    if st is None:
        for p in list(run_dir.glob("genomes_r*.pkl.gz")) + [ckpt_path, run_dir / "champions.pkl.gz",
                                                             run_dir / "parents.csv.gz", csv_path]:
            if p.exists():
                p.unlink()
        if args.set:
            build_config(args.set, cfg_path)
        else:
            shutil.copyfile(netfn.DEFAULT_CONFIG, cfg_path)
    config = netfn.load_config(cfg_path)
    controller = ThresholdController(config, *band) if band else None
    if hasattr(config, "generations_of_backprop"):
        sys.exit("config sets generations_of_backprop, which would override --epochs")

    commit, dirty = git_state()
    if st is None:
        rng = np.random.default_rng(args.seed)
        buffer = SP.ReplayBuffer(args.buffer)
        champions, start_round, wall_before, pop = {}, 0, 0.0, None
        meta = dict(study="c4_selfplay", name=name, args=vars(args), code_commit=commit, code_dirty=dirty,
                    config_text=cfg_path.read_text(), python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__, machine=platform.machine(),
                    torch_threads=TORCH_THREADS, start_time=datetime.now(timezone.utc).isoformat(),
                    resumes=[])
    else:
        rng = np.random.default_rng()
        rng.bit_generator.state = st["selfplay_rng"]
        buffer, champions = st["buffer"], st["champions"]
        start_round, wall_before = st["round"] + 1, st["wall_s"]
        Xb, yb = buffer.data()
        if controller:
            config.species_set_config.compatibility_threshold = st["threshold"]
        pop = restore_population(st["pop"], config, Xb, yb, workers=args.workers)
        if controller:  # the unpickled species set holds its own copy of the species config
            pop.species.species_set_config = config.species_set_config
        random.setstate(st["rng"]["random"])
        np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"])
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["round"]) < start_round]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.setdefault("resumes", []).append(dict(round=start_round, args=vars(args), code_commit=commit,
                                                   code_dirty=dirty,
                                                   time=datetime.now(timezone.utc).isoformat()))
        logging.info("resumed %s at round %d (generation %d)", name, start_round, pop.generation)
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    logging.info("%s: rounds %d..%d, games %d, sims %d, gens %d, epochs %d, buffer %d, temp_moves %d, "
                 "dirichlet %s, commit %s%s", name, start_round, args.rounds - 1, args.games, args.sims,
                 args.gens, args.epochs, args.buffer, args.temp_moves, dirichlet, commit[:9],
                 " (dirty)" if dirty else "")
    if v2_on or args.workers > 1:
        logging.info("v2 options: %s, workers %d", v2_on, args.workers)

    def attach(pop):
        rec = GenerationRecorder(pop)
        pop.add_reporter(rec)
        if controller:
            pop.add_reporter(SpeciesBand(pop, controller))
        return rec

    recorder = attach(pop) if pop is not None else None
    t0 = time.time()
    try:
        for r in range(start_round, args.rounds):
            player = (netfn.ConstValue(0.5) if r == 0
                      else netfn.ValueFn(champions[r - 1]["genome"], config))
            ts = time.time()
            X, y, games = SP.play_games(player, args.games, args.sims, rng, temp_moves=args.temp_moves,
                                        dirichlet=dirichlet, value_mix=args.value_mix)
            selfplay_s = time.time() - ts
            hold = None
            if args.holdout_frac > 0:
                Xh, yh, hkeys = holdout_split(X, y, args.holdout_frac, args.seed, r)
                hold = (Xh, yh) if len(Xh) else None
            buffer.add(r, X, y)  # the held-out positions are in the buffer for later rounds
            Xb, yb = buffer.data()
            if hold is not None:
                Xb, yb = drop_positions(Xb, yb, hkeys)
            if pop is None:
                pop = make_population(config, Xb, yb, args.workers)
                if config.genome_config.num_hidden > 0:
                    align_initial_hidden(pop, config)
                recorder = attach(pop)
            else:
                set_training_data(pop, Xb, yb)
            recorder.round, recorder.generations, recorder.n_species = r, {}, {}
            pop.best_genome = None
            pop.best_generation = None
            reset_stagnation(pop)
            gen_start = pop.generation
            tt = time.time()
            pop.run(neg_bce_fitness, n=args.gens, nEpochs=args.epochs)
            train_s = time.time() - tt
            hold_fit = None
            if hold is not None:
                champ, champ_gen, hold_fit = holdout_champion(recorder.generations, config, *hold)
            else:
                champ, champ_gen, _ = pick_champion(pop, recorder.generations, config)
            champions[r] = {"genome": champ, "generation": champ_gen, "key": champ.key}

            gsize = dump(recorder.generations, run_dir / f"genomes_r{r:03d}.pkl.gz")
            dump(champions, run_dir / "champions.pkl.gz")
            write_parents(pop, run_dir / "parents.csv.gz")

            res = np.array([g["result_first"] for g in games])
            bst = buffer.stats()
            nodes, conns = genome_size(champ)
            row = dict(round=r, gen_start=gen_start, gen_end=pop.generation - 1, buffer_raw=bst["raw"],
                       buffer_unique=bst["unique"], champion_key=champ.key,
                       champion_generation=champ_gen, champion_fitness=repr(champ.fitness),
                       champion_nodes=nodes, champion_conns=conns, selfplay_s=round(selfplay_s, 2),
                       train_s=round(train_s, 2), games=len(games),
                       first_win_share=float(np.mean(res == 1.0)), draw_share=float(np.mean(res == 0.5)),
                       mean_len=float(np.mean([g["n_moves"] for g in games])))
            if v2_on:
                row.update(n_species=recorder.n_species[pop.generation - 1],
                           threshold=repr(config.species_set_config.compatibility_threshold),
                           champion_holdout_fitness="" if hold_fit is None else repr(hold_fit),
                           value_mix=args.value_mix)
            new = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new:
                    w.writeheader()
                w.writerow(row)

            state = dict(args=run_args, round=r, pop=checkpoint_state(pop, config), buffer=buffer,
                         champions=champions, selfplay_rng=rng.bit_generator.state,
                         rng=dict(random=random.getstate(), numpy=np.random.get_state(),
                                  torch=torch.get_rng_state()),
                         wall_s=time.time() - t0 + wall_before)
            if controller:
                state["threshold"] = config.species_set_config.compatibility_threshold
            csize = dump(state, ckpt_path)
            logging.info("round %d  gens %d-%d  buffer %d raw / %d unique  champion %d fit %.5f nodes %d conns %d  "
                         "selfplay %.1fs train %.1fs  games %d first-win %.2f draw %.2f len %.1f  "
                         "genomes file %.2f MB, ckpt %.2f MB", r, gen_start, pop.generation - 1, bst["raw"],
                         bst["unique"], champ.key, champ.fitness, nodes, conns, selfplay_s, train_s, len(games),
                         row["first_win_share"], row["draw_share"], row["mean_len"], gsize / 1e6, csize / 1e6)
            if v2_on:
                logging.info("round %d  species %d  threshold %.4f  holdout %s", r, row["n_species"],
                             config.species_set_config.compatibility_threshold,
                             "off" if hold_fit is None else "%d positions, champion %.5f" % (len(hold[0]), hold_fit))
    finally:
        if isinstance(pop, ParallelBackpropPopulation):
            pop.close()
    logging.info("%s done: %d rounds, wall %.1fs", name, args.rounds, time.time() - t0 + wall_before)


if __name__ == "__main__":
    main()
