"""Lineage yardstick scoring and jump detection for one self-play run.

    PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/jumps.py --name main-s0 \
        [--runs-dir R] [--results-dir D] [--sims 100] [--workers 6] \
        [--parts uci,moves,ladder] [--every 1] [--seed 0] \
        [--positions v1|10k] [--openings 2ply|3ply] [--ladder-every 1]

Inputs: ``<runs-dir>/<name>/`` (``parents.csv.gz``, ``genomes_rNNN.pkl.gz``,
``champions.pkl.gz``; default ``$EXPLANEAT_HOME/c4_selfplay_runs``) and
``<results-dir>/<name>/`` (``rounds.csv``, ``config.ini``). The yardstick files are read from
``<results-dir>/yardstick/`` as in ``yardstick.py``.

1. **Chain.** Target = the final round's champion (its key and generation). Its ancestor chain
   runs back to generation 0, one lineage row ``(generation, key)`` per generation; the genome
   of a row is ``genomes[generation][key]`` (elites keep their key across generations while
   backprop keeps changing their weights, so the key alone does not name a genome). Same
   semantics as ``explaneat.analysis.lineage.get_lineage(mode="fitter_parent")``, which needs
   database genome ids and so cannot be used here: an elite copy steps to the same key one
   generation earlier; offspring step to the parent with the higher training fitness in the
   previous generation (ties and missing fitness go to parent1; NEAT's crossover takes the
   fitter parent's disjoint/excess genes, so this is the structural line). The walk stops at
   generation 0 or at a row with no recorded parents.
2. **Scores.** Every chain row whose generation is a multiple of ``--every`` (plus the
   champion's) is scored with ``yardstick.score_genome(..., sims, parts, seed)`` in worker
   processes (one torch thread each). Scoring is deterministic per genome (fixed seed, no
   shared state), so the csv is identical whatever ``--workers``.
   ``<results-dir>/<name>/lineage_yardstick.csv``: one row per scored genome, sorted by
   generation; columns ``generation, key, round, fitness, n_nodes, n_conns, is_elite_copy,
   is_round_champion, valid`` then the metrics in sorted order. Rewritten after each genome;
   a rerun skips rows already present (the settings are recorded in
   ``lineage_yardstick.meta.json`` and must match, including positions / openings /
   ladder-every; ``--fresh`` starts over). A genome NeuralNeat rejects gets ``valid=0`` and
   empty metrics.
   Yardstick v2 options: ``--positions`` / ``--openings`` as in ``yardstick.py champions``
   (defaults v1 / 2ply). Non-default settings write every output with the suffix
   ``_<positions>_<openings>`` (``lineage_yardstick_10k_3ply.csv``, ``jumps_10k_3ply.json``,
   ...), so v1 and v2 results sit side by side. ``--ladder-every k`` scores the ladder part
   only on scored rows whose generation is a multiple of k, on round champions and on the
   final champion; the other rows get NaN for every ``ladder_*`` metric and
   ``ladder_scored=0`` (the ladder dominates the cost of a genome).
3. **Jumps** (``jumps.json``, ``jumps.md``): for each metric in ``JUMP_METRICS`` present,
   ``lineage.detect_jumps(rows, metric, method="top_k", k=5)`` over the scored rows that have
   a finite value of the metric (a step is between consecutive such rows, so with
   ``--ladder-every`` a ladder jump spans the rows between two ladder-scored rows). Each jump: from/to generation, key and round, values,
   delta, ``boundary`` (the step crosses a round boundary: the next round's buffer) or
   ``within`` (evolution + backprop on the same buffer), whether the key changed (a
   reproduction event vs an elite carried over), and the gene diff (``diff_genome_data``):
   added/removed nodes, added/removed connections, connections enabled/disabled, and the
   largest weight and bias changes.
4. **Summary** (``lineage_summary.json``): chain length, generations covered, distinct keys,
   the generations where the key changes (reproduction) and the number of elite steps, round
   champions on the chain.

All outputs are deterministic (sorted keys, no wall-clock fields); timings go to stdout.
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
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

DEFAULT_RESULTS = HERE / "results"
ID_COLS = ["generation", "key", "round", "fitness", "n_nodes", "n_conns", "is_elite_copy",
           "is_round_champion", "valid"]
JUMP_METRICS = ("move_acc_net", "uci_auc", "ladder_ab4_mcts", "move_acc_mcts")
TOP_K = 5
N_WEIGHT_CHANGES = 5        # design choice: largest weight / bias changes listed per jump


def default_runs_dir() -> Path:
    home = Path(os.environ.get("EXPLANEAT_HOME", Path.home() / ".explaneat")).expanduser()
    return home / "c4_selfplay_runs"


def _load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# --- inputs ---------------------------------------------------------------------------

def read_parents(path):
    """``{(generation, key): (parent1_key, parent2_key, is_elite_copy)}`` from parents.csv.gz."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out = {}
    with opener(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            p1 = int(r["parent1_key"]) if r["parent1_key"] not in ("", None) else None
            p2 = int(r["parent2_key"]) if r["parent2_key"] not in ("", None) else None
            el = str(r.get("is_elite_copy", "0")).strip().lower() in ("1", "true")
            out[(int(r["generation"]), int(r["genome_key"]))] = (p1, p2, el)
    return out


def read_rounds(path):
    """rounds.csv -> list of (round, gen_start, gen_end), by round."""
    with open(path, newline="") as f:
        rows = [(int(r["round"]), int(r["gen_start"]), int(r["gen_end"]))
                for r in csv.DictReader(f)]
    return sorted(rows)


def round_of(generation, rounds):
    for r, a, b in rounds:
        if a <= generation <= b:
            return r
    raise KeyError(f"generation {generation} is in no round of rounds.csv")


def load_genomes(run_dir):
    """{generation: {key: genome}} over every genomes_rNNN.pkl.gz."""
    out = {}
    for p in sorted(Path(run_dir).glob("genomes_r*.pkl.gz")):
        for gen, pop in _load(p).items():
            out[int(gen)] = pop
    return out


def load_champions(run_dir):
    """{round: (genome, generation, key)}. New format {"genome","generation","key"}; the old
    format (round -> genome) has no generation (None)."""
    raw = _load(Path(run_dir) / "champions.pkl.gz")
    out = {}
    for r, v in raw.items():
        if isinstance(v, dict):
            out[int(r)] = (v["genome"], int(v["generation"]), int(v["key"]))
        else:
            out[int(r)] = (v, None, int(v.key))
    return out


def champion_generations(champions, rounds, genomes):
    """{round: (generation, key)}; for an old-format champion, the generation of its round
    whose genome under that key has the champion's fitness (else the latest one holding it)."""
    spans = {r: (a, b) for r, a, b in rounds}
    out = {}
    for r, (g, gen, key) in champions.items():
        if gen is None:
            a, b = spans[r]
            cands = [x for x in range(a, b + 1) if key in genomes.get(x, {})]
            if not cands:
                raise KeyError(f"round {r} champion {key} not found in its generations")
            exact = [x for x in cands if genomes[x][key].fitness == g.fitness]
            gen = (exact or cands)[-1]
        out[r] = (gen, key)
    return out


# --- chain ----------------------------------------------------------------------------

def build_chain(parents, fitness, target):
    """Ancestor chain of ``target`` = (generation, key), oldest first (see module docstring).

    ``parents``: {(gen, key): (p1, p2, is_elite_copy)}; ``fitness``: {(gen, key): float}.
    Rows: generation, key, parent1_key, parent2_key, is_elite_copy, followed ("elite",
    "parent1", "parent2" or None at the root: how the walk left this row towards the past).
    """
    chain = []
    gk = target
    while True:
        g, k = gk
        p1, p2, elite = parents.get(gk, (None, None, False))
        row = dict(generation=g, key=k, parent1_key=p1, parent2_key=p2,
                   is_elite_copy=bool(elite), followed=None)
        chain.append(row)
        if g == 0 or (p1 is None and p2 is None and not elite):
            break
        if elite:
            nxt, followed = (g - 1, k), "elite"
        elif p2 is None or p2 == p1:
            nxt, followed = (g - 1, p1), "parent1"
        else:
            f1 = fitness.get((g - 1, p1))
            f2 = fitness.get((g - 1, p2))
            f1 = -math.inf if f1 is None else f1
            f2 = -math.inf if f2 is None else f2
            nxt, followed = ((g - 1, p2), "parent2") if f2 > f1 else ((g - 1, p1), "parent1")
        if nxt not in fitness:
            raise LookupError(f"parent {nxt} of {gk} not found among the saved genomes")
        row["followed"] = followed
        gk = nxt
    chain.reverse()
    return chain


def summarise_chain(chain, rounds, champ_gk):
    key_changes = [b["generation"] for a, b in zip(chain[:-1], chain[1:]) if a["key"] != b["key"]]
    champs = {gk: r for r, gk in champ_gk.items()}
    on_chain = sorted(champs[(c["generation"], c["key"])] for c in chain
                      if (c["generation"], c["key"]) in champs)
    return dict(
        chain_length=len(chain),
        first_generation=chain[0]["generation"],
        last_generation=chain[-1]["generation"],
        reaches_generation_0=chain[0]["generation"] == 0,
        target=dict(generation=chain[-1]["generation"], key=chain[-1]["key"],
                    round=round_of(chain[-1]["generation"], rounds)),
        n_distinct_keys=len({c["key"] for c in chain}),
        key_change_generations=key_changes,
        n_key_changes=len(key_changes),
        n_elite_steps=sum(1 for c in chain[1:] if c["is_elite_copy"]),
        n_parent2_steps=sum(1 for c in chain if c["followed"] == "parent2"),
        round_champions_on_chain=on_chain,
        keys=[[c["generation"], c["key"]] for c in chain],
    )


# --- scoring --------------------------------------------------------------------------

_W = {}


def _init_worker(config_path):
    import torch

    torch.set_num_threads(1)
    _W["config"] = netfn.load_config(Path(config_path))


def _score_task(task):
    """(generation, key, genome, sims, parts, seed, results_dir, positions, openings) ->
    (gen, key, metrics|None, seconds)."""
    import yardstick as Y
    from explaneat.core.errors import GenomeNotValidError

    gen, key, genome, sims, parts, seed, results_dir, positions, openings = task
    t0 = time.time()
    try:
        m = Y.score_genome(genome, _W["config"], sims=sims, parts=parts, seed=seed,
                           results_dir=results_dir, positions=positions, openings=openings)
    except GenomeNotValidError:
        m = None
    return gen, key, m, time.time() - t0


def genome_size(g):
    return len(g.nodes), sum(1 for c in g.connections.values() if c.enabled)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return repr(v)
    return v


def write_rows(path, rows):
    rows = _ladder_nan([dict(r) for r in rows])
    metrics = sorted({k for r in rows for k in r} - set(ID_COLS))
    header = ID_COLS + metrics
    rows = sorted(rows, key=lambda r: (r["generation"], r["key"]))
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(r.get(k)) for k in header])
    os.replace(tmp, path)


def _num(s):
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s          # a setting column such as positions_set = "v1"


def read_rows(path):
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def scored_generations(chain, every):
    gens = [c["generation"] for c in chain]
    keep = {g for g in gens if g % every == 0}
    keep.add(gens[-1])
    return keep


def ladder_generations(chain, ladder_every, champ_set):
    """Generations whose row gets the ladder part: multiples of ``ladder_every``, round
    champions on the chain and the final champion (the last chain row)."""
    keep = {c["generation"] for c in chain if c["generation"] % ladder_every == 0
            or (c["generation"], c["key"]) in champ_set}
    keep.add(chain[-1]["generation"])
    return keep


def _ladder_nan(rows):
    """Rows scored without the ladder (``ladder_scored`` 0) get NaN for every ladder metric
    another row has."""
    keys = sorted({k for r in rows for k in r if k.startswith("ladder_") and k != "ladder_scored"})
    for r in rows:
        if r.get("ladder_scored") == 0 and r.get("valid") == 1:
            for k in keys:
                if r.get(k) is None:
                    r[k] = math.nan
    return rows


def score_lineage(chain, genomes, rounds, champ_gk, config_path, out_path, sims, parts, seed,
                  results_dir, workers, every, positions="v1", openings="2ply",
                  ladder_every=1, log=print):
    champ_set = set(champ_gk.values())
    base = {}
    for c in chain:
        g = genomes[c["generation"]][c["key"]]
        nn, nc = genome_size(g)
        base[(c["generation"], c["key"])] = dict(
            generation=c["generation"], key=c["key"], round=round_of(c["generation"], rounds),
            fitness=None if g.fitness is None else float(g.fitness), n_nodes=nn, n_conns=nc,
            is_elite_copy=int(c["is_elite_copy"]),
            is_round_champion=int((c["generation"], c["key"]) in champ_set))
    want = scored_generations(chain, every)
    with_ladder = ladder_generations(chain, ladder_every, champ_set) if "ladder" in parts \
        else set()

    def task_parts(gen):
        return tuple(p for p in parts if p != "ladder" or gen in with_ladder)

    rows = {}
    if out_path.exists():
        for r in read_rows(out_path):
            rows[(r["generation"], r["key"])] = r
    tasks = [(c["generation"], c["key"], genomes[c["generation"]][c["key"]], sims,
              task_parts(c["generation"]), seed, str(results_dir), positions, openings)
             for c in chain if c["generation"] in want and (c["generation"], c["key"]) not in rows]
    log(f"lineage {len(chain)} rows, {len(want)} to score, {len(rows)} already scored, "
        f"{len(tasks)} to do ({workers} workers)")
    times = []
    if tasks:
        t_all = time.time()
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx,
                                 initializer=_init_worker, initargs=(str(config_path),)) as ex:
            futs = [ex.submit(_score_task, t) for t in tasks]
            for fut in as_completed(futs):
                gen, key, m, dt = fut.result()
                times.append(dt)
                row = dict(base[(gen, key)])
                row["valid"] = int(m is not None)
                row.update(m or {})
                if m is not None:
                    row["ladder_scored"] = int("ladder" in parts and gen in with_ladder)
                rows[(gen, key)] = row
                write_rows(out_path, list(rows.values()))
                log(f"  gen {gen} key {key} scored in {dt:.1f} s ({len(times)}/{len(tasks)})")
        log(f"scored {len(tasks)} genomes in {time.time() - t_all:.1f} s wall, "
            f"{sum(times) / len(times):.1f} s per genome (worker time)")
    elif not out_path.exists():
        write_rows(out_path, [])
    return [rows[k] for k in sorted(rows) if k in base], times


# --- jumps ----------------------------------------------------------------------------

def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def gene_diff(ga, gb):
    """Structural and parameter diff a -> b (``lineage.diff_genome_data`` on serialised
    genomes), condensed."""
    from explaneat.analysis.lineage import diff_genome_data
    from explaneat.db.serialization import serialize_genome

    d = diff_genome_data(serialize_genome(ga), serialize_genome(gb))
    enabled = sorted(c for c, (o, n) in d["enabled_changed"].items() if n and not o)
    disabled = sorted(c for c, (o, n) in d["enabled_changed"].items() if o and not n)
    wch = sorted(((c, o, n) for c, (o, n) in d["weight_changes"].items()),
                 key=lambda t: (-abs(t[2] - t[1]), t[0]))[:N_WEIGHT_CHANGES]
    bch = sorted(((nid, o, n) for nid, (o, n) in d["bias_changes"].items()),
                 key=lambda t: (-abs(t[2] - t[1]), t[0]))[:N_WEIGHT_CHANGES]
    return dict(
        structural_change=bool(d["structural_change"]),
        added_nodes=d["added_nodes"], removed_nodes=d["removed_nodes"],
        added_connections=d["added_connections"], removed_connections=d["removed_connections"],
        enabled_connections=enabled, disabled_connections=disabled,
        n_weight_changes=d["summary"]["n_weight_changes"],
        n_bias_changes=d["summary"]["n_bias_changes"],
        max_abs_weight_change=d["summary"]["max_abs_weight_change"],
        largest_weight_changes=[dict(connection=c, old=o, new=n, delta=n - o) for c, o, n in wch],
        largest_bias_changes=[dict(node=nid, old=o, new=n, delta=n - o) for nid, o, n in bch],
    )


def find_jumps(rows, genomes, metrics=JUMP_METRICS, k=TOP_K):
    """{metric: [jump records]} over the scored, valid lineage rows (generation order)."""
    from explaneat.analysis.lineage import detect_jumps

    valid = [r for r in sorted(rows, key=lambda r: r["generation"]) if r.get("valid") == 1]
    lin = [dict(r, genome_id=f"{r['generation']}:{r['key']}") for r in valid]
    out = {}
    for metric in metrics:
        # only rows with a finite value: a ladder metric under --ladder-every is NaN between
        # ladder-scored rows, and a step must join two measured rows
        sub = [r for r in lin if isinstance(r.get(metric), (int, float))
               and math.isfinite(r[metric])]
        if not sub:
            continue
        by_gen = {r["generation"]: i for i, r in enumerate(sub)}
        jumps = []
        for j in detect_jumps(sub, metric=metric, method="top_k", k=k):
            b = sub[by_gen[j["generation"]]]
            a = sub[by_gen[j["generation"]] - 1]
            ga, gb = genomes[a["generation"]][a["key"]], genomes[b["generation"]][b["key"]]
            jumps.append(dict(
                metric=metric, delta=j["delta"],
                from_generation=a["generation"], from_key=a["key"], from_round=a["round"],
                to_generation=b["generation"], to_key=b["key"], to_round=b["round"],
                from_value=a[metric], to_value=b[metric],
                from_fitness=a["fitness"], to_fitness=b["fitness"],
                from_size=[a["n_nodes"], a["n_conns"]], to_size=[b["n_nodes"], b["n_conns"]],
                gap_generations=b["generation"] - a["generation"],
                round_boundary=a["round"] != b["round"],
                when="boundary" if a["round"] != b["round"] else "within",
                key_changed=a["key"] != b["key"],
                to_is_elite_copy=bool(b["is_elite_copy"]),
                to_is_round_champion=bool(b["is_round_champion"]),
                diff=gene_diff(ga, gb)))
        out[metric] = jumps
    return out


def _check_csv_settings(path, positions, openings):
    """Refuse a lineage csv whose rows record another positions / openings set (a csv left
    without its meta file, or edited by hand): rows of different settings never mix."""
    if not Path(path).exists():
        return
    for r in read_rows(path):
        if r.get("valid") != 1:
            continue
        got = (r.get("positions_set", "v1"), r.get("openings_set", "2ply"))
        if got != (positions, openings):
            raise SystemExit(f"{path}: row generation {r['generation']} was scored with "
                             f"positions={got[0]}, openings={got[1]}, not {positions}, "
                             f"{openings}; rerun with those settings or --fresh")


def _dump_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _struct_text(d):
    parts = []
    for lab, k in (("+n", "added_nodes"), ("-n", "removed_nodes"), ("+c", "added_connections"),
                   ("-c", "removed_connections"), ("en", "enabled_connections"),
                   ("dis", "disabled_connections")):
        if d[k]:
            parts.append(f"{lab} {','.join(str(x) for x in d[k])}")
    return "; ".join(parts) if parts else "none"


def jumps_markdown(name, jumps, summary, settings=None):
    lines = [f"# Lineage jumps: {name}", "",
             f"Generated by experiments/c4_selfplay/jumps.py. Lineage of the final champion "
             f"(generation {summary['target']['generation']}, key {summary['target']['key']}): "
             f"{summary['chain_length']} rows, {summary['n_distinct_keys']} distinct keys, "
             f"{summary['n_key_changes']} reproduction steps, {summary['n_elite_steps']} elite "
             f"steps. Top {TOP_K} rises per metric between consecutive scored rows; "
             f"'boundary' = the step enters a new round (new buffer), 'within' = same buffer."
             + ("" if settings is None else
                f" Yardstick: positions {settings.get('positions', 'v1')}, openings "
                f"{settings.get('openings', '2ply')}, sims {settings.get('sims')}, ladder every "
                f"{settings.get('ladder_every', 1)} generation(s)."), ""]
    for metric, js in jumps.items():
        lines += [f"## {metric}", "",
                  "| gen from -> to | key from -> to | round | when | value from -> to | delta "
                  "| size (nodes, conns) | structure change | largest weight change |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for j in js:
            w = j["diff"]["largest_weight_changes"]
            wtxt = f"{w[0]['connection']} {w[0]['old']:+.3f} -> {w[0]['new']:+.3f}" if w else "-"
            rnd = (f"{j['from_round']}" if j["from_round"] == j["to_round"]
                   else f"{j['from_round']} -> {j['to_round']}")
            lines.append(
                f"| {j['from_generation']} -> {j['to_generation']} | {j['from_key']} -> "
                f"{j['to_key']} | {rnd} | {j['when']} | {j['from_value']:.4f} -> "
                f"{j['to_value']:.4f} | {j['delta']:+.4f} | {tuple(j['from_size'])} -> "
                f"{tuple(j['to_size'])} | {_struct_text(j['diff'])} | {wtxt} |")
        if not js:
            lines.append("| (no rises) | | | | | | | | |")
        lines.append("")
    return "\n".join(lines)


# --- main -----------------------------------------------------------------------------

def output_names(positions="v1", openings="2ply"):
    """Output file names for a yardstick setting (v1 / 2ply: the original names)."""
    import yardstick as Y

    sfx = Y.settings_suffix(positions, openings)
    return dict(csv=f"lineage_yardstick{sfx}.csv", meta=f"lineage_yardstick{sfx}.meta.json",
                summary=f"lineage_summary{sfx}.json", json=f"jumps{sfx}.json",
                md=f"jumps{sfx}.md")


def run(name, runs_dir=None, results_dir=DEFAULT_RESULTS, sims=100, workers=6,
        parts=("uci", "moves", "ladder"), every=1, seed=0, fresh=False, positions="v1",
        openings="2ply", ladder_every=1, log=print):
    import yardstick as Y

    Y.check_settings(positions, openings)
    if int(ladder_every) < 1:
        raise ValueError("ladder_every must be >= 1")
    ladder_every = int(ladder_every)
    names = output_names(positions, openings)
    runs_dir = Path(runs_dir) if runs_dir else default_runs_dir()
    results_dir = Path(results_dir)
    run_dir, res = runs_dir / name, results_dir / name
    config_path = res / "config.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} missing (the run's NEAT config)")
    rounds = read_rounds(res / "rounds.csv")
    parents = read_parents(run_dir / "parents.csv.gz")
    genomes = load_genomes(run_dir)
    champions = load_champions(run_dir)
    champ_gk = champion_generations(champions, rounds, genomes)
    last = max(champions)
    target = champ_gk[last]
    cg = champions[last][0]
    tg = genomes[target[0]][target[1]]
    if {k: c.weight for k, c in cg.connections.items()} != \
            {k: c.weight for k, c in tg.connections.items()}:
        log(f"warning: champion of round {last} differs from genomes[{target[0]}][{target[1]}]")
    fitness = {(g, k): gn.fitness for g, pop in genomes.items() for k, gn in pop.items()}
    chain = build_chain(parents, fitness, target)
    summary = summarise_chain(chain, rounds, champ_gk)

    out_path = res / names["csv"]
    meta_path = res / names["meta"]
    meta = dict(name=name, sims=sims, parts=list(parts), seed=seed, every=every,
                target=list(target), positions=positions, openings=openings,
                ladder_every=ladder_every)
    if fresh:
        for p in (out_path, meta_path):
            if p.exists():
                p.unlink()
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        # a meta written before the v2 options has the v1 settings
        old = dict(dict(positions="v1", openings="2ply", ladder_every=1), **old)
        if old != meta:
            raise SystemExit(f"{meta_path} records other settings {old}; rerun with the same "
                             f"settings or --fresh")
    _check_csv_settings(out_path, positions, openings)
    _dump_json(meta_path, meta)
    rows, times = score_lineage(chain, genomes, rounds, champ_gk, config_path, out_path, sims,
                                parts, seed, results_dir, workers, every, positions=positions,
                                openings=openings, ladder_every=ladder_every, log=log)
    jumps = find_jumps(rows, genomes)
    summary["n_scored"] = len(rows)
    _dump_json(res / names["summary"], summary)
    _dump_json(res / names["json"], dict(name=name, top_k=TOP_K, metrics=jumps,
                                         settings=meta))
    (res / names["md"]).write_text(jumps_markdown(name, jumps, summary, meta))
    log(f"wrote {out_path}, jumps.json, jumps.md, lineage_summary.json in {res}")
    return dict(chain=chain, rows=rows, jumps=jumps, summary=summary, times=times)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--parts", default="uci,moves,ladder")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--positions", choices=["10k", "v1"], default="v1",
                    help="move-accuracy positions: v1 (2,000) or 10k (10,000)")
    ap.add_argument("--openings", choices=["2ply", "3ply"], default="2ply",
                    help="ladder openings: 2ply (42 games / opponent) or 3ply (294)")
    ap.add_argument("--ladder-every", type=int, default=1,
                    help="ladder only every k-th generation (+ round champions, final)")
    a = ap.parse_args(argv)
    run(a.name, a.runs_dir, a.results_dir, sims=a.sims, workers=a.workers,
        parts=tuple(p for p in a.parts.split(",") if p), every=max(1, a.every), seed=a.seed,
        fresh=a.fresh, positions=a.positions, openings=a.openings,
        ladder_every=a.ladder_every)


if __name__ == "__main__":
    main()
