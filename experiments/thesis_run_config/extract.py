"""Read the configuration every Chapter 7 run family used (layer 2: read-only extractor).

    uv run python experiments/thesis_run_config/extract.py     # -> data.json (~10 s)

One record per run family (a study's seeds that share one configuration) behind a network Chapter 7
presents (21 Nov 2026: the WHAS500 and GBSG2 Cox runs (GBSG2 a side note), the Churn case; no
300-generation or sigmoid-unit synthetic runs; ExConNEAT only the presented MNIST 10-class network's
phases). Every value is read from the
record the runs left behind, never typed:

* runs stored in the database (``explaneat_dev``, READ ONLY transactions; ``PGOPTIONS
  default_transaction_read_only=on`` as well): ``experiments.config_json`` (the NEAT config the
  run was built from), ``hardware_info.run_params`` (generations, epochs, fitness, phantom mask,
  hidden activation), the number of stored generations, the config template a UI run was launched
  from (and, for the original Adult run, the options it changed from that template), and
  ``neat_config_text`` where the run stored one. Which experiments belong to a family is
  read from the study's own run list (``runs*.csv``) or
  from the explained genome (``experiments/benchmarks/results/propneat.json``; the original networks
  also ``experiments/10seed/runs_paper_reference.csv``, asserted equal where both hold one). Experiment 2's
  configs are also read from ``explaneat_v4`` and asserted equal;
* in-memory runs: their run records (Cox: ``experiments/survival/cox_case/runs/<tag>.json`` with
  the NEAT config text the run used; Connect-4: ``logs/<name>.json`` plus the config file, or for
  the capacity study the base config with the recorded overrides, asserted equal to the config
  text stored when the selected run was imported; XOR: the config and settings in
  ``experiments/xor/run_xor.py``; ExConNEAT: ``config.json`` / ``results.json`` of each phase of the
  presented MNIST 10-class network: the 10k run, its pruning phase and the 50k warm start);
* the training code each run used: ``explaneat/core/backproppop.py`` at the run's recorded commit
  (``git show``; optimiser and learning rate, whether a phantom-weight mask existed and its
  default), and whether the training-path files differ between the commits of one family;
* Experiment 3's fine-tuning: the constants in ``experiments/churn/ext_common.py`` and
  ``genome_surgery.py`` (read with ``ast``, not imported), checked against the columns of the
  result CSVs of the in-scope studies.

Every family's runs are asserted to share one configuration (all NEAT settings except
``num_inputs``, and the run parameters except seed and commit). Nothing is written to any database.
"""
from __future__ import annotations

import ast
import configparser
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PGOPTIONS", "-c default_transaction_read_only=on")

import neat  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiments"
OUT = HERE / "data.json"
DBS = ("explaneat_dev", "explaneat_v4")
TRAINING_PATH = ["explaneat/core/backproppop.py", "explaneat/core/neuralneat.py",
                 "explaneat/db/population.py", "explaneat/evaluators"]
# NEAT settings that describe the data, not the run (the datasets table has them)
DATA_KEYS = {"num_inputs"}

# ------------------------------------------------------------------ helpers


class DB:
    def __init__(self):
        self.engines = {d: create_engine(f"postgresql://localhost/{d}") for d in DBS}

    def rows(self, dbname, sql, **kw):
        with self.engines[dbname].connect() as c:
            c.execute(text("SET TRANSACTION READ ONLY"))
            out = [dict(r._mapping) for r in c.execute(text(sql), kw)]
            c.rollback()
        return out

    def one(self, dbname, sql, **kw):
        r = self.rows(dbname, sql, **kw)
        assert len(r) == 1, (sql, kw, len(r))
        return r[0]


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def code_facts(commit):
    """Optimiser and phantom mask of the training loop at ``commit``."""
    src = git("show", f"{commit}:explaneat/core/backproppop.py")
    lrs = set(re.findall(r"optim\.Adadelta\(\s*net\.parameters\(\)\s*,\s*lr=([0-9.]+)", src))
    assert len(lrs) == 1, (commit, lrs)
    sig = re.search(r"def __init__\((.*?)\):", src, re.S).group(1)
    m = re.search(r"mask_phantom\s*=\s*(True|False)", sig)
    return {"commit": commit, "optimiser": "Adadelta", "lr": float(lrs.pop()),
            "batch": "full", "has_mask_phantom": m is not None,
            "mask_phantom_default": (m.group(1) == "True") if m else None,
            "weight_decay_param": "weight_decay" in sig}


def training_path_diff(commits):
    """Files of the training path that differ between any two commits of a family."""
    commits = sorted(set(commits))
    diff = set()
    for c in commits[1:]:
        diff |= set(git("diff", "--name-only", commits[0], c, "--", *TRAINING_PATH).split())
    return sorted(diff)


def canon(v):
    if isinstance(v, str):
        if v in ("True", "False"):
            return v == "True"
        try:
            f = float(v)
            return int(f) if f == int(f) and "." not in v and "e" not in v.lower() else f
        except ValueError:
            return v
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return int(v) if float(v) == int(v) and not isinstance(v, float) else float(v)
    if isinstance(v, (list, tuple)):
        return [canon(x) for x in v]
    return v


def num_equal(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


def neat_defaults():
    """neat-python's defaults for the genome options that have one (``ConfigParameter.default``)."""
    from neat.genes import DefaultConnectionGene, DefaultNodeGene
    out = {"single_structural_mutation": False, "structural_mutation_surer": "default"}
    src = Path(neat.genome.__file__).read_text()
    assert "ConfigParameter('single_structural_mutation', bool, 'false')" in src
    assert "ConfigParameter('structural_mutation_surer', str, 'default')" in src
    for p in DefaultNodeGene.get_config_params() + DefaultConnectionGene.get_config_params():
        if p.default is not None:
            out[p.name] = canon(p.default)
    return out


def flat_from_json(cj, fill=None):
    """The app's serialised NEAT config (``experiments.config_json``) as one flat dict. The app does
    not store options left at their neat-python default; ``fill`` (a dict) collects the ones filled in."""
    out = {k: canon(v) for k, v in cj["genome"].items()}
    out["compatibility_threshold"] = canon(cj["species"]["compatibility_threshold"])
    for sec in ("stagnation", "reproduction"):
        out.update({k: canon(v) for k, v in cj[sec].items()})
    for k in ("pop_size", "fitness_criterion", "fitness_threshold", "reset_on_extinction",
              "no_fitness_termination"):
        out[k] = canon(cj[k])
    for k, v in neat_defaults().items():
        if k not in out:
            out[k] = v
            if fill is not None:
                fill[k] = v
    return out


def flat_from_neat(path):
    """A NEAT .ini as neat-python reads it (its defaults filled in), as one flat dict."""
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                      neat.DefaultStagnation, str(path))
    g = cfg.genome_config
    out = {}
    for p in g._params:
        out[p.name] = canon(getattr(g, p.name))
    out["compatibility_threshold"] = canon(cfg.species_set_config.compatibility_threshold)
    for k in ("species_fitness_func", "max_stagnation", "species_elitism"):
        out[k] = canon(getattr(cfg.stagnation_config, k))
    for k in ("elitism", "survival_threshold", "min_species_size"):
        out[k] = canon(getattr(cfg.reproduction_config, k))
    for k in ("pop_size", "fitness_criterion", "fitness_threshold", "reset_on_extinction",
              "no_fitness_termination"):
        out[k] = canon(getattr(cfg, k))
    # neat-python 0.92 stores bool options of the gene attributes as strings
    return {k: canon(str(v)) if isinstance(v, str) else v for k, v in out.items()}


def flat_from_text(txt):
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as f:
        f.write(txt)
    try:
        return flat_from_neat(f.name)
    finally:
        os.unlink(f.name)


def ini_with_overrides(base, overrides):
    """configs/<base>.ini with ``Section.key=value`` overrides (as c4_capacity/run_seed.build_config)."""
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(base)
    for o in overrides:
        key, val = o.split("=", 1)
        sec, k = key.split(".", 1)
        assert cp.has_option(sec, k), o
        cp.set(sec, k, val)
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as f:
        cp.write(f)
    try:
        return flat_from_neat(f.name)
    finally:
        os.unlink(f.name)


def same_config(flats, what):
    """All runs of a family share one NEAT config (except the data keys); returns it."""
    first = {k: v for k, v in flats[0].items() if k not in DATA_KEYS}
    for f in flats[1:]:
        other = {k: v for k, v in f.items() if k not in DATA_KEYS}
        assert first.keys() == other.keys(), (what, first.keys() ^ other.keys())
        bad = [k for k in first if not num_equal(first[k], other[k])]
        assert not bad, (what, {k: (first[k], other[k]) for k in bad})
    return first


def assert_flat_equal(a, b, what):
    assert set(a) - DATA_KEYS == set(b) - DATA_KEYS, (what, (set(a) ^ set(b)) - DATA_KEYS)
    keys = (set(a) & set(b)) - DATA_KEYS
    bad = [k for k in sorted(keys) if not num_equal(a[k], b[k])]
    assert not bad, (what, {k: (a[k], b[k]) for k in bad})
    return sorted(keys)


def module_constant(path, name):
    """A module-level constant (literal, or ``dict(k=literal, ...)``), read with ast."""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, "id", None) == "dict":
                return {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords}
            return ast.literal_eval(v)
    raise KeyError((path, name))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rel(p):
    return str(Path(p).relative_to(REPO))


# ------------------------------------------------------------------ database families

Q_EXP = ("SELECT id, name, random_seed, git_commit_sha, config_json, hardware_info, neat_config_text, "
         "config_template_id, status, "
         "(SELECT count(*) FROM populations p WHERE p.experiment_id = e.id) AS n_pop, "
         "(SELECT min(generation) FROM populations p WHERE p.experiment_id = e.id) AS g_min, "
         "(SELECT max(generation) FROM populations p WHERE p.experiment_id = e.id) AS g_max "
         "FROM experiments e WHERE id = CAST(:i AS uuid)")
Q_GENOME_EXP = ("SELECT p.experiment_id FROM genomes g JOIN populations p ON g.population_id = p.id "
                "WHERE g.id = CAST(:g AS uuid)")


def db_runs(db, exp_ids, dbname="explaneat_dev"):
    runs = []
    for i in exp_ids:
        r = db.one(dbname, Q_EXP, i=str(i))
        assert r["n_pop"] == r["g_max"] + 1 and r["g_min"] == 0, (r["name"], "generations not contiguous")
        runs.append(r)
    return runs


def db_family(db, key, label, experiment, exp_ids, source, check_v4=False):
    runs = db_runs(db, exp_ids)
    filled = {}
    flats = [flat_from_json(r["config_json"], filled) for r in runs]
    neat_cfg = same_config(flats, key)
    rp = [dict((r["hardware_info"] or {}).get("run_params") or {}) for r in runs]
    for p in rp:
        for k in ("seed", "code_commit", "label", "wall_clock_s", "best_generation", "dataset"):
            p.pop(k, None)
    assert all(p == rp[0] for p in rp), (key, "run_params differ")
    training = [r["config_json"].get("training") or {} for r in runs]
    assert all(t == training[0] for t in training), (key, "config_json.training differs")
    gens = sorted({r["n_pop"] for r in runs})
    commits = sorted({r["git_commit_sha"] for r in runs})
    # a stored NEAT config text, when present, says the same as config_json
    texts = [r for r in runs if r["neat_config_text"]]
    text_checked = []
    for r in texts:
        text_checked = assert_flat_equal(flat_from_text(r["neat_config_text"]), flat_from_json(r["config_json"]),
                                         (key, r["name"], "neat_config_text"))
    if check_v4:
        for r in runs:
            v4 = db.one("explaneat_v4", "SELECT config_json, hardware_info FROM experiments WHERE id = CAST(:i AS uuid)",
                        i=str(r["id"]))
            assert v4["config_json"] == r["config_json"] and v4["hardware_info"] == r["hardware_info"], (key, r["name"])
    rec = {
        "key": key, "label": label, "experiment": experiment, "source": source,
        "store": "explaneat_dev" + (" (= explaneat_v4)" if check_v4 else ""),
        "runs": [{"name": r["name"], "experiment_id": str(r["id"]), "seed": r["random_seed"],
                  "commit": r["git_commit_sha"], "status": r["status"], "generations_stored": r["n_pop"],
                  "num_inputs": flats[j]["num_inputs"]} for j, r in enumerate(runs)],
        "seeds": sorted(r["random_seed"] for r in runs) if all(r["random_seed"] is not None for r in runs) else None,
        "n_runs": len(runs),
        "neat": neat_cfg, "run_params": rp[0], "config_json_training": training[0],
        "generations_stored": gens, "commits": commits,
        "code": [code_facts(c) for c in commits], "training_path_diff": training_path_diff(commits),
        "neat_config_text_checked_keys": len(text_checked),
        "neat_defaults_filled": filled,
    }
    # the config file the run read (run_params.neat_config) says the same, every key
    ini = rec["run_params"].get("neat_config")
    if ini:
        rec["config_file"] = ini
        rec["config_file_checked_keys"] = len(assert_flat_equal(flat_from_neat(REPO / ini), neat_cfg, (key, ini)))
    return rec, runs


def exp_ids_from_csv(path, filt=None):
    rows = read_csv(path)
    return [r["experiment_id"] for r in rows if filt is None or filt(r)], rows


def original_genome(tag):
    """The explained genome of an original network: experiments/benchmarks/results/propneat.json, or, for one
    it does not hold yet (the original Adult network, 21 Nov 2026), experiments/10seed/runs_paper_reference.csv
    (the three original networks). When both hold it they must agree."""
    pn = json.loads((EXP / "benchmarks" / "results" / "propneat.json").read_text())["exp1"].get(tag)
    ref = [r for r in read_csv(EXP / "10seed" / "runs_paper_reference.csv")
           if r["seed"] == "paper" and f"paper-{r['dataset']}" == tag]
    assert len(ref) == 1, (tag, len(ref))
    if pn is not None:
        assert pn["genome_id"] == ref[0]["genome_id"], (tag, pn["genome_id"], ref[0]["genome_id"])
        return pn["genome_id"], "experiments/benchmarks/results/propneat.json"
    return ref[0]["genome_id"], "experiments/10seed/runs_paper_reference.csv"


def paper_family(db, key, label, tag, ini, template_overrides=()):
    """An original network (Experiment 1): the run that produced the explained genome. ``template_overrides``:
    the NEAT options the run changed from the template it was launched from (experiments/10seed/PREREG.md,
    "Configs": Adult "template bb576521 + overrides"); asserted to be exactly the options that differ."""
    gid, gsrc = original_genome(tag)
    eid = db.one("explaneat_dev", Q_GENOME_EXP, g=gid)["experiment_id"]
    rec, runs = db_family(db, key, label, "1", [eid], f"genome {gid} ({gsrc}) -> its experiment")
    r = runs[0]
    rec["genome_id"] = gid
    # the explained genome has the run's highest training fitness
    fit = db.one("explaneat_dev", "SELECT g.fitness, (SELECT max(g2.fitness) FROM genomes g2 JOIN populations p2 "
                 "ON g2.population_id = p2.id WHERE p2.experiment_id = p.experiment_id) AS best FROM genomes g "
                 "JOIN populations p ON g.population_id = p.id WHERE g.id = CAST(:g AS uuid)", g=gid)
    assert fit["fitness"] == fit["best"], (key, fit)
    rec["selected"] = {"rule": "highest training fitness of the run", "checked": True}
    # experiments/10seed/PREREG.md: the ten-seed config files equal the original runs' stored configs
    rec["config_file"] = ini
    rec["config_file_checked_keys"] = len(assert_flat_equal(flat_from_neat(REPO / ini), rec["neat"], (key, ini)))
    if r["config_template_id"]:
        t = db.one("explaneat_dev", "SELECT name, config FROM config_templates WHERE id = :i",
                   i=r["config_template_id"])
        rec["template"] = {"id": str(r["config_template_id"]), "name": t["name"],
                           "training": t["config"]["training"], "backprop": t["config"]["backprop"]}
        # the template's NEAT block agrees with the config the run stored, except the run's overrides
        diff = sorted(k for k, v in t["config"]["neat"].items() if not num_equal(canon(v), rec["neat"][k]))
        assert diff == sorted(template_overrides), (key, diff)
        if diff:
            rec["template"]["overrides"] = {k: {"template": canon(t["config"]["neat"][k]), "run": rec["neat"][k]}
                                            for k in diff}
        assert t["config"]["training"]["population_size"] == rec["neat"]["pop_size"]
    return rec


# ------------------------------------------------------------------ file families


# Cox datasets whose network Chapter 7 presents, in the order of the table's name (GBSG2 is back in
# Experiment 1 as a side note, 21 Nov 2026; tab:exp1-networks has both) -- curated
COX_PRESENTED = ("whas500", "gbsg2")


def cox_family():
    d = EXP / "survival" / "cox_case"
    sel = {k: v for k, v in json.loads((d / "selected.json").read_text()).items() if k in COX_PRESENTED}
    assert set(sel) == set(COX_PRESENTED), sel
    runs, flats, rps = [], [], []
    for ds in COX_PRESENTED:
        for s in range(5):
            r = json.loads((d / "runs" / f"{ds}-s{s}.json").read_text())
            runs.append({"name": r["tag"], "seed": r["seed"], "commit": r["code_commit"], "code_dirty": r["code_dirty"],
                         "generations_stored": len(r["champion_trace"]), "num_inputs": r["n_features"]})
            flats.append(flat_from_text(r["neat_config_text"]))
            rps.append(r["run_params"])
    assert all(p == rps[0] for p in rps), "Cox run_params differ"
    rp = rps[0]
    assert {r["generations_stored"] for r in runs} == {rp["generations"]}
    neat_cfg = same_config(flats, "cox")
    assert neat_cfg["pop_size"] == rp["pop_size"]
    # the run's config is the study config file with num_inputs set
    assert_flat_equal(flat_from_neat(REPO / rp["config_file"]), neat_cfg, "cox config file")
    commits = sorted({r["commit"] for r in runs})
    return {
        "key": "cox", "label": "WHAS500, GBSG2 (Cox)", "experiment": "1", "datasets": list(COX_PRESENTED),
        "source": "experiments/survival/cox_case/runs/{whas500,gbsg2}-s{0..4}.json (run_params, neat_config_text)",
        "store": "run records (in memory, not in the database)",
        "runs": runs, "seeds": sorted(r["seed"] for r in runs), "n_runs": len(runs),
        "neat": neat_cfg, "run_params": rp, "generations_stored": [rp["generations"]],
        "commits": commits, "code_dirty": sorted({r["code_dirty"] for r in runs}),
        "code": [code_facts(c) for c in commits], "training_path_diff": training_path_diff(commits),
        "selected": {k: {"tag": v["tag"], "rule": v["rule"]} for k, v in sel.items()},
    }


def c4_log_runs(logs, names):
    out = []
    for n in names:
        j = json.loads((logs / f"{n}.json").read_text())
        out.append(j)
    return out


def c4_family(db, key, label, study, names, flats_of, import_name, selected_name, runs_csv, csv_pick):
    logs = EXP / study / "logs"
    recs = c4_log_runs(logs, names)
    flats = [flats_of(j) for j in recs]
    neat_cfg = same_config(flats, key)
    keep = ("n_generations", "n_epochs_backprop", "fitness_function", "criterion", "output_activation",
            "mask_phantom", "variant", "val_frac", "dynamic_threshold", "torch_threads")
    rps = [{k: j.get(k) for k in keep} for j in recs]
    ov = [j["overrides"] if isinstance(j["overrides"], list)
          else [f"{k}={v}" for k, v in j["overrides"].items()] for j in recs]
    assert all(o == ov[0] for o in ov), (key, "overrides differ")
    assert all(p == rps[0] for p in rps), (key, "run settings differ")
    assert neat_cfg["pop_size"] == int(dict(o.split("=") for o in ov[0])["NEAT.pop_size"])
    # the selected run, imported into explaneat_dev, stored the same config
    imp = db.one("explaneat_dev", "SELECT id, neat_config_text, config_json, hardware_info FROM experiments WHERE name = :n",
                 n=import_name)
    assert_flat_equal(flat_from_text(imp["neat_config_text"]), neat_cfg, (key, "imported neat_config_text"))
    assert_flat_equal(flat_from_json(imp["config_json"]), neat_cfg, (key, "imported config_json"))
    # selection: the selected run has the largest training AUC of the runs it was chosen from
    rows = [r for r in read_csv(runs_csv) if csv_pick(r)]
    best = max(rows, key=lambda r: float(r["train_auc"]))
    assert best["name"] == selected_name, (key, best["name"])
    commits = sorted({j["code_commit"] for j in recs})
    return {
        "key": key, "label": label, "experiment": "1",
        "source": f"experiments/{study}/logs/<run>.json (+ config, see overrides); imported run {import_name}",
        "store": "run records (in memory); the selected run imported into explaneat_dev",
        "runs": [{"name": j["name"], "seed": j["seed"], "commit": j["code_commit"], "code_dirty": j["code_dirty"],
                  "generations_stored": j["n_generations"], "num_inputs": flats[i]["num_inputs"]}
                 for i, j in enumerate(recs)],
        "seeds": sorted(j["seed"] for j in recs), "n_runs": len(recs),
        "neat": neat_cfg, "run_params": rps[0], "overrides": ov[0],
        "generations_stored": sorted({j["n_generations"] for j in recs}),
        "commits": commits, "code_dirty": sorted({j["code_dirty"] for j in recs}),
        "code": [code_facts(c) for c in commits], "training_path_diff": training_path_diff(commits),
        "selected": {"name": selected_name, "from": [r["name"] for r in rows], "rule": "largest training AUC"},
    }


def xor_family():
    sys.path.insert(0, str(EXP / "xor"))
    import run_xor
    src = (EXP / "xor" / "run_xor.py").read_text()
    res = json.loads((EXP / "xor" / "xor_results.json").read_text())
    neat_cfg = flat_from_text(run_xor.CONFIG)
    epochs = {int(x) for x in re.findall(r"pop\.run\(bce_fitness, n=1, nEpochs=(\d+)\)", src)}
    assert len(epochs) == 1
    gen_cap = int(re.search(r'"--generations", type=int, default=(\d+)', src).group(1))
    n_seeds = int(re.search(r'"--seeds", type=int, default=(\d+)', src).group(1))
    assert res["summary"]["n_seeds"] == n_seeds == len(res["runs"])
    solved = [r["solved_generation"] for r in res["runs"]]
    assert all(g is not None and g < gen_cap for g in solved)
    assert "g.fitness = float(1.0 / loss(p, ys))" in src and "torch.nn.BCELoss()" in src
    mask = re.findall(r"SafePopulation\(config, X, Y, mask_phantom=(True|False)\)", src)
    assert mask == ["True"], mask
    assert "set_output_activation" not in src and "set_hidden_activation" not in src
    # the code of the commit that wrote the results
    commit = git("log", "-1", "--format=%H", "--", "experiments/xor/xor_results.json").strip()
    return {
        "key": "xor", "label": "XOR", "experiment": "1",
        "source": "experiments/xor/run_xor.py (CONFIG, argparse defaults, run_seed) and xor_results.json",
        "store": "run records (in memory)",
        "runs": [{"name": f"seed {r['seed']}", "seed": r["seed"], "solved_generation": r["solved_generation"]}
                 for r in res["runs"]],
        "seeds": sorted(r["seed"] for r in res["runs"]), "n_runs": len(res["runs"]),
        "neat": neat_cfg,
        "run_params": {"n_epochs_backprop": epochs.pop(), "generation_cap": gen_cap,
                       "stop": "best genome classifies all four rows", "fitness_function": "1/BCE",
                       "mask_phantom": True, "criterion": "bce", "output_activation": "sigmoid"},
        "solved_generation_range": [min(solved), max(solved)],
        "commits": [commit], "code": [code_facts(commit)], "training_path_diff": [],
        "code_commit_note": "last commit of xor_results.json",
    }


# ------------------------------------------------------------------ Experiment 3 fine-tuning


def exp3_finetune():
    ch = EXP / "churn"
    ec = ch / "ext_common.py"
    gs = ch / "genome_surgery.py"
    const = {n: module_constant(ec, n) for n in ("EPOCHS_PRIMARY", "EPOCHS_SECONDARY", "ES_PATIENCE",
                                                  "ES_MIN_DELTA", "ES_MAX_EPOCHS", "VAL_FRACTION", "VAL_RANDOM_STATE")}
    const.update({n: module_constant(gs, n) for n in ("FT_OPTIMIZER", "FT_LR", "FT_LOSS")})
    const["E_PRIMARY_reinit"] = module_constant(ch / "reinit.py", "E_PRIMARY")
    assert const["E_PRIMARY_reinit"] == const["EPOCHS_PRIMARY"]
    # the in-scope studies' result files: repair (B), knockout-and-rescue (D2), re-initialisation (E), MONK-2 (F2)
    fixed = {"repair.csv": "epochs", "rescue2.csv": "epochs", "reinit_network.csv": "epochs",
             "reinit_feature.csv": "epochs", "monk2graft_e500.csv": "ft_epochs"}
    es = ["repair_es.csv", "rescue2_es.csv", "reinit_network_es.csv"]
    seen_e, phantom, es_params = set(), set(), {"es_patience": set(), "es_min_delta": set(), "es_max_epochs": set()}
    for f, col in fixed.items():
        rows = read_csv(ch / f)
        seen_e |= {int(r[col]) for r in rows if r[col] not in ("", "0")}
        if rows and "ft_phantom" in rows[0]:
            phantom |= {r["ft_phantom"] for r in rows if r["ft_phantom"]}
    for f in es:
        rows = read_csv(ch / f)
        for k in es_params:
            es_params[k] |= {r[k] for r in rows if r[k]}
        if "ft_phantom" in rows[0]:
            phantom |= {r["ft_phantom"] for r in rows if r["ft_phantom"]}
    assert es_params["es_patience"] == {str(const["ES_PATIENCE"])}
    assert {float(x) for x in es_params["es_min_delta"]} == {const["ES_MIN_DELTA"]}
    assert es_params["es_max_epochs"] == {str(const["ES_MAX_EPOCHS"])}
    assert phantom == {"masked"}, phantom
    # the early-stopping validation rows
    vals = {k: json.loads((ch / f).read_text()) for k, f in (("churn", "val_split.json"), ("monk2", "monk2_val_split.json"))}
    for v in vals.values():
        assert abs(v["n_val"] / v["n_train"] - const["VAL_FRACTION"]) < 0.01
    return {
        "key": "exp3_ft", "label": "Fine-tuning (grafting)", "experiment": "3",
        "source": "experiments/churn/ext_common.py, genome_surgery.py, reinit.py (constants); result CSVs "
                  + ", ".join(list(fixed) + es),
        "constants": const, "fixed_epochs_seen": sorted(seen_e), "ft_phantom": sorted(phantom),
        "es_params_seen": {k: sorted(v) for k, v in es_params.items()},
        "validation": {k: {"n_val": v["n_val"], "n_train": v["n_train"]} for k, v in vals.items()},
    }


# ------------------------------------------------------------------ Experiment 4 (ExConNEAT)


def stop_text(s):
    """A run's stop reason up to its first semicolon (what stopped it, without the run log's comment)."""
    return s.split(";")[0].strip()


def conv_phase(key, label, d, prev=None):
    cfg = json.loads((d / "config.json").read_text())
    res = json.loads((d / "results.json").read_text())
    c, a = cfg["config"], cfg["args"]
    out = {
        "key": key, "label": label, "experiment": "4", "source": f"{rel(d)}/config.json, results.json",
        "config": c, "args": a, "sizes": cfg["sizes"], "device": cfg["device"], "torch": cfg["torch"],
        "generations_run": res["generations_run"], "stop_reason": stop_text(res["stop_reason"]),
        "selected": {k: res["selected"][k] for k in ("generation", "val_acc", "complexity") if k in res["selected"]},
        "warm_start_info": cfg.get("warm_start_info"),
    }
    if prev is not None:
        out["generations_in_phase"] = res["generations_run"] - cfg["warm_start_info"]["source_generation"]
    return out


def conv_families():
    cn = EXP / "conv_neat"
    p1 = conv_phase("conv_p1", "MNIST 10-class, 10k run", cn / "results" / "sweep_p50_m4_e5")
    p2 = conv_phase("conv_p2", "MNIST 10-class, pruning", cn / "results" / "mnist10_pruned", prev=p1)
    p3 = conv_phase("conv_p3", "MNIST 10-class, 50k warm start", cn / "results" / "mnist10_ws50k_halving", prev=p2)
    head = json.loads((cn / "results" / "mnist10_pruned" / "headline_results.json").read_text())
    p2["headline_rule"] = head["rule"]
    p2["headline_generation"] = head["generation"]
    # the phases chain: each warm start is the previous phase's checkpoint
    assert p2["warm_start_info"]["from"] == "results/sweep_p50_m4_e5/checkpoint.pkl"
    assert p3["warm_start_info"]["from"] == "results/mnist10_pruned/checkpoint.pkl"
    # mutation scale: the run's structural rates over the code's defaults
    from explaneat.conv.genome import ConvNeatConfig
    dflt = ConvNeatConfig()
    for p in (p1, p2, p3):
        p["mut_scale_check"] = {f: p["config"][f] / getattr(dflt, f) for f in ("p_add_node", "p_add_conn", "p_kernel_size")}
    # the scarcity phase's settings in the code (run_long.adopt_schedule)
    src = (cn / "run_long.py").read_text()
    m = re.search(r'Phase\("scarcity", 1, struct_scale=([0-9.]+), other_struct=(\w+), weight_scale=([0-9.]+), '
                  r'epochs=a\.epochs,\s*penalty_scale=a\.scarcity_penalty_scale, p_delete_conn=([0-9.]+), '
                  r'p_delete_node=([0-9.]+)\)', src)
    assert m, "scarcity phase not found in run_long.py"
    p2["scarcity_phase"] = {"struct_scale": float(m.group(1)), "other_struct": m.group(2) == "True",
                            "weight_scale": float(m.group(3)), "p_delete_conn": float(m.group(4)),
                            "p_delete_node": float(m.group(5))}
    # fitness (documented in each runner's protocol) and the training code: Adam, cross-entropy
    assert ("fitness (--fitness train, default) = accuracy on the generation's training rows - lambda * (hidden\n"
            "  nodes + enabled edges)") in src
    assert all(p["args"]["fitness"] == "train" for p in (p1, p2, p3))
    net = (REPO / "explaneat" / "conv" / "network.py").read_text()
    assert "F.cross_entropy(logits, y.long())" in net
    assert "torch.optim.Adam(net.parameters(), lr=cfg.lr)" in src
    return [p1, p2, p3]


# ------------------------------------------------------------------ main


def main():
    db = DB()
    fam = {}
    # Experiment 1: original networks
    fam["paper_heart"] = paper_family(db, "paper_heart", "Heart, original network", "paper-heart",
                                      "experiments/10seed/configs/heart.ini")
    # the original Adult run ("No pressure") was launched from the Heart run's template with its structural
    # mutation and speciation options changed (RESULTS.md, "Author's account of the config families")
    fam["paper_adult"] = paper_family(db, "paper_adult", "Adult, original network", "paper-adult",
                                      "experiments/10seed/configs/adult.ini",
                                      template_overrides=("node_add_prob", "node_delete_prob", "conn_add_prob",
                                                          "conn_delete_prob", "compatibility_threshold",
                                                          "compatibility_disjoint_coefficient"))
    fam["paper_monk2"] = paper_family(db, "paper_monk2", "MONK-2, original network", "paper-monk2",
                                      "experiments/10seed/configs/monk2.ini")
    # the MONK-2 run was launched by the ingest script: its training settings
    ing = EXP / "aaai27" / "ingest_monk2.py"
    fam["paper_monk2"]["launch"] = {"file": rel(ing), "TRAIN": module_constant(ing, "TRAIN")}
    assert fam["paper_monk2"]["launch"]["TRAIN"]["population_size"] == fam["paper_monk2"]["neat"]["pop_size"]
    # the Churn case (one seeded run, experiments/churn_case/runs.csv): the churn configuration, 200 generations
    ids, rows = exp_ids_from_csv(EXP / "churn_case" / "runs.csv")
    fam["churn_case"], _ = db_family(db, "churn_case", "Churn case", "1", ids, "experiments/churn_case/runs.csv")
    fam["churn_case"]["selected"] = {"rule": "highest training fitness of the run", "genome_id": rows[0]["genome_id"]}
    assert len(rows) == 1 and fam["churn_case"]["runs"][0]["seed"] == int(rows[0]["seed"])
    fit = db.one("explaneat_dev", "SELECT g.fitness, (SELECT max(g2.fitness) FROM genomes g2 JOIN populations p2 "
                 "ON g2.population_id = p2.id WHERE p2.experiment_id = p.experiment_id) AS best FROM genomes g "
                 "JOIN populations p ON g.population_id = p.id WHERE g.id = CAST(:g AS uuid)", g=rows[0]["genome_id"])
    assert fit["fitness"] == fit["best"], ("churn_case", fit)   # the explained genome is the run's best
    # Cox
    fam["cox"] = cox_family()
    # Connect-4
    c4base = EXP / "c4_capacity" / "configs" / "c4_base.ini"
    fam["c4_cells"] = c4_family(
        db, "c4_cells", "Connect-4, cells", "c4_capacity", [f"main-h10-e50-s{s}" for s in range(3)],
        lambda j: ini_with_overrides(c4base, j["overrides"]), "c4cap-main-h10-e50-s1", "main-h10-e50-s1",
        EXP / "c4_capacity" / "runs.csv", lambda r: r["name"].startswith("main-"))
    fam["c4_cells"]["config_file"] = rel(c4base) + " + overrides (configs/main-*.ini are not committed)"
    fam["c4_lines"] = c4_family(
        db, "c4_lines", "Connect-4, lines", "c4_line_features", [f"lines-heavy-s{s}" for s in range(3)],
        lambda j: flat_from_neat(REPO / j["neat_config"]), "c4lf-lines-heavy-s0", "lines-heavy-s0",
        EXP / "c4_line_features" / "runs.csv", lambda r: r["inputs"] == "lines")
    # XOR
    fam["xor"] = xor_family()
    # synthetic
    us = EXP / "ushape_saddle"
    ids, rows = exp_ids_from_csv(us / "runs.csv")
    fam["shapes"], _ = db_family(db, "shapes", "Synthetic U, saddle, U + saddle", "1", ids,
                                 "experiments/ushape_saddle/runs.csv (15 experiment ids)")
    fam["shapes"]["datasets"] = sorted({r["dataset"] for r in rows})
    pn = json.loads((EXP / "benchmarks" / "results" / "propneat.json").read_text())["exp1"]
    sel = {}
    for tag, ds in (("shape-A", "u"), ("shape-B", "saddle"), ("shape-C", "both")):
        best = max((r for r in rows if r["dataset"] == ds), key=lambda r: float(r["train_auc"]))
        assert best["genome_id"] == pn[tag]["genome_id"], tag
        sel[ds] = best["experiment_name"]
    fam["shapes"]["selected"] = {"per_dataset": sel, "rule": "largest training AUC of the five seeds"}
    # the 300-generation and sigmoid-unit variants (g300_scores.csv, sig_gens.json) are left out: no
    # network of theirs is presented (they appear only in the options figures)
    # Experiment 2
    ids, _ = exp_ids_from_csv(EXP / "10seed" / "runs.csv", lambda r: r["dataset"] == "adult")
    fam["adult"], _ = db_family(db, "adult", "Adult", "2", ids, "experiments/10seed/runs.csv (dataset adult)",
                                check_v4=True)
    ids, _ = exp_ids_from_csv(EXP / "churn" / "runs_churn100.csv")
    fam["churn"], _ = db_family(db, "churn", "Churn", "2", ids, "experiments/churn/runs_churn100.csv", check_v4=True)
    ids, _ = exp_ids_from_csv(EXP / "10seed" / "runs_heart100.csv")
    fam["heart100"], _ = db_family(db, "heart100", "Heart-100", "2", ids, "experiments/10seed/runs_heart100.csv",
                                   check_v4=True)
    # Experiment 3: the MONK-2 recipients and donors (ten seeds), and the fine-tuning
    ids, _ = exp_ids_from_csv(EXP / "10seed" / "runs.csv", lambda r: r["dataset"] == "monk2")
    fam["monk2_10"], _ = db_family(db, "monk2_10", "MONK-2, ten seeds", "3", ids,
                                   "experiments/10seed/runs.csv (dataset monk2)")
    fam["exp3_ft"] = exp3_finetune()
    for p in conv_families():
        fam[p["key"]] = p
    data = {"generated_by": "experiments/thesis_run_config/extract.py", "families": fam,
            "neat_defaults": neat_defaults()}
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    print("wrote", rel(OUT))
    for k, v in fam.items():
        print(f"{k:12s} runs={v.get('n_runs', '-')} gens={v.get('generations_stored', v.get('generations_run'))}"
              f" commits={len(v.get('commits', []))} diff={v.get('training_path_diff')}")


if __name__ == "__main__":
    main()
