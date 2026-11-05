# Reproducing the 10-seed consistency experiment (thesis Ch 7)

What this directory holds:
- the design and its dated amendments: `PREREG.md`;
- the results and interpretation: `../../results/10seed-consistency.md`;
- per-metric statistics: `summary.md`.

This file covers how to reproduce every number from the configs, seeds and code commits.

The pre-registered runs are 30: Heart (40 generations), MONK-2 and Adult, seeds 0–9.
There are three additional analyses, all reported separately and not counted towards the
threshold:
- **Heart-100:** Heart at 100 generations.
- **churn100** and **churn200:** see `../churn/PLAN.md`.

## 1. Environment

- Python via `uv`, from the repo root: `uv venv && uv pip install -e .`. Always run Python
  with `uv run`.
- Reported runs used an Apple M4 Pro with 14 cores and 48 GB, CPU only. Each run used float64
  and one torch thread (`TORCH_THREADS = 1` in `run_seed.py`). Each run's Python, torch and
  numpy versions are in its Experiment's `hardware_info`.
- PostgreSQL. The default is `postgresql://localhost/explaneat_dev`; override it with
  `DATABASE_URL`.
- `PYTHONHASHSEED=0` is required. `run_seed.py` refuses to start without it.

## 2. Database setup

The commands are in the repo `CLAUDE.md` under "Database". Run them yourself; this guide
deliberately does not run migrations:

```bash
uv run python -m explaneat db create-db
uv run python -m explaneat db init      # or: db upgrade, for an existing database
```

## 3. Datasets and splits

`run_seed.py` loads each dataset and split by id. It fits a StandardScaler on the training
rows in memory and asserts that the result equals the split's stored scaler.

| Run set | Dataset (id) | Split (id) | How it was built |
|---|---|---|---|
| Heart, Heart-100 | `Heart Disease (binary, prepared)` (`eea247c2-…`) | `165cebbd-…` | One-hot "prepare dataset" route (`explaneat/api/routes/datasets.py`) on Cleveland Heart; 80:20 split, `random_state` 42, not stratified |
| MONK-2 | `monk2_onehot` (`c2340904-…`) | `63257a45-…` | `scripts/aaai27/ingest_monk2.py`: 432 unique rows, one-hot, stratified 80:20, seed 42. It also launches a paper training run, so on a fresh DB stop it after ingestion or use only steps 1–4 |
| Adult | `adult_ohe_v2` (`46c63b17-…`) | `1df50169-…` | One-hot Adult, 80:20 split, `random_state` 42 |
| churn100/200 | `churn_onehot` (`802129c2-…`) | `45962ea6-…` | `uv run python experiments/churn/ingest_churn.py`: idempotent; drops phone number; one-hot state and area code; stratified 80:20, seed 42; also writes the LR/GBM baselines (`../churn/baselines.json`) |

The full ids are in `PREREG.md` and `run_seed.py` (`DATASETS`).

The paper datasets also ship offline in `reproducibility/{heart,monk2,adult}/`:
- `split.json` holds the split parameters and scaler;
- `data.npz` holds the exact z-normalised arrays.

On a fresh database, rebuild these datasets so that the scaler assertion passes. The new ids
must then be written into `DATASETS`.

## 4. Training runs

Each process runs one seed and creates one Experiment. The NEAT configs are
`configs/{heart,monk2,adult}.ini` and `../churn/configs/churn.ini`, all selected by
`run_seed.py`. The `--mask-phantom` flag was **off for every reported run**: it defaults
to off, and none of the commands below pass it.

`run_seed.py` passes `mask_phantom` explicitly. It therefore stays unmasked even though the
library default changed to masked in `929fd76`. The masked runs, `churn100mask`, are a
separate extension study (`../churn/EXTENSIONS_RESULTS.md`).

```bash
# Pre-registered: Heart (40 gen), MONK-2 (100), Adult (100); seeds 0-9.
# Experiment names default to 10seed-<dataset>-s<seed>.
for s in 0 1 2 3 4 5 6 7 8 9; do
  PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py heart $s
  PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py monk2 $s
  PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py adult $s
done
# (launch.sh <dataset> starts all 10 seeds of one dataset in parallel, logs in logs/.)

# Additional: Heart-100 (100 gen, default name 10seed-heart100-s<seed>)
PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py heart100 $s

# Additional: churn100 / churn200
PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py churn $s --name churn100-s$s
PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py churn $s --generations 200 --name churn200-s$s
```

Notes:
- **churn100 seeds 0–2** were run as the pilot under the default name `churn-pilot-s<seed>`,
  then renamed `churn100-s<seed>` (`../churn/PLAN.md`).
- **Queue:** `../churn/run_queue.sh` is the exact queue used for churn100 seeds 3–9 and all
  of churn200 (at most 4 concurrent).
- **`--generations`:** only churn200 overrides the dataset default (`DATASETS[...]["n_generations"]`).
- **Crashes:** adult-s1, s3 and s8 crashed at start-up, before any Experiment row was written,
  and were re-run with the same seed (`rerun_note` in `runs.csv`; `RUN_INCIDENTS.md`).
- **Adult pause:** Adult runs were paused and throttled with SIGSTOP/SIGCONT. Their raw wall
  clock includes those intervals. `make_figures.py` subtracts them.

### Code commits of the reported runs

Each run records `git rev-parse HEAD` (`code_commit` in `runs*.csv`, and
`hardware_info.run_params.code_commit` in the DB):

| Runs | Commit | Seeds |
|---|---|---|
| Heart (40 gen) | `0fd368ed895830eeeadb5ceee866ca86a9c916b4` | 0–9 |
| MONK-2 | `4126fe2508c64707c305172bf76d3cf3aa7f4617` | 0–9 |
| Adult | `4126fe2508c64707c305172bf76d3cf3aa7f4617` | 0, 2, 4, 5, 6, 7, 9 |
| Adult (re-runs after the start-up crash) | `cc48129c298763b02f716c310a3d94b0f41e9867` | 1, 3, 8 |
| Heart-100 | `4126fe2508c64707c305172bf76d3cf3aa7f4617` | 0–9 |
| churn100 | `901fad4b564131e118162d1d65d6859c5bf6885a` | 0, 1, 2 (pilot) |
| churn100 | `ad20d2dafe50ce3051feb84302f34d638181dac3` | 3–9 |
| churn200 | `3d59753eb09c1635b7c820f157354d1107d08b97` | 0, 6 |
| churn200 | `92d1f1191d8804aba50d3f16e2a971ce772d5f36` | 1, 2, 4, 7 |
| churn200 | `df2322599c3b920d664eff2b67c51025d7d8fa14` | 3, 5, 8, 9 |
| Paper networks (reference) | Heart `b21658f`, MONK-2 `fff5bb0`, Adult `0bd2e54` | — |

Several commits appear because the runs were launched while other work landed on the branch.

None of the training-path files (`explaneat/core/backproppop.py`, `core/neuralneat.py`,
`db/population.py`, `evaluators/`) changed the training computation between these commits:
- `9259bf1` changed only metric logging, and was checked bit-identical (PREREG amendment of
  2026-11-03).
- The `mask_phantom` commits add an opt-in path. They were checked bit-identical when off
  (500/500 genomes, `../churn/PLAN.md`).
- `4e1e51e` only changes behaviour when evaluation raises (`RUN_INCIDENTS.md`).

For an exact replay, check out the recorded commit.

### Expected reproduction: bit-identical

A seeded run is deterministic. It uses `random`, `numpy` and `torch` seeds, `PYTHONHASHSEED=0`,
one thread, float64 and CPU. Re-running a seed on the same platform should reproduce every
genome and fitness exactly. This was verified three ways:
- **Concurrent runs (2026-11-02):** two concurrent runs of the same seed (Heart, seed 1000,
  3 generations) matched 750/750 genomes (`PREREG.md`).
- **Prefix checks:** a longer run with the same seed must reproduce the shorter run's
  generations exactly. `verify_prefix_reproduction.py` compares each genome's NEAT key,
  fitness and an md5 of its full serialised genome, generation by generation. It is
  read-only.

  ```bash
  uv run python experiments/10seed/verify_prefix_reproduction.py heart heart100     # 100,000/100,000 OK
  uv run python experiments/10seed/verify_prefix_reproduction.py churn100 churn200  # 250,000/250,000 OK
  ```

  Results, re-run 2026-11-06:
  - Heart-100's first 40 generations reproduce Heart for all 10 seeds. That holds across the
    `9259bf1` code change: Heart ran on `0fd368e`, Heart-100 on `4126fe2`.
  - churn200's first 100 generations reproduce churn100 for all 10 seeds. That holds across
    five different commits.
- **Stored fitness check:** `compute_runs.py` recomputes each best genome's train AUC from
  scratch and checks it against the stored fitness (`fitness_match_1e-6` in `runs*.csv`).
  This is true for every run.

Bit-identity is only expected on the same platform and library versions. Different BLAS,
torch or CPU builds can change floating-point results.

## 5. Metrics (`runs*.csv`)

Definitions are in `PREREG.md` ("Metrics") and the `compute_runs.py` docstring. Every
script below only reads the database:

```bash
uv run python experiments/10seed/compute_runs.py                    # runs.csv, runs_heart100.csv, runs_paper_reference.csv
uv run python experiments/churn/compute_churn_runs.py --prefix churn100   # ../churn/runs_churn100.csv
uv run python experiments/churn/compute_churn_runs.py --prefix churn200   # ../churn/runs_churn200.csv
```

Each script picks the best genome by training fitness. Ties go to the earliest generation,
then the lowest NEAT key.

The test-AUC CI is a percentile 95% interval from 10,000 paired bootstrap resamples, with
`RandomState(42)` index draws shared across models on a dataset.

## 6. Explanations (`explanations*.csv`, `explanations/*.json`)

Every best genome was explained by a Claude agent, one agent per genome, following
`EXPLAIN_PROTOCOL.md` (protocol v3). The protocol is the paper workflow with
function-preserving operations only:
- splits, identity nodes, consolidations and renames;
- leaf annotations with closed form, hypothesis and training-split evidence;
- compositions up to a single root containing the output.

Each agent had a 3-hour box. The orders were fixed before any structure was inspected:
- the pre-registered 30: `PREREG.md`;
- Heart-100: its PREREG amendment;
- churn: `../churn/PLAN.md`.

The tools:
- `xn.py <tool> '<json>'`: calls an ExplaNEAT MCP tool function from the shell (the same
  code as the MCP server). Every call is logged to `logs/mcp_calls.jsonl`. All operations
  land in the genome's `Explanation.operations` event stream, so an explanation can be
  replayed with `ModelStateEngine`.
- `rename_inputs.py <genome_id> <dataset_id>`: names input nodes after their features. It
  is run once, before any split.
- `check_explanation.py <genome_id> [--json]`: the automatic checks, run on the training
  split:
  - function preservation (max |Δ output|);
  - per-leaf numeric and displayed closed-form error;
  - operation counts;
  - the hierarchy check (a single root that contains the output, with every annotation
    beneath it).
- `verify_record.py <report.json>`: checks an agent's report independently, after the
  agent has finished. It re-runs `get_coverage` and `check_explanation.py`, writes the
  `verified_by_orchestrator` block, flags any disagreement, and saves
  `explanations/<label>.json`. All 60 records agreed with their agents' reports.
- `build_explanations_csv.py`: rebuilds `explanations.csv`, `explanations_heart100.csv` and
  `../churn/explanations_churn{100,200}.csv` from the JSON records, with no database. It
  was re-run on 2026-11-06 and its output was unchanged.

"Full coverage" is defined in the PREREG amendment of 2026-11-03: tool structural = 1.0,
tool compositional = 1.0, `hierarchy.complete`, and function preserved to < 1e-9.

**Mike's verification** is not done yet: `verified_by_mike` is `no` for every network.
To record a verification, add this to that network's `explanations/<label>.json`:
`"mike_verification": {"verified": true, "minutes": 25, "changes": "..."}`.
Then rerun `build_explanations_csv.py`. The script reads verification from the JSON records
and nowhere else, so a rebuild never erases it.

## 7. Figures, table and summary

```bash
uv run python experiments/10seed/make_figures.py
```

The script reads only the CSVs and `../churn/baselines.json`. It runs in seconds, with no
database. It writes:
- `figures/*.{pdf,png}`, with captions in `figures/README.md`;
- `ch7_table.tex`, which needs `\usepackage{booktabs}`;
- `summary.md`.

The outputs are byte-stable across re-runs. `ch7_table.tex` compiles cleanly, with no
overfull boxes, in an 11pt `report` with 2.5 cm margins, using `pdflatex`.
