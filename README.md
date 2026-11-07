# ExplaNEAT

ExplaNEAT is a comprehensive suite of tools for creating explanations of neural networks trained using the PropNEAT (Propagation-based NeuroEvolution of Augmenting Topologies) algorithm. This library combines the power of evolutionary neural networks with modern gradient-based training techniques to create interpretable and efficient models.

## Features

- **Evolutionary Neural Networks**: Build networks using NEAT topology evolution
- **GPU-Accelerated Training**: Efficient backpropagation on PyTorch
- **Network Analysis**: Tools for measuring network depth, density, and "skippiness"
- **Experiment Management**: Comprehensive experiment tracking and results management
- **Visualization**: Network topology visualization and analysis
- **Data Wrangling**: Built-in data preprocessing and management utilities

## Install and run

Prerequisites: [Git](https://git-scm.com), [uv](https://docs.astral.sh/uv/getting-started/installation/)
(it fetches a suitable Python, 3.11 or 3.12, if you have none) and
[Node.js LTS](https://nodejs.org) (only to build the web UI). No GPU and no
database server are needed. Windows: see [docs/install_windows.md](docs/install_windows.md).

```bash
git clone https://github.com/mikethemerry/explaneat.git
cd explaneat
uv sync --extra survival
cd web/react-explorer && npm ci && npm run build && cd ../..
uv run python -m explaneat setup     # local SQLite DB + deploys the built UI
uv run python -m explaneat demo      # hello world, under a minute on a laptop CPU
uv run python -m explaneat api       # then open http://127.0.0.1:8000
```

- `explaneat setup` creates the data directory (`~/.explaneat`, or `EXPLANEAT_HOME`),
  a SQLite database with every table, and copies `web/react-explorer/dist` into
  `explaneat/static/react_explorer` (pure Python, so it works on Windows too). It
  never migrates or writes to a Postgres database.
- `explaneat doctor` checks Python, uv, Node, torch, the database and the UI build,
  and prints a fix for anything missing.
- `explaneat demo` runs the whole pipeline on synthetic data and writes
  `demo_output/report.md` and `report.html`, with PASS/FAIL per stage; it exits
  non-zero if any stage fails. `--tiny` runs it in seconds, `--out DIR` moves it,
  `--no-db` keeps everything in memory.
- `uv run explaneat ...` works too (the package installs an `explaneat` command).

The demo's stages: (1) the SQLite database; (2) synthetic data, a classification
task with a known hinge feature `|x0|` and a survival task with a known
non-linear hazard (a Gaussian bump in x0, x1 that linear Cox cannot see), both
saved with their splits; (3) two short seeded runs per task, recorded in the DB
with ancestry (Cox partial-likelihood training for survival); (4) a scripted
explanation of each best genome (node splits, then leaf and composition
annotations through `ModelStateEngine`), checking that coverage is 1.0 and that
the explained model computes exactly the trained network's function, then
storing it on the genome and replaying it; (5) evidence and metrics (test AUC
against logistic regression and the true logit, per-feature sensitivity;
Harrell/Uno C, td-AUC, IBS and calibration against linear Cox and the true
log-risk); (6) the lineage of each best genome (`explaneat.analysis.lineage`);
(7) the report. After the demo, `explaneat api` shows the four demo experiments
in the UI.

### Database

| | How |
|---|---|
| Local (default) | `explaneat setup` creates `~/.explaneat/explaneat.db` and records it in `~/.explaneat/config.json` |
| Postgres | `export DATABASE_URL=postgresql://localhost/explaneat_dev`, then `explaneat db create-db` and `explaneat db upgrade` (Alembic) |

`DATABASE_URL` always wins, then `config.json`, then the historical default
`postgresql://localhost/explaneat_dev` (so an existing Postgres setup keeps
working without `setup`). Alembic migrations are PostgreSQL-only; `explaneat db
upgrade` refuses to run against a SQLite URL. The other `db` commands (`init`,
`drop`, `current`, `history`, `revision`, `create-db`) are unchanged. See
[DATABASE_SETUP.md](DATABASE_SETUP.md) for Postgres.

### Dependencies

`pyproject.toml` is the source of truth and `uv.lock` pins every package
(Python 3.11-3.12; torch 2.2.x with numpy 1.26). Extras: `survival`
(scikit-survival, lifelines; the demo and `doctor` expect it) and `dev` (pytest, httpx; also installed by a plain
`uv sync` as uv's default group). On Linux and Windows uv installs the CPU-only
torch build; for CUDA, change the `pytorch-cpu` index in `pyproject.toml` to
e.g. `https://download.pytorch.org/whl/cu121` and run `uv lock`.

`requirements.txt` is generated from the lock for pip users:

```bash
uv export --format requirements-txt --no-hashes --no-emit-project --no-dev --extra survival -o requirements.txt
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && pip install -e .
```

### Data governance (restricted cohorts)

For use with restricted data, e.g. the VAREANZ / PREDICT cohort under a DAP and
HDEC approval:

- **No phoning home.** Installing pulls packages from PyPI, the PyTorch index
  and npm; after that, `setup`, `demo`, `doctor` and `api` make no network
  requests and there is no telemetry. The built UI bundles every script, style
  and font (KaTeX fonts included) and loads nothing from a CDN; FastAPI's
  Swagger pages, which do load from a CDN, are off unless `EXPLANEAT_API_DOCS=1`.
  The only outbound feature is the explicit "download from PMLB" dataset action.
  The bundle check is recorded in [docs/clean_install_check.md](docs/clean_install_check.md).
- **Restricted mode.** `EXPLANEAT_RESTRICTED=1` makes the API refuse every
  route that would make an outbound call (PMLB / UCI search and download) and
  the MCP server refuse genome and dataset access; datasets flagged
  `restricted` are refused on the Claude-assisted (MCP) path unless
  `EXPLANEAT_ALLOW_ASSISTED_RESTRICTED=1` (`explaneat/governance.py`,
  `GET /api/governance`).
- **Localhost only.** `explaneat api` binds to `127.0.0.1` by default and warns
  if given another `--host`.
- **Data outside the repository.** The database and config live in
  `EXPLANEAT_HOME` (default `~/.explaneat`); `doctor` warns if it is inside the
  checkout. Demo outputs go to `--out` (default `./demo_output`, git-ignored),
  and the demo uses synthetic data only.
- See `docs/survival.md` for the cohort checklist (local DB, no rows to external
  services).

### Continuous integration

`.github/workflows/clean-install.yml` runs the commands above, `explaneat doctor`,
the tiny demo, `pytest -m smoke`, a fast test subset and an API check on
Windows, Linux and macOS. It is **manual only** (no push or PR trigger):

```bash
gh workflow run clean-install.yml --ref exp/install
gh workflow run clean-install.yml --ref exp/install -f full_demo=true   # also the full demo
gh run watch
```

Locally: `uv run pytest -m smoke` (seconds).

### Developing the UI

`cd web/react-explorer && npm run dev` serves the UI on port 3001 with `/api`
proxied to `explaneat api` on 8000. After `npm run build`, rerun `explaneat
setup` (or `explaneat setup --skip-db`) to deploy it.

## Quick Start (Python API)

Here's a basic example of how to use ExplaNEAT:

```python
import numpy as np
import torch
import neat
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.backproppop import BackpropPopulation
from explaneat.evaluators.evaluators import binary_cross_entropy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your data
X_train = np.random.randn(100, 10)  # Example data
y_train = np.random.randint(0, 2, (100, 1))

# Configure NEAT
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction, 
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-file.cfg'  # Your NEAT config file
)

# Create population
population = BackpropPopulation(config, X_train, y_train)

# Evolve the network
winner = population.run(binary_cross_entropy, max_generations=50)

# Create explainable model
explainer = ExplaNEAT(winner, config)

# Analyze the network
print(f"Network depth: {explainer.depth()}")
print(f"Network density: {explainer.density()}")
print(f"Network skippiness: {explainer.skippines()}")
print(f"Number of parameters: {explainer.n_genome_params()}")
```

## Core Components

### 1. ExplaNEAT Class

The main class for network analysis and explanation:

```python
from explaneat.core.explaneat import ExplaNEAT

# Initialize with a trained genome and config
explainer = ExplaNEAT(genome, config)

# Network analysis methods
explainer.depth()           # Get network depth
explainer.density()         # Calculate parameter density 
explainer.skippines()       # Measure skip connections
explainer.n_genome_params() # Count total parameters
```

### 2. BackpropPopulation

Manages population evolution with backpropagation:

```python
from explaneat.core.backproppop import BackpropPopulation
import torch.nn as nn

population = BackpropPopulation(
    config, 
    X_train, 
    y_train, 
    criterion=nn.BCELoss()
)
```

### 3. Generic Experiment Framework

Comprehensive experiment management:

```python
from explaneat.experimenter.experiment import GenericExperiment

experiment = GenericExperiment(
    'config.json', 
    confirm_path_creation=False, 
    ref_file='reference.json'
)

# Access experiment components
logger = experiment.logger
results_db = experiment.results_database
```

### 4. Data Wranglers

Built-in data preprocessing utilities:

```python
from explaneat.data.wranglers import GENERIC_WRANGLER

wrangler = GENERIC_WRANGLER('/path/to/data')
X_train, y_train = wrangler.train_sets_as_np
X_test, y_test = wrangler.test_sets_as_np
```

## Advanced Usage

### Network Retraining

```python
# Reinitialize network weights and retrain
explainer.net.reinitialise_network_weights()
explainer.net.retrain(
    X_train, 
    y_train,
    n_epochs=100,
    choose_best=True,
    validate_split=0.3,
    random_seed=42
)
```

### Visualization

```python
from explaneat.visualization import visualize

# Generate network visualization
network_graph = visualize.draw_net(config, genome)
print(network_graph.source)  # GraphViz source code
```

## Interactive React Explorer

- Run `python genome_explorer_cli.py --interactive` and use `ni` (or `network-interactive`) to launch the new React-based explorer. It opens a static HTML file that works over `file://` and mirrors the Pyvis features with a modern UI.
- Use `ni-py` (or `network-interactive-py`) whenever you want the legacy Pyvis rendering.
- To rebuild the frontend bundle after making changes under `web/react-explorer/`:

```bash
cd web/react-explorer && npm ci && npm run build && cd ../..
uv run python -m explaneat setup --skip-db     # copies dist/ into explaneat/static/react_explorer
```

See `docs/react_explorer_requirements.md` for the detailed feature contract and `docs/arc42/react_explorer_schema.md` for the JSON payload specification.

### Results Management

```python
from explaneat.experimenter.results import Result, ResultsDatabase

# Create and store results
result = Result(
    data=model_predictions,
    result_type="predictions",
    experiment_name="my_experiment",
    dataset_name="my_dataset",
    experiment_sha="abc123",
    iteration=0,
    meta={"accuracy": 0.95}
)

# Save to database
results_db = ResultsDatabase("results.json")
results_db.add_result(result)
results_db.save()
```

## Configuration

ExplaNEAT uses NEAT configuration files. Here's a minimal example:

```ini
[NEAT]
fitness_criterion     = max
fitness_threshold     = 3.9
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# Network parameters
num_inputs            = 10
num_outputs           = 1
num_hidden            = 0
initial_connection    = full_direct
connection_add_prob   = 0.5
connection_delete_prob = 0.5
node_add_prob         = 0.2
node_delete_prob      = 0.2

# Connection parameters
enabled_default       = True
enabled_mutate_rate   = 0.01
weight_init_mean      = 0.0
weight_init_stdev     = 1.0
weight_max_value      = 30
weight_min_value      = -30
weight_mutate_power   = 0.5
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
```

## Dependencies

See [Install and run](#install-and-run): `pyproject.toml` lists them and `uv.lock` pins them.

## Behind ExplaNEAT

The PropNEAT algorithm uses graph analysis methods to create a way of training models with NEAT using backpropagation in a way that is efficient on GPUs. Key innovations include:

1. **Hybrid Evolution**: Combines NEAT's topology evolution with gradient-based weight optimization
2. **GPU Efficiency**: Vectorized operations for population-based training
3. **Explainability**: Built-in metrics for understanding network structure and behavior
4. **Flexibility**: Supports various network architectures and training strategies

## Contributing

We welcome contributions! Please see our contributing guidelines for details on how to submit pull requests, report issues, and contribute to the codebase.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, bug reports, and feature requests, please open an issue on GitHub or contact the maintainers.
