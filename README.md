# ExplaNEAT

![ExplaNEAT Logo text](image.png)

ExplaNEAT is a comprehensive suite of tools for creating explanations of neural networks trained using the PropNEAT (Propagation-based NeuroEvolution of Augmenting Topologies) algorithm. This library combines the power of evolutionary neural networks with modern gradient-based training techniques to create interpretable and efficient models.

## Features

- **Evolutionary Neural Networks**: Build networks using NEAT topology evolution
- **GPU-Accelerated Training**: Efficient backpropagation on PyTorch
- **Network Analysis**: Tools for measuring network depth, density, and "skippiness"
- **Experiment Management**: Comprehensive experiment tracking and results management
- **Visualization**: Network topology visualization and analysis
- **Data Wrangling**: Built-in data preprocessing and management utilities

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for large-scale experiments)
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer (install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`)

### Install from source

```bash
# Clone the repository
git clone https://github.com/mikethemerry/explaneat.git
cd explaneat

# Create virtual environment and install dependencies with uv
uv venv
uv pip install -e .

# Run scripts using uv run (recommended - no activation needed)
uv run python run_working_backache.py --generations=50

# Or manually activate for interactive use
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python run_working_backache.py --generations=50
```

### Using uv (when available on PyPI)

```bash
uv pip install explaneat
```

## Quick Start

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

The library requires the following key dependencies:

- **PyTorch** (>=1.9.0): Neural network operations and GPU acceleration
- **NEAT-Python** (>=0.92): Base NEAT algorithm implementation
- **NumPy** (>=1.19.0): Numerical computations
- **Pandas** (>=1.3.0): Data manipulation and analysis
- **Scikit-learn** (>=1.0.0): Machine learning utilities
- **Matplotlib** (>=3.3.0): Plotting and visualization
- **GraphViz** (>=0.16): Network topology visualization

See `pyproject.toml` for the complete list of dependencies.

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

## Citation

If you use ExplaNEAT in your research, please cite:

```bibtex
@software{explaneat2024,
  title={ExplaNEAT: Explainable Neural Evolution with Augmenting Topologies},
  author={Mike the Merry},
  year={2024},
  url={https://github.com/mikethemerry/explaneat}
}
```

## Support

For questions, bug reports, and feature requests, please open an issue on GitHub or contact the maintainers.
