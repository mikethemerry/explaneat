# Getting Started with ExplaNEAT

This guide will help you get up and running with ExplaNEAT quickly.

## What is ExplaNEAT?

ExplaNEAT (Explainable NeuroEvolution of Augmenting Topologies) is a Python library that combines the power of evolutionary neural networks with modern gradient-based training techniques. It provides:

- **Evolution-based topology optimization** using NEAT
- **GPU-accelerated backpropagation** for efficient weight training
- **Network analysis tools** for understanding evolved architectures
- **Comprehensive experiment management** for research workflows

## Prerequisites

Before installing ExplaNEAT, ensure you have:

- Python 3.8 or higher
- A CUDA-capable GPU (recommended for performance)
- Git (for installation from source)

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/mikethemerry/explaneat.git
cd explaneat

# Create and activate a virtual environment
python -m venv explaneat-env
source explaneat-env/bin/activate  # On Windows: explaneat-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install ExplaNEAT in development mode
pip install -e .
```

### Option 2: Install from PyPI (When Available)

```bash
pip install explaneat
```

## Verify Installation

Test that ExplaNEAT is properly installed:

```python
import explaneat
from explaneat.core.explaneat import ExplaNEAT
print("ExplaNEAT installed successfully!")
```

## Your First ExplaNEAT Example

Let's create a simple example that evolves a neural network for binary classification:

### 1. Create a NEAT Configuration File

Save this as `simple_config.cfg`:

```ini
[NEAT]
fitness_criterion     = max
fitness_threshold     = 3.9
pop_size              = 50
reset_on_extinction   = False

[DefaultGenome]
num_inputs            = 4
num_outputs           = 1
num_hidden            = 0
initial_connection    = full_direct
connection_add_prob   = 0.5
connection_delete_prob = 0.5
node_add_prob         = 0.2
node_delete_prob      = 0.2

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

### 2. Create Your First Script

Save this as `first_example.py`:

```python
import numpy as np
import torch
import neat
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.backproppop import BackpropPopulation
from explaneat.evaluators.evaluators import binary_cross_entropy

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Generate sample data
X, y = make_classification(n_samples=500, n_features=4, n_redundant=0, 
                          n_informative=4, n_clusters_per_class=1, 
                          random_state=42)

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1).astype(np.float32)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Load NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'simple_config.cfg')

# Create population
population = BackpropPopulation(config, X, y)

# Add progress reporter
population.add_reporter(neat.StdOutReporter(True))

# Evolve the network
print("Starting evolution...")
winner = population.run(binary_cross_entropy, n_generations=20, nEpochs=5)

# Analyze the evolved network
explainer = ExplaNEAT(winner, config)
print(f"\nNetwork Analysis:")
print(f"Depth: {explainer.depth()}")
print(f"Parameters: {explainer.n_genome_params()}")
print(f"Density: {explainer.density():.4f}")
print(f"Skippiness: {explainer.skippines():.4f}")

# Make predictions
X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    predictions = explainer.net.forward(X_tensor)
    accuracy = torch.mean((predictions > 0.5).float() == torch.tensor(y)).item()

print(f"Training accuracy: {accuracy:.4f}")
print("Evolution completed successfully!")
```

### 3. Run Your Example

```bash
python first_example.py
```

You should see evolution progress and final network statistics!

## Understanding the Output

The evolution process will show:
- **Generation progress**: Population fitness improvements
- **Species information**: How the population diversifies
- **Best fitness**: The fitness of the best individual each generation

The final analysis shows:
- **Depth**: Number of layers in the evolved network
- **Parameters**: Total nodes and connections
- **Density**: How sparse the network is compared to a fully connected network
- **Skippiness**: Average number of layers skipped by connections

## Next Steps

Now that you have ExplaNEAT working, explore:

### 1. Run the Provided Examples

```bash
# Basic usage patterns
python examples/basic_usage.py

# Experiment management
python examples/experiment_management.py
```

### 2. Experiment with Configurations

Try modifying the NEAT configuration:
- Change `pop_size` for different population sizes
- Adjust mutation rates to control evolution speed
- Modify network parameters for different architectures

### 3. Use Your Own Data

Replace the synthetic data with your dataset:

```python
# Load your data
X_train = np.load('your_features.npy')
y_train = np.load('your_targets.npy')

# Make sure to update num_inputs in your config file
# to match X_train.shape[1]
```

### 4. Advanced Features

Explore advanced ExplaNEAT features:
- **Network retraining**: Fine-tune evolved networks
- **Experiment management**: Track multiple experiment runs
- **Visualization**: Generate network topology diagrams
- **Custom fitness functions**: Implement domain-specific objectives

## Common Issues and Solutions

### Installation Problems

**Problem**: `ModuleNotFoundError` for dependencies
**Solution**: Ensure all requirements are installed: `pip install -r requirements.txt`

**Problem**: CUDA not available warnings
**Solution**: This is normal if no GPU is available. ExplaNEAT will use CPU automatically.

### Runtime Problems

**Problem**: Config file not found
**Solution**: Ensure the config file path is correct and the file exists

**Problem**: Slow evolution
**Solution**: Reduce population size or enable GPU acceleration if available

**Problem**: Network doesn't learn
**Solution**: Try adjusting learning rates, population size, or fitness function

## Getting Help

- **Documentation**: Check the `API_REFERENCE.md` for detailed API information
- **Examples**: Look at the `examples/` directory for more usage patterns  
- **Issues**: Open an issue on GitHub for bugs or questions
- **Community**: Join discussions on the project repository

## What's Next?

Now you're ready to:
- Explore the full ExplaNEAT API
- Run your own evolutionary experiments
- Analyze and understand evolved networks
- Integrate ExplaNEAT into your research workflow

Happy evolving! 🧬🤖