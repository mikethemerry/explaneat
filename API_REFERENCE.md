# API Reference

This document provides detailed API reference for the ExplaNEAT library.

## Core Module

### ExplaNEAT Class

The main class for network analysis and explanation.

```python
from explaneat.core.explaneat import ExplaNEAT
```

#### Constructor

```python
ExplaNEAT(genome, config, neat_class=NeuralNeat)
```

**Parameters:**
- `genome`: NEAT genome object representing the evolved network topology
- `config`: NEAT configuration object
- `neat_class`: Neural network implementation class (default: NeuralNeat)

#### Methods

##### `shapes()`
Returns the shapes of network layers.

**Returns:** Dictionary mapping layer indices to shape tuples

##### `n_genome_params()`
Calculate the total number of parameters in the genome.

**Returns:** Integer count of nodes + connections

##### `density()`
Calculate parameter density as the ratio of actual parameters to dense network parameters.

**Returns:** Float representing parameter density (0.0 to 1.0)

##### `depth()`
Get the depth (number of layers) of the network.

**Returns:** Integer representing network depth

##### `node_depth(nodeId)`
Get the depth of a specific node.

**Parameters:**
- `nodeId`: Node identifier

**Returns:** Integer depth of the specified node

##### `skippines()` 
Measure the "skippiness" of connections - average number of layers skipped by connections.

**Returns:** Float representing average skip distance

---

### NeuralNeat Class

PyTorch-based neural network implementation from NEAT genomes.

```python
from explaneat.core.neuralneat import NeuralNeat
```

#### Key Methods

##### `reinitialise_network_weights()`
Reinitialize all network weights randomly.

##### `retrain(X_train, y_train, n_epochs=100, choose_best=True, validate_split=0.3, random_seed=None)`
Retrain the network using backpropagation.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training targets
- `n_epochs`: Number of training epochs
- `choose_best`: Whether to keep the best model based on validation loss
- `validate_split`: Fraction of data to use for validation
- `random_seed`: Random seed for reproducibility

##### `forward(x)`
Forward pass through the network.

**Parameters:**
- `x`: Input tensor

**Returns:** Output tensor

---

## Population Module

### BackpropPopulation Class

Population class that combines NEAT evolution with backpropagation training.

```python
from explaneat.core.backproppop import BackpropPopulation
```

#### Constructor

```python
BackpropPopulation(config, X_train, y_train, criterion=None)
```

**Parameters:**
- `config`: NEAT configuration object
- `X_train`: Training features (numpy array or tensor)
- `y_train`: Training targets (numpy array or tensor)
- `criterion`: PyTorch loss function (default: BCELoss)

#### Methods

##### `run(fitness_function, n_generations, nEpochs=10)`
Run the evolutionary algorithm.

**Parameters:**
- `fitness_function`: Function to evaluate genome fitness
- `n_generations`: Maximum number of generations
- `nEpochs`: Number of backprop epochs per generation

**Returns:** Best genome found

---

## Experiment Module

### GenericExperiment Class

Comprehensive experiment management framework.

```python
from explaneat.experimenter.experiment import GenericExperiment
```

#### Constructor

```python
GenericExperiment(config_file, confirm_path_creation=True, ref_file=None)
```

**Parameters:**
- `config_file`: Path to experiment configuration JSON file
- `confirm_path_creation`: Whether to confirm before creating directories
- `ref_file`: Path to reference configuration file

#### Properties

- `logger`: Configured logger instance
- `config`: Experiment configuration dictionary
- `results_database`: ResultsDatabase instance for storing results
- `experiment_sha`: Unique experiment identifier
- `data_folder`: Path to data directory

#### Methods

##### `create_logging_header(message, width=50)`
Create a formatted logging header.

**Parameters:**
- `message`: Header message
- `width`: Width of the header line

---

## Results Module

### Result Class

Container for experiment results.

```python
from explaneat.experimenter.results import Result
```

#### Constructor

```python
Result(data, result_type, experiment_name, dataset_name, experiment_sha, iteration, meta=None)
```

**Parameters:**
- `data`: Result data (can be any serializable type)
- `result_type`: String identifying the type of result
- `experiment_name`: Name of the experiment
- `dataset_name`: Name of the dataset used
- `experiment_sha`: Unique experiment identifier
- `iteration`: Iteration number
- `meta`: Additional metadata dictionary

### ResultsDatabase Class

Database for storing and retrieving experiment results.

```python
from explaneat.experimenter.results import ResultsDatabase
```

#### Constructor

```python
ResultsDatabase(file_path)
```

**Parameters:**
- `file_path`: Path to the results database file (JSON format)

#### Methods

##### `add_result(result)`
Add a result to the database.

**Parameters:**
- `result`: Result instance to add

##### `save()`
Save the database to file.

##### `load()`
Load the database from file.

---

## Data Module

### GENERIC_WRANGLER Class

Generic data wrangling utility.

```python
from explaneat.data.wranglers import GENERIC_WRANGLER
```

#### Constructor

```python
GENERIC_WRANGLER(data_path)
```

**Parameters:**
- `data_path`: Path to the data directory

#### Properties

- `train_sets_as_np`: Training data as numpy arrays (X_train, y_train)
- `test_sets_as_np`: Test data as numpy arrays (X_test, y_test)

---

## Evaluators Module

### Binary Cross Entropy Evaluator

```python
from explaneat.evaluators.evaluators import binary_cross_entropy

# Use as fitness function
winner = population.run(binary_cross_entropy, n_generations=50)
```

---

## Visualization Module

### Network Visualization

```python
from explaneat.visualization import visualize

# Create network diagram
graph = visualize.draw_net(config, genome)
print(graph.source)  # GraphViz source
```

#### `draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False, node_colors=None, fmt='svg')`

Create a GraphViz visualization of the network.

**Parameters:**
- `config`: NEAT configuration object
- `genome`: NEAT genome to visualize
- `view`: Whether to display the graph immediately
- `filename`: Output filename (optional)
- `node_names`: Dictionary of custom node names
- `show_disabled`: Whether to show disabled connections
- `prune_unused`: Whether to remove unused nodes
- `node_colors`: Dictionary of node colors
- `fmt`: Output format ('svg', 'png', etc.)

**Returns:** GraphViz Digraph object

---

## Utility Functions

### One-Hot Encoding

```python
from explaneat.core.utility import one_hot_encode

encoded = one_hot_encode(labels, num_classes=None)
```

**Parameters:**
- `labels`: Array of class labels
- `num_classes`: Number of classes (auto-detected if None)

**Returns:** One-hot encoded array

---

## Configuration Schema

ExplaNEAT experiments are configured using JSON files. Here's the expected schema:

```json
{
  "experiment": {
    "name": "my_experiment",
    "description": "Experiment description"
  },
  "random_seed": 42,
  "model": {
    "propneat": {
      "base_config_path": "path/to/neat_config.ini",
      "population_size": 150,
      "n_iterations": 5,
      "max_n_generations": 100,
      "epochs_per_generation": 10
    },
    "propneat_retrain": {
      "n_iterations": 3,
      "n_epochs": 50
    }
  },
  "data": {
    "path": "path/to/data",
    "format": "generic"
  }
}
```

---

## Error Handling

### Common Exceptions

#### `GenomeNotValidError`
```python
from explaneat.core.errors import GenomeNotValidError
```

Raised when a genome cannot be converted to a valid neural network.

---

This API reference covers the main components of ExplaNEAT. For more specific implementation details, refer to the source code and docstrings within each module.