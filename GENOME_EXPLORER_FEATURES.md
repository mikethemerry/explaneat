# Genome Explorer Features

## Overview

The Genome Explorer system provides comprehensive tools for analyzing and visualizing NEAT evolutionary experiments. It offers three main components:

1. **Enhanced Experiment Summary** - Detailed analysis at experiment completion
2. **CLI Explorer** - Interactive command-line interface for database exploration
3. **Enhanced Visualization** - Improved plotting with ancestor tracking

## Features

### 1. Enhanced Experiment Summary

When an experiment completes, the system now provides a comprehensive summary including:

- **Basic Information**: Experiment ID, best genome ID, fitness, generation
- **Network Structure**: Nodes, connections, depth, width
- **Ancestry Analysis**: Fitness progression, network growth across generations
- **Performance Context**: Rank in generation, comparison with population
- **Next Steps**: Commands for further analysis

**Usage**: Automatically displayed at the end of `run_working_backache.py`

### 2. CLI Explorer (`genome_explorer_cli.py`)

Interactive command-line interface for exploring genome data:

#### Command Line Usage:
```bash
# List all experiments
python genome_explorer_cli.py --list

# Interactive mode
python genome_explorer_cli.py --interactive

# Direct analysis of specific experiment
python genome_explorer_cli.py --experiment-id <ID> --summary --network --evolution

# Export data
python genome_explorer_cli.py --experiment-id <ID> --export data.json
```

#### Interactive Mode Commands:
- `list` - List all experiments
- `select <experiment_id>` - Select experiment to explore
- `summary` - Show genome summary
- `network` - Show network structure
- `training` - Plot training metrics
- `ancestry [max_gen]` - Plot ancestry fitness
- `evolution [max_gen]` - Plot evolution progression
- `export [filename]` - Export genome data
- `help` - Show available commands
- `quit` - Exit

### 3. Enhanced Visualization

#### A. Training Metrics Plotting
- **Purpose**: Shows training progress (epochs) within a single genome
- **Shows**: Loss and accuracy over training epochs
- **Usage**: `explorer.plot_training_metrics()`

#### B. Ancestry Fitness Plotting
- **Purpose**: Shows evolution across generations for a specific lineage
- **Shows**: Fitness progression and network complexity evolution
- **Features**: 
  - Duplicate removal and proper sorting
  - Generation range information
  - Summary statistics
- **Usage**: `explorer.plot_ancestry_fitness(max_generations=10)`

#### C. Evolution Progression Plotting (Enhanced)
- **Purpose**: Shows population-level evolution across all generations
- **Shows**: 
  - Population fitness evolution (best, mean, std dev)
  - Population size changes
  - Cumulative fitness improvement
  - Generation-to-generation improvements
  - **NEW**: Best ancestor fitness overlay
- **Features**:
  - 4-panel comprehensive visualization
  - Ancestor fitness tracking
  - Detailed statistics
- **Usage**: `explorer.plot_evolution_progression(max_generations=50)`

## Key Improvements

### 1. Ancestor Fitness Tracking
The evolution progression plot now includes a "Best Ancestor Fitness" line that shows the fitness of the best genome that was an ancestor of the final best genome. This provides insight into:
- How the lineage of the best genome evolved
- Whether the best genome came from consistently good ancestors
- The relationship between population best and lineage best

### 2. Enhanced Data Processing
- Better duplicate removal in ancestry analysis
- Improved error handling for insufficient data
- More informative plot titles and labels
- Comprehensive summary statistics

### 3. CLI Interface
- Interactive exploration of database
- Command-line automation support
- Data export capabilities
- User-friendly help system

## Usage Examples

### Basic Analysis
```python
from explaneat.analysis.genome_explorer import GenomeExplorer

# Load best genome from experiment
explorer = GenomeExplorer.load_best_genome(experiment_id)

# Show comprehensive summary
explorer.summary()

# Visualize network
explorer.show_network(figsize=(12, 8))

# Plot training progress (epochs within genome)
explorer.plot_training_metrics()

# Plot lineage evolution (generations in ancestry)
explorer.plot_ancestry_fitness()

# Plot population evolution (all generations with ancestor tracking)
explorer.plot_evolution_progression()

# Export all data
data = explorer.export_genome_data()
```

### CLI Usage
```bash
# Start interactive exploration
python genome_explorer_cli.py --interactive

# Quick analysis of specific experiment
python genome_explorer_cli.py --experiment-id abc123 --summary --network --evolution

# Export data for external analysis
python genome_explorer_cli.py --experiment-id abc123 --export my_analysis.json
```

## File Structure

```
explaneat/
├── analysis/
│   ├── genome_explorer.py          # Main explorer class (enhanced)
│   ├── ancestry_analyzer.py        # Ancestry analysis
│   └── visualization.py            # Visualization utilities
├── run_working_backache.py         # Enhanced with summary
└── genome_explorer_cli.py          # New CLI interface
```

## Testing

Run the test script to see all features in action:

```bash
python test_genome_explorer_features.py
```

This will demonstrate:
- Enhanced summary functionality
- Improved plotting with ancestor tracking
- Network visualization
- Data export capabilities

## Next Steps

The system is now ready for:
1. **Interactive exploration** of experiment results
2. **Comparative analysis** across different experiments
3. **Data export** for external analysis tools
4. **Automated reporting** of experiment outcomes

The enhanced ancestor tracking in the evolution progression plot provides valuable insights into how the best genomes evolved and whether they came from consistently good lineages.
