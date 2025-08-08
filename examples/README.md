# Examples

This directory contains example scripts demonstrating various aspects of ExplaNEAT usage.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the fundamental ExplaNEAT workflow:
- Creating synthetic data
- Setting up NEAT configuration
- Evolving neural networks with BackpropPopulation
- Analyzing evolved networks with ExplaNEAT
- Network retraining
- Basic visualization

**Run with:**
```bash
python examples/basic_usage.py
```

**Key concepts covered:**
- ExplaNEAT class usage
- BackpropPopulation for evolution
- Network analysis metrics (depth, density, skippiness)
- Weight reinitialization and retraining
- Basic network visualization

### 2. Experiment Management (`experiment_management.py`)

Shows how to use the comprehensive experiment framework:
- GenericExperiment setup and configuration
- Results database management
- Experiment logging and tracking
- Result storage and retrieval

**Run with:**
```bash
python examples/experiment_management.py
```

**Key concepts covered:**
- GenericExperiment class
- Configuration file management
- Results database operations
- Experiment logging
- Metadata tracking

## Requirements

Before running the examples, ensure you have installed ExplaNEAT and its dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- The examples use synthetic data for demonstration purposes
- GPU acceleration will be used if available
- Some visualization features require GraphViz to be installed system-wide
- Experiment artifacts are created in temporary directories during example runs

## Extending the Examples

Feel free to modify these examples to:
- Use your own datasets
- Experiment with different NEAT configurations
- Test various network architectures
- Implement custom fitness functions
- Add additional analysis metrics

For more advanced usage patterns, refer to the provided reference training system in the main documentation.