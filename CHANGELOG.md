# Changelog

All notable changes to the ExplaNEAT project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-XX

### Added
- Initial release of ExplaNEAT library
- Core ExplaNEAT class for network analysis and explanation
- BackpropPopulation for combining NEAT evolution with backpropagation
- NeuralNeat class for PyTorch-based neural network implementation
- GenericExperiment framework for comprehensive experiment management
- Results database system for storing and retrieving experiment results
- Data wrangling utilities for preprocessing and data management
- Binary cross-entropy evaluator for fitness assessment
- Network visualization using GraphViz
- Comprehensive API documentation
- Getting started guide with examples
- Setup and installation scripts
- Requirements specification

### Features
- Network topology evolution using NEAT algorithm
- GPU-accelerated backpropagation training
- Network analysis metrics:
  - Depth calculation
  - Parameter density measurement
  - Skippiness analysis (skip connection patterns)
  - Parameter counting
- Network weight reinitialization and retraining
- Experiment tracking with metadata
- Results serialization and storage
- Data preprocessing utilities
- Network topology visualization

### Documentation
- Complete README with installation and usage instructions
- API reference documentation
- Getting started tutorial
- Example scripts demonstrating key features
- Setup and configuration guides

### Dependencies
- PyTorch (>=1.9.0) for neural network operations
- NEAT-Python (>=0.92) for evolutionary algorithms
- NumPy (>=1.19.0) for numerical computations
- Pandas (>=1.3.0) for data manipulation
- Scikit-learn (>=1.0.0) for machine learning utilities
- Matplotlib (>=3.3.0) for visualization
- GraphViz (>=0.16) for network topology diagrams
- Additional utilities for experiment management

## [Unreleased]

### Planned Features
- Enhanced visualization options
- Additional fitness functions
- Multi-objective optimization support
- Hyperparameter optimization utilities
- Integration with popular ML frameworks
- Performance optimizations
- Extended data format support

---

## Release Notes

### Version 0.1.0

This is the initial release of ExplaNEAT, providing a complete framework for evolutionary neural network training and analysis. The library focuses on combining the strengths of NEAT (topology evolution) with modern backpropagation techniques for efficient GPU-based training.

**Key Highlights:**
- Full integration between NEAT and PyTorch
- Comprehensive experiment management system
- Built-in network analysis and explanation tools
- Extensive documentation and examples
- Production-ready codebase with proper error handling

**Getting Started:**
New users should begin with the Getting Started guide and run the provided examples to familiarize themselves with the library's capabilities.

**Breaking Changes:**
This is the initial release, so no breaking changes apply.

**Migration Guide:**
N/A - Initial release

---

For detailed changes in each release, see the Git commit history and tagged releases.