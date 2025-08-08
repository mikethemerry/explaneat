# ExplaNEAT Documentation Overview

Welcome to the ExplaNEAT documentation! This page provides an overview of all available documentation and guides you to the right resources for your needs.

## 📚 Documentation Structure

### Getting Started
- **[README.md](README.md)** - Main project overview, features, and quick start
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Comprehensive tutorial for new users
- **[requirements.txt](requirements.txt)** - Complete list of dependencies
- **[setup.py](setup.py)** - Package installation and metadata

### API Documentation
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API reference with all classes and methods
- **[explaneat/__init__.py](explaneat/__init__.py)** - Package imports and version information

### Examples and Tutorials
- **[examples/](examples/)** - Directory with practical examples
  - **[basic_usage.py](examples/basic_usage.py)** - Basic ExplaNEAT workflow
  - **[experiment_management.py](examples/experiment_management.py)** - Advanced experiment tracking
  - **[README.md](examples/README.md)** - Examples overview and instructions

### Development and Contributing
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guidelines for contributors
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes
- **[.gitignore](.gitignore)** - Git ignore patterns for the project

## 🎯 Documentation for Different Users

### For New Users
1. Start with **[README.md](README.md)** to understand what ExplaNEAT is
2. Follow the **[GETTING_STARTED.md](GETTING_STARTED.md)** tutorial
3. Run the **[basic usage example](examples/basic_usage.py)**
4. Explore the **[examples directory](examples/)** for more use cases

### For Researchers and Power Users
1. Review the **[API_REFERENCE.md](API_REFERENCE.md)** for detailed method documentation
2. Study the **[experiment management example](examples/experiment_management.py)**
3. Check the **[reference training system](#reference-training-system)** below
4. Customize configurations based on your research needs

### For Developers and Contributors
1. Read the **[CONTRIBUTING.md](CONTRIBUTING.md)** guidelines
2. Review the **[CHANGELOG.md](CHANGELOG.md)** for recent changes
3. Set up your development environment as described in the contributing guide
4. Check the source code documentation in each module

## 🧬 Core Concepts

### ExplaNEAT Framework
ExplaNEAT combines three key concepts:

1. **NEAT Evolution**: Topology optimization through evolutionary algorithms
2. **Backpropagation Training**: Efficient weight optimization using gradients
3. **Network Analysis**: Tools for understanding evolved architectures

### Key Components

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| `ExplaNEAT` | Network analysis and explanation | [API Reference](API_REFERENCE.md#explaneat-class) |
| `BackpropPopulation` | Evolution with backprop training | [API Reference](API_REFERENCE.md#backproppopulation-class) |
| `GenericExperiment` | Experiment management framework | [API Reference](API_REFERENCE.md#genericexperiment-class) |
| `NeuralNeat` | PyTorch-based network implementation | [API Reference](API_REFERENCE.md#neuralneat-class) |
| `Result/ResultsDatabase` | Results storage and retrieval | [API Reference](API_REFERENCE.md#results-module) |

## 🔧 Reference Training System

The documentation includes a comprehensive reference example (from the original issue) that demonstrates:

```python
# Core workflow from the reference training system
from explaneat.experimenter.experiment import GenericExperiment
from explaneat.data.wranglers import GENERIC_WRANGLER
from explaneat.core.backproppop import BackpropPopulation
from explaneat.core.explaneat import ExplaNEAT
from explaneat.evaluators.evaluators import binary_cross_entropy

# 1. Set up experiment framework
experiment = GenericExperiment(args.conf_file, ref_file=args.ref_file)

# 2. Load and prepare data
wrangler = GENERIC_WRANGLER(processed_data_location)
X_train, y_train = wrangler.train_sets_as_np

# 3. Configure NEAT and create population
population = BackpropPopulation(config, X_train, y_train)

# 4. Evolve networks
winner = population.run(binary_cross_entropy, max_generations)

# 5. Analyze evolved network
explainer = ExplaNEAT(winner, config)
depth = explainer.depth()
density = explainer.density()
skippiness = explainer.skippines()

# 6. Retrain and evaluate
explainer.net.retrain(X_train, y_train, n_epochs=100)
predictions = explainer.net.forward(X_test_tensor)
```

This pattern forms the basis for most ExplaNEAT applications.

## 📖 Quick Reference

### Installation
```bash
git clone https://github.com/mikethemerry/explaneat.git
cd explaneat
pip install -r requirements.txt
pip install -e .
```

### Basic Usage
```python
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.backproppop import BackpropPopulation
# ... see examples for complete workflows
```

### Key Metrics
- **Depth**: Network layer count (`explainer.depth()`)
- **Density**: Parameter efficiency (`explainer.density()`)  
- **Skippiness**: Skip connection analysis (`explainer.skippines()`)
- **Parameters**: Total parameter count (`explainer.n_genome_params()`)

## 🔗 External Resources

### Dependencies Documentation
- [PyTorch](https://pytorch.org/docs/) - Neural network framework
- [NEAT-Python](https://neat-python.readthedocs.io/) - Base evolutionary algorithm
- [NumPy](https://numpy.org/doc/) - Numerical computing
- [Scikit-learn](https://scikit-learn.org/stable/) - Machine learning utilities
- [Matplotlib](https://matplotlib.org/stable/) - Plotting and visualization

### Related Research
ExplaNEAT is based on research in:
- NeuroEvolution of Augmenting Topologies (NEAT)
- Evolutionary neural networks
- Explainable AI and network interpretability
- Hybrid evolution-gradient optimization

## 🆘 Getting Help

### Common Questions
1. **Installation issues**: Check [GETTING_STARTED.md](GETTING_STARTED.md#common-issues-and-solutions)
2. **Usage examples**: See the [examples directory](examples/)
3. **API details**: Consult [API_REFERENCE.md](API_REFERENCE.md)
4. **Contributing**: Read [CONTRIBUTING.md](CONTRIBUTING.md)

### Support Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community interaction
- **Documentation**: This comprehensive documentation set

## 📝 Documentation Standards

This documentation follows these principles:
- **Accessibility**: Clear explanations for users at all levels
- **Completeness**: Comprehensive coverage of all features
- **Examples**: Practical code examples for key concepts
- **Maintenance**: Regular updates with new releases

---

**Happy evolving with ExplaNEAT!** 🧬🤖

For questions about this documentation or suggestions for improvements, please open an issue on GitHub.