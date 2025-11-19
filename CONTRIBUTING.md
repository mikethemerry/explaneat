# Contributing to ExplaNEAT

We welcome contributions to ExplaNEAT! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic familiarity with NEAT, PyTorch, and evolutionary algorithms
- Understanding of neural networks and machine learning concepts

### Areas for Contribution

We welcome contributions in several areas:

- **Core Algorithm Improvements**: Enhancements to the NEAT-backpropagation integration
- **New Features**: Additional analysis tools, fitness functions, or visualization options
- **Performance Optimizations**: Speed improvements, memory optimizations, GPU acceleration
- **Documentation**: API documentation, tutorials, examples, and guides
- **Testing**: Unit tests, integration tests, and benchmarking
- **Bug Fixes**: Identifying and fixing issues in the codebase
- **Examples**: New usage examples and application demonstrations

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/explaneat.git
cd explaneat

# Add the upstream repository
git remote add upstream https://github.com/mikethemerry/explaneat.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode with dev dependencies
uv pip install -e ".[dev]"
```

### 3. Verify Setup

```bash
# Run basic import test
python -c "from explaneat.core.explaneat import ExplaNEAT; print('Setup successful!')"

# Run existing tests (if available)
python -m pytest tests/
```

## Making Contributions

### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Branch Naming Conventions

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation improvements
- `test/description` - Test additions
- `refactor/description` - Code refactoring

### 3. Make Your Changes

Follow these guidelines when making changes:

- **Keep changes focused**: One feature or fix per branch
- **Write clear commit messages**: Describe what and why, not just how
- **Update documentation**: Update docstrings and documentation files as needed
- **Add tests**: Include tests for new functionality
- **Follow code style**: Use the project's coding conventions

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some specific preferences:

```python
# Use descriptive variable names
population_size = 50  # Good
pop_sz = 50          # Avoid

# Function and method docstrings
def calculate_density(self):
    """
    Calculate parameter density as ratio of actual to dense parameters.
    
    Returns:
        float: Parameter density between 0.0 and 1.0
    """
    pass

# Class docstrings
class ExplaNEAT:
    """
    Main class for neural network analysis and explanation.
    
    This class provides tools for analyzing evolved neural networks,
    including depth calculation, density measurement, and skippiness analysis.
    
    Args:
        genome: NEAT genome object
        config: NEAT configuration object
        neat_class: Neural network implementation class
    """
    pass
```

### Code Formatting

We use Black for code formatting:

```bash
# Format your code before committing
black explaneat/ tests/ examples/

# Check with flake8
flake8 explaneat/ tests/ examples/
```

### Type Hints

Use type hints where possible:

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

def analyze_network(genome: neat.DefaultGenome, 
                   config: neat.Config) -> Dict[str, float]:
    """Analyze network properties."""
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=explaneat

# Run specific test file
python -m pytest tests/test_core.py
```

### Writing Tests

Create tests for new functionality:

```python
import pytest
import numpy as np
from explaneat.core.explaneat import ExplaNEAT

class TestExplaNEAT:
    def test_depth_calculation(self):
        """Test network depth calculation."""
        # Setup test data
        # Create assertions
        assert explainer.depth() > 0
    
    def test_density_calculation(self):
        """Test parameter density calculation."""
        # Test implementation
        assert 0.0 <= explainer.density() <= 1.0
```

## Documentation

### Docstring Style

Use NumPy/SciPy docstring format:

```python
def retrain(self, X_train: np.ndarray, y_train: np.ndarray, 
           n_epochs: int = 100, choose_best: bool = True) -> None:
    """
    Retrain the network using backpropagation.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features with shape (n_samples, n_features)
    y_train : np.ndarray  
        Training targets with shape (n_samples, n_outputs)
    n_epochs : int, default=100
        Number of training epochs
    choose_best : bool, default=True
        Whether to keep the best model based on validation loss
    
    Examples
    --------
    >>> explainer.net.retrain(X_train, y_train, n_epochs=50)
    """
    pass
```

### README Updates

Update the README when adding new features:
- Add to the features list
- Include usage examples
- Update API reference links

## Submitting Changes

### 1. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add skippiness calculation for network analysis"
git commit -m "Fix memory leak in BackpropPopulation training"
git commit -m "Update documentation for ExplaNEAT class methods"

# Avoid unclear messages
git commit -m "Fix bug"
git commit -m "Update code"
```

### 2. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
# Include a clear description of your changes
```

### 3. Pull Request Guidelines

Your PR should include:

- **Clear title and description**: Explain what changes you made and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update docs if you changed APIs or added features
- **Changelog entry**: Add an entry to CHANGELOG.md if appropriate

### 4. Pull Request Template

Use this template for your PR description:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated if needed
- [ ] No breaking changes (or clearly documented)
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers and answer questions
- Give credit where credit is due

### Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Documentation**: Check existing documentation before asking questions

### Recognition

Contributors are recognized in several ways:
- Listed in the contributors section
- Credited in release notes for significant contributions
- Mentioned in academic papers using the library

## Questions?

If you have questions about contributing:

1. Check the existing documentation
2. Look at closed issues and PRs for similar questions
3. Open a new issue with the "question" label
4. Join community discussions

Thank you for contributing to ExplaNEAT! 🧬🤖