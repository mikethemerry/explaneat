"""
ExplaNEAT: Explainable NeuroEvolution of Augmenting Topologies

A comprehensive suite of tools for creating explanations of neural networks 
trained using the PropNEAT algorithm, which combines NEAT evolution with 
backpropagation for efficient GPU-based training.

Key Components:
- ExplaNEAT: Main analysis and explanation class
- BackpropPopulation: NEAT population with backprop training
- GenericExperiment: Comprehensive experiment management
- Data wranglers and visualization tools

Example:
    >>> from explaneat.core.explaneat import ExplaNEAT
    >>> from explaneat.core.backproppop import BackpropPopulation
    >>> # Set up evolution and analysis...
"""

__version__ = "0.1.0"
__author__ = "Mike the Merry"
__email__ = "your-email@example.com"

# Core imports for convenience
try:
    from .core.explaneat import ExplaNEAT
    from .core.backproppop import BackpropPopulation
    from .experimenter.experiment import GenericExperiment
    from .experimenter.results import Result, ResultsDatabase
    
    __all__ = [
        "ExplaNEAT",
        "BackpropPopulation", 
        "GenericExperiment",
        "Result",
        "ResultsDatabase",
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some ExplaNEAT components not available: {e}", ImportWarning)
    __all__ = []