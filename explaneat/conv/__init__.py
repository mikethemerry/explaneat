"""ConvNEAT prototype: NEAT over single-channel feature-map nodes.

Self-contained research prototype (no DB, no changes to the rest of
ExplaNEAT). See experiments/conv_neat/RESULTS.md.
"""

from .genome import ConnGene, ConvGenome, ConvNeatConfig, InnovationTracker, NodeGene, minimal_genome
from .network import NaiveConvNet, build_engine, pick_device
from .propneat import PropNEATConvNet

__all__ = ["ConnGene", "ConvGenome", "ConvNeatConfig", "InnovationTracker", "NodeGene",
           "minimal_genome", "NaiveConvNet", "PropNEATConvNet", "build_engine", "pick_device"]
