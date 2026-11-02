"""Parity between the training engine (NeuralNeat) and StructureNetwork.

NeuralNeat applies ReLU on hidden layers and a sigmoid on the output layer,
regardless of the activation stored on the genome's output node gene.  The
evidence pipeline (StructureNetwork / ActivationExtractor / AnnotationFunction)
must compute exactly the same function on the phenotype, otherwise every
downstream evidence number describes a different model from the trained one.
"""

import random

import numpy as np
import pytest
import torch

from explaneat.analysis.activation_extractor import ActivationExtractor
from explaneat.analysis.annotation_function import AnnotationFunction
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.neuralneat import NeuralNeat
from explaneat.core.structure_network import StructureNetwork


def _make_genome(config, seed=0, n_add_node=4, n_add_conn=6):
    """Build a small evolved-looking genome: full init plus some hidden nodes."""
    random.seed(seed)
    genome = config.genome_type(1)
    genome.configure_new(config.genome_config)
    for _ in range(n_add_node):
        genome.mutate_add_node(config.genome_config)
    for _ in range(n_add_conn):
        genome.mutate_add_connection(config.genome_config)
    rng = np.random.default_rng(seed)
    for conn in genome.connections.values():
        conn.weight = float(rng.normal(0, 1.5))
        conn.enabled = True
    for node in genome.nodes.values():
        node.bias = float(rng.normal(0, 1.0))
    # Push the output pre-activation negative for many rows so that a ReLU
    # output (the bug) would clip to 0 where the trained sigmoid does not.
    genome.nodes[0].bias = -2.0
    assert genome.nodes[0].activation == "relu"  # genome gene says relu
    return genome


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_structure_network_matches_neuralneat(neat_config, seed):
    genome = _make_genome(neat_config, seed=seed)
    X = np.random.default_rng(100 + seed).normal(size=(300, 10))

    nn_out = (
        NeuralNeat(genome, neat_config)
        .forward(torch.tensor(X, dtype=torch.float64))
        .detach()
        .numpy()
        .ravel()
    )

    phenotype = ExplaNEAT(genome, neat_config).get_phenotype_network()
    sn_out = (
        StructureNetwork(phenotype)
        .forward(torch.tensor(X, dtype=torch.float64))
        .detach()
        .numpy()
        .ravel()
    )

    # The bug produced relu(z) here (many exact zeros); the trained model
    # produces sigmoid(z) which is strictly inside (0, 1).
    assert np.all(sn_out > 0) and np.all(sn_out < 1)
    np.testing.assert_allclose(sn_out, nn_out, atol=1e-9, rtol=0)


def test_activation_extractor_output_matches_neuralneat(neat_config):
    genome = _make_genome(neat_config, seed=3)
    X = np.random.default_rng(7).normal(size=(200, 10))
    nn_out = (
        NeuralNeat(genome, neat_config)
        .forward(torch.tensor(X, dtype=torch.float64))
        .detach()
        .numpy()
        .ravel()
    )
    phenotype = ExplaNEAT(genome, neat_config).get_phenotype_network()
    out_ids = list(phenotype.output_node_ids)
    _, exit_acts = ActivationExtractor.from_structure(phenotype).extract(
        X, {"entry_nodes": out_ids, "exit_nodes": out_ids}
    )
    np.testing.assert_allclose(exit_acts.ravel(), nn_out, atol=1e-9, rtol=0)


def test_annotation_function_whole_model_matches_neuralneat(neat_config):
    """An annotation whose exit is the model output computes the sigmoid output."""
    genome = _make_genome(neat_config, seed=4)
    X = np.random.default_rng(8).normal(size=(200, 10))
    nn_out = (
        NeuralNeat(genome, neat_config)
        .forward(torch.tensor(X, dtype=torch.float64))
        .detach()
        .numpy()
        .ravel()
    )
    phenotype = ExplaNEAT(genome, neat_config).get_phenotype_network()
    ann = {
        "entry_nodes": list(phenotype.input_node_ids),
        "exit_nodes": list(phenotype.output_node_ids),
        "subgraph_nodes": [n.id for n in phenotype.nodes],
        "subgraph_connections": [],
    }
    fn = AnnotationFunction.from_structure(ann, phenotype)
    np.testing.assert_allclose(fn(X).ravel(), nn_out, atol=1e-9, rtol=0)


def test_formula_renders_sigmoid_at_output(neat_config):
    """get_formula/AnnotationFunction must render what the model computes."""
    genome = _make_genome(neat_config, seed=5, n_add_node=0, n_add_conn=0)
    phenotype = ExplaNEAT(genome, neat_config).get_phenotype_network()
    ann = {
        "entry_nodes": ["-1", "-2"],
        "exit_nodes": ["0"],
        "subgraph_nodes": ["-1", "-2", "0"],
        "subgraph_connections": [["-1", "0"], ["-2", "0"]],
    }
    latex = AnnotationFunction.from_structure(ann, phenotype).to_latex(force=True)
    assert "relu" not in latex
    assert "e^{" in latex  # sigmoid renders as 1/(1+e^{-z})
