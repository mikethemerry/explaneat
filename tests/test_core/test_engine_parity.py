"""Parity between the training engine (NeuralNeat) and StructureNetwork.

NeuralNeat applies ReLU on hidden layers and its configured output activation
(sigmoid by default; identity for a Cox log-risk, see the second half of this
file) on the output layer, regardless of the activation stored on the genome's
output node gene.  The
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


# --- Identity output activation (Cox log-risk, explaneat.survival) -----------
#
# A survival model's NeuralNeat applies NO output squashing: the output is the
# log-risk g(x).  Every engine that reproduces the trained function must follow
# the configured output activation, or formulas and evidence describe a
# sigmoid-squashed model that was never trained.


def _identity_config(neat_config):
    import copy

    from explaneat.core.neuralneat import set_output_activation

    cfg = copy.deepcopy(neat_config)
    set_output_activation(cfg, "identity")
    return cfg


def _nn_out(genome, config, X):
    return (
        NeuralNeat(genome, config)
        .forward(torch.tensor(X, dtype=torch.float64))
        .detach()
        .numpy()
        .ravel()
    )


def test_output_activation_default_is_sigmoid_and_metadata_unchanged(neat_config):
    from explaneat.core.neuralneat import get_output_activation

    genome = _make_genome(neat_config, seed=0)
    assert get_output_activation(neat_config) == "sigmoid"
    assert NeuralNeat(genome, neat_config).output_activation == "sigmoid"
    ph = ExplaNEAT(genome, neat_config).get_phenotype_network()
    assert "output_activation" not in ph.metadata  # default rows unchanged
    assert StructureNetwork(ph).model_output_activation == "sigmoid"


def test_identity_output_neuralneat_is_linear(neat_config):
    cfg = _identity_config(neat_config)
    genome = _make_genome(neat_config, seed=1)
    X = np.random.default_rng(1).normal(size=(300, 10))
    lin = _nn_out(genome, cfg, X)
    sig = _nn_out(genome, neat_config, X)
    # identity output is exactly the pre-activation of the sigmoid model
    np.testing.assert_allclose(1.0 / (1.0 + np.exp(-lin)), sig, atol=1e-12, rtol=0)
    assert lin.min() < 0  # not squashed into (0, 1)
    # an explicit argument overrides the config
    explicit = NeuralNeat(genome, neat_config, output_activation="identity")
    np.testing.assert_array_equal(
        explicit.forward(torch.tensor(X, dtype=torch.float64)).detach().numpy().ravel(), lin)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_identity_structure_network_matches_neuralneat(neat_config, seed):
    cfg = _identity_config(neat_config)
    genome = _make_genome(neat_config, seed=seed)
    X = np.random.default_rng(100 + seed).normal(size=(300, 10))
    nn_out = _nn_out(genome, cfg, X)
    ph = ExplaNEAT(genome, cfg).get_phenotype_network()
    assert ph.metadata["output_activation"] == "identity"
    out_nodes = [n for n in ph.nodes if n.id in ph.output_node_ids]
    assert all(n.activation == "identity" for n in out_nodes)
    sn = StructureNetwork(ph)
    assert sn.model_output_activation == "identity"
    np.testing.assert_allclose(
        sn.forward(torch.tensor(X, dtype=torch.float64)).detach().numpy().ravel(),
        nn_out, atol=1e-9, rtol=0)
    # the evidence tools' override keeps the model's activation
    sn.override_output_activation(sn.model_output_activation)
    sn.override_hidden_activation("relu")
    sn_out = sn.forward(torch.tensor(X, dtype=torch.float64)).detach().numpy().ravel()
    np.testing.assert_allclose(sn_out, nn_out, atol=1e-9, rtol=0)


def test_identity_activation_extractor_and_annotation_function(neat_config):
    cfg = _identity_config(neat_config)
    genome = _make_genome(neat_config, seed=3)
    X = np.random.default_rng(7).normal(size=(200, 10))
    nn_out = _nn_out(genome, cfg, X)
    ph = ExplaNEAT(genome, cfg).get_phenotype_network()
    out_ids = list(ph.output_node_ids)
    _, exit_acts = ActivationExtractor.from_structure(ph).extract(
        X, {"entry_nodes": out_ids, "exit_nodes": out_ids})
    np.testing.assert_allclose(exit_acts.ravel(), nn_out, atol=1e-9, rtol=0)

    ann = {
        "entry_nodes": list(ph.input_node_ids),
        "exit_nodes": out_ids,
        "subgraph_nodes": [n.id for n in ph.nodes],
        "subgraph_connections": [],
    }
    fn = AnnotationFunction.from_structure(ann, ph)
    np.testing.assert_allclose(fn(X).ravel(), nn_out, atol=1e-9, rtol=0)


def test_identity_survives_model_state_replay(neat_config):
    from explaneat.core.model_state import ModelStateEngine

    cfg = _identity_config(neat_config)
    genome = _make_genome(neat_config, seed=2)
    X = np.random.default_rng(2).normal(size=(100, 10))
    ph = ExplaNEAT(genome, cfg).get_phenotype_network()
    engine = ModelStateEngine(ph)
    out = ph.output_node_ids[0]
    src = next(c.from_node for c in ph.connections if c.to_node == out and c.enabled)
    engine.add_operation("add_identity_node", {
        "target_node": out,
        "connections": [[src, out]],
        "new_node_id": "identity_t",
    })
    state = engine.current_state
    assert state.metadata.get("output_activation") == "identity"
    sn = StructureNetwork(state)
    sn.override_output_activation(sn.model_output_activation)
    sn.override_hidden_activation("relu")
    np.testing.assert_allclose(
        sn.forward(torch.tensor(X, dtype=torch.float64)).detach().numpy().ravel(),
        _nn_out(genome, cfg, X), atol=1e-9, rtol=0)


def test_identity_formula_has_no_sigmoid(neat_config):
    cfg = _identity_config(neat_config)
    genome = _make_genome(neat_config, seed=5, n_add_node=0, n_add_conn=0)
    ph = ExplaNEAT(genome, cfg).get_phenotype_network()
    ann = {
        "entry_nodes": ["-1", "-2"],
        "exit_nodes": ["0"],
        "subgraph_nodes": ["-1", "-2", "0"],
        "subgraph_connections": [["-1", "0"], ["-2", "0"]],
    }
    latex = AnnotationFunction.from_structure(ann, ph).to_latex(force=True)
    assert "e^{" not in latex and "relu" not in latex  # purely linear in x1, x2


def test_identity_legacy_genome_mode(neat_config):
    """AnnotationFunction built from (genome, config) follows the config."""
    cfg = _identity_config(neat_config)
    genome = _make_genome(neat_config, seed=6, n_add_node=0, n_add_conn=0)
    X = np.random.default_rng(6).normal(size=(50, 10))
    nn_out = _nn_out(genome, cfg, X)
    ins = [str(k) for k in cfg.genome_config.input_keys]
    ann = {
        "entry_nodes": ins, "exit_nodes": ["0"],
        "subgraph_nodes": ins + ["0"],
        "subgraph_connections": [[i, "0"] for i in ins],
    }
    fn = AnnotationFunction(ann, genome, cfg)
    np.testing.assert_allclose(fn(X).ravel(), nn_out, atol=1e-9, rtol=0)


def test_load_neat_config_reads_output_activation(neat_config):
    """A genome loaded from the DB is rebuilt with the trained output activation."""
    from explaneat.core.config_utils import load_neat_config
    from explaneat.core.neuralneat import get_output_activation

    from explaneat.db.serialization import serialize_population_config

    cj = serialize_population_config(neat_config)
    assert get_output_activation(load_neat_config("", cj)) == "sigmoid"
    assert get_output_activation(
        load_neat_config("", {**cj, "training": {"mask_phantom": True}})) == "sigmoid"
    cfg = load_neat_config("", {**cj, "training": {"output_activation": "identity"}})
    assert get_output_activation(cfg) == "identity"
    genome = _make_genome(neat_config, seed=0)
    ph = ExplaNEAT(genome, cfg).get_phenotype_network()
    assert ph.metadata["output_activation"] == "identity"


def test_output_activation_validation():
    from explaneat.core.neuralneat import normalise_output_activation

    assert normalise_output_activation("linear") == "identity"
    assert normalise_output_activation(None) == "sigmoid"
    with pytest.raises(ValueError):
        normalise_output_activation("softmax")
