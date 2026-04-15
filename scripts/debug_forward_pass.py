"""Compare forward pass outputs between NeuralNeat and StructureNetwork.

Loads the Backache genome from the database and runs the same input data
through both engines, then reports numerical differences and per-layer
diagnostics.

Usage:
    uv run python scripts/debug_forward_pass.py
"""

import sys
import uuid

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from explaneat.db import db, Experiment, Genome, Population, Dataset, DatasetSplit
from explaneat.db.serialization import deserialize_genome
from explaneat.core.config_utils import load_neat_config
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.neuralneat import NeuralNeat
from explaneat.core.structure_network import StructureNetwork


GENOME_PREFIX = "37774b2a"
EXPERIMENT_NAME = "Working Backache 50gen"


# ── Helpers ──────────────────────────────────────────────────────────────


def find_genome(session):
    """Find the genome by UUID prefix or experiment name."""
    # Try prefix match on genome UUID
    all_genomes = session.query(Genome).all()
    for g in all_genomes:
        if str(g.id).startswith(GENOME_PREFIX):
            return g

    # Fallback: best genome from the experiment
    experiment = (
        session.query(Experiment)
        .filter(Experiment.name == EXPERIMENT_NAME)
        .first()
    )
    if not experiment:
        print(f"ERROR: Experiment '{EXPERIMENT_NAME}' not found")
        sys.exit(1)

    population = (
        session.query(Population)
        .filter(Population.experiment_id == experiment.id)
        .order_by(Population.generation.desc())
        .first()
    )
    if not population:
        print("ERROR: No populations found for experiment")
        sys.exit(1)

    genome = (
        session.query(Genome)
        .filter(Genome.population_id == population.id)
        .order_by(Genome.fitness.desc().nullslast())
        .first()
    )
    if not genome:
        print("ERROR: No genomes found")
        sys.exit(1)

    return genome


def load_dataset_split(session, experiment):
    """Load the dataset split linked to the experiment."""
    split = (
        session.query(DatasetSplit)
        .filter(DatasetSplit.experiment_id == experiment.id)
        .first()
    )
    if not split:
        print("ERROR: No dataset split found for experiment")
        sys.exit(1)

    dataset = session.query(Dataset).filter_by(id=split.dataset_id).first()
    if not dataset:
        print("ERROR: Dataset not found")
        sys.exit(1)

    data = dataset.get_data()
    if data is None:
        print("ERROR: Dataset has no stored data")
        sys.exit(1)

    X_full, y_full = data

    # Use test split
    indices = split.test_indices
    if not indices:
        indices = split.train_indices
    if not indices:
        print("ERROR: No split indices found")
        sys.exit(1)

    X = X_full[indices]
    y = y_full[indices]
    return X, y, dataset


def infer_num_classes(y, db_num_classes):
    """Infer num_classes from data if not set in DB (matches API logic)."""
    if db_num_classes is not None:
        return db_num_classes
    y_flat = y.ravel()
    if np.all(y_flat == y_flat.astype(int)):
        n_unique = len(np.unique(y_flat.astype(int)))
        if n_unique <= 20:
            return n_unique
    return None


def print_separator(title):
    """Print a visual section separator."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def build_nn_node_to_layer_col(nn_model):
    """Build mapping from NEAT node ID to (layer_id, col_index) for NeuralNeat."""
    result = {}
    for layer_id in range(nn_model.n_layers):
        layer = nn_model.layers[layer_id]
        for node_id, node in layer["nodes"].items():
            result[node_id] = (layer_id, node["layer_index"])
    return result


def build_sn_node_to_depth_col(struct_net):
    """Build mapping from string node ID to (depth, col_index) for StructureNetwork."""
    result = {}
    for nid, info in struct_net.node_info.items():
        result[nid] = (info["depth"], info["layer_index"])
    return result


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    db.init_db()

    with db.session_scope() as session:
        # ── 1. Load genome ───────────────────────────────────────────────
        print_separator("1. Loading Genome")
        genome_db = find_genome(session)
        print(f"Genome UUID: {genome_db.id}")
        print(f"Genome ID (NEAT): {genome_db.genome_id}")
        print(f"Fitness: {genome_db.fitness}")

        population = session.query(Population).filter_by(
            id=genome_db.population_id
        ).first()
        experiment = session.query(Experiment).filter_by(
            id=population.experiment_id
        ).first()
        print(f"Experiment: {experiment.name}")
        print(f"Generation: {population.generation}")

        # ── 2. Load config and NEAT genome ───────────────────────────────
        print_separator("2. Building NEAT Config")
        config = load_neat_config(
            experiment.neat_config_text or "",
            experiment.config_json,
        )
        neat_genome = deserialize_genome(genome_db.genome_data, config)
        print(f"Input keys:  {config.genome_config.input_keys}")
        print(f"Output keys: {config.genome_config.output_keys}")
        print(f"Genome nodes: {sorted(neat_genome.nodes.keys())}")
        print(f"Genome connections: {len(neat_genome.connections)} total, "
              f"{sum(1 for c in neat_genome.connections.values() if c.enabled)} enabled")

        # ── 3. Load dataset ──────────────────────────────────────────────
        print_separator("3. Loading Dataset")
        X, y, dataset = load_dataset_split(session, experiment)
        num_classes = infer_num_classes(y, dataset.num_classes)
        print(f"Dataset: {dataset.name}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"y unique values: {np.unique(y)}")
        print(f"DB num_classes: {dataset.num_classes}")
        print(f"Inferred num_classes: {num_classes}")

        # ── 4. NeuralNeat forward pass ───────────────────────────────────
        print_separator("4. NeuralNeat Forward Pass")
        nn_model = NeuralNeat(neat_genome, config)
        print(f"NeuralNeat layers: {nn_model.n_layers}")
        print(f"NeuralNeat shapes: {nn_model.shapes()}")
        print(f"Valid nodes: {sorted(nn_model.valid_nodes)}")
        if nn_model.unreachable_nodes:
            print(f"Unreachable nodes: {sorted(nn_model.unreachable_nodes)}")

        x_tensor = torch.tensor(X, dtype=torch.float64)
        with torch.no_grad():
            nn_output = nn_model.forward(x_tensor)
        nn_preds = nn_output.detach().cpu().numpy().ravel()

        print(f"Output shape: {nn_output.shape}")
        print(f"Output range: [{nn_preds.min():.6f}, {nn_preds.max():.6f}]")
        print(f"Output mean:  {nn_preds.mean():.6f}")

        # Print per-layer node ordering
        print()
        print("NeuralNeat layer node ordering:")
        for layer_id in range(nn_model.n_layers):
            layer = nn_model.layers[layer_id]
            layer_type = nn_model.layer_types[layer_id]
            node_ids = sorted(layer["nodes"].keys())
            act = "relu" if layer_type == "CONNECTED" else (
                "sigmoid" if layer_type == "OUTPUT" else "input"
            )
            print(f"  Layer {layer_id} ({layer_type}, {act}): "
                  f"nodes={node_ids}, "
                  f"input_layers={layer['input_layers']}")

        # ── 5. StructureNetwork forward pass ─────────────────────────────
        print_separator("5. StructureNetwork Forward Pass")
        explainer = ExplaNEAT(neat_genome, config)
        phenotype = explainer.get_phenotype_network()

        # Show which nodes are pruned vs kept
        phenotype_node_ids = {n.id for n in phenotype.nodes}
        print(f"Phenotype nodes: {len(phenotype.nodes)} "
              f"(pruned {len([n for n in phenotype.nodes]) - len(phenotype_node_ids)} "
              f"from genotype)")
        print(f"Input nodes:  {phenotype.input_node_ids}")
        print(f"Output nodes: {phenotype.output_node_ids}")

        struct_net = StructureNetwork(phenotype)

        # Override output activation to sigmoid for binary classification
        is_binary = (
            num_classes is not None
            and num_classes == 2
            and len(phenotype.output_node_ids) == 1
        )
        if is_binary:
            print("Binary classification: overriding output activation to sigmoid")
            struct_net.override_output_activation("sigmoid")
        else:
            print(f"NOT overriding output activation (num_classes={num_classes}, "
                  f"n_outputs={len(phenotype.output_node_ids)})")

        # Show output layer activation
        for d in struct_net._layer_order:
            layer = struct_net._layers[d]
            if layer["is_output"]:
                print(f"Output layer (depth {d}) activations: {layer['activations']}")

        print()
        print("StructureNetwork layer node ordering:")
        for d in struct_net._layer_order:
            layer = struct_net._layers[d]
            role = "INPUT" if layer["is_input"] else (
                "OUTPUT" if layer["is_output"] else "HIDDEN"
            )
            print(f"  Depth {d} ({role}): "
                  f"nodes={layer['node_ids']}, "
                  f"activations={layer['activations']}, "
                  f"input_depths={layer['input_depths']}")

        with torch.no_grad():
            sn_output = struct_net.forward(x_tensor)
        sn_preds = sn_output.detach().cpu().numpy().ravel()

        print(f"\nOutput shape: {sn_output.shape}")
        print(f"Output range: [{sn_preds.min():.6f}, {sn_preds.max():.6f}]")
        print(f"Output mean:  {sn_preds.mean():.6f}")

        # ── 6. Comparison ────────────────────────────────────────────────
        print_separator("6. Output Comparison")
        diff = nn_preds - sn_preds
        abs_diff = np.abs(diff)
        print(f"Max absolute difference:  {abs_diff.max():.10f}")
        print(f"Mean absolute difference: {abs_diff.mean():.10f}")
        print(f"Std of differences:       {abs_diff.std():.10f}")

        print(f"\nFirst 10 predictions:")
        print(f"  {'Idx':>3}  {'NeuralNeat':>14}  {'StructureNet':>14}  {'Diff':>14}")
        for i in range(min(10, len(nn_preds))):
            marker = " <--" if abs(diff[i]) > 1e-6 else ""
            print(f"  {i:3d}  {nn_preds[i]:14.8f}  {sn_preds[i]:14.8f}  "
                  f"{diff[i]:14.8f}{marker}")

        # Correlation
        if len(nn_preds) > 1 and np.std(nn_preds) > 0 and np.std(sn_preds) > 0:
            correlation = np.corrcoef(nn_preds, sn_preds)[0, 1]
            print(f"\nPearson correlation: {correlation:.10f}")

        # ── 7. AUC for both ──────────────────────────────────────────────
        print_separator("7. AUC Comparison")
        y_int = y.astype(int).ravel()
        unique_classes = np.unique(y_int)

        if len(unique_classes) == 2:
            try:
                nn_auc = roc_auc_score(y_int, nn_preds)
                print(f"NeuralNeat AUC:      {nn_auc:.6f}")
            except Exception as e:
                print(f"NeuralNeat AUC error: {e}")

            try:
                sn_auc = roc_auc_score(y_int, sn_preds)
                print(f"StructureNetwork AUC: {sn_auc:.6f}")
            except Exception as e:
                print(f"StructureNetwork AUC error: {e}")

            nn_classes = (nn_preds > 0.5).astype(int)
            sn_classes = (sn_preds > 0.5).astype(int)
            nn_acc = np.mean(nn_classes == y_int)
            sn_acc = np.mean(sn_classes == y_int)
            print(f"NeuralNeat accuracy:      {nn_acc:.6f}")
            print(f"StructureNetwork accuracy: {sn_acc:.6f}")
        else:
            print(f"Not binary classification (classes: {unique_classes}), "
                  "skipping AUC")

        # ── 8. Per-layer divergence ──────────────────────────────────────
        if abs_diff.max() > 1e-8:
            print_separator("8. Per-Node Divergence Analysis")

            # Run both forward passes to capture intermediates
            nn_model._outputs = {}
            with torch.no_grad():
                nn_model.forward(x_tensor)
            nn_intermediates = {
                layer_id: output.detach().cpu().numpy()
                for layer_id, output in nn_model._outputs.items()
            }

            struct_net._outputs = {}
            with torch.no_grad():
                struct_net.forward(x_tensor)
            sn_intermediates = {
                depth: output.detach().cpu().numpy()
                for depth, output in struct_net._outputs.items()
            }

            # Build node-level activation maps
            nn_node_map = build_nn_node_to_layer_col(nn_model)
            sn_node_map = build_sn_node_to_depth_col(struct_net)

            # Find all nodes present in both engines
            nn_node_ids = set(nn_node_map.keys())
            sn_node_ids_str = set(sn_node_map.keys())
            sn_node_ids_int = {}
            for nid_str in sn_node_ids_str:
                try:
                    sn_node_ids_int[int(nid_str)] = nid_str
                except ValueError:
                    pass

            common_nodes = nn_node_ids & set(sn_node_ids_int.keys())
            print(f"Nodes in NeuralNeat: {len(nn_node_ids)}")
            print(f"Nodes in StructureNetwork: {len(sn_node_ids_str)}")
            print(f"Common nodes (comparable): {len(common_nodes)}")

            nn_only = nn_node_ids - set(sn_node_ids_int.keys())
            sn_only = set(sn_node_ids_int.keys()) - nn_node_ids
            if nn_only:
                print(f"NeuralNeat-only nodes: {sorted(nn_only)}")
            if sn_only:
                print(f"StructureNetwork-only nodes: {sorted(sn_only)}")

            # Compare activations per node
            print(f"\nPer-node activation comparison "
                  f"(first sample, all {len(common_nodes)} common nodes):")
            print(f"  {'Node':>6}  {'NNDepth':>7}  {'SNDepth':>7}  "
                  f"{'NN val[0]':>14}  {'SN val[0]':>14}  {'MaxDiff':>14}  Status")

            divergent_nodes = []
            for node_id in sorted(common_nodes):
                nn_layer, nn_col = nn_node_map[node_id]
                sn_depth, sn_col = sn_node_map[sn_node_ids_int[node_id]]

                nn_vals = nn_intermediates[nn_layer][:, nn_col]
                sn_vals = sn_intermediates[sn_depth][:, sn_col]

                max_diff = np.abs(nn_vals - sn_vals).max()
                status = "OK" if max_diff < 1e-8 else "DIFFERS"

                print(f"  {node_id:6d}  {nn_layer:7d}  {sn_depth:7d}  "
                      f"{nn_vals[0]:14.8f}  {sn_vals[0]:14.8f}  "
                      f"{max_diff:14.8f}  {status}")

                if max_diff > 1e-8:
                    divergent_nodes.append(node_id)

            if divergent_nodes:
                print(f"\nDivergent nodes: {divergent_nodes}")

                # Detailed analysis of first divergent node
                print_separator("9. Detailed Divergence: First Divergent Node")
                first_div = divergent_nodes[0]
                print(f"Analyzing node {first_div}...")

                nn_layer, nn_col = nn_node_map[first_div]
                sn_depth, sn_col = sn_node_map[sn_node_ids_int[first_div]]

                nn_vals = nn_intermediates[nn_layer][:, nn_col]
                sn_vals = sn_intermediates[sn_depth][:, sn_col]

                print(f"\n  NeuralNeat: layer {nn_layer}, col {nn_col}")
                print(f"  StructureNetwork: depth {sn_depth}, col {sn_col}")

                # Show weight column for this node in both engines
                nn_w = nn_model.weights[nn_layer].detach().cpu().numpy()[:, nn_col]
                nn_b = nn_model.biases[nn_layer].detach().cpu().numpy()[nn_col]
                sn_w_layer = struct_net._layers[sn_depth]
                sn_w = sn_w_layer["weights"].detach().cpu().numpy()[:, sn_col]
                sn_b = sn_w_layer["bias"].detach().cpu().numpy()[sn_col]

                print(f"\n  NN weight column (len={len(nn_w)}):")
                nonzero_nn = [(i, v) for i, v in enumerate(nn_w) if abs(v) > 1e-10]
                for idx, val in nonzero_nn:
                    print(f"    [{idx}] = {val:.8f}")
                print(f"  NN bias = {nn_b:.8f}")

                print(f"\n  SN weight column (len={len(sn_w)}):")
                nonzero_sn = [(i, v) for i, v in enumerate(sn_w) if abs(v) > 1e-10]
                for idx, val in nonzero_sn:
                    print(f"    [{idx}] = {val:.8f}")
                print(f"  SN bias = {sn_b:.8f}")

                # Show input layers
                nn_input_layers = nn_model.layers[nn_layer]["input_layers"]
                sn_input_depths = sn_w_layer["input_depths"]
                print(f"\n  NN input layers: {nn_input_layers}")
                print(f"  SN input depths: {sn_input_depths}")

                # Show the concatenated input for both at sample 0
                nn_cat_input = np.concatenate(
                    [nn_intermediates[ll][0, :] for ll in nn_input_layers]
                )
                sn_cat_input = np.concatenate(
                    [sn_intermediates[dd][0, :] for dd in sn_input_depths]
                )
                print(f"\n  NN concat input (sample 0, len={len(nn_cat_input)}):")
                nonzero_nn_in = [(i, v) for i, v in enumerate(nn_cat_input) if abs(v) > 1e-10]
                for idx, val in nonzero_nn_in:
                    print(f"    [{idx}] = {val:.8f}")
                print(f"\n  SN concat input (sample 0, len={len(sn_cat_input)}):")
                nonzero_sn_in = [(i, v) for i, v in enumerate(sn_cat_input) if abs(v) > 1e-10]
                for idx, val in nonzero_sn_in:
                    print(f"    [{idx}] = {val:.8f}")

                # Compute z = input @ weights + bias for both
                nn_z = nn_cat_input @ nn_w + nn_b
                sn_z = sn_cat_input @ sn_w + sn_b
                print(f"\n  NN z (pre-activation) = {nn_z:.8f}")
                print(f"  SN z (pre-activation) = {sn_z:.8f}")
                print(f"  NN output = {nn_vals[0]:.8f}")
                print(f"  SN output = {sn_vals[0]:.8f}")

                # Check if the divergence is in the input, weights, or activation
                input_diff = np.abs(nn_cat_input - sn_cat_input).max() if len(nn_cat_input) == len(sn_cat_input) else -1
                weight_diff = np.abs(nn_w - sn_w).max() if len(nn_w) == len(sn_w) else -1
                bias_diff = abs(nn_b - sn_b)
                print(f"\n  Input diff (max): {input_diff:.10f}")
                print(f"  Weight diff (max): {weight_diff:.10f}")
                print(f"  Bias diff: {bias_diff:.10f}")

                if input_diff < 1e-8 and weight_diff < 1e-8 and bias_diff < 1e-8:
                    print("  Inputs, weights, and biases all match -- "
                          "divergence is in activation function application")
                elif input_diff > 1e-8:
                    print("  CAUSE: Concatenated inputs differ between engines")
                    if len(nn_cat_input) != len(sn_cat_input):
                        print(f"  Input lengths differ: "
                              f"NN={len(nn_cat_input)} vs SN={len(sn_cat_input)}")
                elif weight_diff > 1e-8:
                    print("  CAUSE: Weight values differ between engines")
                elif bias_diff > 1e-8:
                    print("  CAUSE: Bias values differ between engines")
            else:
                print("\nAll individual node activations match.")
                print("Divergence must be in output layer assembly or "
                      "activation function.")

                # Check the output layer specifically
                print_separator("9. Output Layer Analysis")
                nn_out_layer = nn_model.n_layers - 1
                sn_out_depth = struct_net._layer_order[-1]

                nn_out_type = nn_model.layer_types[nn_out_layer]
                sn_out_acts = struct_net._layers[sn_out_depth]["activations"]
                print(f"NeuralNeat output layer type: {nn_out_type}")
                print(f"StructureNetwork output activations: {sn_out_acts}")

                nn_out_z = nn_intermediates[nn_out_layer][0, 0]
                sn_out_z = sn_intermediates[sn_out_depth][0, 0]
                print(f"\nNN output (sample 0): {nn_out_z:.8f}")
                print(f"SN output (sample 0): {sn_out_z:.8f}")

                if nn_out_type == "OUTPUT":
                    print("NeuralNeat applies sigmoid to output layer")
                if "sigmoid" in sn_out_acts:
                    print("StructureNetwork applies sigmoid to output layer")
                elif "relu" in sn_out_acts:
                    print("StructureNetwork applies RELU to output layer "
                          "(MISMATCH: should be sigmoid for binary classification)")
        else:
            print_separator("8. No Divergence Detected")
            print("Outputs match within 1e-8. No further analysis needed.")

        # ── Summary ──────────────────────────────────────────────────────
        print_separator("Summary")
        match_status = "MATCH" if abs_diff.max() < 1e-6 else "MISMATCH"
        print(f"Result: {match_status}")
        print(f"Max difference: {abs_diff.max():.10f}")
        if abs_diff.max() >= 1e-6:
            print("The two engines produce DIFFERENT outputs for the same input.")
            print("See the per-node analysis above for root cause.")
        else:
            print("The two engines produce effectively IDENTICAL outputs.")


if __name__ == "__main__":
    main()
