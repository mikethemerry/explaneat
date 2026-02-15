"""Forward-pass network built from a NetworkStructure.

Unlike NeuralNeat which works with NEAT genome objects (integer node IDs),
this works with NetworkStructure objects that may contain identity nodes,
split nodes, etc. with string node IDs.
"""

from typing import Dict, List, Set

import numpy as np
import torch

from .genome_network import NetworkStructure, NodeType


class StructureNetwork:
    """Build a layered feedforward network from a NetworkStructure and run
    forward passes.  Stores per-node activations keyed by string node ID.
    """

    def __init__(self, structure: NetworkStructure):
        self.structure = structure
        self._outputs = None  # depth -> tensor
        self.node_info: Dict[str, dict] = {}  # node_id -> {depth, layer_index, activation}
        self._layers: Dict[int, dict] = {}
        self._layer_order: List[int] = []
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        nodes_by_id = {n.id: n for n in self.structure.nodes}
        enabled_conns = [c for c in self.structure.connections if c.enabled]

        input_ids = set(self.structure.input_node_ids)
        output_ids = set(self.structure.output_node_ids)
        # Map each input_node_id to its column index in the input tensor
        self._input_col_map = {
            nid: i for i, nid in enumerate(self.structure.input_node_ids)
        }

        # Detect split input nodes: nodes with type INPUT that aren't in
        # input_node_ids.  This happens when apply_split_node splits an input
        # node (e.g. -20 -> -20_a, -20_b) — the split nodes inherit
        # NodeType.INPUT but input_node_ids still lists the original.
        # Map each split input to the same tensor column as its base node.
        for node in self.structure.nodes:
            if node.type == NodeType.INPUT and node.id not in input_ids:
                base_id = self._get_base_node_id(node.id)
                if base_id and base_id in self._input_col_map:
                    self._input_col_map[node.id] = self._input_col_map[base_id]
                    input_ids.add(node.id)

        all_node_ids = {n.id for n in self.structure.nodes}

        # Adjacency ---------------------------------------------------------
        children: Dict[str, List[str]] = {nid: [] for nid in all_node_ids}
        parents: Dict[str, List[str]] = {nid: [] for nid in all_node_ids}
        for conn in enabled_conns:
            if conn.from_node in all_node_ids and conn.to_node in all_node_ids:
                children[conn.from_node].append(conn.to_node)
                parents[conn.to_node].append(conn.from_node)

        # Depth (topological) -----------------------------------------------
        depth: Dict[str, int] = {}
        for nid in input_ids:
            depth[nid] = 0

        changed = True
        max_iter = len(all_node_ids) * 2
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for nid in all_node_ids:
                if nid in input_ids:
                    continue
                parent_depths = [depth[p] for p in parents[nid] if p in depth]
                if parent_depths:
                    new_d = max(parent_depths) + 1
                    if nid not in depth or new_d > depth[nid]:
                        depth[nid] = new_d
                        changed = True

        # Valid = reachable from inputs AND backwards from outputs -----------
        reachable_forward = set(depth.keys())

        backward_reachable: Set[str] = set()
        stack = [nid for nid in output_ids if nid in reachable_forward]
        visited: Set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            backward_reachable.add(nid)
            for p in parents.get(nid, []):
                if p in reachable_forward:
                    stack.append(p)

        valid_nodes = reachable_forward & backward_reachable

        # Group by depth ----------------------------------------------------
        layers_by_depth: Dict[int, List[str]] = {}
        for nid in valid_nodes:
            d = depth[nid]
            layers_by_depth.setdefault(d, []).append(nid)

        # Deterministic ordering within each layer.
        # Input layer: preserve input_node_ids order so columns match the tensor.
        for d in layers_by_depth:
            if all(nid in input_ids for nid in layers_by_depth[d]):
                ordered = [
                    nid for nid in self.structure.input_node_ids
                    if nid in valid_nodes and depth.get(nid) == d
                ]
                for nid in layers_by_depth[d]:
                    if nid not in ordered:
                        ordered.append(nid)
                layers_by_depth[d] = ordered
            else:
                layers_by_depth[d].sort()

        # Assign node info --------------------------------------------------
        for d, node_list in layers_by_depth.items():
            for i, nid in enumerate(node_list):
                node_obj = nodes_by_id.get(nid)
                if nid in input_ids:
                    act = "input"
                elif nid in output_ids:
                    act = "sigmoid"
                elif node_obj and node_obj.activation == "identity":
                    act = "identity"
                else:
                    act = "relu"

                self.node_info[nid] = {
                    "depth": d,
                    "layer_index": i,
                    "activation": act,
                }

        # Build per-layer data -----------------------------------------------
        sorted_depths = sorted(layers_by_depth.keys())
        self._layer_order = sorted_depths

        for d in sorted_depths:
            node_list = layers_by_depth[d]
            n_nodes = len(node_list)

            is_input = all(nid in input_ids for nid in node_list)
            is_output = any(nid in output_ids for nid in node_list)

            # Which prior layers feed into this one?
            input_depths_set: Set[int] = set()
            for nid in node_list:
                for p in parents.get(nid, []):
                    if p in valid_nodes:
                        input_depths_set.add(depth[p])
            input_depths = sorted(input_depths_set)

            input_size = sum(len(layers_by_depth[dd]) for dd in input_depths)

            # Weight matrix and bias vector
            weights = np.zeros((max(input_size, 1), n_nodes), dtype=np.float64)
            bias = np.zeros(n_nodes, dtype=np.float64)
            activations = []

            for j, nid in enumerate(node_list):
                node_obj = nodes_by_id.get(nid)
                if node_obj and node_obj.bias is not None:
                    bias[j] = node_obj.bias
                activations.append(self.node_info[nid]["activation"])

            # Fill weights from connections
            for conn in enabled_conns:
                if conn.from_node not in valid_nodes or conn.to_node not in valid_nodes:
                    continue
                if depth.get(conn.to_node) != d:
                    continue
                from_d = depth.get(conn.from_node)
                if from_d not in input_depths:
                    continue

                offset = sum(
                    len(layers_by_depth[dd]) for dd in input_depths if dd < from_d
                )
                from_idx = offset + self.node_info[conn.from_node]["layer_index"]
                to_idx = self.node_info[conn.to_node]["layer_index"]
                weights[from_idx, to_idx] = conn.weight

            self._layers[d] = {
                "node_ids": node_list,
                "is_input": is_input,
                "is_output": is_output,
                "input_depths": input_depths,
                "weights": torch.tensor(weights, dtype=torch.float64),
                "bias": torch.tensor(bias, dtype=torch.float64),
                "activations": activations,
                "n_nodes": n_nodes,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_base_node_id(node_id: str) -> str | None:
        """Extract base node ID from a split node ID (e.g. '-20_a' -> '-20')."""
        if "_" not in node_id:
            return node_id
        parts = node_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isalpha():
            return parts[0]
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass storing intermediate activations.

        Args:
            x: shape (batch_size, n_inputs)

        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        self._outputs = {}

        for d in self._layer_order:
            layer = self._layers[d]

            if layer["is_input"]:
                # Select columns matching valid input nodes (phenotype may
                # prune some inputs that don't reach the output).
                col_indices = [
                    self._input_col_map[nid] for nid in layer["node_ids"]
                ]
                self._outputs[d] = x[:, col_indices]
                continue

            layer_input = torch.cat(
                [self._outputs[dd] for dd in layer["input_depths"]], dim=1
            )

            z = torch.matmul(layer_input, layer["weights"]) + layer["bias"]

            # Per-node activation
            output = torch.zeros_like(z)
            for j, act in enumerate(layer["activations"]):
                if act == "relu":
                    output[:, j] = torch.relu(z[:, j])
                elif act == "sigmoid":
                    output[:, j] = torch.sigmoid(z[:, j])
                else:  # identity / input / unknown
                    output[:, j] = z[:, j]

            self._outputs[d] = output

            if layer["is_output"]:
                out_ids = set(self.structure.output_node_ids)
                indices = [
                    j for j, nid in enumerate(layer["node_ids"]) if nid in out_ids
                ]
                return output[:, indices] if indices else output

        # Fallback: return last layer
        return self._outputs[self._layer_order[-1]]

    # ------------------------------------------------------------------
    # Node activation lookup
    # ------------------------------------------------------------------

    def get_node_activation(self, node_id: str) -> np.ndarray:
        """Get activation values for a node from the last forward pass.

        Returns array of shape (n_samples,).
        """
        if self._outputs is None:
            raise RuntimeError("Must call forward() before extracting activations")

        info = self.node_info.get(node_id)
        if info is None:
            raise ValueError(f"Node '{node_id}' not found in network")

        d = info["depth"]
        idx = info["layer_index"]

        if d not in self._outputs:
            raise ValueError(f"Layer {d} not in outputs for node '{node_id}'")

        return self._outputs[d][:, idx].detach().cpu().numpy()
