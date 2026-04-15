"""Trainable forward-pass network built from a NetworkStructure.

Like StructureNetwork but as a proper PyTorch nn.Module with trainable
parameters. Supports freezing specific nodes (e.g., annotated subgraphs)
and writing trained weights back to the NetworkStructure.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from .activations import get_torch_activation
from .genome_network import NetworkStructure, NodeType


class TrainableStructureNetwork(nn.Module):
    """A trainable feedforward network built from a NetworkStructure.

    Mirrors StructureNetwork._build() and forward() logic, but uses
    nn.Parameter for weights/biases and differentiable torch activations.

    Args:
        structure: The NetworkStructure to build from.
        frozen_nodes: Optional set of node IDs whose weights/biases should
            not receive gradient updates (e.g., nodes in annotated subgraphs).
    """

    def __init__(
        self,
        structure: NetworkStructure,
        frozen_nodes: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.structure = structure
        self._frozen_nodes = frozen_nodes or set()
        self._layer_data: Dict[int, dict] = {}
        self._layer_order: List[int] = []
        self.node_info: Dict[str, dict] = {}

        # Maps for input column assignment (same logic as StructureNetwork)
        self._input_col_map: Dict[str, int] = {}

        # Connection metadata for update_structure_weights
        self._conn_map: Dict[int, List[Tuple[int, int, str, str]]] = {}
        # depth -> list of (from_idx_in_concat, to_col, from_node, to_node)
        self._node_col_map: Dict[int, List[Tuple[str, int]]] = {}
        # depth -> list of (node_id, col_index)

        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        nodes_by_id = {n.id: n for n in self.structure.nodes}
        enabled_conns = [c for c in self.structure.connections if c.enabled]

        input_ids = set(self.structure.input_node_ids)
        output_ids = set(self.structure.output_node_ids)

        # Input column mapping (same as StructureNetwork)
        base_col: Dict[str, int] = {}
        col_idx = 0
        for nid in self.structure.input_node_ids:
            base_id = self._get_base_node_id(nid) or nid
            if base_id not in base_col:
                base_col[base_id] = col_idx
                col_idx += 1
            self._input_col_map[nid] = base_col[base_id]

        for node in self.structure.nodes:
            if node.type == NodeType.INPUT and node.id not in input_ids:
                base_id = self._get_base_node_id(node.id)
                if base_id and base_id in base_col:
                    self._input_col_map[node.id] = base_col[base_id]
                    input_ids.add(node.id)

        all_node_ids = {n.id for n in self.structure.nodes}

        # Adjacency
        children: Dict[str, List[str]] = {nid: [] for nid in all_node_ids}
        parents: Dict[str, List[str]] = {nid: [] for nid in all_node_ids}
        for conn in enabled_conns:
            if conn.from_node in all_node_ids and conn.to_node in all_node_ids:
                children[conn.from_node].append(conn.to_node)
                parents[conn.to_node].append(conn.from_node)

        # Depth (topological)
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

        # Valid = reachable from inputs AND backwards from outputs
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

        # Group by depth
        layers_by_depth: Dict[int, List[str]] = {}
        for nid in valid_nodes:
            d = depth[nid]
            layers_by_depth.setdefault(d, []).append(nid)

        # Deterministic ordering
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

        # Assign node info
        for d, node_list in layers_by_depth.items():
            for col, nid in enumerate(node_list):
                node_obj = nodes_by_id.get(nid)
                if nid in input_ids:
                    act = "input"
                elif node_obj and node_obj.activation:
                    act = node_obj.activation
                elif nid in output_ids:
                    act = "sigmoid"
                else:
                    act = "relu"

                self.node_info[nid] = {
                    "depth": d,
                    "layer_index": col,
                    "activation": act,
                }

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

            # Weight and bias initialization
            weights_np = np.zeros((max(input_size, 1), n_nodes), dtype=np.float64)
            bias_np = np.zeros(n_nodes, dtype=np.float64)
            activations = []

            self._node_col_map[d] = []

            for col, nid in enumerate(node_list):
                node_obj = nodes_by_id.get(nid)
                if node_obj and node_obj.bias is not None:
                    bias_np[col] = node_obj.bias
                activations.append(self.node_info[nid]["activation"])
                self._node_col_map[d].append((nid, col))

            # Fill weights
            conn_metadata = []
            for conn in enabled_conns:
                if conn.from_node not in valid_nodes or conn.to_node not in valid_nodes:
                    continue
                if depth.get(conn.to_node) != d:
                    continue
                from_d = depth.get(conn.from_node)
                if from_d not in input_depths:
                    continue

                offset = sum(len(layers_by_depth[dd]) for dd in input_depths if dd < from_d)
                from_col = self.node_info[conn.from_node]["layer_index"]
                from_idx = offset + from_col
                to_idx = self.node_info[conn.to_node]["layer_index"]
                weights_np[from_idx, to_idx] = conn.weight
                conn_metadata.append((from_idx, to_idx, conn.from_node, conn.to_node))

            self._conn_map[d] = conn_metadata

            # Determine which parameters should be frozen
            # A layer's bias col is frozen if the corresponding node is frozen
            # Weights are frozen if the target node is frozen
            any_frozen = any(nid in self._frozen_nodes for nid in node_list)

            weights_param = nn.Parameter(
                torch.tensor(weights_np, dtype=torch.float64),
                requires_grad=not is_input,
            )
            bias_param = nn.Parameter(
                torch.tensor(bias_np, dtype=torch.float64),
                requires_grad=not is_input,
            )

            # Register as named parameters
            if not is_input:
                self.register_parameter(f"weight_{d}", weights_param)
                self.register_parameter(f"bias_{d}", bias_param)

                # Freeze individual node parameters if needed
                if any_frozen:
                    # We freeze the entire layer param but only if ALL nodes are frozen
                    # For partial freezing, we'll zero gradients in a hook
                    all_frozen = all(nid in self._frozen_nodes for nid in node_list)
                    if all_frozen:
                        weights_param.requires_grad_(False)
                        bias_param.requires_grad_(False)

            self._layer_data[d] = {
                "node_ids": node_list,
                "is_input": is_input,
                "is_output": is_output,
                "input_depths": input_depths,
                "weights": weights_param,
                "bias": bias_param,
                "activations": activations,
                "n_nodes": n_nodes,
            }

        # Register gradient hooks for partially frozen layers
        self._register_freeze_hooks()

    def _register_freeze_hooks(self):
        """Register backward hooks to zero gradients for frozen nodes."""
        for d, layer in self._layer_data.items():
            if layer["is_input"]:
                continue
            node_list = layer["node_ids"]
            frozen_cols = [
                i for i, nid in enumerate(node_list)
                if nid in self._frozen_nodes
            ]
            if not frozen_cols or len(frozen_cols) == len(node_list):
                continue  # All frozen (handled by requires_grad) or none

            # Hook to zero gradients for frozen columns
            cols = frozen_cols

            def make_hook(columns):
                def hook(grad):
                    grad = grad.clone()
                    grad[:, columns] = 0
                    return grad
                return hook

            layer["weights"].register_hook(make_hook(cols))

            def make_bias_hook(columns):
                def hook(grad):
                    grad = grad.clone()
                    grad[columns] = 0
                    return grad
                return hook

            layer["bias"].register_hook(make_bias_hook(cols))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_base_node_id(node_id: str) -> str | None:
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
        """Forward pass with differentiable torch activations.

        Args:
            x: shape (batch_size, n_inputs), dtype float64

        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        outputs: Dict[int, torch.Tensor] = {}

        for d in self._layer_order:
            layer = self._layer_data[d]

            if layer["is_input"]:
                col_indices = [
                    self._input_col_map[nid] for nid in layer["node_ids"]
                ]
                outputs[d] = x[:, col_indices]
                continue

            layer_input = torch.cat(
                [outputs[dd] for dd in layer["input_depths"]], dim=1
            )

            z = torch.matmul(layer_input, layer["weights"]) + layer["bias"]

            # Per-node activation using torch functions
            activated_cols = []
            for j, act in enumerate(layer["activations"]):
                if act == "input":
                    activated_cols.append(z[:, j:j+1])
                else:
                    torch_fn = get_torch_activation(act)
                    activated_cols.append(torch_fn(z[:, j:j+1]))

            output = torch.cat(activated_cols, dim=1)
            outputs[d] = output

            if layer["is_output"]:
                out_ids = set(self.structure.output_node_ids)
                indices = [
                    self.node_info[nid]["layer_index"]
                    for nid in layer["node_ids"]
                    if nid in out_ids
                ]
                return output[:, indices] if indices else output

        # Fallback: return last layer
        return outputs[self._layer_order[-1]]

    # ------------------------------------------------------------------
    # Weight updates
    # ------------------------------------------------------------------

    def update_structure_weights(self) -> Dict[str, Dict]:
        """Copy trained weights and biases back to the NetworkStructure.

        Returns a dict with 'weight_updates' and 'bias_updates' suitable
        for storing as a 'retrain' operation.
        """
        weight_updates: Dict[str, float] = {}  # "from,to" -> new_weight
        bias_updates: Dict[str, float] = {}

        for d, layer in self._layer_data.items():
            if layer["is_input"]:
                continue

            weights_tensor = layer["weights"].detach()
            bias_tensor = layer["bias"].detach()

            # Update connection weights
            for from_idx, to_idx, from_node, to_node in self._conn_map.get(d, []):
                new_weight = float(weights_tensor[from_idx, to_idx])
                # Find the connection and update it
                for conn in self.structure.connections:
                    if conn.from_node == from_node and conn.to_node == to_node:
                        conn.weight = new_weight
                        break
                weight_updates[f"{from_node},{to_node}"] = new_weight

            # Update biases
            for nid, col in self._node_col_map.get(d, []):
                new_bias = float(bias_tensor[col])
                node = self.structure.get_node_by_id(nid)
                if node is not None and node.bias is not None:
                    node.bias = new_bias
                    bias_updates[nid] = new_bias

        return {
            "weight_updates": weight_updates,
            "bias_updates": bias_updates,
        }
