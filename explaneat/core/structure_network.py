"""Forward-pass network built from a NetworkStructure.

Unlike NeuralNeat which works with NEAT genome objects (integer node IDs),
this works with NetworkStructure objects that may contain identity nodes,
split nodes, etc. with string node IDs.

Supports FUNCTION nodes (collapsed annotations) by reconstructing an
AnnotationFunction from the FunctionNodeMetadata and evaluating it during
the forward pass.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .activations import get_numpy_activation
from .genome_network import FunctionNodeMetadata, NetworkStructure, NodeType


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
        # FUNCTION node support
        self._function_nodes: Dict[str, dict] = {}
        self._function_output_columns: Dict[Tuple[str, int], int] = {}
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        nodes_by_id = {n.id: n for n in self.structure.nodes}
        enabled_conns = [c for c in self.structure.connections if c.enabled]

        input_ids = set(self.structure.input_node_ids)
        output_ids = set(self.structure.output_node_ids)

        # Map each input_node_id to its column index in the input tensor.
        # Split input nodes (e.g. -20_a, -20_b from splitting -20) share
        # the same dataset column as their base node.  We assign one column
        # per unique base node and map all variants to it.
        self._input_col_map: Dict[str, int] = {}
        base_col: Dict[str, int] = {}  # base_node_id -> column index
        col_idx = 0
        for nid in self.structure.input_node_ids:
            base_id = self._get_base_node_id(nid) or nid
            if base_id not in base_col:
                base_col[base_id] = col_idx
                col_idx += 1
            self._input_col_map[nid] = base_col[base_id]

        # Also handle split input nodes that appear in the node list but
        # aren't in input_node_ids (e.g. added via later operations).
        for node in self.structure.nodes:
            if node.type == NodeType.INPUT and node.id not in input_ids:
                base_id = self._get_base_node_id(node.id)
                if base_id and base_id in base_col:
                    self._input_col_map[node.id] = base_col[base_id]
                    input_ids.add(node.id)

        # Identify FUNCTION nodes
        function_node_ids: Set[str] = set()
        for node in self.structure.nodes:
            if node.type == NodeType.FUNCTION and node.function_metadata is not None:
                function_node_ids.add(node.id)

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

        # Build AnnotationFunctions for FUNCTION nodes ----------------------
        self._build_function_nodes(nodes_by_id, valid_nodes)

        # Compute expanded column counts.  A FUNCTION node with n_outputs > 1
        # occupies n_outputs columns in the layer output tensor.
        expanded_node_list: Dict[int, List[Tuple[str, int]]] = {}
        for d, node_list in layers_by_depth.items():
            expanded = []
            for nid in node_list:
                if nid in self._function_nodes:
                    n_out = self._function_nodes[nid]["n_outputs"]
                    expanded.append((nid, n_out))
                else:
                    expanded.append((nid, 1))
            expanded_node_list[d] = expanded

        # Assign node info with expanded column positions -------------------
        for d, expanded in expanded_node_list.items():
            col = 0
            for nid, span in expanded:
                node_obj = nodes_by_id.get(nid)
                if nid in input_ids:
                    act = "input"
                elif nid in function_node_ids:
                    act = "function"
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

                if nid in self._function_nodes:
                    for out_idx in range(span):
                        self._function_output_columns[(nid, out_idx)] = col + out_idx

                col += span

        # Build per-layer data -----------------------------------------------
        sorted_depths = sorted(layers_by_depth.keys())
        self._layer_order = sorted_depths

        for d in sorted_depths:
            node_list = layers_by_depth[d]
            expanded = expanded_node_list[d]
            n_cols = sum(span for _, span in expanded)

            is_input = all(nid in input_ids for nid in node_list)
            is_output = any(nid in output_ids for nid in node_list)

            # Which prior layers feed into this one?
            input_depths_set: Set[int] = set()
            for nid in node_list:
                for p in parents.get(nid, []):
                    if p in valid_nodes:
                        input_depths_set.add(depth[p])
            input_depths = sorted(input_depths_set)

            # Input size accounting for expanded function nodes in source layers
            input_size = 0
            for dd in input_depths:
                input_size += sum(span for _, span in expanded_node_list[dd])

            # Weight matrix and bias vector
            weights = np.zeros((max(input_size, 1), n_cols), dtype=np.float64)
            bias = np.zeros(n_cols, dtype=np.float64)
            activations = []

            for nid, span in expanded:
                node_obj = nodes_by_id.get(nid)
                if nid in function_node_ids:
                    # FUNCTION nodes use placeholder activation; computed separately.
                    for _ in range(span):
                        activations.append("function")
                else:
                    if node_obj and node_obj.bias is not None:
                        col_idx = self.node_info[nid]["layer_index"]
                        bias[col_idx] = node_obj.bias
                    activations.append(self.node_info[nid]["activation"])

            # Fill weights from connections (skip connections TO function nodes)
            for conn in enabled_conns:
                if conn.from_node not in valid_nodes or conn.to_node not in valid_nodes:
                    continue
                if depth.get(conn.to_node) != d:
                    continue
                if conn.to_node in function_node_ids:
                    continue
                from_d = depth.get(conn.from_node)
                if from_d not in input_depths:
                    continue

                # Compute source column in the concatenated input
                offset = sum(
                    sum(span for _, span in expanded_node_list[dd])
                    for dd in input_depths if dd < from_d
                )
                from_col = self.node_info[conn.from_node]["layer_index"]
                # If source is a function node, use output_index to find column
                if conn.from_node in function_node_ids and conn.output_index is not None:
                    from_col = self._function_output_columns.get(
                        (conn.from_node, conn.output_index),
                        from_col,
                    )
                from_idx = offset + from_col
                to_idx = self.node_info[conn.to_node]["layer_index"]
                weights[from_idx, to_idx] = conn.weight

            # Collect function node info for this layer
            layer_function_nodes = []
            for nid, span in expanded:
                if nid in function_node_ids:
                    layer_function_nodes.append({
                        "node_id": nid,
                        "col_start": self.node_info[nid]["layer_index"],
                        "n_outputs": span,
                        "entry_nodes": self._function_nodes[nid]["entry_nodes"],
                        "ann_func": self._function_nodes[nid]["ann_func"],
                    })

            self._layers[d] = {
                "node_ids": node_list,
                "is_input": is_input,
                "is_output": is_output,
                "input_depths": input_depths,
                "weights": torch.tensor(weights, dtype=torch.float64),
                "bias": torch.tensor(bias, dtype=torch.float64),
                "activations": activations,
                "n_nodes": n_cols,
                "function_nodes": layer_function_nodes,
            }

    # ------------------------------------------------------------------
    # FUNCTION node support
    # ------------------------------------------------------------------

    def _build_function_nodes(
        self,
        nodes_by_id: Dict[str, "NetworkNode"],
        valid_nodes: Set[str],
    ) -> None:
        """Build AnnotationFunction evaluators for FUNCTION nodes.

        Reconstructs a mini NetworkStructure from the metadata (which
        stores node properties and connection weights) and creates an
        AnnotationFunction for each valid FUNCTION node.
        """
        for node in self.structure.nodes:
            if node.type != NodeType.FUNCTION or node.function_metadata is None:
                continue
            if node.id not in valid_nodes:
                continue

            meta = node.function_metadata
            ann_func = self._build_annotation_function_from_metadata(
                meta, nodes_by_id
            )

            self._function_nodes[node.id] = {
                "ann_func": ann_func,
                "entry_nodes": list(meta.input_names),
                "n_outputs": meta.n_outputs,
                "output_names": list(meta.output_names),
            }

    def _build_annotation_function_from_metadata(
        self,
        meta: FunctionNodeMetadata,
        nodes_by_id: Dict[str, "NetworkNode"],
    ) -> "AnnotationFunction":
        """Build an AnnotationFunction from FunctionNodeMetadata.

        Reconstructs a mini NetworkStructure from stored node properties
        and connection weights, then creates an AnnotationFunction.
        Entry nodes that still exist in the collapsed structure are used
        directly; removed internal nodes use properties from the metadata.
        Nested fn_* nodes are reconstructed as FUNCTION nodes using the
        stored child_function_metadata so recursive evaluation is correct.
        """
        from ..analysis.annotation_function import AnnotationFunction
        from .genome_network import (
            NetworkConnection as NC,
            NetworkNode as NN,
            NetworkStructure as NS,
        )
        from .model_state import AnnotationData

        node_props = meta.node_properties or {}
        conn_weights = meta.connection_weights or {}
        child_meta = meta.child_function_metadata or {}

        # Reconstruct subgraph nodes.  fn_* nodes that have stored child
        # metadata are reconstructed as FUNCTION nodes so AnnotationFunction
        # can dispatch to StructureNetwork for their evaluation.
        sub_nodes = []
        for nid in meta.subgraph_nodes:
            if nid in child_meta:
                # Nested function node — restore its FUNCTION type and metadata
                sub_nodes.append(NN(
                    id=nid,
                    type=NodeType.FUNCTION,
                    function_metadata=child_meta[nid],
                ))
            elif nid in nodes_by_id:
                orig = nodes_by_id[nid]
                sub_nodes.append(NN(
                    id=orig.id,
                    type=orig.type,
                    bias=orig.bias,
                    activation=orig.activation,
                    response=orig.response,
                    aggregation=orig.aggregation,
                ))
            elif nid in node_props:
                props = node_props[nid]
                sub_nodes.append(NN(
                    id=nid,
                    type=NodeType.HIDDEN,
                    bias=props.get("bias", 0.0),
                    activation=props.get("activation", "relu"),
                ))
            else:
                sub_nodes.append(NN(
                    id=nid,
                    type=NodeType.HIDDEN,
                    bias=0.0,
                    activation="relu",
                ))

        # Reconstruct subgraph connections with stored weights.
        # Connections to/from fn_* nodes carry output_index information
        # that must be preserved for correct column routing in StructureNetwork.
        sub_conns = []
        for from_n, to_n in meta.subgraph_connections:
            w = conn_weights.get((from_n, to_n), 0.0)
            # Determine output_index: connections from a fn_* child node
            # need the index to select the right output column.
            output_index = None
            if from_n in child_meta:
                child = child_meta[from_n]
                if len(child.output_names) > 1:
                    # Find which output of the child feeds this target
                    try:
                        output_index = list(child.output_names).index(to_n)
                    except ValueError:
                        output_index = 0
                else:
                    output_index = 0
            sub_conns.append(NC(
                from_node=from_n,
                to_node=to_n,
                weight=w,
                enabled=True,
                output_index=output_index,
            ))

        mini_structure = NS(
            nodes=sub_nodes,
            connections=sub_conns,
            input_node_ids=list(meta.input_names),
            output_node_ids=list(meta.output_names),
        )

        ann_data = AnnotationData(
            name=meta.annotation_name,
            hypothesis=meta.hypothesis,
            entry_nodes=list(meta.input_names),
            exit_nodes=list(meta.output_names),
            subgraph_nodes=list(meta.subgraph_nodes),
            subgraph_connections=list(meta.subgraph_connections),
        )

        return AnnotationFunction.from_structure(ann_data, mini_structure)

    @classmethod
    def _build_from_child_meta(
        cls,
        meta: "FunctionNodeMetadata",
        parent_nodes_by_id: Dict[str, "NetworkNode"],
    ) -> "StructureNetwork":
        """Build a StructureNetwork from a child FunctionNodeMetadata.

        Reconstructs the mini NetworkStructure the child function node
        represents (including any further nested fn_* nodes) and returns
        a fully built StructureNetwork ready to call forward() on.
        """
        from .genome_network import (
            NetworkConnection as NC,
            NetworkNode as NN,
            NetworkStructure as NS,
        )

        node_props = meta.node_properties or {}
        conn_weights = meta.connection_weights or {}
        child_meta_map = meta.child_function_metadata or {}

        sub_nodes = []
        for nid in meta.subgraph_nodes:
            if nid in child_meta_map:
                sub_nodes.append(NN(
                    id=nid,
                    type=NodeType.FUNCTION,
                    function_metadata=child_meta_map[nid],
                ))
            elif nid in parent_nodes_by_id:
                orig = parent_nodes_by_id[nid]
                sub_nodes.append(NN(
                    id=orig.id,
                    type=orig.type,
                    bias=orig.bias,
                    activation=orig.activation,
                    response=orig.response,
                    aggregation=orig.aggregation,
                ))
            elif nid in node_props:
                props = node_props[nid]
                sub_nodes.append(NN(
                    id=nid,
                    type=NodeType.HIDDEN,
                    bias=props.get("bias", 0.0),
                    activation=props.get("activation", "relu"),
                ))
            else:
                sub_nodes.append(NN(id=nid, type=NodeType.HIDDEN, bias=0.0, activation="relu"))

        sub_conns = []
        for from_n, to_n in meta.subgraph_connections:
            w = conn_weights.get((from_n, to_n), 0.0)
            output_index = None
            if from_n in child_meta_map:
                child = child_meta_map[from_n]
                if len(child.output_names) > 1:
                    try:
                        output_index = list(child.output_names).index(to_n)
                    except ValueError:
                        output_index = 0
                else:
                    output_index = 0
            sub_conns.append(NC(
                from_node=from_n,
                to_node=to_n,
                weight=w,
                enabled=True,
                output_index=output_index,
            ))

        mini_structure = NS(
            nodes=sub_nodes,
            connections=sub_conns,
            input_node_ids=list(meta.input_names),
            output_node_ids=list(meta.output_names),
        )
        return cls(mini_structure)

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
        # Per-node activation storage (used by function nodes to look up
        # entry node activations from earlier layers)
        self._node_activations: Dict[str, torch.Tensor] = {}

        for d in self._layer_order:
            layer = self._layers[d]

            if layer["is_input"]:
                # Select columns matching valid input nodes (phenotype may
                # prune some inputs that don't reach the output).
                col_indices = [
                    self._input_col_map[nid] for nid in layer["node_ids"]
                ]
                self._outputs[d] = x[:, col_indices]
                for j, nid in enumerate(layer["node_ids"]):
                    self._node_activations[nid] = self._outputs[d][:, j]
                continue

            layer_input = torch.cat(
                [self._outputs[dd] for dd in layer["input_depths"]], dim=1
            )

            z = torch.matmul(layer_input, layer["weights"]) + layer["bias"]

            # Per-node activation
            n_cols = layer["n_nodes"]
            output = torch.zeros((x.shape[0], n_cols), dtype=x.dtype)
            for j, act in enumerate(layer["activations"]):
                if act == "input":
                    output[:, j] = z[:, j]
                elif act == "function":
                    # Handled below by function node processing
                    pass
                else:
                    np_fn = get_numpy_activation(act)
                    z_np = z[:, j].detach().numpy()
                    output[:, j] = torch.from_numpy(np_fn(z_np).copy()).to(z.dtype)

            # Process FUNCTION nodes in this layer
            for fn_info in layer.get("function_nodes", []):
                fn_node_id = fn_info["node_id"]
                entry_nodes = fn_info["entry_nodes"]
                ann_func = fn_info["ann_func"]
                n_outputs = fn_info["n_outputs"]
                col_start = fn_info["col_start"]

                # Gather entry node activations from earlier layers
                entry_acts = []
                for en_id in entry_nodes:
                    if en_id in self._node_activations:
                        entry_acts.append(
                            self._node_activations[en_id].detach().numpy()
                        )
                    else:
                        entry_acts.append(np.zeros(x.shape[0]))

                fn_input = np.column_stack(entry_acts) if entry_acts else np.zeros((x.shape[0], 0))

                # Evaluate the annotation function
                fn_output = ann_func(fn_input)  # (batch, n_outputs)
                if fn_output.ndim == 1:
                    fn_output = fn_output.reshape(-1, 1)

                # Place outputs into the layer tensor
                for out_idx in range(n_outputs):
                    if out_idx < fn_output.shape[1]:
                        output[:, col_start + out_idx] = torch.from_numpy(
                            fn_output[:, out_idx].copy()
                        ).to(x.dtype)

            self._outputs[d] = output

            # Store per-node activations for all nodes in this layer
            for nid in layer["node_ids"]:
                info = self.node_info[nid]
                col = info["layer_index"]
                self._node_activations[nid] = output[:, col]

            if layer["is_output"]:
                out_ids = set(self.structure.output_node_ids)
                indices = [
                    self.node_info[nid]["layer_index"]
                    for nid in layer["node_ids"]
                    if nid in out_ids
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
