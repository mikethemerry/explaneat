"""Extract callable functions from annotation subgraphs."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core.genome_network import NetworkStructure
from ..core.neuralneat import NeuralNeat, LAYER_TYPE_OUTPUT


class AnnotationFunction:
    """Callable function f: R^n -> R^m extracted from an annotation subgraph.

    Given an annotation with entry/exit nodes, this class extracts the
    weight submatrices and biases to build a standalone function representing
    the annotation's computation.

    Supports two modes:
    - Legacy: from a NeuralNeat network (raw NEAT genome)
    - Structure: from a NetworkStructure with operations applied
    """

    def __init__(
        self,
        annotation,
        genome=None,
        config=None,
        *,
        structure: Optional[NetworkStructure] = None,
    ):
        if structure is not None:
            self._mode = "structure"
            self._structure = structure
        elif genome is not None and config is not None:
            self._mode = "neat"
            self.net = NeuralNeat(genome, config)
            self.node_tracker = self.net.node_tracker
            self.layers = self.net.layers
        else:
            raise ValueError("Must provide either (genome, config) or structure=")

        # Extract annotation info (support both dict and model)
        if isinstance(annotation, dict):
            self.entry_nodes = [str(n) for n in annotation["entry_nodes"]]
            self.exit_nodes = [str(n) for n in annotation["exit_nodes"]]
            self.subgraph_nodes = [str(n) for n in annotation["subgraph_nodes"]]
            self.subgraph_connections = [
                (str(c[0]), str(c[1])) for c in annotation.get("subgraph_connections", [])
            ]
        else:
            self.entry_nodes = [str(n) for n in annotation.entry_nodes]
            self.exit_nodes = [str(n) for n in annotation.exit_nodes]
            self.subgraph_nodes = [str(n) for n in annotation.subgraph_nodes]
            self.subgraph_connections = [
                (str(c[0]), str(c[1])) for c in (annotation.subgraph_connections or [])
            ]

        self.n_inputs = len(self.entry_nodes)
        self.n_outputs = len(self.exit_nodes)

        # Build the computation sequence
        self._build_computation_graph()

    @classmethod
    def from_structure(
        cls, annotation, structure: NetworkStructure
    ) -> "AnnotationFunction":
        """Create from a NetworkStructure (with operations applied)."""
        return cls(annotation, structure=structure)

    # ------------------------------------------------------------------
    # Build computation graph
    # ------------------------------------------------------------------

    def _build_computation_graph(self):
        if self._mode == "structure":
            self._build_from_structure()
        else:
            self._build_from_neat()

    def _build_from_structure(self):
        """Build computation graph directly from NetworkStructure."""
        nodes_by_id = {n.id: n for n in self._structure.nodes}
        enabled_conns = {
            (c.from_node, c.to_node): c.weight
            for c in self._structure.connections
            if c.enabled
        }

        output_ids = set(self._structure.output_node_ids)
        subgraph_set = set(self.subgraph_nodes)

        # Build predecessors within the subgraph
        predecessors = {n: [] for n in self.subgraph_nodes}
        for from_n, to_n in self.subgraph_connections:
            if from_n in subgraph_set and to_n in subgraph_set:
                predecessors.setdefault(to_n, []).append(from_n)

        # Compute depth within subgraph (entry nodes = depth 0)
        depth = {n: 0 for n in self.entry_nodes}
        changed = True
        while changed:
            changed = False
            for n in self.subgraph_nodes:
                if n in self.entry_nodes:
                    continue
                parent_depths = [depth[p] for p in predecessors.get(n, []) if p in depth]
                if parent_depths:
                    new_d = max(parent_depths) + 1
                    if n not in depth or new_d > depth[n]:
                        depth[n] = new_d
                        changed = True

        self._node_locations = {n: (depth.get(n, 0), 0) for n in self.subgraph_nodes}
        self._exit_is_output = {n: n in output_ids for n in self.exit_nodes}

        # Order internal nodes by depth
        internal_nodes = [n for n in self.subgraph_nodes if n not in self.entry_nodes]
        internal_nodes.sort(key=lambda n: depth.get(n, 0))

        self._steps = []
        for node_str in internal_nodes:
            node_obj = nodes_by_id.get(node_str)
            bias_val = node_obj.bias if node_obj and node_obj.bias is not None else 0.0

            # Determine activation
            is_output_node = self._exit_is_output.get(node_str, False) and node_str in output_ids
            if node_obj and node_obj.activation == "identity":
                activation = "identity"
            elif is_output_node:
                activation = "sigmoid"
            else:
                activation = "relu"

            # Get input connections within subgraph
            input_node_strs = []
            weights = []
            for conn in self.subgraph_connections:
                if conn[1] == node_str:
                    input_node_strs.append(conn[0])
                    w = enabled_conns.get(conn, 0.0)
                    weights.append(w)

            self._steps.append({
                "node": node_str,
                "input_nodes": input_node_strs,
                "weights": np.array(weights, dtype=np.float64),
                "bias": float(bias_val),
                "activation": activation,
            })

    def _build_from_neat(self):
        """Build computation graph from NeuralNeat (legacy path)."""
        # Map annotation nodes to (layer_depth, layer_index) in NeuralNeat
        self._node_locations = {}  # node_str -> (depth, layer_index)
        for node_str in self.subgraph_nodes:
            resolved = self._resolve_node_id(node_str)
            if resolved is not None and resolved in self.node_tracker:
                tracker = self.node_tracker[resolved]
                self._node_locations[node_str] = (
                    tracker["depth"],
                    tracker["layer_index"],
                )

        # Determine if exit nodes are genome output nodes
        output_node_ids = set()
        for layer_id, layer in self.layers.items():
            if layer.get("is_output_layer"):
                output_node_ids.update(layer["nodes"].keys())
        self._exit_is_output = {}
        for node_str in self.exit_nodes:
            resolved = self._resolve_node_id(node_str)
            self._exit_is_output[node_str] = resolved in output_node_ids

        # Order internal nodes by depth for topological computation
        internal_nodes = [
            n for n in self.subgraph_nodes if n not in self.entry_nodes
        ]
        internal_nodes.sort(key=lambda n: self._node_locations.get(n, (0, 0))[0])

        # Get the connection map from NodeMapping for weight lookups
        self._connection_map = self.net.node_mapping.connection_map

        # Extract per-node weights and biases from NeuralNeat layers
        self._steps = []  # list of (node_str, weight_row, bias, activation)
        for node_str in internal_nodes:
            resolved = self._resolve_node_id(node_str)
            if resolved is None:
                continue
            loc = self._node_locations.get(node_str)
            if loc is None:
                continue
            depth, layer_idx = loc
            if depth not in self.layers:
                continue
            layer = self.layers[depth]

            # Determine activation
            is_output = node_str in self._exit_is_output and self._exit_is_output[node_str]
            layer_is_output = layer.get("is_output_layer", False)
            use_sigmoid = is_output and layer_is_output
            activation = "sigmoid" if use_sigmoid else "relu"

            # Extract bias
            bias_val = float(self.net.biases[depth][layer_idx].detach().cpu())

            # Extract relevant weights from input nodes in the subgraph
            input_node_strs = []
            for conn in self.subgraph_connections:
                if conn[1] == node_str:
                    input_node_strs.append(conn[0])

            weights = []
            for in_node_str in input_node_strs:
                in_resolved = self._resolve_node_id(in_node_str)
                if in_resolved is None:
                    weights.append(0.0)
                    continue

                conn_key = (in_resolved, resolved)
                if conn_key in self._connection_map:
                    layer_id, in_idx, out_idx = self._connection_map[conn_key]
                    w = float(
                        self.net.weights[layer_id][in_idx][out_idx].detach().cpu()
                    )
                    weights.append(w)
                else:
                    weights.append(0.0)

            self._steps.append({
                "node": node_str,
                "input_nodes": input_node_strs,
                "weights": np.array(weights, dtype=np.float64),
                "bias": bias_val,
                "activation": activation,
            })

    def _resolve_node_id(self, node_str: str):
        """Resolve a node ID string to the NeuralNeat node tracker key (legacy)."""
        try:
            int_id = int(node_str)
            if int_id in self.node_tracker:
                return int_id
        except ValueError:
            pass

        if "_" in node_str:
            base = node_str.split("_")[0]
            try:
                int_id = int(base)
                if int_id in self.node_tracker:
                    return int_id
            except ValueError:
                pass

        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the subgraph function.

        Args:
            x: Input array of shape (n_samples, n_inputs) or (n_inputs,)

        Returns:
            Output array of shape (n_samples, n_outputs) or (n_outputs,)
        """
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        # Map entry nodes to their activation values
        activations = {}
        for i, node_str in enumerate(self.entry_nodes):
            activations[node_str] = x[:, i]

        # Evaluate each step in topological order
        for step in self._steps:
            # Gather inputs
            inputs = np.column_stack(
                [activations.get(n, np.zeros(len(x))) for n in step["input_nodes"]]
            ) if step["input_nodes"] else np.zeros((len(x), 0))

            # Compute: weights @ inputs + bias
            if len(step["weights"]) > 0 and inputs.shape[1] > 0:
                z = inputs @ step["weights"] + step["bias"]
            else:
                z = np.full(len(x), step["bias"])

            # Apply activation
            if step["activation"] == "relu":
                activations[step["node"]] = np.maximum(0, z)
            elif step["activation"] == "identity":
                activations[step["node"]] = z
            else:  # sigmoid
                activations[step["node"]] = 1.0 / (1.0 + np.exp(-z))

        # Gather outputs
        outputs = np.column_stack(
            [activations.get(n, np.zeros(len(x))) for n in self.exit_nodes]
        )

        if single:
            return outputs[0]
        return outputs

    def to_sympy(self) -> Optional[Dict[str, "sympy.Expr"]]:
        """Extract symbolic expressions for each output.

        Returns None if the subgraph is too complex (>5 inputs or >3 internal layers).
        """
        if self.n_inputs > 5:
            return None

        # Count distinct depths in internal nodes
        depths = set()
        for step in self._steps:
            loc = self._node_locations.get(step["node"])
            if loc:
                depths.add(loc[0])
        if len(depths) > 3:
            return None

        try:
            import sympy
        except ImportError:
            return None

        # Create input symbols
        input_syms = {}
        for i, node_str in enumerate(self.entry_nodes):
            input_syms[node_str] = sympy.Symbol(f"x_{i}")

        node_exprs = dict(input_syms)

        for step in self._steps:
            # Build expression: sum(w_i * input_i) + bias
            expr = sympy.Float(step["bias"])
            for j, in_node in enumerate(step["input_nodes"]):
                w = step["weights"][j]
                if in_node in node_exprs and abs(w) > 1e-12:
                    expr += sympy.Float(w) * node_exprs[in_node]

            # Apply activation
            if step["activation"] == "relu":
                expr = sympy.Piecewise((expr, expr > 0), (0, True))
            elif step["activation"] == "identity":
                pass  # no transformation
            else:  # sigmoid
                expr = 1 / (1 + sympy.exp(-expr))

            node_exprs[step["node"]] = sympy.simplify(expr)

        # Collect output expressions
        result = {}
        for i, node_str in enumerate(self.exit_nodes):
            if node_str in node_exprs:
                result[f"y_{i}"] = node_exprs[node_str]

        return result if result else None

    def to_latex(self) -> Optional[str]:
        """Get LaTeX representation of the function.

        Returns None if sympy extraction fails or is intractable.
        """
        exprs = self.to_sympy()
        if exprs is None:
            return None

        try:
            import sympy
            parts = []
            for name, expr in exprs.items():
                parts.append(f"{name} = {sympy.latex(expr)}")
            return " \\\\ ".join(parts)
        except Exception:
            return None

    @property
    def dimensionality(self) -> Tuple[int, int]:
        return (self.n_inputs, self.n_outputs)
