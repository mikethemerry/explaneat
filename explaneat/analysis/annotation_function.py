"""Extract callable functions from annotation subgraphs."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core.activations import get_numpy_activation, get_sympy_activation
from ..core.genome_network import NetworkStructure, NodeType
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
        """Build computation graph directly from NetworkStructure.

        FUNCTION nodes within the subgraph are handled by building nested
        StructureNetwork evaluators rather than treating them as scalar nodes.
        """
        nodes_by_id = {n.id: n for n in self._structure.nodes}
        enabled_conns = {
            (c.from_node, c.to_node): c.weight
            for c in self._structure.connections
            if c.enabled
        }
        # Track output_index on connections from function nodes
        conn_output_index = {
            (c.from_node, c.to_node): c.output_index
            for c in self._structure.connections
            if c.enabled
        }

        output_ids = set(self._structure.output_node_ids)
        subgraph_set = set(self.subgraph_nodes)

        # Identify FUNCTION nodes within this subgraph
        function_node_ids = {
            n.id
            for n in self._structure.nodes
            if n.type == NodeType.FUNCTION and n.function_metadata is not None
            and n.id in subgraph_set
        }

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
        # Maps fn_node_id -> StructureNetwork for recursive evaluation
        self._nested_networks: Dict[str, object] = {}

        for node_str in internal_nodes:
            node_obj = nodes_by_id.get(node_str)

            if node_str in function_node_ids:
                # Build a nested StructureNetwork from the child's metadata
                # so we can evaluate it during __call__.
                from ..core.structure_network import StructureNetwork as SN
                child_meta = node_obj.function_metadata
                child_net = SN._build_from_child_meta(child_meta, nodes_by_id)
                self._nested_networks[node_str] = child_net

                n_outputs = child_meta.n_outputs
                input_node_strs = list(child_meta.input_names)
                self._steps.append({
                    "node": node_str,
                    "type": "function",
                    "input_nodes": input_node_strs,
                    "n_outputs": n_outputs,
                    "output_names": list(child_meta.output_names),
                    "function_metadata": child_meta,
                })
                continue

            bias_val = node_obj.bias if node_obj and node_obj.bias is not None else 0.0

            # Determine activation
            is_output_node = self._exit_is_output.get(node_str, False) and node_str in output_ids
            if node_obj and node_obj.activation:
                activation = node_obj.activation
            elif is_output_node:
                activation = "sigmoid"
            else:
                activation = "relu"

            # Get input connections within subgraph, including output_index
            # for connections from function nodes
            input_node_strs = []
            weights = []
            output_indices = []
            for conn in self.subgraph_connections:
                if conn[1] == node_str:
                    input_node_strs.append(conn[0])
                    w = enabled_conns.get(conn, 0.0)
                    weights.append(w)
                    output_indices.append(conn_output_index.get(conn))

            self._steps.append({
                "node": node_str,
                "type": "scalar",
                "input_nodes": input_node_strs,
                "weights": np.array(weights, dtype=np.float64),
                "bias": float(bias_val),
                "activation": activation,
                "output_indices": output_indices,
            })

    def _build_from_neat(self):
        """Build computation graph from NeuralNeat (legacy path)."""
        self._nested_networks = {}
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
            node_str = step["node"]

            if step.get("type") == "function":
                # Delegate to the nested StructureNetwork
                nested_net = self._nested_networks[node_str]
                fn_input_cols = [
                    activations.get(n, np.zeros(len(x)))
                    for n in step["input_nodes"]
                ]
                fn_input = np.column_stack(fn_input_cols) if fn_input_cols else np.zeros((len(x), 0))
                fn_tensor = torch.tensor(fn_input, dtype=torch.float64)
                fn_out = nested_net.forward(fn_tensor).detach().numpy()
                if fn_out.ndim == 1:
                    fn_out = fn_out.reshape(-1, 1)
                # Store each output column under the corresponding output name
                for out_idx, out_name in enumerate(step["output_names"]):
                    if out_idx < fn_out.shape[1]:
                        activations[out_name] = fn_out[:, out_idx]
                # Also store the whole output keyed by node_id for connections
                # that reference the node directly (output_index selects column)
                activations[node_str] = fn_out
                continue

            # Scalar node: gather inputs, respecting output_index for fn sources
            input_vals = []
            for conn_idx, in_node in enumerate(step["input_nodes"]):
                out_idx = step["output_indices"][conn_idx] if "output_indices" in step else None
                val = activations.get(in_node)
                if val is None:
                    input_vals.append(np.zeros(len(x)))
                elif isinstance(val, np.ndarray) and val.ndim == 2:
                    # Source is a function node result — select the right column
                    col = out_idx if out_idx is not None else 0
                    input_vals.append(val[:, col])
                else:
                    input_vals.append(val)

            inputs = np.column_stack(input_vals) if input_vals else np.zeros((len(x), 0))

            # Compute: inputs @ weights + bias
            if len(step["weights"]) > 0 and inputs.shape[1] > 0:
                z = inputs @ step["weights"] + step["bias"]
            else:
                z = np.full(len(x), step["bias"])

            act_fn = get_numpy_activation(step["activation"])
            activations[node_str] = act_fn(z)

        # Gather outputs — exit nodes may be direct scalar activations or
        # named outputs of a nested function node
        output_cols = []
        for n in self.exit_nodes:
            val = activations.get(n)
            if val is None:
                output_cols.append(np.zeros(len(x)))
            elif isinstance(val, np.ndarray) and val.ndim == 2:
                output_cols.append(val[:, 0])
            else:
                output_cols.append(val)

        outputs = np.column_stack(output_cols)

        if single:
            return outputs[0]
        return outputs

    def to_sympy(self, expand: bool = True) -> Optional[Dict[str, "sympy.Expr"]]:
        """Extract symbolic expressions for each output.

        Args:
            expand: If True (default), recursively inline child FUNCTION node
                expressions down to primitive activations.  If False, represent
                child FUNCTION nodes as named ``sympy.Function`` applications
                (e.g. ``child1(x_0)``).

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
            if step.get("type") == "function":
                self._sympy_handle_function_step(
                    step, node_exprs, expand, sympy
                )
                continue

            # Build expression: sum(w_i * input_i) + bias
            expr = sympy.Float(step["bias"])
            for j, in_node in enumerate(step["input_nodes"]):
                w = step["weights"][j]
                # Resolve input: if it's a function node output, look up by
                # output_index using the __out_N key pattern
                in_expr = self._resolve_sympy_input(
                    in_node, step, j, node_exprs
                )
                if in_expr is not None and abs(w) > 1e-12:
                    expr += sympy.Float(w) * in_expr

            # Apply activation
            act_sym_fn = get_sympy_activation(step["activation"])
            expr = act_sym_fn(expr)

            node_exprs[step["node"]] = sympy.simplify(expr)

        # Collect output expressions
        result = {}
        for i, node_str in enumerate(self.exit_nodes):
            if node_str in node_exprs:
                result[f"y_{i}"] = node_exprs[node_str]

        return result if result else None

    def _resolve_sympy_input(
        self, in_node: str, step: dict, conn_idx: int, node_exprs: dict
    ):
        """Resolve an input expression for a scalar step, handling function node outputs.

        If the input comes from a function node, the expression is stored
        under ``(fn_node_id, "__out_N")`` where N is the output_index from
        the connection.  Regular nodes are looked up directly.
        """
        # Check if there is an output_index pointing to a function node output
        output_indices = step.get("output_indices", [])
        out_idx = output_indices[conn_idx] if conn_idx < len(output_indices) else None

        if out_idx is not None:
            key = (in_node, f"__out_{out_idx}")
            if key in node_exprs:
                return node_exprs[key]

        return node_exprs.get(in_node)

    def _sympy_handle_function_step(
        self,
        step: dict,
        node_exprs: dict,
        expand: bool,
        sympy,
    ) -> None:
        """Handle a FUNCTION step in to_sympy.

        Populates ``node_exprs`` with the function node's output expressions
        keyed both by output name and by ``(fn_node_id, "__out_N")``.
        """
        meta = step.get("function_metadata")
        if meta is None:
            return

        # Gather the sympy expressions for the function node's inputs
        fn_input_exprs = []
        for in_node in step["input_nodes"]:
            fn_input_exprs.append(node_exprs.get(in_node, sympy.Symbol(in_node)))

        if expand:
            # Build a child AnnotationFunction and get its sympy expressions
            child_af = self._build_child_annotation_function(meta)
            if child_af is not None:
                child_exprs = child_af.to_sympy(expand=True)
                if child_exprs is not None:
                    # child_exprs uses x_0, x_1, ... as input symbols.
                    # Substitute them with the parent's input expressions.
                    child_input_syms = [
                        sympy.Symbol(f"x_{i}")
                        for i in range(len(step["input_nodes"]))
                    ]
                    for out_key, child_expr in child_exprs.items():
                        # Substitute child x_i -> parent input expressions
                        subs = {
                            child_input_syms[i]: fn_input_exprs[i]
                            for i in range(min(len(child_input_syms), len(fn_input_exprs)))
                        }
                        substituted = child_expr.subs(subs)
                        # Extract output index from y_N key
                        if not isinstance(out_key, str) or not out_key.startswith("y_"):
                            continue
                        out_idx = int(out_key[2:])
                        # Store under output name and positional key
                        if out_idx < len(step["output_names"]):
                            out_name = step["output_names"][out_idx]
                            node_exprs[out_name] = sympy.simplify(substituted)
                        node_exprs[(step["node"], f"__out_{out_idx}")] = sympy.simplify(substituted)
                    return

            # Fallback: if child AnnotationFunction couldn't be built, treat as opaque
            self._sympy_function_opaque(step, fn_input_exprs, node_exprs, sympy)
        else:
            # expand=False: represent as a named sympy Function
            self._sympy_function_opaque(step, fn_input_exprs, node_exprs, sympy)

    def _sympy_function_opaque(
        self, step: dict, fn_input_exprs: list, node_exprs: dict, sympy
    ) -> None:
        """Create opaque sympy Function symbols for a FUNCTION step."""
        meta = step.get("function_metadata")
        fn_name = meta.annotation_name if meta else step["node"]
        fn_sym = sympy.Function(fn_name)

        n_outputs = step["n_outputs"]
        if n_outputs == 1:
            expr = fn_sym(*fn_input_exprs)
            if step["output_names"]:
                node_exprs[step["output_names"][0]] = expr
            node_exprs[(step["node"], "__out_0")] = expr
        else:
            for out_idx in range(n_outputs):
                # Create indexed function: fn_name_{out_idx}(inputs)
                indexed_fn = sympy.Function(f"{fn_name}_{out_idx}")
                expr = indexed_fn(*fn_input_exprs)
                if out_idx < len(step["output_names"]):
                    node_exprs[step["output_names"][out_idx]] = expr
                node_exprs[(step["node"], f"__out_{out_idx}")] = expr

    def _build_child_annotation_function(self, meta) -> Optional["AnnotationFunction"]:
        """Build an AnnotationFunction from FunctionNodeMetadata for sympy extraction."""
        try:
            from ..core.genome_network import (
                NetworkConnection as NC,
                NetworkNode as NN,
                NetworkStructure as NS,
            )
            from ..core.model_state import AnnotationData

            node_props = meta.node_properties or {}
            conn_weights = meta.connection_weights or {}
            child_meta_map = meta.child_function_metadata or {}
            nodes_by_id = {n.id: n for n in self._structure.nodes}

            sub_nodes = []
            for nid in meta.subgraph_nodes:
                if nid in child_meta_map:
                    sub_nodes.append(NN(
                        id=nid,
                        type=NodeType.FUNCTION,
                        function_metadata=child_meta_map[nid],
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

            ann_data = AnnotationData(
                name=meta.annotation_name,
                hypothesis=meta.hypothesis,
                entry_nodes=list(meta.input_names),
                exit_nodes=list(meta.output_names),
                subgraph_nodes=list(meta.subgraph_nodes),
                subgraph_connections=list(meta.subgraph_connections),
            )

            return AnnotationFunction.from_structure(ann_data, mini_structure)
        except Exception:
            return None

    def to_latex(self, expand: bool = True) -> Optional[str]:
        """Get LaTeX representation of the function.

        Args:
            expand: If True (default), inline child FUNCTION node expressions
                to primitives.  If False, represent them as named functions.

        Returns None if sympy extraction fails or is intractable.
        """
        exprs = self.to_sympy(expand=expand)
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
