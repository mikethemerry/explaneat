"""Extract intermediate activations from a network at annotation nodes."""

from typing import Optional, Tuple

import numpy as np
import torch

from ..core.genome_network import NetworkStructure
from ..core.neuralneat import NeuralNeat
from ..core.structure_network import StructureNetwork


class ActivationExtractor:
    """Forward-pass real data through a network and extract activations at
    specific annotation entry/exit nodes.

    Supports two modes:
    - Legacy: from a raw NEAT genome + config (integer node IDs only)
    - Structure: from a NetworkStructure with operations applied (supports
      identity nodes, split nodes, etc.)
    """

    def __init__(
        self,
        genome=None,
        config=None,
        *,
        structure: Optional[NetworkStructure] = None,
    ):
        if structure is not None:
            self._net = StructureNetwork(structure)
            self._mode = "structure"
        elif genome is not None and config is not None:
            self._net = NeuralNeat(genome, config)
            self._mode = "neat"
        else:
            raise ValueError("Must provide either (genome, config) or structure=")

    @classmethod
    def from_structure(cls, structure: NetworkStructure) -> "ActivationExtractor":
        """Create from a NetworkStructure (with operations applied)."""
        return cls(structure=structure)

    def _get_activation(self, node_str: str) -> np.ndarray:
        """Get activations for a node from the last forward pass."""
        if self._mode == "structure":
            return self._net.get_node_activation(node_str)

        # Legacy NeuralNeat path
        resolved = self._resolve_node_id(node_str)
        if resolved is None:
            raise ValueError(f"Cannot resolve node '{node_str}' in network")

        tracker = self._net.node_tracker[resolved]
        depth = tracker["depth"]
        layer_index = tracker["layer_index"]

        if self._net._outputs is None:
            raise RuntimeError("Must call forward pass before extracting activations")
        if depth not in self._net._outputs:
            raise ValueError(f"Layer {depth} not in outputs (node '{node_str}')")

        return self._net._outputs[depth][:, layer_index].detach().cpu().numpy()

    def _resolve_node_id(self, node_str: str):
        """Resolve a string node ID to NeuralNeat tracker key (legacy mode)."""
        try:
            int_id = int(node_str)
            if int_id in self._net.node_tracker:
                return int_id
        except ValueError:
            pass

        if node_str.startswith("identity_"):
            base = node_str[len("identity_"):]
            try:
                int_id = int(base)
                if int_id in self._net.node_tracker:
                    return int_id
            except ValueError:
                pass

        if "_" in node_str:
            base = node_str.split("_")[0]
            try:
                int_id = int(base)
                if int_id in self._net.node_tracker:
                    return int_id
            except ValueError:
                pass

        return None

    def extract(
        self, X: np.ndarray, annotation
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass X through the network and extract entry/exit activations.

        Args:
            X: Input data of shape (n_samples, n_features)
            annotation: Annotation dict or object with entry_nodes and exit_nodes

        Returns:
            Tuple of (entry_acts, exit_acts):
                entry_acts: shape (n_samples, n_entry)
                exit_acts: shape (n_samples, n_exit)
        """
        if isinstance(annotation, dict):
            entry_nodes = [str(n) for n in annotation["entry_nodes"]]
            exit_nodes = [str(n) for n in annotation["exit_nodes"]]
        else:
            entry_nodes = [str(n) for n in annotation.entry_nodes]
            exit_nodes = [str(n) for n in annotation.exit_nodes]

        x_tensor = torch.tensor(X, dtype=torch.float64)
        with torch.no_grad():
            self._net.forward(x_tensor)

        entry_acts = np.column_stack(
            [self._get_activation(n) for n in entry_nodes]
        )
        exit_acts = np.column_stack(
            [self._get_activation(n) for n in exit_nodes]
        )

        return entry_acts, exit_acts
