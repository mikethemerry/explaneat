"""
Visualization classes for genome and ancestry analysis

Provides visualization tools for:
- Network structure visualization
- Ancestry tree visualization
- Gene origin timelines
- Performance comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import tempfile
import webbrowser
import os
import logging
import json
import shutil
import re
from datetime import datetime, timezone
from pathlib import Path
from importlib import resources as pkg_resources

try:
    from pyvis.network import Network

    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    Network = None


logger = logging.getLogger(__name__)


class GenomeVisualizer:
    """
    Visualizes NEAT genome network structures.
    """

    def __init__(self, neat_genome, config):
        self.genome = neat_genome
        self.config = config

    def plot_network(
        self,
        figsize: Tuple[int, int] = (14, 10),
        node_size: int = 800,
        show_weights: bool = True,
        show_node_labels: bool = True,
        show_layers: bool = True,
        layout: str = "layered",
    ) -> None:
        """
        Plot the network structure of the genome.

        Args:
            figsize: Figure size (auto-adjusted based on network size)
            node_size: Size of nodes in the plot
            show_weights: Whether to show connection weights
            show_node_labels: Whether to show node IDs
            show_layers: Whether to show layer/depth annotations
            layout: Layout algorithm ('layered', 'spring', 'circular')
                   'layered' (default): Left-to-right with depth-based layers
                   'spring': Force-directed layout
                   'circular': Circular layout
        """
        # Create networkx graph
        G = nx.DiGraph()

        # Add nodes
        for node_id in self.genome.nodes:
            G.add_node(node_id)

        # Add connections
        edge_weights = []
        for conn_key, conn in self.genome.connections.items():
            if conn.enabled:
                G.add_edge(conn_key[0], conn_key[1])
                edge_weights.append(abs(conn.weight))

        if not G.nodes():
            print("No nodes to visualize")
            return

        # Choose layout
        if layout in ["layered", "hierarchical"]:
            pos = self._layered_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        # Auto-adjust figure size for layered layout
        if layout in ["layered", "hierarchical"] and pos:
            # Calculate network dimensions
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            width = max(x_coords) - min(x_coords) + 4
            height = max(y_coords) - min(y_coords) + 4
            # Adjust figure size to maintain aspect ratio
            figsize = (max(12, width * 1.5), max(8, height * 1.2))

        # Set up the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Identify node types
        input_nodes = [n for n in G.nodes() if n < 0]
        output_nodes = [n for n in G.nodes() if n == 0]
        hidden_nodes = [n for n in G.nodes() if n > 0]

        # Draw nodes by type with different shapes
        if input_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=input_nodes,
                node_color="#4A90E2",  # Blue
                node_size=node_size,
                node_shape="s",  # Square for inputs
                edgecolors="black",
                linewidths=2,
                ax=ax,
                label="Input Nodes",
            )

        if output_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=output_nodes,
                node_color="#E24A4A",  # Red
                node_size=node_size,
                node_shape="h",  # Hexagon for outputs
                edgecolors="black",
                linewidths=2,
                ax=ax,
                label="Output Nodes",
            )

        if hidden_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=hidden_nodes,
                node_color="#50C878",  # Green
                node_size=node_size * 0.8,
                node_shape="o",  # Circle for hidden
                edgecolors="black",
                linewidths=1.5,
                ax=ax,
                label="Hidden Nodes",
            )

        # Draw edges with improved styling
        if edge_weights:
            # Calculate node depths for edge styling
            if layout in ["layered", "hierarchical"]:
                input_node_list = [n for n in G.nodes() if n < 0] or [
                    n for n in G.nodes() if G.in_degree(n) == 0
                ]
                node_depths = self._calculate_node_depths(G, input_node_list)
            else:
                node_depths = {n: 0 for n in G.nodes()}

            # Separate regular connections from skip connections
            regular_edges = []
            skip_edges = []
            regular_widths = []
            skip_widths = []
            regular_colors = []
            skip_colors = []

            max_weight = max(edge_weights) if edge_weights else 1

            for conn_key, conn in self.genome.connections.items():
                if conn.enabled:
                    from_node, to_node = conn_key
                    depth_diff = abs(
                        node_depths.get(to_node, 0) - node_depths.get(from_node, 0)
                    )

                    # Edge width based on weight magnitude
                    width = 0.5 + 2.5 * abs(conn.weight) / max_weight

                    # Edge color based on weight sign
                    if conn.weight >= 0:
                        color = "#2E86DE"  # Blue for positive
                    else:
                        color = "#EE5A6F"  # Red for negative

                    # Skip connections (span more than 1 layer)
                    if depth_diff > 1:
                        skip_edges.append(conn_key)
                        skip_widths.append(width)
                        skip_colors.append(color)
                    else:
                        regular_edges.append(conn_key)
                        regular_widths.append(width)
                        regular_colors.append(color)

            # Draw regular edges
            if regular_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=regular_edges,
                    width=regular_widths,
                    edge_color=regular_colors,
                    alpha=0.6,
                    connectionstyle="arc3,rad=0.05",
                    ax=ax,
                )

            # Draw skip connections with curved edges
            if skip_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=skip_edges,
                    width=skip_widths,
                    edge_color=skip_colors,
                    alpha=0.4,
                    connectionstyle="arc3,rad=0.2",
                    style="dashed",
                    ax=ax,
                )
        else:
            nx.draw_networkx_edges(
                G, pos, alpha=0.5, connectionstyle="arc3,rad=0.05", ax=ax
            )

        # Add node labels
        if show_node_labels:
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

        # Add weight labels (only if not too many connections)
        if (
            show_weights
            and self.genome.connections
            and len(self.genome.connections) < 30
        ):
            edge_labels = {}
            for conn_key, conn in self.genome.connections.items():
                if conn.enabled:
                    edge_labels[conn_key] = f"{conn.weight:.2f}"

            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

        # Add layer annotations for layered layout
        if show_layers and layout in ["layered", "hierarchical"] and pos:
            input_node_list = [n for n in G.nodes() if n < 0] or [
                n for n in G.nodes() if G.in_degree(n) == 0
            ]
            node_depths = self._calculate_node_depths(G, input_node_list)

            # Get unique depths and their x positions
            depth_x_positions = {}
            for node, depth in node_depths.items():
                if depth not in depth_x_positions and node in pos:
                    depth_x_positions[depth] = pos[node][0]

            # Draw vertical lines for each layer
            y_coords = [p[1] for p in pos.values()]
            y_min, y_max = min(y_coords), max(y_coords)
            y_range = y_max - y_min
            y_min -= y_range * 0.1
            y_max += y_range * 0.1

            for depth, x in sorted(depth_x_positions.items()):
                ax.axvline(x=x, color="gray", linestyle=":", alpha=0.3, linewidth=1)
                ax.text(
                    x,
                    y_max + y_range * 0.05,
                    f"Layer {depth}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="gray",
                )

        # Add improved legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#4A90E2",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label="Input Nodes",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="h",
                color="w",
                markerfacecolor="#E24A4A",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label="Output Nodes",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#50C878",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label="Hidden Nodes",
            ),
            plt.Line2D(
                [0], [0], color="#2E86DE", linewidth=2, label="Positive Weights"
            ),
            plt.Line2D(
                [0], [0], color="#EE5A6F", linewidth=2, label="Negative Weights"
            ),
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=2,
                linestyle="dashed",
                alpha=0.4,
                label="Skip Connections",
            ),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9
        )

        # Enhanced title with network statistics
        enabled_connections = len(
            [c for c in self.genome.connections.values() if c.enabled]
        )
        disabled_connections = len(self.genome.connections) - enabled_connections

        if layout in ["layered", "hierarchical"] and pos:
            input_node_list = [n for n in G.nodes() if n < 0] or [
                n for n in G.nodes() if G.in_degree(n) == 0
            ]
            node_depths = self._calculate_node_depths(G, input_node_list)
            max_depth = max(node_depths.values()) if node_depths else 0
            title_suffix = f", Depth: {max_depth + 1} layers"
        else:
            title_suffix = ""

        ax.set_title(
            f"Network Structure - Genome {self.genome.key}\n"
            f"Fitness: {self.genome.fitness:.3f} | "
            f"Nodes: {len(self.genome.nodes)} | "
            f"Connections: {enabled_connections}/{len(self.genome.connections)}{title_suffix}",
            fontsize=12,
            fontweight="bold",
            pad=20,
        )

        # Set axis properties with proper limits to show everything
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add padding around the network
            x_padding = (x_max - x_min) * 0.15 if x_max > x_min else 1
            y_padding = (y_max - y_min) * 0.15 if y_max > y_min else 1

            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

        ax.axis("off")
        ax.set_aspect("equal")

        # Enable interactive zoom and pan
        # Store initial limits for reset functionality
        if pos:
            initial_xlim = ax.get_xlim()
            initial_ylim = ax.get_ylim()

            def on_key(event):
                """Handle keyboard shortcuts for zoom and pan"""
                if event.key == "+" or event.key == "=":
                    # Zoom in
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    x_range = (xlim[1] - xlim[0]) * 0.8 / 2
                    y_range = (ylim[1] - ylim[0]) * 0.8 / 2
                    ax.set_xlim(x_center - x_range, x_center + x_range)
                    ax.set_ylim(y_center - y_range, y_center + y_range)
                    fig.canvas.draw()
                elif event.key == "-" or event.key == "_":
                    # Zoom out
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    x_range = (xlim[1] - xlim[0]) * 1.25 / 2
                    y_range = (ylim[1] - ylim[0]) * 1.25 / 2
                    ax.set_xlim(x_center - x_range, x_center + x_range)
                    ax.set_ylim(y_center - y_range, y_center + y_range)
                    fig.canvas.draw()
                elif event.key == "r" or event.key == "home":
                    # Reset to initial view
                    ax.set_xlim(initial_xlim)
                    ax.set_ylim(initial_ylim)
                    fig.canvas.draw()
                elif event.key == "left":
                    # Pan left
                    xlim = ax.get_xlim()
                    shift = (xlim[1] - xlim[0]) * 0.1
                    ax.set_xlim(xlim[0] - shift, xlim[1] - shift)
                    fig.canvas.draw()
                elif event.key == "right":
                    # Pan right
                    xlim = ax.get_xlim()
                    shift = (xlim[1] - xlim[0]) * 0.1
                    ax.set_xlim(xlim[0] + shift, xlim[1] + shift)
                    fig.canvas.draw()
                elif event.key == "up":
                    # Pan up
                    ylim = ax.get_ylim()
                    shift = (ylim[1] - ylim[0]) * 0.1
                    ax.set_ylim(ylim[0] + shift, ylim[1] + shift)
                    fig.canvas.draw()
                elif event.key == "down":
                    # Pan down
                    ylim = ax.get_ylim()
                    shift = (ylim[1] - ylim[0]) * 0.1
                    ax.set_ylim(ylim[0] - shift, ylim[1] - shift)
                    fig.canvas.draw()
                elif event.key == "f":
                    # Fit to window (reset view)
                    ax.set_xlim(initial_xlim)
                    ax.set_ylim(initial_ylim)
                    fig.canvas.draw()

            # Connect keyboard handler
            fig.canvas.mpl_connect("key_press_event", on_key)

            # Add scroll wheel zoom
            def on_scroll(event):
                """Handle mouse scroll for zooming"""
                if event.inaxes != ax:
                    return

                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # Get mouse position
                x_data = event.xdata
                y_data = event.ydata

                if x_data is None or y_data is None:
                    return

                # Zoom factor
                zoom_factor = 0.9 if event.button == "up" else 1.1

                # Calculate new limits centered on mouse position
                x_range = (xlim[1] - xlim[0]) * zoom_factor
                y_range = (ylim[1] - ylim[0]) * zoom_factor

                # Zoom towards mouse position
                x_ratio = (x_data - xlim[0]) / (xlim[1] - xlim[0])
                y_ratio = (y_data - ylim[0]) / (ylim[1] - ylim[0])

                ax.set_xlim(
                    x_data - x_range * x_ratio, x_data + x_range * (1 - x_ratio)
                )
                ax.set_ylim(
                    y_data - y_range * y_ratio, y_data + y_range * (1 - y_ratio)
                )
                fig.canvas.draw()

            # Connect scroll handler
            fig.canvas.mpl_connect("scroll_event", on_scroll)

            # Add instructions to title
            instructions = "\nControls: +/- zoom | Arrow keys pan | R reset | Scroll wheel zoom | Click+drag pan"
            current_title = ax.get_title()
            ax.set_title(
                current_title + f"\n{instructions}",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )

        plt.tight_layout()
        plt.show()

    def _calculate_node_depths(
        self, G: nx.DiGraph, input_nodes: List[int]
    ) -> Dict[int, int]:
        """
        Calculate depth of each node as maximum path length from inputs.

        Uses PropNEAT algorithm: depth = max(all path lengths from input nodes).
        This matches the GPU layer mapping approach.

        Enforces:
        - Input nodes are always depth 0
        - Hidden nodes have minimum depth 1
        - Output nodes are always at maximum depth

        Args:
            G: Directed graph of the network
            input_nodes: List of input node IDs

        Returns:
            Dictionary mapping node_id to depth (layer number)
        """
        # Get output nodes from config
        output_nodes = (
            set(self.config.genome_config.output_keys)
            if hasattr(self.config, "genome_config")
            else set()
        )
        # Also identify by convention (node ID == 0)
        output_nodes.update([n for n in G.nodes() if n == 0])

        # Identify hidden nodes (positive IDs, not input or output)
        input_node_set = set(input_nodes)
        hidden_nodes = set([n for n in G.nodes() if n > 0 and n not in output_nodes])

        depths = {node: [] for node in G.nodes()}

        # Initialize input nodes at depth 0
        for input_node in input_nodes:
            if input_node in G.nodes():
                depths[input_node] = [0]

        # Breadth-first search from all input nodes
        queue = [(node, 0) for node in input_nodes if node in G.nodes()]

        while queue:
            node, depth = queue.pop(0)

            # Propagate to all successors
            for successor in G.successors(node):
                depths[successor].append(depth + 1)
                queue.append((successor, depth + 1))

        # Take maximum depth for each node (longest path from inputs)
        node_depths = {node: max(d) if d else 0 for node, d in depths.items()}

        # Post-process to enforce constraints
        max_calculated_depth = max(node_depths.values()) if node_depths else 0

        for node in G.nodes():
            # Enforce: inputs always depth 0
            if node in input_node_set:
                node_depths[node] = 0
            # Enforce: hidden nodes minimum depth 1
            elif node in hidden_nodes:
                node_depths[node] = max(1, node_depths.get(node, 0))
            # Enforce: outputs always at maximum depth
            elif node in output_nodes:
                node_depths[node] = max_calculated_depth

        return node_depths

    def _minimize_crossings_barycenter(
        self, G: nx.DiGraph, layers: Dict[int, List[int]], max_iterations: int = 4
    ) -> Dict[int, List[int]]:
        """
        Minimize edge crossings using the barycenter heuristic.

        This is a computationally efficient approximation that gives good results.
        The algorithm iteratively reorders nodes within each layer based on the
        average position of their neighbors in adjacent layers.

        Args:
            G: Directed graph of the network
            layers: Dictionary mapping depth -> list of node IDs at that depth
            max_iterations: Number of forward/backward passes

        Returns:
            Dictionary mapping depth -> ordered list of node IDs
        """
        # Create a copy to work with
        ordered_layers = {depth: list(nodes) for depth, nodes in layers.items()}
        layer_depths = sorted(ordered_layers.keys())

        # Create position lookup for quick access
        def get_positions(layer_depth):
            return {node: idx for idx, node in enumerate(ordered_layers[layer_depth])}

        # Barycenter calculation
        def calculate_barycenter(node, neighbor_positions):
            """Calculate weighted average position of neighbors"""
            if not neighbor_positions:
                return 0.0
            return sum(neighbor_positions) / len(neighbor_positions)

        # Iterate forward and backward passes
        for iteration in range(max_iterations):
            # Forward pass: order based on predecessors
            if iteration % 2 == 0:
                for i in range(1, len(layer_depths)):
                    depth = layer_depths[i]
                    prev_depth = layer_depths[i - 1]
                    prev_positions = get_positions(prev_depth)

                    barycenters = []
                    for node in ordered_layers[depth]:
                        predecessors = list(G.predecessors(node))
                        pred_positions = [
                            prev_positions[p]
                            for p in predecessors
                            if p in prev_positions
                        ]
                        bc = calculate_barycenter(node, pred_positions)
                        barycenters.append((bc, node))

                    # Sort by barycenter
                    barycenters.sort()
                    ordered_layers[depth] = [node for _, node in barycenters]

            # Backward pass: order based on successors
            else:
                for i in range(len(layer_depths) - 2, -1, -1):
                    depth = layer_depths[i]
                    next_depth = layer_depths[i + 1]
                    next_positions = get_positions(next_depth)

                    barycenters = []
                    for node in ordered_layers[depth]:
                        successors = list(G.successors(node))
                        succ_positions = [
                            next_positions[s] for s in successors if s in next_positions
                        ]
                        bc = calculate_barycenter(node, succ_positions)
                        barycenters.append((bc, node))

                    # Sort by barycenter
                    barycenters.sort()
                    ordered_layers[depth] = [node for _, node in barycenters]

        return ordered_layers

    def _layered_layout(self, G: nx.DiGraph) -> Dict:
        """
        Create a left-to-right layered layout with crossing minimization.

        Uses PropNEAT depth calculation and barycenter heuristic for ordering.

        Args:
            G: Directed graph of the network

        Returns:
            Dictionary mapping node_id to (x, y) position
        """
        if not G.nodes():
            return {}

        # Identify node types from config if available
        if hasattr(self.config, "genome_config"):
            input_nodes = sorted(
                [n for n in self.config.genome_config.input_keys if n in G.nodes()]
            )
            output_nodes = sorted(
                [n for n in self.config.genome_config.output_keys if n in G.nodes()]
            )
        else:
            # Fallback to convention-based identification
            input_nodes = sorted([n for n in G.nodes() if n < 0])
            output_nodes = sorted([n for n in G.nodes() if n == 0])

        if not input_nodes:
            # If no input nodes, use nodes with no predecessors
            input_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

        # Calculate depths using PropNEAT algorithm
        node_depths = self._calculate_node_depths(G, input_nodes)

        # Group nodes by depth (layer)
        layers = {}
        for node, depth in node_depths.items():
            if depth not in layers:
                layers[depth] = []
            layers[depth].append(node)

        # Sort nodes within each layer initially
        for depth in layers:
            layers[depth].sort()

        # Apply crossing minimization
        if len(layers) > 1:
            layers = self._minimize_crossings_barycenter(G, layers)

        # Calculate positions
        pos = {}
        max_depth = max(layers.keys()) if layers else 0

        # Spacing parameters
        horizontal_spacing = 3.0  # Space between layers
        vertical_spacing = 1.5  # Space between nodes in a layer

        for depth, nodes in layers.items():
            x = depth * horizontal_spacing

            # Center nodes vertically
            num_nodes = len(nodes)
            total_height = (num_nodes - 1) * vertical_spacing
            start_y = -total_height / 2

            for i, node in enumerate(nodes):
                y = start_y + i * vertical_spacing
                pos[node] = (x, y)

        return pos

    def _hierarchical_layout(self, G) -> Dict:
        """
        Create a hierarchical layout for feedforward networks.

        Note: This is now an alias for _layered_layout which provides
        better depth calculation and crossing minimization.
        """
        return self._layered_layout(G)

    def plot_node_properties(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot distribution of node properties"""
        if not self.genome.nodes:
            print("No nodes to analyze")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Collect node properties
        biases = [node.bias for node in self.genome.nodes.values()]
        responses = [node.response for node in self.genome.nodes.values()]
        activations = [node.activation for node in self.genome.nodes.values()]

        # Plot bias distribution
        axes[0].hist(biases, bins=10, alpha=0.7, edgecolor="black")
        axes[0].set_title("Node Bias Distribution")
        axes[0].set_xlabel("Bias Value")
        axes[0].set_ylabel("Count")

        # Plot response distribution
        axes[1].hist(responses, bins=10, alpha=0.7, edgecolor="black", color="orange")
        axes[1].set_title("Node Response Distribution")
        axes[1].set_xlabel("Response Value")
        axes[1].set_ylabel("Count")

        # Plot activation function usage
        activation_counts = pd.Series(activations).value_counts()
        axes[2].bar(
            activation_counts.index, activation_counts.values, alpha=0.7, color="green"
        )
        axes[2].set_title("Activation Function Usage")
        axes[2].set_xlabel("Activation Function")
        axes[2].set_ylabel("Count")
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle(f"Node Properties - Genome {self.genome.key}")
        plt.tight_layout()
        plt.show()

    def plot_connection_properties(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """Plot distribution of connection properties"""
        if not self.genome.connections:
            print("No connections to analyze")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Collect connection properties
        weights = [
            conn.weight for conn in self.genome.connections.values() if conn.enabled
        ]
        disabled_count = sum(
            1 for conn in self.genome.connections.values() if not conn.enabled
        )
        enabled_count = len(weights)

        if weights:
            # Plot weight distribution
            axes[0].hist(weights, bins=15, alpha=0.7, edgecolor="black")
            axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Zero")
            axes[0].set_title("Connection Weight Distribution")
            axes[0].set_xlabel("Weight Value")
            axes[0].set_ylabel("Count")
            axes[0].legend()

        # Plot enabled vs disabled connections
        labels = ["Enabled", "Disabled"]
        sizes = [enabled_count, disabled_count]
        colors = ["lightgreen", "lightcoral"]

        axes[1].pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
        )
        axes[1].set_title("Connection Status")

        plt.suptitle(f"Connection Properties - Genome {self.genome.key}")
        plt.tight_layout()
        plt.show()


class AncestryVisualizer:
    """
    Visualizes ancestry trees and evolutionary patterns.
    """

    def __init__(self, ancestry_analyzer):
        self.analyzer = ancestry_analyzer

    def plot_ancestry_tree(
        self, max_generations: int = 10, figsize: Tuple[int, int] = (14, 10)
    ) -> None:
        """
        Plot the ancestry tree as a directed graph.

        Args:
            max_generations: Maximum generations to show
            figsize: Figure size
        """
        ancestry_df = self.analyzer.get_ancestry_tree(max_generations)

        if ancestry_df.empty:
            print("No ancestry data to visualize")
            return

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes with attributes
        for _, ancestor in ancestry_df.iterrows():
            G.add_node(
                ancestor["genome_id"],
                generation=ancestor["generation"],
                fitness=ancestor["fitness"],
                neat_id=ancestor["neat_genome_id"],
            )

        # Add edges (parent relationships)
        for _, ancestor in ancestry_df.iterrows():
            if ancestor["parent1_id"]:
                G.add_edge(ancestor["parent1_id"], ancestor["genome_id"])
            if ancestor["parent2_id"]:
                G.add_edge(ancestor["parent2_id"], ancestor["genome_id"])

        # Create layout based on generations
        pos = {}
        generation_groups = ancestry_df.groupby("generation")

        for gen, group in generation_groups:
            gen_nodes = group["genome_id"].tolist()
            for i, node_id in enumerate(gen_nodes):
                pos[node_id] = (gen, i - len(gen_nodes) / 2)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Color nodes by fitness
        fitnesses = [G.nodes[node]["fitness"] for node in G.nodes()]

        # Draw the graph
        nx.draw_networkx_nodes(
            G, pos, node_color=fitnesses, node_size=300, cmap="viridis", ax=ax
        )

        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=20, ax=ax
        )

        # Add labels with NEAT genome IDs
        labels = {node: str(G.nodes[node]["neat_id"]) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(vmin=min(fitnesses), vmax=max(fitnesses))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Fitness")

        ax.set_title("Ancestry Tree\n(Arrows point from parent to child)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Position in Generation")

        plt.tight_layout()
        plt.show()

    def plot_lineage_progression(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot multiple metrics showing lineage progression over generations.
        """
        ancestry_df = self.analyzer.get_ancestry_tree()

        if ancestry_df.empty:
            print("No ancestry data to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Fitness over generations
        axes[0, 0].plot(
            ancestry_df["generation"],
            ancestry_df["fitness"],
            "o-",
            linewidth=2,
            markersize=6,
        )
        axes[0, 0].set_title("Fitness Progression")
        axes[0, 0].set_xlabel("Generation")
        axes[0, 0].set_ylabel("Fitness")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Network complexity over generations
        axes[0, 1].plot(
            ancestry_df["generation"],
            ancestry_df["num_nodes"],
            "g-o",
            label="Nodes",
            linewidth=2,
        )
        axes[0, 1].plot(
            ancestry_df["generation"],
            ancestry_df["num_connections"],
            "b-s",
            label="Connections",
            linewidth=2,
        )
        axes[0, 1].set_title("Network Complexity")
        axes[0, 1].set_xlabel("Generation")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Enabled vs total connections
        enabled_ratio = (
            ancestry_df["num_enabled_connections"] / ancestry_df["num_connections"]
        )
        enabled_ratio = enabled_ratio.fillna(0)

        axes[1, 0].plot(ancestry_df["generation"], enabled_ratio, "r-o", linewidth=2)
        axes[1, 0].set_title("Connection Efficiency")
        axes[1, 0].set_xlabel("Generation")
        axes[1, 0].set_ylabel("Enabled Connections Ratio")
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Network dimensions
        axes[1, 1].plot(
            ancestry_df["generation"],
            ancestry_df["network_depth"],
            "purple",
            marker="o",
            label="Depth",
            linewidth=2,
        )
        axes[1, 1].plot(
            ancestry_df["generation"],
            ancestry_df["network_width"],
            "orange",
            marker="s",
            label="Width",
            linewidth=2,
        )
        axes[1, 1].set_title("Network Dimensions")
        axes[1, 1].set_xlabel("Generation")
        axes[1, 1].set_ylabel("Size")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Lineage Analysis Over Generations", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_gene_origins_timeline(
        self, current_genome, figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot when genes were introduced in the lineage.
        """
        gene_origins_df = self.analyzer.trace_gene_origins(current_genome)

        if gene_origins_df.empty:
            print("No gene origin data to visualize")
            return

        # Separate nodes and connections
        nodes_df = gene_origins_df[gene_origins_df["gene_type"] == "node"]
        connections_df = gene_origins_df[gene_origins_df["gene_type"] == "connection"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot node origins
        if not nodes_df.empty:
            node_origins = nodes_df["origin_generation"].dropna()
            if len(node_origins) > 0:
                ax1.hist(
                    node_origins,
                    bins=max(1, len(node_origins.unique())),
                    alpha=0.7,
                    edgecolor="black",
                    color="lightgreen",
                )
            ax1.set_title("Node Introduction Timeline")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Number of Nodes Added")
            ax1.grid(True, alpha=0.3)

        # Plot connection origins
        if not connections_df.empty:
            conn_origins = connections_df["origin_generation"].dropna()
            if len(conn_origins) > 0:
                ax2.hist(
                    conn_origins,
                    bins=max(1, len(conn_origins.unique())),
                    alpha=0.7,
                    edgecolor="black",
                    color="lightblue",
                )
            ax2.set_title("Connection Introduction Timeline")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Number of Connections Added")
            ax2.grid(True, alpha=0.3)

        plt.suptitle("Gene Introduction Timeline")
        plt.tight_layout()
        plt.show()

    def plot_fitness_vs_complexity(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot fitness vs network complexity across the lineage.
        """
        ancestry_df = self.analyzer.get_ancestry_tree()

        if ancestry_df.empty:
            print("No ancestry data to visualize")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Fitness vs number of connections
        ax1.scatter(
            ancestry_df["num_connections"],
            ancestry_df["fitness"],
            c=ancestry_df["generation"],
            cmap="viridis",
            s=60,
        )
        ax1.set_xlabel("Number of Connections")
        ax1.set_ylabel("Fitness")
        ax1.set_title("Fitness vs Connection Count")
        ax1.grid(True, alpha=0.3)

        # Fitness vs number of nodes
        scatter = ax2.scatter(
            ancestry_df["num_nodes"],
            ancestry_df["fitness"],
            c=ancestry_df["generation"],
            cmap="viridis",
            s=60,
        )
        ax2.set_xlabel("Number of Nodes")
        ax2.set_ylabel("Fitness")
        ax2.set_title("Fitness vs Node Count")
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=[ax1, ax2])
        cbar.set_label("Generation")

        plt.suptitle("Fitness vs Network Complexity")
        plt.tight_layout()
        plt.show()


class InteractiveNetworkViewer:
    """
    Interactive web-based network visualization using Pyvis.

    Provides subgraph-based filtering, annotation highlighting,
    and flexible node/edge hiding controls.
    """

    def __init__(self, neat_genome, config, annotations: Optional[List[Any]] = None):
        """
        Initialize interactive network viewer.

        Args:
            neat_genome: NEAT genome object
            config: NEAT config object
            annotations: Optional list of Annotation objects from database
        """
        if not PYVIS_AVAILABLE:
            raise ImportError(
                "Pyvis is required for interactive visualization. "
                "Install it with: pip install pyvis"
            )

        self.genome = neat_genome
        self.config = config
        self.annotations = annotations or []

        # Build network graph
        self.G = nx.DiGraph()
        self._build_graph()

        # Calculate node properties
        self.node_depths = self._calculate_depths()
        self.direct_connected_inputs = self._identify_direct_connected_inputs()
        # Identify all nodes and edges that are part of direct input-to-output connections
        self.direct_connection_nodes, self.direct_connection_edges = (
            self._identify_direct_connection_subgraph()
        )

    def _get_genome_viewer_js_path(self) -> Path:
        """
        Return the absolute path to the shared genome viewer JavaScript bundle.
        """
        js_path = (
            Path(__file__).resolve().parent.parent
            / "static"
            / "js"
            / "genome_viewer.js"
        )
        if not js_path.exists():
            raise FileNotFoundError(
                f"Genome viewer script not found at {js_path}. "
                "Ensure explaneat/static/js/genome_viewer.js exists."
            )
        return js_path

    def _prepare_annotation_sets(self):
        """Build lookup tables for annotation membership."""
        annotation_colors = {}
        annotation_node_sets: Dict[str, set] = {}
        annotation_edge_sets: Dict[str, set] = {}
        annotation_colors_list = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#FFA07A",
            "#98D8C8",
            "#F7DC6F",
            "#BB8FCE",
            "#85C1E2",
            "#F8B739",
            "#52BE80",
        ]

        for idx, annotation in enumerate(self.annotations):
            color = annotation_colors_list[idx % len(annotation_colors_list)]
            if isinstance(annotation, dict):
                ann_id = str(annotation.get("id", ""))
                subgraph_nodes = annotation.get("subgraph_nodes") or []
                subgraph_connections = annotation.get("subgraph_connections") or []
            else:
                ann_id = str(annotation.id)
                subgraph_nodes = annotation.subgraph_nodes or []
                subgraph_connections = annotation.subgraph_connections or []

            nodes = set(subgraph_nodes)
            edges = set()
            for conn in subgraph_connections:
                if isinstance(conn, (list, tuple)):
                    edges.add(tuple(conn))
                else:
                    edges.add(tuple(conn))

            annotation_node_sets[ann_id] = nodes
            annotation_edge_sets[ann_id] = edges

            for node_id in nodes:
                annotation_colors[node_id] = color

        return annotation_node_sets, annotation_edge_sets, annotation_colors

    def _build_graph(self):
        """Build NetworkX graph from genome."""
        # Add nodes
        for node_id in self.genome.nodes:
            self.G.add_node(node_id)

        # Add connections
        for conn_key, conn in self.genome.connections.items():
            if conn.enabled:
                self.G.add_edge(conn_key[0], conn_key[1], weight=conn.weight)

    def _calculate_depths(self) -> Dict[int, int]:
        """Calculate node depths using the same logic as GenomeVisualizer."""
        # Get input nodes from config
        if hasattr(self.config, "genome_config"):
            input_nodes = [
                n for n in self.config.genome_config.input_keys if n in self.G.nodes()
            ]
        else:
            input_nodes = [n for n in self.G.nodes() if n < 0]

        if not input_nodes:
            input_nodes = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]

        # Get output nodes
        if hasattr(self.config, "genome_config"):
            output_nodes = set(self.config.genome_config.output_keys)
        else:
            output_nodes = set([n for n in self.G.nodes() if n == 0])
        output_nodes.update([n for n in self.G.nodes() if n == 0])

        # Identify hidden nodes
        input_node_set = set(input_nodes)
        hidden_nodes = set(
            [n for n in self.G.nodes() if n > 0 and n not in output_nodes]
        )

        depths = {node: [] for node in self.G.nodes()}

        # Initialize input nodes at depth 0
        for input_node in input_nodes:
            if input_node in self.G.nodes():
                depths[input_node] = [0]

        # BFS from all input nodes
        queue = [(node, 0) for node in input_nodes if node in self.G.nodes()]

        while queue:
            node, depth = queue.pop(0)
            for successor in self.G.successors(node):
                depths[successor].append(depth + 1)
                queue.append((successor, depth + 1))

        # Take maximum depth
        node_depths = {node: max(d) if d else 0 for node, d in depths.items()}

        # Post-process to enforce constraints
        max_calculated_depth = max(node_depths.values()) if node_depths else 0

        for node in self.G.nodes():
            if node in input_node_set:
                node_depths[node] = 0
            elif node in hidden_nodes:
                node_depths[node] = max(1, node_depths.get(node, 0))
            elif node in output_nodes:
                node_depths[node] = max_calculated_depth

        return node_depths

    def _identify_direct_connected_inputs(self) -> List[int]:
        """
        Identify input nodes that only connect directly to output (no hidden nodes).

        Returns:
            List of input node IDs that are direct-connected
        """
        # Get input nodes
        if hasattr(self.config, "genome_config"):
            input_nodes = [
                n for n in self.config.genome_config.input_keys if n in self.G.nodes()
            ]
        else:
            input_nodes = [n for n in self.G.nodes() if n < 0]

        # Get output nodes
        if hasattr(self.config, "genome_config"):
            output_nodes = set(self.config.genome_config.output_keys)
        else:
            output_nodes = set([n for n in self.G.nodes() if n == 0])
        output_nodes.update([n for n in self.G.nodes() if n == 0])

        direct_connected = []

        for input_node in input_nodes:
            # Check all successors of this input
            successors = list(self.G.successors(input_node))

            # If all successors are outputs (no hidden nodes), it's direct-connected
            if successors and all(succ in output_nodes for succ in successors):
                direct_connected.append(input_node)

        return direct_connected

    def _identify_direct_connection_subgraph(self) -> Tuple[set, set]:
        """
        Identify all nodes and edges that are part of direct input-to-output connections.

        Returns:
            Tuple of (set of node IDs, set of edge tuples (from, to))
        """
        direct_nodes = set()
        direct_edges = set()

        # Get input and output nodes
        if hasattr(self.config, "genome_config"):
            input_nodes = [
                n for n in self.config.genome_config.input_keys if n in self.G.nodes()
            ]
            output_nodes = set(self.config.genome_config.output_keys)
        else:
            input_nodes = [n for n in self.G.nodes() if n < 0]
            output_nodes = set([n for n in self.G.nodes() if n == 0])
        output_nodes.update([n for n in self.G.nodes() if n == 0])

        # Find all direct connections (input -> output, no hidden nodes in between)
        for input_node in input_nodes:
            for output_node in output_nodes:
                # Check if there's a direct edge from input to output
                if self.G.has_edge(input_node, output_node):
                    # Check if this is truly direct (no path through hidden nodes)
                    # A direct connection means: input -> output with no intermediate hidden nodes
                    # Since we already have the edge, we just need to verify it's enabled
                    for conn_key, conn in self.genome.connections.items():
                        if (
                            conn_key[0] == input_node
                            and conn_key[1] == output_node
                            and conn.enabled
                        ):
                            direct_nodes.add(input_node)
                            direct_nodes.add(output_node)
                            direct_edges.add((input_node, output_node))
                            break

        return direct_nodes, direct_edges

    def _get_node_type(self, node_id: int) -> str:
        """Get node type: 'input', 'output', or 'hidden'."""
        if hasattr(self.config, "genome_config"):
            if node_id in self.config.genome_config.input_keys:
                return "input"
            if node_id in self.config.genome_config.output_keys:
                return "output"
        else:
            if node_id < 0:
                return "input"
            if node_id == 0:
                return "output"

        return "hidden"

    def _get_node_color(self, node_id: int, annotation_colors: Dict[int, str]) -> str:
        """Get color for a node based on type and annotations."""
        # Check if node is in an annotation
        if node_id in annotation_colors:
            return annotation_colors[node_id]

        # Default colors by type
        node_type = self._get_node_type(node_id)
        if node_type == "input":
            return "#4A90E2"  # Blue
        elif node_type == "output":
            return "#E24A4A"  # Red
        else:
            return "#50C878"  # Green

    def _get_edge_color(self, weight: float) -> str:
        """Get color for edge based on weight."""
        if weight >= 0:
            return "#2E86DE"  # Blue for positive
        else:
            return "#EE5A6F"  # Red for negative

    def _is_skip_connection(self, from_node: int, to_node: int) -> bool:
        """Check if connection is a skip connection (spans more than 1 layer)."""
        from_depth = self.node_depths.get(from_node, 0)
        to_depth = self.node_depths.get(to_node, 0)
        return abs(to_depth - from_depth) > 1

    def _calculate_layered_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Calculate left-to-right layered positions for nodes.

        Uses the same algorithm as GenomeVisualizer._layered_layout.

        Returns:
            Dictionary mapping node_id to (x, y) position
        """
        if not self.G.nodes():
            return {}

        # Get input nodes
        if hasattr(self.config, "genome_config"):
            input_nodes = sorted(
                [n for n in self.config.genome_config.input_keys if n in self.G.nodes()]
            )
        else:
            input_nodes = sorted([n for n in self.G.nodes() if n < 0])

        if not input_nodes:
            input_nodes = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]

        # Group nodes by depth (layer)
        layers = {}
        for node, depth in self.node_depths.items():
            if depth not in layers:
                layers[depth] = []
            layers[depth].append(node)

        # Sort nodes within each layer
        for depth in layers:
            layers[depth].sort()

        # Apply crossing minimization (simplified version)
        if len(layers) > 1:
            layers = self._minimize_crossings_barycenter(layers)

        # Calculate positions
        pos = {}
        max_depth = max(layers.keys()) if layers else 0

        # Spacing parameters (scaled for Pyvis)
        horizontal_spacing = 200.0  # Space between layers
        vertical_spacing = 100.0  # Space between nodes in a layer

        for depth, nodes in layers.items():
            x = depth * horizontal_spacing

            # Center nodes vertically
            num_nodes = len(nodes)
            total_height = (num_nodes - 1) * vertical_spacing
            start_y = -total_height / 2

            for i, node in enumerate(nodes):
                y = start_y + i * vertical_spacing
                pos[node] = (x, y)

        return pos

    def _minimize_crossings_barycenter(
        self, layers: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        """
        Minimize edge crossings using the barycenter heuristic.

        Simplified version for InteractiveNetworkViewer.
        """
        ordered_layers = {depth: list(nodes) for depth, nodes in layers.items()}
        layer_depths = sorted(ordered_layers.keys())

        def get_positions(layer_depth):
            return {node: idx for idx, node in enumerate(ordered_layers[layer_depth])}

        def calculate_barycenter(node, neighbor_positions):
            if not neighbor_positions:
                return 0.0
            return sum(neighbor_positions) / len(neighbor_positions)

        # Iterate forward and backward passes
        for iteration in range(4):
            if iteration % 2 == 0:
                # Forward pass
                for i in range(1, len(layer_depths)):
                    depth = layer_depths[i]
                    prev_depth = layer_depths[i - 1]
                    prev_positions = get_positions(prev_depth)

                    barycenters = []
                    for node in ordered_layers[depth]:
                        predecessors = list(self.G.predecessors(node))
                        pred_positions = [
                            prev_positions[p]
                            for p in predecessors
                            if p in prev_positions
                        ]
                        bc = calculate_barycenter(node, pred_positions)
                        barycenters.append((bc, node))

                    barycenters.sort()
                    ordered_layers[depth] = [node for _, node in barycenters]
            else:
                # Backward pass
                for i in range(len(layer_depths) - 2, -1, -1):
                    depth = layer_depths[i]
                    next_depth = layer_depths[i + 1]
                    next_positions = get_positions(next_depth)

                    barycenters = []
                    for node in ordered_layers[depth]:
                        successors = list(self.G.successors(node))
                        succ_positions = [
                            next_positions[s] for s in successors if s in next_positions
                        ]
                        bc = calculate_barycenter(node, succ_positions)
                        barycenters.append((bc, node))

                    barycenters.sort()
                    ordered_layers[depth] = [node for _, node in barycenters]

        return ordered_layers

    def visualize(
        self,
        output_file: Optional[str] = None,
        height: str = "800px",
        width: str = "100%",
        show_weights: bool = True,
        physics: bool = False,
        layout: str = "layered",
    ) -> str:
        """
        Generate interactive network visualization.

        Args:
            output_file: Path to output HTML file (if None, uses temp file)
            height: Height of visualization
            width: Width of visualization
            show_weights: Whether to show edge weights
            physics: Whether to enable physics simulation (only used if layout='physics')
            layout: Layout algorithm ('layered', 'hierarchical')
                   'layered': Left-to-right hierarchical layout (default)
                   'hierarchical': Alias for layered

        Returns:
            Path to generated HTML file
        """
        logger.debug(
            "InteractiveNetworkViewer.visualize start: nodes=%d edges=%d annotations=%d",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
            len(self.annotations),
        )
        # Always use layered layout (physics removed)
        use_layered = True
        node_positions = self._calculate_layered_positions()
        physics = False  # Physics disabled - using drag-and-drop

        # Create Pyvis network
        net = Network(
            height=height,
            width=width,
            directed=True,
            bgcolor="#ffffff",
            font_color="black",
        )

        # Configure for layered layout: disable physics, enable drag-and-drop
        # Nodes will be positioned by the layered algorithm and can be dragged
        # Use straight lines (no curves) for edges
        net.set_options(
            """
        {
          "physics": {
            "enabled": false,
            "stabilization": {
              "enabled": false,
              "iterations": 0
            }
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "selectConnectedEdges": true
          },
          "layout": {
            "improvedLayout": false,
            "hierarchical": {
              "enabled": false
            }
          },
          "edges": {
            "smooth": false
          }
        }
        """
        )

        (
            annotation_node_sets,
            annotation_edge_sets,
            annotation_colors,
        ) = self._prepare_annotation_sets()

        # Add nodes
        for node_id in self.G.nodes():
            node_type = self._get_node_type(node_id)
            depth = self.node_depths.get(node_id, 0)
            color = self._get_node_color(node_id, annotation_colors)
            is_direct_connected = node_id in self.direct_connected_inputs

            # Node shape
            if node_type == "input":
                shape = "box"
            elif node_type == "output":
                shape = "diamond"
            else:
                shape = "circle"

            # Build title with metadata
            title_parts = [f"Node: {node_id}", f"Type: {node_type}", f"Depth: {depth}"]
            if is_direct_connected:
                title_parts.append("Direct-connected input")

            # Add annotation info if applicable
            for ann_id, ann_nodes in annotation_node_sets.items():
                if node_id in ann_nodes:
                    try:
                        ann = next(
                            a
                            for a in self.annotations
                            if str(a.id if hasattr(a, "id") else a.get("id", ""))
                            == str(ann_id)
                        )
                        ann_name = (
                            ann.name
                            if hasattr(ann, "name")
                            else ann.get("name", "Unnamed")
                        )
                        ann_hypothesis = (
                            ann.hypothesis
                            if hasattr(ann, "hypothesis")
                            else ann.get("hypothesis", "")
                        )
                        title_parts.append(f"Annotation: {ann_name or 'Unnamed'}")
                        if ann_hypothesis:
                            title_parts.append(f"Hypothesis: {ann_hypothesis[:100]}")
                    except StopIteration:
                        pass

            title = "\\n".join(title_parts)

            # Get position if using layered layout
            x_pos = None
            y_pos = None
            if use_layered and node_positions and node_id in node_positions:
                x_pos, y_pos = node_positions[node_id]

            # Check if node is in direct connection subgraph
            is_in_direct_connection = node_id in self.direct_connection_nodes

            # Add node with metadata
            node_data = {
                "label": str(node_id),
                "color": color,
                "shape": shape,
                "title": title,
                "size": 20 if node_type == "hidden" else 25,
                # Store metadata for filtering
                "node_type": node_type,
                "depth": depth,
                "is_direct_connected": str(is_direct_connected).lower(),
                "is_in_direct_connection": str(is_in_direct_connection).lower(),
                "annotation_ids": ",".join(
                    [
                        str(aid)
                        for aid, nodes in annotation_node_sets.items()
                        if node_id in nodes
                    ]
                ),
            }

            # Add position from layered layout
            # Note: We set positions here, but JavaScript will also apply them
            # to ensure they're respected even if Pyvis tries to reposition
            if x_pos is not None and y_pos is not None:
                node_data["x"] = float(x_pos)
                node_data["y"] = float(y_pos)
                # Don't set 'fixed' - let JavaScript handle it for better control

            net.add_node(node_id, **node_data)

        # Add edges
        for from_node, to_node, data in self.G.edges(data=True):
            weight = data.get("weight", 0.0)
            is_skip = self._is_skip_connection(from_node, to_node)
            color = self._get_edge_color(weight)

            # Check if edge is in annotation
            edge_in_annotation = False
            annotation_info = []
            for ann_id, ann_edges in annotation_edge_sets.items():
                edge_tuple = (from_node, to_node)
                if edge_tuple in ann_edges or tuple(reversed(edge_tuple)) in ann_edges:
                    edge_in_annotation = True
                    try:
                        ann = next(
                            a
                            for a in self.annotations
                            if str(a.id if hasattr(a, "id") else a.get("id", ""))
                            == str(ann_id)
                        )
                        ann_name = (
                            ann.name
                            if hasattr(ann, "name")
                            else ann.get("name", "Unnamed")
                        )
                        annotation_info.append(ann_name or "Unnamed")
                    except StopIteration:
                        pass

            # Edge title
            title_parts = [f"Weight: {weight:.3f}"]
            if is_skip:
                title_parts.append("Skip connection")
            if edge_in_annotation:
                title_parts.append(f"In annotation(s): {', '.join(annotation_info)}")

            title = "\\n".join(title_parts)

            # Edge width based on weight magnitude
            width_val = 1 + abs(weight) * 2

            # Check if edge is in direct connection subgraph
            edge_tuple = (from_node, to_node)
            is_in_direct_connection = edge_tuple in self.direct_connection_edges

            net.add_edge(
                from_node,
                to_node,
                title=title,
                color=color,
                width=width_val,
                # Store metadata for filtering
                weight_sign="positive" if weight >= 0 else "negative",
                is_skip=str(is_skip).lower(),
                is_in_direct_connection=str(is_in_direct_connection).lower(),
                annotation_ids=",".join(
                    [
                        str(aid)
                        for aid, edges in annotation_edge_sets.items()
                        if (from_node, to_node) in edges
                        or (to_node, from_node) in edges
                    ]
                ),
            )

        # Prepare filter metadata for JavaScript
        node_filter_metadata = {}
        edge_filter_metadata = {}

        # Build node filter metadata
        for node_id in self.G.nodes():
            is_in_direct_connection = node_id in self.direct_connection_nodes
            annotation_ids = [
                str(aid)
                for aid, nodes in annotation_node_sets.items()
                if node_id in nodes
            ]
            node_filter_metadata[str(node_id)] = {
                "is_in_direct_connection": is_in_direct_connection,
                "annotation_ids": annotation_ids,
            }

        # Build edge filter metadata
        for from_node, to_node in self.G.edges():
            edge_tuple = (from_node, to_node)
            is_in_direct_connection = edge_tuple in self.direct_connection_edges
            annotation_ids = [
                str(aid)
                for aid, edges in annotation_edge_sets.items()
                if edge_tuple in edges or tuple(reversed(edge_tuple)) in edges
            ]
            edge_key = f"{from_node},{to_node}"
            edge_filter_metadata[edge_key] = {
                "is_in_direct_connection": is_in_direct_connection,
                "annotation_ids": annotation_ids,
            }

        logger.debug(
            "Filter metadata prepared: node_meta=%d edge_meta=%d direct_nodes=%d direct_edges=%d",
            len(node_filter_metadata),
            len(edge_filter_metadata),
            len(self.direct_connection_nodes),
            len(self.direct_connection_edges),
        )

        # Generate HTML with filtering controls and annotation CRUD UI
        try:
            html_content = self._generate_html_with_controls(
                net,
                annotation_node_sets,
                annotation_edge_sets,
                use_layered=True,
                node_positions=node_positions,
                node_filter_metadata=node_filter_metadata,
                edge_filter_metadata=edge_filter_metadata,
            )
        except Exception:
            logger.exception(
                "Failed generating interactive HTML (nodes=%d edges=%d annotations=%d)",
                self.G.number_of_nodes(),
                self.G.number_of_edges(),
                len(self.annotations),
            )
            raise

        # Write to file
        if output_file is None:
            fd, output_file = tempfile.mkstemp(suffix=".html", prefix="genome_network_")
            os.close(fd)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_file

    def _build_react_payload(self) -> Dict[str, Any]:
        (
            annotation_node_sets,
            annotation_edge_sets,
            annotation_colors,
        ) = self._prepare_annotation_sets()
        node_positions = self._calculate_layered_positions()

        nodes_payload = []
        for node_id in self.G.nodes():
            annotation_ids = [
                ann_id
                for ann_id, nodes in annotation_node_sets.items()
                if node_id in nodes
            ]
            node_type = self._get_node_type(node_id)
            nodes_payload.append(
                {
                    "id": node_id,
                    "label": str(node_id),
                    "type": node_type,
                    "depth": self.node_depths.get(node_id, 0),
                    "color": self._get_node_color(node_id, annotation_colors),
                    "annotationIds": annotation_ids,
                    "isDirectConnection": node_id in self.direct_connection_nodes,
                    "position": (
                        {
                            "x": node_positions[node_id][0],
                            "y": node_positions[node_id][1],
                        }
                        if node_id in node_positions
                        else None
                    ),
                }
            )

        edges_payload = []
        for idx, (from_node, to_node, data) in enumerate(self.G.edges(data=True)):
            annotation_ids = [
                ann_id
                for ann_id, edges in annotation_edge_sets.items()
                if (from_node, to_node) in edges or (to_node, from_node) in edges
            ]
            weight = data.get("weight", 0.0)
            edges_payload.append(
                {
                    "id": f"edge_{idx}_{from_node}_{to_node}",
                    "from": from_node,
                    "to": to_node,
                    "weight": weight,
                    "color": self._get_edge_color(weight),
                    "annotationIds": annotation_ids,
                    "isDirectConnection": (from_node, to_node)
                    in self.direct_connection_edges,
                    "isSkip": self._is_skip_connection(from_node, to_node),
                }
            )

        annotations_payload = []
        for annotation in self.annotations:
            if isinstance(annotation, dict):
                ann_id = str(annotation.get("id", ""))
                name = annotation.get("name")
                hypothesis = annotation.get("hypothesis")
                nodes = annotation.get("subgraph_nodes") or []
                edges = annotation.get("subgraph_connections") or []
            else:
                ann_id = str(annotation.id)
                name = getattr(annotation, "name", None)
                hypothesis = getattr(annotation, "hypothesis", None)
                nodes = annotation.subgraph_nodes or []
                edges = annotation.subgraph_connections or []

            annotations_payload.append(
                {
                    "id": ann_id,
                    "name": name,
                    "hypothesis": hypothesis,
                    "nodes": nodes,
                    "edges": edges,
                }
            )

        layout_dims = None
        if node_positions:
            xs = [pos[0] for pos in node_positions.values()]
            ys = [pos[1] for pos in node_positions.values()]
            layout_dims = {
                "width": max(xs) - min(xs) if xs else 0,
                "height": max(ys) - min(ys) if ys else 0,
            }

        genome_info = getattr(self, "genome_info", None)
        genome_id = str(genome_info.genome_id) if genome_info else "UNKNOWN"

        payload = {
            "metadata": {
                "genomeId": genome_id,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "schemaVersion": 1,
                "layout": {"type": "layered", "dimensions": layout_dims},
            },
            "nodes": nodes_payload,
            "edges": edges_payload,
            "annotations": annotations_payload,
        }

        logger.debug(
            "React payload prepared: nodes=%d edges=%d annotations=%d",
            len(nodes_payload),
            len(edges_payload),
            len(annotations_payload),
        )
        return payload

    def visualize_react(self, output_dir: Optional[str] = None) -> str:
        """Generate React-based visualization assets."""
        payload = self._build_react_payload()
        if output_dir is None:
            output_path = Path(tempfile.mkdtemp(prefix="genome_network_react_"))
        else:
            output_path = Path(output_dir)
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        static_root = pkg_resources.files("explaneat.static.react_explorer")
        with pkg_resources.as_file(static_root) as src_dir:
            shutil.copytree(src_dir, output_path, dirs_exist_ok=True)

        index_path = output_path / "index.html"
        html = index_path.read_text(encoding="utf-8")

        # Remove favicon reference (not available in packaged assets)
        html = re.sub(r'\s*<link rel="icon"[^>]+>\s*', "\n", html)

        payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
        payload_tag = f'<script id="explorer-data" type="application/json">{payload_json}</script>'

        def inline_stylesheet(match):
            href = match.group("href")
            asset_path = output_path / href.lstrip("./")
            if not asset_path.exists():
                logger.warning("Stylesheet asset not found: %s", asset_path)
                return ""
            css_content = asset_path.read_text(encoding="utf-8")
            return f"<style>\n{css_content}\n</style>"

        html = re.sub(
            r'<link\s+rel="stylesheet"[^>]*href="(?P<href>[^"]+)"[^>]*>',
            inline_stylesheet,
            html,
        )

        payload_injected = {"done": False}

        def inline_module_script(match):
            src = match.group("src")
            asset_path = output_path / src.lstrip("./")
            if not asset_path.exists():
                logger.warning("Script asset not found: %s", asset_path)
                return ""
            script_content = asset_path.read_text(encoding="utf-8").replace(
                "</script>", "<\\/script>"
            )
            prefix = ""
            if not payload_injected["done"]:
                prefix = payload_tag + "\n"
                payload_injected["done"] = True
            return f'{prefix}<script type="module">\n{script_content}\n</script>'

        html = re.sub(
            r'<script\s+type="module"[^>]*src="(?P<src>[^"]+)"[^>]*></script>',
            inline_module_script,
            html,
        )

        if not payload_injected["done"]:
            if "</body>" in html:
                html = html.replace("</body>", f"    {payload_tag}</body>")
            else:
                html = f"{html}\n{payload_tag}"
        index_path.write_text(html, encoding="utf-8")

        logger.info("React explorer generated at %s", index_path)
        return str(index_path)

    def show_react(
        self, auto_open: bool = True, output_dir: Optional[str] = None
    ) -> str:
        """Display the React-based interactive visualization."""
        logger.debug(
            "InteractiveNetworkViewer.show_react called auto_open=%s", auto_open
        )
        output_file = self.visualize_react(output_dir=output_dir)

        if auto_open:
            webbrowser.open(f"file://{os.path.abspath(output_file)}")
            print(f"React network visualization opened in browser: {output_file}")
        else:
            print(f"React network visualization saved to: {output_file}")

        return output_file

    def _generate_html_with_controls(
        self,
        net: Network,
        annotation_node_sets: Dict,
        annotation_edge_sets: Dict,
        use_layered: bool = False,
        node_positions: Optional[Dict] = None,
        node_filter_metadata: Optional[Dict] = None,
        edge_filter_metadata: Optional[Dict] = None,
    ) -> str:
        """Generate HTML with interactive filtering controls."""
        logger.debug(
            "Generating HTML with controls: nodes=%d edges=%d annotations=%d",
            len(net.nodes),
            len(net.edges),
            len(self.annotations),
        )
        # Generate base HTML from Pyvis
        html = net.generate_html()

        # Note: We don't remove inline handlers from Pyvis HTML as it might break Pyvis functionality
        # Instead, we define global wrapper functions that can handle any inline handlers
        # and also use event listeners for better control

        processed_annotations: List[Dict[str, Any]] = []
        annotation_data_payload: Dict[str, Dict[str, Any]] = {}
        genome_info = getattr(self, "genome_info", None)
        genome_id = (
            str(genome_info.genome_id) if genome_info else "GENOME_ID_PLACEHOLDER"
        )

        for idx, annotation in enumerate(self.annotations):
            if isinstance(annotation, dict):
                ann_id = str(annotation.get("id", ""))
                ann_name = annotation.get("name") or f"Annotation {idx + 1}"
                hypothesis = annotation.get("hypothesis", "")
                nodes = annotation.get("subgraph_nodes") or []
                raw_edges = annotation.get("subgraph_connections") or []
            else:
                ann_id = str(annotation.id)
                ann_name = annotation.name or f"Annotation {idx + 1}"
                hypothesis = annotation.hypothesis or ""
                nodes = annotation.subgraph_nodes or []
                raw_edges = annotation.subgraph_connections or []

            edges = [
                list(edge) if isinstance(edge, tuple) else edge for edge in raw_edges
            ]

            processed_annotations.append(
                {
                    "id": ann_id,
                    "name": ann_name,
                    "hypothesis": hypothesis,
                    "nodes": nodes,
                    "edges": edges,
                }
            )

            annotation_data_payload[ann_id] = {
                "genome_id": genome_id,
                "nodes": nodes,
                "edges": edges,
                "name": ann_name if ann_name else None,
                "hypothesis": hypothesis,
            }

        # Extract the network HTML and add filtering controls and annotation CRUD UI
        # Find where to insert controls (before closing body tag)
        control_html = """
        <!-- Filter Controls -->
        <div id="filter-controls" style="position: fixed; top: 10px; right: 10px; background: white; padding: 15px; border: 2px solid #333; border-radius: 5px; z-index: 1000; max-width: 300px; max-height: 80vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0;">Filter Controls</h3>
            
            <div style="margin-bottom: 15px;">
                <strong>Direct Connections:</strong><br>
                <label>
                    <input type="checkbox" id="show_direct_connections" checked>
                    Show Direct Input→Output Connections
                </label>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>Annotations:</strong><br>
        """

        # Add annotation filter checkboxes
        if processed_annotations:
            for ann in processed_annotations:
                ann_name = ann["name"]
                ann_id = ann["id"]
                # Escape the ID for use in HTML/JS
                ann_id_escaped = ann_id.replace("'", "\\'").replace('"', '\\"')
                control_html += f"""
                <label>
                    <input type="checkbox" id="show_annotation_{ann_id_escaped}" checked class="annotation-filter-checkbox">
                    {ann_name}
                </label><br>
                """
        else:
            control_html += (
                "<p style='color: #666; font-size: 12px;'>No annotations available</p>"
            )

        control_html += """
            </div>
            
            <button id="reset-filters-btn" style="margin-top: 10px; padding: 5px 10px; cursor: pointer; width: 100%;">Reset All Filters</button>
        </div>
        
        <!-- Annotation CRUD UI -->
        <div id="annotation-panel" style="position: fixed; top: 10px; left: 10px; background: white; padding: 15px; border: 2px solid #333; border-radius: 5px; z-index: 1000; max-width: 350px; max-height: 90vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: none;">
            <h3 style="margin-top: 0;">Annotation Manager</h3>
            <button onclick="toggleAnnotationPanel()" style="float: right; padding: 2px 8px; cursor: pointer;">×</button>
            <div style="clear: both;"></div>
            
            <div style="margin-bottom: 15px;">
                <button onclick="showCreateAnnotation()" style="padding: 8px 15px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 3px; width: 100%;">+ Create New Annotation</button>
            </div>
            
            <div id="annotation-list" style="margin-bottom: 15px;">
                <strong>Existing Annotations:</strong><br>
        """

        # Add existing annotations to the list
        if processed_annotations:
            for ann in processed_annotations:
                ann_id = ann["id"]
                ann_name = ann["name"]
                hypothesis = ann["hypothesis"]
                # Escape for JavaScript
                ann_id_escaped = ann_id.replace("'", "\\'").replace('"', '\\"')
                control_html += f"""
                <div id="ann-item-{ann_id_escaped}" style="border: 1px solid #ddd; padding: 8px; margin: 5px 0; border-radius: 3px;">
                    <strong>{ann_name}</strong><br>
                    <small>{hypothesis[:50] if hypothesis else 'No hypothesis'}...</small><br>
                    <button onclick="editAnnotation('{ann_id_escaped}')" style="padding: 3px 8px; margin: 3px 0; cursor: pointer; font-size: 11px;">Edit</button>
                    <button onclick="deleteAnnotation('{ann_id_escaped}')" style="padding: 3px 8px; margin: 3px 0; cursor: pointer; font-size: 11px; background: #f44336; color: white; border: none;">Delete</button>
                    <button onclick="exportAnnotation('{ann_id_escaped}')" style="padding: 3px 8px; margin: 3px 0; cursor: pointer; font-size: 11px; background: #2196F3; color: white; border: none;">Export</button>
                </div>
                """
        else:
            control_html += "<p style='color: #666; font-size: 12px;'>No annotations yet. Create one to get started!</p>"

        control_html += """
            </div>
            
            <!-- Create/Edit Annotation Form -->
            <div id="annotation-form" style="display: none; border-top: 2px solid #ddd; padding-top: 15px; margin-top: 15px;">
                <h4 id="form-title">Create Annotation</h4>
                <form id="annotation-form-element" onsubmit="saveAnnotation(event)">
                    <input type="hidden" id="annotation-id" value="">
                    <label>Name: <input type="text" id="annotation-name" style="width: 100%; padding: 5px; margin: 5px 0;" placeholder="Optional name"></label><br>
                    <label>Hypothesis: <textarea id="annotation-hypothesis" style="width: 100%; padding: 5px; margin: 5px 0; min-height: 60px;" placeholder="Describe what this subgraph does..." required></textarea></label><br>
                    <label>Selected Nodes: <span id="selected-nodes-display" style="color: #666; font-size: 11px;">Click nodes to select</span></label><br>
                    <label>Selected Edges: <span id="selected-edges-display" style="color: #666; font-size: 11px;">Click edges to select</span></label><br>
                    <div style="margin-top: 10px;">
                        <button type="submit" style="padding: 8px 15px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 3px; margin-right: 5px;">Save</button>
                        <button type="button" onclick="cancelAnnotationForm()" style="padding: 8px 15px; cursor: pointer; background: #f44336; color: white; border: none; border-radius: 3px;">Cancel</button>
                    </div>
                </form>
            </div>
            
            <div style="margin-top: 15px; padding-top: 15px; border-top: 2px solid #ddd;">
                <button onclick="exportAllAnnotations()" style="padding: 8px 15px; cursor: pointer; background: #2196F3; color: white; border: none; border-radius: 3px; width: 100%;">Export All as Python Code</button>
            </div>
        </div>
        
        <button id="annotation-toggle-btn" onclick="toggleAnnotationPanel()" style="position: fixed; top: 10px; left: 10px; padding: 10px 15px; background: #2196F3; color: white; border: none; border-radius: 5px; z-index: 999; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">📝 Annotations</button>
        """

        initial_positions_payload: Dict[str, Dict[str, float]] = {}
        if node_positions:
            for node_id, (x, y) in node_positions.items():
                initial_positions_payload[str(node_id)] = {
                    "x": float(x),
                    "y": float(y),
                }

        node_filter_metadata = node_filter_metadata or {}
        edge_filter_metadata = edge_filter_metadata or {}

        data_payload = {
            "initialPositions": initial_positions_payload,
            "nodeFilterMetadata": node_filter_metadata,
            "edgeFilterMetadata": edge_filter_metadata,
            "annotationData": annotation_data_payload,
        }

        js_bundle_uri = self._get_genome_viewer_js_path().resolve().as_uri()

        control_html += f"""
        <script>
        window.GENOME_VIEWER_DATA = {json.dumps(data_payload)};
        </script>
        <script src="{js_bundle_uri}"></script>
        <script>
        (function initGenomeViewer() {{
            function start() {{
                if (window.GenomeViewer && typeof window.GenomeViewer.init === 'function') {{
                    window.GenomeViewer.init();
                }} else {{
                    setTimeout(start, 50);
                }}
            }}
            start();
        }})();
        </script>
        """

        # Insert controls before closing body tag
        html = html.replace("</body>", control_html + "</body>")

        # CRITICAL: Make sure the network variable is globally accessible
        # Pyvis creates 'var network = ...' which should be global, but let's ensure it
        # Also inject code to store it on window and trigger our initialization
        # Find where Pyvis creates the network and make it globally accessible
        # Pattern 1: var network = new vis.Network(...)
        network_init_pattern = (
            r"(var\s+network\s*=\s*new\s+vis\.Network\([^)]+\)[^;]*;)"
        )

        def make_network_global(match):
            # Add code to store network on window for external scripts
            return (
                match.group(0)
                + "\n        window.network = network; // Make globally accessible"
            )

        if re.search(network_init_pattern, html):
            html = re.sub(network_init_pattern, make_network_global, html, count=1)
        else:
            # Pattern 2: network = new vis.Network(...) (without var)
            network_init_pattern2 = (
                r"(\s+network\s*=\s*new\s+vis\.Network\([^)]+\)[^;]*;)"
            )
            if re.search(network_init_pattern2, html):
                html = re.sub(network_init_pattern2, make_network_global, html, count=1)
            else:
                # Pattern 3: Just look for vis.Network instantiation anywhere
                vis_network_pattern = r"(new\s+vis\.Network\([^)]+\)[^;]*;)"
                if re.search(vis_network_pattern, html):

                    def inject_after_vis_network(match):
                        return (
                            match.group(0)
                            + "\n        window.network = network; // Make globally accessible"
                        )

                    html = re.sub(
                        vis_network_pattern, inject_after_vis_network, html, count=1
                    )

        utils_tag = '<script src="lib/bindings/utils.js"></script>'
        if utils_tag in html:
            try:
                utils_res = (
                    pkg_resources.files("pyvis")
                    / "templates"
                    / "lib"
                    / "bindings"
                    / "utils.js"
                )
                with pkg_resources.as_file(utils_res) as utils_path:
                    utils_js = utils_path.read_text(encoding="utf-8")
                html = html.replace(utils_tag, f"<script>\n{utils_js}\n</script>")
            except FileNotFoundError:
                logger.warning(
                    "Could not inline pyvis utils.js asset; leaving original tag"
                )

        return html

    def show(self, auto_open: bool = True, **kwargs) -> str:
        """
        Generate and display interactive network.

        Args:
            auto_open: Whether to automatically open in browser
            **kwargs: Additional arguments passed to visualize()

        Returns:
            Path to generated HTML file
        """
        logger.debug("InteractiveNetworkViewer.show called auto_open=%s", auto_open)
        output_file = self.visualize(**kwargs)

        if auto_open:
            webbrowser.open(f"file://{os.path.abspath(output_file)}")
            logger.info("Interactive network visualization opened: %s", output_file)
            print(f"Interactive network visualization opened in browser: {output_file}")
        else:
            logger.info("Interactive network visualization saved: %s", output_file)
            print(f"Interactive network visualization saved to: {output_file}")

        return output_file
