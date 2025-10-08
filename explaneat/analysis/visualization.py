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


class GenomeVisualizer:
    """
    Visualizes NEAT genome network structures.
    """
    
    def __init__(self, neat_genome, config):
        self.genome = neat_genome
        self.config = config
        
    def plot_network(self,
                    figsize: Tuple[int, int] = (14, 10),
                    node_size: int = 800,
                    show_weights: bool = True,
                    show_node_labels: bool = True,
                    show_layers: bool = True,
                    layout: str = 'layered') -> None:
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
        if layout in ['layered', 'hierarchical']:
            pos = self._layered_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        # Auto-adjust figure size for layered layout
        if layout in ['layered', 'hierarchical'] and pos:
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
            nx.draw_networkx_nodes(G, pos,
                                  nodelist=input_nodes,
                                  node_color='#4A90E2',  # Blue
                                  node_size=node_size,
                                  node_shape='s',  # Square for inputs
                                  edgecolors='black',
                                  linewidths=2,
                                  ax=ax,
                                  label='Input Nodes')

        if output_nodes:
            nx.draw_networkx_nodes(G, pos,
                                  nodelist=output_nodes,
                                  node_color='#E24A4A',  # Red
                                  node_size=node_size,
                                  node_shape='h',  # Hexagon for outputs
                                  edgecolors='black',
                                  linewidths=2,
                                  ax=ax,
                                  label='Output Nodes')

        if hidden_nodes:
            nx.draw_networkx_nodes(G, pos,
                                  nodelist=hidden_nodes,
                                  node_color='#50C878',  # Green
                                  node_size=node_size * 0.8,
                                  node_shape='o',  # Circle for hidden
                                  edgecolors='black',
                                  linewidths=1.5,
                                  ax=ax,
                                  label='Hidden Nodes')
        
        # Draw edges with improved styling
        if edge_weights:
            # Calculate node depths for edge styling
            if layout in ['layered', 'hierarchical']:
                input_node_list = [n for n in G.nodes() if n < 0] or [n for n in G.nodes() if G.in_degree(n) == 0]
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
                    depth_diff = abs(node_depths.get(to_node, 0) - node_depths.get(from_node, 0))

                    # Edge width based on weight magnitude
                    width = 0.5 + 2.5 * abs(conn.weight) / max_weight

                    # Edge color based on weight sign
                    if conn.weight >= 0:
                        color = '#2E86DE'  # Blue for positive
                    else:
                        color = '#EE5A6F'  # Red for negative

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
                nx.draw_networkx_edges(G, pos,
                                      edgelist=regular_edges,
                                      width=regular_widths,
                                      edge_color=regular_colors,
                                      alpha=0.6,
                                      connectionstyle='arc3,rad=0.05',
                                      ax=ax)

            # Draw skip connections with curved edges
            if skip_edges:
                nx.draw_networkx_edges(G, pos,
                                      edgelist=skip_edges,
                                      width=skip_widths,
                                      edge_color=skip_colors,
                                      alpha=0.4,
                                      connectionstyle='arc3,rad=0.2',
                                      style='dashed',
                                      ax=ax)
        else:
            nx.draw_networkx_edges(G, pos,
                                  alpha=0.5,
                                  connectionstyle='arc3,rad=0.05',
                                  ax=ax)
        
        # Add node labels
        if show_node_labels:
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

        # Add weight labels (only if not too many connections)
        if show_weights and self.genome.connections and len(self.genome.connections) < 30:
            edge_labels = {}
            for conn_key, conn in self.genome.connections.items():
                if conn.enabled:
                    edge_labels[conn_key] = f"{conn.weight:.2f}"

            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                        font_size=7, ax=ax)

        # Add layer annotations for layered layout
        if show_layers and layout in ['layered', 'hierarchical'] and pos:
            input_node_list = [n for n in G.nodes() if n < 0] or [n for n in G.nodes() if G.in_degree(n) == 0]
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
                ax.axvline(x=x, color='gray', linestyle=':', alpha=0.3, linewidth=1)
                ax.text(x, y_max + y_range * 0.05, f'Layer {depth}',
                       ha='center', va='bottom', fontsize=9, color='gray')

        # Add improved legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4A90E2',
                      markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                      label='Input Nodes'),
            plt.Line2D([0], [0], marker='h', color='w', markerfacecolor='#E24A4A',
                      markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                      label='Output Nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#50C878',
                      markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                      label='Hidden Nodes'),
            plt.Line2D([0], [0], color='#2E86DE', linewidth=2, label='Positive Weights'),
            plt.Line2D([0], [0], color='#EE5A6F', linewidth=2, label='Negative Weights'),
            plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='dashed',
                      alpha=0.4, label='Skip Connections')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        
        # Enhanced title with network statistics
        enabled_connections = len([c for c in self.genome.connections.values() if c.enabled])
        disabled_connections = len(self.genome.connections) - enabled_connections

        if layout in ['layered', 'hierarchical'] and pos:
            input_node_list = [n for n in G.nodes() if n < 0] or [n for n in G.nodes() if G.in_degree(n) == 0]
            node_depths = self._calculate_node_depths(G, input_node_list)
            max_depth = max(node_depths.values()) if node_depths else 0
            title_suffix = f', Depth: {max_depth + 1} layers'
        else:
            title_suffix = ''

        ax.set_title(f'Network Structure - Genome {self.genome.key}\n'
                    f'Fitness: {self.genome.fitness:.3f} | '
                    f'Nodes: {len(self.genome.nodes)} | '
                    f'Connections: {enabled_connections}/{len(self.genome.connections)}{title_suffix}',
                    fontsize=12, fontweight='bold', pad=20)

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

        ax.axis('off')
        ax.set_aspect('equal')

        # Enable interactive zoom and pan
        # Store initial limits for reset functionality
        if pos:
            initial_xlim = ax.get_xlim()
            initial_ylim = ax.get_ylim()

            def on_key(event):
                """Handle keyboard shortcuts for zoom and pan"""
                if event.key == '+' or event.key == '=':
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
                elif event.key == '-' or event.key == '_':
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
                elif event.key == 'r' or event.key == 'home':
                    # Reset to initial view
                    ax.set_xlim(initial_xlim)
                    ax.set_ylim(initial_ylim)
                    fig.canvas.draw()
                elif event.key == 'left':
                    # Pan left
                    xlim = ax.get_xlim()
                    shift = (xlim[1] - xlim[0]) * 0.1
                    ax.set_xlim(xlim[0] - shift, xlim[1] - shift)
                    fig.canvas.draw()
                elif event.key == 'right':
                    # Pan right
                    xlim = ax.get_xlim()
                    shift = (xlim[1] - xlim[0]) * 0.1
                    ax.set_xlim(xlim[0] + shift, xlim[1] + shift)
                    fig.canvas.draw()
                elif event.key == 'up':
                    # Pan up
                    ylim = ax.get_ylim()
                    shift = (ylim[1] - ylim[0]) * 0.1
                    ax.set_ylim(ylim[0] + shift, ylim[1] + shift)
                    fig.canvas.draw()
                elif event.key == 'down':
                    # Pan down
                    ylim = ax.get_ylim()
                    shift = (ylim[1] - ylim[0]) * 0.1
                    ax.set_ylim(ylim[0] - shift, ylim[1] - shift)
                    fig.canvas.draw()
                elif event.key == 'f':
                    # Fit to window (reset view)
                    ax.set_xlim(initial_xlim)
                    ax.set_ylim(initial_ylim)
                    fig.canvas.draw()

            # Connect keyboard handler
            fig.canvas.mpl_connect('key_press_event', on_key)

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
                zoom_factor = 0.9 if event.button == 'up' else 1.1

                # Calculate new limits centered on mouse position
                x_range = (xlim[1] - xlim[0]) * zoom_factor
                y_range = (ylim[1] - ylim[0]) * zoom_factor

                # Zoom towards mouse position
                x_ratio = (x_data - xlim[0]) / (xlim[1] - xlim[0])
                y_ratio = (y_data - ylim[0]) / (ylim[1] - ylim[0])

                ax.set_xlim(x_data - x_range * x_ratio, x_data + x_range * (1 - x_ratio))
                ax.set_ylim(y_data - y_range * y_ratio, y_data + y_range * (1 - y_ratio))
                fig.canvas.draw()

            # Connect scroll handler
            fig.canvas.mpl_connect('scroll_event', on_scroll)

            # Add instructions to title
            instructions = "\nControls: +/- zoom | Arrow keys pan | R reset | Scroll wheel zoom | Click+drag pan"
            current_title = ax.get_title()
            ax.set_title(current_title + f"\n{instructions}", fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.show()
    
    def _calculate_node_depths(self, G: nx.DiGraph, input_nodes: List[int]) -> Dict[int, int]:
        """
        Calculate depth of each node as maximum path length from inputs.

        Uses PropNEAT algorithm: depth = max(all path lengths from input nodes).
        This matches the GPU layer mapping approach.

        Args:
            G: Directed graph of the network
            input_nodes: List of input node IDs

        Returns:
            Dictionary mapping node_id to depth (layer number)
        """
        depths = {node: [] for node in G.nodes()}

        # Initialize input nodes at depth 0
        for input_node in input_nodes:
            depths[input_node] = [0]

        # Breadth-first search from all input nodes
        queue = [(node, 0) for node in input_nodes]

        while queue:
            node, depth = queue.pop(0)

            # Propagate to all successors
            for successor in G.successors(node):
                depths[successor].append(depth + 1)
                queue.append((successor, depth + 1))

        # Take maximum depth for each node (longest path from inputs)
        return {node: max(d) if d else 0 for node, d in depths.items()}

    def _minimize_crossings_barycenter(self,
                                       G: nx.DiGraph,
                                       layers: Dict[int, List[int]],
                                       max_iterations: int = 4) -> Dict[int, List[int]]:
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
                        pred_positions = [prev_positions[p] for p in predecessors if p in prev_positions]
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
                        succ_positions = [next_positions[s] for s in successors if s in next_positions]
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

        # Identify node types
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
        vertical_spacing = 1.5    # Space between nodes in a layer

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
        axes[0].hist(biases, bins=10, alpha=0.7, edgecolor='black')
        axes[0].set_title('Node Bias Distribution')
        axes[0].set_xlabel('Bias Value')
        axes[0].set_ylabel('Count')
        
        # Plot response distribution
        axes[1].hist(responses, bins=10, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_title('Node Response Distribution')
        axes[1].set_xlabel('Response Value')
        axes[1].set_ylabel('Count')
        
        # Plot activation function usage
        activation_counts = pd.Series(activations).value_counts()
        axes[2].bar(activation_counts.index, activation_counts.values, alpha=0.7, color='green')
        axes[2].set_title('Activation Function Usage')
        axes[2].set_xlabel('Activation Function')
        axes[2].set_ylabel('Count')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'Node Properties - Genome {self.genome.key}')
        plt.tight_layout()
        plt.show()
    
    def plot_connection_properties(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """Plot distribution of connection properties"""
        if not self.genome.connections:
            print("No connections to analyze")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Collect connection properties
        weights = [conn.weight for conn in self.genome.connections.values() if conn.enabled]
        disabled_count = sum(1 for conn in self.genome.connections.values() if not conn.enabled)
        enabled_count = len(weights)
        
        if weights:
            # Plot weight distribution
            axes[0].hist(weights, bins=15, alpha=0.7, edgecolor='black')
            axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero')
            axes[0].set_title('Connection Weight Distribution')
            axes[0].set_xlabel('Weight Value')
            axes[0].set_ylabel('Count')
            axes[0].legend()
        
        # Plot enabled vs disabled connections
        labels = ['Enabled', 'Disabled']
        sizes = [enabled_count, disabled_count]
        colors = ['lightgreen', 'lightcoral']
        
        axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Connection Status')
        
        plt.suptitle(f'Connection Properties - Genome {self.genome.key}')
        plt.tight_layout()
        plt.show()


class AncestryVisualizer:
    """
    Visualizes ancestry trees and evolutionary patterns.
    """
    
    def __init__(self, ancestry_analyzer):
        self.analyzer = ancestry_analyzer
    
    def plot_ancestry_tree(self, 
                          max_generations: int = 10,
                          figsize: Tuple[int, int] = (14, 10)) -> None:
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
            G.add_node(ancestor['genome_id'], 
                      generation=ancestor['generation'],
                      fitness=ancestor['fitness'],
                      neat_id=ancestor['neat_genome_id'])
        
        # Add edges (parent relationships)
        for _, ancestor in ancestry_df.iterrows():
            if ancestor['parent1_id']:
                G.add_edge(ancestor['parent1_id'], ancestor['genome_id'])
            if ancestor['parent2_id']:
                G.add_edge(ancestor['parent2_id'], ancestor['genome_id'])
        
        # Create layout based on generations
        pos = {}
        generation_groups = ancestry_df.groupby('generation')
        
        for gen, group in generation_groups:
            gen_nodes = group['genome_id'].tolist()
            for i, node_id in enumerate(gen_nodes):
                pos[node_id] = (gen, i - len(gen_nodes)/2)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Color nodes by fitness
        fitnesses = [G.nodes[node]['fitness'] for node in G.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, 
                              node_color=fitnesses,
                              node_size=300,
                              cmap='viridis',
                              ax=ax)
        
        nx.draw_networkx_edges(G, pos, 
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              ax=ax)
        
        # Add labels with NEAT genome IDs
        labels = {node: str(G.nodes[node]['neat_id']) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=min(fitnesses), 
                                                    vmax=max(fitnesses)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Fitness')
        
        ax.set_title('Ancestry Tree\n(Arrows point from parent to child)')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Position in Generation')
        
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
        axes[0, 0].plot(ancestry_df['generation'], ancestry_df['fitness'], 
                       'o-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Fitness Progression')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Network complexity over generations
        axes[0, 1].plot(ancestry_df['generation'], ancestry_df['num_nodes'], 
                       'g-o', label='Nodes', linewidth=2)
        axes[0, 1].plot(ancestry_df['generation'], ancestry_df['num_connections'], 
                       'b-s', label='Connections', linewidth=2)
        axes[0, 1].set_title('Network Complexity')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Enabled vs total connections
        enabled_ratio = ancestry_df['num_enabled_connections'] / ancestry_df['num_connections']
        enabled_ratio = enabled_ratio.fillna(0)
        
        axes[1, 0].plot(ancestry_df['generation'], enabled_ratio, 
                       'r-o', linewidth=2)
        axes[1, 0].set_title('Connection Efficiency')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Enabled Connections Ratio')
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Network dimensions
        axes[1, 1].plot(ancestry_df['generation'], ancestry_df['network_depth'], 
                       'purple', marker='o', label='Depth', linewidth=2)
        axes[1, 1].plot(ancestry_df['generation'], ancestry_df['network_width'], 
                       'orange', marker='s', label='Width', linewidth=2)
        axes[1, 1].set_title('Network Dimensions')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Size')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Lineage Analysis Over Generations', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_gene_origins_timeline(self, current_genome, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot when genes were introduced in the lineage.
        """
        gene_origins_df = self.analyzer.trace_gene_origins(current_genome)
        
        if gene_origins_df.empty:
            print("No gene origin data to visualize")
            return
        
        # Separate nodes and connections
        nodes_df = gene_origins_df[gene_origins_df['gene_type'] == 'node']
        connections_df = gene_origins_df[gene_origins_df['gene_type'] == 'connection']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot node origins
        if not nodes_df.empty:
            node_origins = nodes_df['origin_generation'].dropna()
            if len(node_origins) > 0:
                ax1.hist(node_origins, bins=max(1, len(node_origins.unique())), 
                        alpha=0.7, edgecolor='black', color='lightgreen')
            ax1.set_title('Node Introduction Timeline')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Number of Nodes Added')
            ax1.grid(True, alpha=0.3)
        
        # Plot connection origins
        if not connections_df.empty:
            conn_origins = connections_df['origin_generation'].dropna()
            if len(conn_origins) > 0:
                ax2.hist(conn_origins, bins=max(1, len(conn_origins.unique())), 
                        alpha=0.7, edgecolor='black', color='lightblue')
            ax2.set_title('Connection Introduction Timeline')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Number of Connections Added')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Gene Introduction Timeline')
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
        ax1.scatter(ancestry_df['num_connections'], ancestry_df['fitness'], 
                   c=ancestry_df['generation'], cmap='viridis', s=60)
        ax1.set_xlabel('Number of Connections')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness vs Connection Count')
        ax1.grid(True, alpha=0.3)
        
        # Fitness vs number of nodes
        scatter = ax2.scatter(ancestry_df['num_nodes'], ancestry_df['fitness'], 
                            c=ancestry_df['generation'], cmap='viridis', s=60)
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Fitness vs Node Count')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=[ax1, ax2])
        cbar.set_label('Generation')
        
        plt.suptitle('Fitness vs Network Complexity')
        plt.tight_layout()
        plt.show()