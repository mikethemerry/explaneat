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
                    figsize: Tuple[int, int] = (12, 8),
                    node_size: int = 500,
                    show_weights: bool = True,
                    show_node_labels: bool = True,
                    layout: str = 'spring') -> None:
        """
        Plot the network structure of the genome.
        
        Args:
            figsize: Figure size
            node_size: Size of nodes in the plot
            show_weights: Whether to show connection weights
            show_node_labels: Whether to show node IDs
            layout: Layout algorithm ('spring', 'hierarchical', 'circular')
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
            
        # Set up the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Choose layout
        if layout == 'hierarchical':
            pos = self._hierarchical_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Color nodes by type
        node_colors = []
        for node_id in G.nodes():
            if node_id < 0:  # Input nodes
                node_colors.append('lightblue')
            elif node_id == 0:  # Output node
                node_colors.append('lightcoral')
            else:  # Hidden nodes
                node_colors.append('lightgreen')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_size,
                              ax=ax)
        
        # Draw edges with weights as thickness
        if edge_weights:
            # Normalize weights for edge thickness
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [3 * abs(w) / max_weight for w in edge_weights]
            
            # Color edges by weight (red for negative, blue for positive)
            edge_colors = []
            for conn_key, conn in self.genome.connections.items():
                if conn.enabled:
                    if conn.weight >= 0:
                        edge_colors.append('blue')
                    else:
                        edge_colors.append('red')
            
            nx.draw_networkx_edges(G, pos,
                                  width=edge_widths,
                                  edge_color=edge_colors,
                                  alpha=0.7,
                                  ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, ax=ax)
        
        # Add node labels
        if show_node_labels:
            nx.draw_networkx_labels(G, pos, ax=ax)
        
        # Add weight labels
        if show_weights and self.genome.connections:
            edge_labels = {}
            for conn_key, conn in self.genome.connections.items():
                if conn.enabled:
                    edge_labels[conn_key] = f"{conn.weight:.2f}"
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                        font_size=8, ax=ax)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Input Nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='Output Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='Hidden Nodes'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Positive Weights'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Negative Weights')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f'Network Structure - Genome {self.genome.key}\n'
                    f'Fitness: {self.genome.fitness:.3f}, '
                    f'Nodes: {len(self.genome.nodes)}, '
                    f'Connections: {len([c for c in self.genome.connections.values() if c.enabled])}')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _hierarchical_layout(self, G) -> Dict:
        """Create a hierarchical layout for feedforward networks"""
        pos = {}
        
        # Separate nodes by type
        input_nodes = [n for n in G.nodes() if n < 0]
        output_nodes = [n for n in G.nodes() if n == 0]
        hidden_nodes = [n for n in G.nodes() if n > 0]
        
        # Position input nodes
        for i, node in enumerate(sorted(input_nodes)):
            pos[node] = (0, i)
        
        # Position hidden nodes (if any)
        if hidden_nodes:
            for i, node in enumerate(sorted(hidden_nodes)):
                pos[node] = (1, i)
        
        # Position output nodes
        for i, node in enumerate(output_nodes):
            pos[node] = (2, i)
        
        return pos
    
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