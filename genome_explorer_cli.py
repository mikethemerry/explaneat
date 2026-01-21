#!/usr/bin/env python3
"""
Genome Explorer CLI - Interactive command-line interface for exploring genome data

This CLI allows you to:
- List experiments and select one
- Load specific genomes by ID or generation
- Visualize network structures and ancestry
- Explore evolutionary progression
- Export data for further analysis
"""

import argparse
import logging
import sys
from typing import Optional, List, Dict, Any, Tuple, Union
import pandas as pd

from explaneat.analysis.genome_explorer import GenomeExplorer
from explaneat.analysis.annotation_manager import AnnotationManager
from explaneat.analysis.explanation_manager import ExplanationManager
from explaneat.analysis.node_splitting import NodeSplitManager
from explaneat.analysis.subgraph_validator import SubgraphValidator
from explaneat.analysis.coverage import (
    compute_structural_coverage,
    compute_compositional_coverage,
    CoverageComputer,
)
from explaneat.db import db

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GenomeExplorerCLI:
    """Interactive CLI for exploring genome data"""

    def __init__(self, debug: bool = False):
        self.current_explorer = None
        self.current_experiment_id = None
        self.current_page = 1
        self.page_size = 20
        self.debug = debug

    def list_experiments(
        self,
        show_numbers: bool = True,
        page: int = 1,
        page_size: Optional[int] = 20,
        interactive: bool = False,
    ) -> Tuple[pd.DataFrame, int]:
        """List all experiments in the database with pagination

        Args:
            show_numbers: If True, show row numbers for easy selection
            page: Page number to display (1-indexed)
            page_size: Number of experiments per page. None or 0 means show all
            interactive: If True, allow interactive navigation
        """
        try:
            experiments_df = GenomeExplorer.list_experiments()
            if experiments_df.empty:
                print("No experiments found in database")
                return experiments_df, 1

            # Sort by creation date (most recent first)
            experiments_df = experiments_df.sort_values(
                "created_at", ascending=False
            ).reset_index(drop=True)

            total_experiments = len(experiments_df)

            # Handle pagination: if page_size is None or 0, show all
            if page_size is None or page_size <= 0:
                page_df = experiments_df
                total_pages = 1
                start_idx = 0
                end_idx = total_experiments
            else:
                total_pages = (
                    total_experiments + page_size - 1
                ) // page_size  # Ceiling division

                # Validate page number
                if page < 1:
                    page = 1
                elif page > total_pages and total_pages > 0:
                    page = total_pages

                # Calculate slice indices
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, total_experiments)
                page_df = experiments_df.iloc[start_idx:end_idx]

            print("\n📊 Available Experiments:")
            print("=" * 90)
            if show_numbers:
                print(
                    f"{'#':<4} {'Name':<30} {'Status':<10} {'Gens':<6} {'Best Fitness':<12} {'ID':<36}"
                )
            else:
                print(
                    f"{'ID':<36} {'Name':<25} {'Status':<10} {'Generations':<12} {'Best Fitness':<12}"
                )
            print("-" * 90)

            for idx, exp in page_df.iterrows():
                fitness_str = (
                    f"{exp['best_fitness']:.3f}" if exp["best_fitness"] else "N/A"
                )
                if show_numbers:
                    # Use the original index from the full dataframe for selection
                    print(
                        f"{idx:<4} {exp['name'][:29]:<30} {exp['status']:<10} {exp['generations']:<6} {fitness_str:<12} {exp['experiment_id']:<36}"
                    )
                else:
                    print(
                        f"{exp['experiment_id']:<36} {exp['name'][:24]:<25} {exp['status']:<10} {exp['generations']:<12} {fitness_str:<12}"
                    )

            # Show pagination info
            print("-" * 90)
            if total_pages > 1:
                print(
                    f"Page {page} of {total_pages} (Showing {start_idx + 1}-{end_idx} of {total_experiments} experiments)"
                )
                if interactive:
                    print(
                        "💡 Navigation: 'n' (next), 'p' (prev), 'page N' (go to page), 'q' (quit)"
                    )
            else:
                print(f"Total: {total_experiments} experiment(s)")

            return experiments_df, page

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return pd.DataFrame(), 1

    def select_experiment(self, selector: str = None) -> bool:
        """Select an experiment to explore

        Args:
            selector: Can be:
                - Full experiment UUID
                - Number from the list (0, 1, 2, etc.)
                - 'latest' or 'best' for most recent experiment
                - Substring of experiment name (e.g., 'backache')
        """
        try:
            if selector:
                # Get experiments list
                experiments_df = GenomeExplorer.list_experiments()
                if experiments_df.empty:
                    print("❌ No experiments found")
                    return False

                # Sort by creation (most recent first) and reset index
                experiments_df = experiments_df.sort_values(
                    "created_at", ascending=False
                ).reset_index(drop=True)

                selected_id = None

                # Try as number first
                if selector.isdigit():
                    idx = int(selector)
                    if 0 <= idx < len(experiments_df):
                        selected_id = experiments_df.iloc[idx]["experiment_id"]
                        print(
                            f"📍 Selected experiment #{idx}: {experiments_df.iloc[idx]['name']}"
                        )
                    else:
                        print(
                            f"❌ Invalid experiment number: {idx} (valid range: 0-{len(experiments_df)-1})"
                        )
                        return False

                # Try special keywords
                elif selector.lower() in ["latest", "best", "recent"]:
                    selected_id = experiments_df.iloc[0]["experiment_id"]
                    print(
                        f"📍 Selected latest experiment: {experiments_df.iloc[0]['name']}"
                    )

                # Try as full UUID
                elif selector in experiments_df["experiment_id"].values:
                    selected_id = selector
                    exp_name = experiments_df[
                        experiments_df["experiment_id"] == selector
                    ].iloc[0]["name"]
                    print(f"📍 Selected experiment: {exp_name}")

                # Try as name substring
                else:
                    matches = experiments_df[
                        experiments_df["name"].str.contains(
                            selector, case=False, na=False
                        )
                    ]
                    if len(matches) == 1:
                        selected_id = matches.iloc[0]["experiment_id"]
                        print(
                            f"📍 Found matching experiment: {matches.iloc[0]['name']}"
                        )
                    elif len(matches) > 1:
                        print(f"❌ Multiple experiments match '{selector}':")
                        for idx, exp in matches.iterrows():
                            print(f"  {idx}: {exp['name']}")
                        print("Please use the number to select.")
                        return False
                    else:
                        print(f"❌ No experiment found matching: {selector}")
                        return False

                # Load the selected experiment
                if selected_id:
                    self.current_explorer = GenomeExplorer.load_best_genome(selected_id)
                    self.current_experiment_id = selected_id
                    print(f"✅ Loaded experiment: {selected_id}")
                    return True

            else:
                # Interactive selection - show all experiments (no pagination)
                experiments_df, _ = self.list_experiments(
                    page_size=None, interactive=False
                )
                if experiments_df.empty:
                    return False

                print(
                    f"\n💡 Enter: number (0-{len(experiments_df)-1}), 'latest', name substring, or full UUID"
                )
                print("   Or 'q' to quit")
                choice = input("> ").strip()

                if choice.lower() == "q":
                    return False

                return self.select_experiment(choice)

        except Exception as e:
            logger.error(f"Failed to select experiment: {e}")
            return False

    def show_summary(self):
        """Show summary of current genome"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        print("\n📋 Genome Summary:")
        print("=" * 50)
        self.current_explorer.summary()

    def show_network(self, **kwargs):
        """Show network structure"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        print("🕸️  Displaying network structure...")
        self.current_explorer.show_network(**kwargs)

    def show_genotype(self):
        """Print genotype network structure (all nodes and connections)"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        if not self.current_explorer.explainer:
            print("❌ ExplaNEAT not initialized.")
            return

        print("\n🧬 Genotype Network Structure")
        print("=" * 70)
        genotype = self.current_explorer.explainer.get_genotype_network()

        print(f"\n📊 Summary:")
        print(f"  Total Nodes: {len(genotype.nodes)}")
        print(f"  Total Connections: {len(genotype.connections)}")
        enabled_conns = len([c for c in genotype.connections if c.enabled])
        disabled_conns = len([c for c in genotype.connections if not c.enabled])
        print(f"  Enabled Connections: {enabled_conns}")
        print(f"  Disabled Connections: {disabled_conns}")
        print(f"  Input Nodes: {len(genotype.input_node_ids)}")
        print(f"  Output Nodes: {len(genotype.output_node_ids)}")

        print(f"\n🔵 Input Nodes: {genotype.input_node_ids}")
        print(f"🔴 Output Nodes: {genotype.output_node_ids}")

        # Group nodes by type
        input_nodes = [n for n in genotype.nodes if n.id in genotype.input_node_ids]
        output_nodes = [n for n in genotype.nodes if n.id in genotype.output_node_ids]
        hidden_nodes = [
            n
            for n in genotype.nodes
            if n.id not in genotype.input_node_ids
            and n.id not in genotype.output_node_ids
        ]

        if input_nodes:
            print(f"\n📥 Input Nodes ({len(input_nodes)}):")
            for node in sorted(input_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(
                    f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}"
                )

        if output_nodes:
            print(f"\n📤 Output Nodes ({len(output_nodes)}):")
            for node in sorted(output_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(
                    f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}"
                )

        if hidden_nodes:
            print(f"\n⚪ Hidden Nodes ({len(hidden_nodes)}):")
            for node in sorted(hidden_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(
                    f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}"
                )

        print(f"\n🔗 Connections ({len(genotype.connections)}):")
        enabled_connections = [c for c in genotype.connections if c.enabled]
        disabled_connections = [c for c in genotype.connections if not c.enabled]

        if enabled_connections:
            print(f"  ✅ Enabled ({len(enabled_connections)}):")
            for conn in sorted(
                enabled_connections, key=lambda c: (c.from_node, c.to_node)
            ):
                print(
                    f"    {conn.from_node:4d} → {conn.to_node:4d}: weight={conn.weight:8.4f}"
                )

        if disabled_connections:
            print(f"  ❌ Disabled ({len(disabled_connections)}):")
            for conn in sorted(
                disabled_connections, key=lambda c: (c.from_node, c.to_node)
            ):
                print(
                    f"    {conn.from_node:4d} → {conn.to_node:4d}: weight={conn.weight:8.4f}"
                )

        print()

    def show_phenotype(self):
        """Print phenotype network structure (active, reachable subgraph)"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        if not self.current_explorer.explainer:
            print("❌ ExplaNEAT not initialized.")
            return

        print("\n🧬 Phenotype Network Structure")
        print("=" * 70)
        phenotype = self.current_explorer.explainer.get_phenotype_network()

        print(f"\n📊 Summary:")
        print(f"  Active Nodes: {len(phenotype.nodes)}")
        print(f"  Active Connections: {len(phenotype.connections)}")
        print(f"  Input Nodes: {len(phenotype.input_node_ids)}")
        print(f"  Output Nodes: {len(phenotype.output_node_ids)}")

        if "pruned_nodes" in phenotype.metadata:
            print(f"  Pruned Nodes: {phenotype.metadata['pruned_nodes']}")
        if "pruned_connections" in phenotype.metadata:
            print(f"  Pruned Connections: {phenotype.metadata['pruned_connections']}")

        print(f"\n🔵 Input Nodes: {phenotype.input_node_ids}")
        print(f"🔴 Output Nodes: {phenotype.output_node_ids}")

        # Group nodes by type
        input_nodes = [n for n in phenotype.nodes if n.id in phenotype.input_node_ids]
        output_nodes = [n for n in phenotype.nodes if n.id in phenotype.output_node_ids]
        hidden_nodes = [
            n
            for n in phenotype.nodes
            if n.id not in phenotype.input_node_ids
            and n.id not in phenotype.output_node_ids
        ]

        if input_nodes:
            print(f"\n📥 Input Nodes ({len(input_nodes)}):")
            for node in sorted(input_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(
                    f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}"
                )

        if output_nodes:
            print(f"\n📤 Output Nodes ({len(output_nodes)}):")
            for node in sorted(output_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(
                    f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}"
                )

        if hidden_nodes:
            print(f"\n⚪ Hidden Nodes ({len(hidden_nodes)}):")
            for node in sorted(hidden_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(
                    f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}"
                )

        print(f"\n🔗 Active Connections ({len(phenotype.connections)}):")
        for conn in sorted(
            phenotype.connections, key=lambda c: (c.from_node, c.to_node)
        ):
            print(
                f"  {conn.from_node:4d} → {conn.to_node:4d}: weight={conn.weight:8.4f}"
            )

        print()

    def plot_training_metrics(self):
        """Plot training metrics"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        print("📈 Plotting training metrics...")
        self.current_explorer.plot_training_metrics()

    def plot_ancestry_fitness(self, max_generations: Optional[int] = None):
        """Plot ancestry fitness progression

        Args:
            max_generations: Maximum generations to trace. None = full history
        """
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        print("🌳 Plotting ancestry fitness progression...")
        self.current_explorer.plot_ancestry_fitness(max_generations)

    def plot_evolution_progression(self, max_generations: int = 50):
        """Plot full evolution progression"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        print("🧬 Plotting full evolution progression...")
        self.current_explorer.plot_evolution_progression(max_generations)

    def export_data(self, filename: str = None):
        """Export genome data"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        print("💾 Exporting genome data...")
        data = self.current_explorer.export_genome_data()

        if filename:
            import json

            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"✅ Data exported to {filename}")
        else:
            print("📊 Data exported (use --filename to save to file)")
            print(f"   Contains {len(data)} sections")

    @staticmethod
    def parse_node_list(node_str: str) -> List[int]:
        """Parse space or comma-separated node list (e.g., '-2 -3 6' or '-2, -3, 6' -> [-2, -3, 6])"""
        try:
            # Strip quotes if present
            node_str = node_str.strip("'\"")
            # Split by comma first, then by whitespace, to handle both formats
            # This allows: "1,2,3", "1 2 3", "1, 2, 3", etc.
            parts = []
            for part in node_str.split(","):
                parts.extend(part.split())
            nodes = [int(n.strip()) for n in parts if n.strip()]
            return nodes
        except ValueError as e:
            raise ValueError(f"Invalid node list format: {node_str}. Error: {e}")

    @staticmethod
    def parse_annotate_command(command: List[str]) -> tuple:
        """Parse annotate command arguments, handling quoted strings"""
        import shlex

        # Rejoin and reparse to handle quotes properly
        full_cmd = " ".join(command)
        try:
            parsed = shlex.split(full_cmd)
            if len(parsed) < 4:
                return None, None, None, None
            start_nodes = parsed[1]
            end_nodes = parsed[2]
            hypothesis = parsed[3] if len(parsed) == 4 else " ".join(parsed[3:-1])
            name = parsed[-1] if len(parsed) > 4 else None
            return start_nodes, end_nodes, hypothesis, name
        except ValueError:
            # Fallback to simple parsing
            if len(command) < 4:
                return None, None, None, None
            start_nodes = command[1].strip("'\"")
            end_nodes = command[2].strip("'\"")
            hypothesis = (
                " ".join(command[3:-1]).strip("'\"")
                if len(command) > 4
                else " ".join(command[3:]).strip("'\"")
            )
            name = command[-1].strip("'\"") if len(command) > 4 else None
            return start_nodes, end_nodes, hypothesis, name

    def create_annotation(
        self,
        start_nodes_str: str,
        end_nodes_str: str,
        hypothesis: str,
        name: str = None,
    ):
        """Create an annotation by finding reachable subgraph between start and end nodes"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return None

        try:
            # Parse node lists
            start_nodes = self.parse_node_list(start_nodes_str)
            end_nodes = self.parse_node_list(end_nodes_str)

            print(
                f"🔍 Finding subgraph from start nodes {start_nodes} to end nodes {end_nodes}..."
            )

            # Find reachable subgraph
            subgraph_result = SubgraphValidator.find_reachable_subgraph(
                self.current_explorer.neat_genome,
                start_nodes,
                end_nodes,
                config=self.current_explorer.config,  # Pass config to access input/output keys
            )

            if not subgraph_result["is_valid"]:
                print(
                    f"❌ Subgraph validation failed: {subgraph_result.get('error_message', 'Unknown error')}"
                )
                if subgraph_result.get("unreachable_ends"):
                    print(
                        f"   Unreachable end nodes: {subgraph_result['unreachable_ends']}"
                    )
                return None

            nodes = subgraph_result["nodes"]
            connections = subgraph_result["connections"]

            print(
                f"✅ Found subgraph with {len(nodes)} nodes and {len(connections)} connections"
            )
            print(f"   Nodes: {nodes}")

            # Create annotation with entry and exit nodes
            annotation_dict = AnnotationManager.create_annotation(
                genome_id=self.current_explorer.genome_info.genome_id,
                nodes=nodes,
                connections=connections,
                hypothesis=hypothesis,
                entry_nodes=start_nodes,  # Start nodes are the entry nodes
                exit_nodes=end_nodes,  # End nodes are the exit nodes
                name=name,
                validate_against_genome=False,  # Already validated
            )

            print(f"✅ Created annotation: {annotation_dict['id']}")
            print(f"   Name: {annotation_dict.get('name') or '(unnamed)'}")
            hypothesis_text = annotation_dict.get("hypothesis", "")
            print(
                f"   Hypothesis: {hypothesis_text[:60]}..."
                if len(hypothesis_text) > 60
                else f"   Hypothesis: {hypothesis_text}"
            )
            return annotation_dict

        except ValueError as e:
            print(f"❌ Error: {e}")
            return None
        except Exception as e:
            print(f"❌ Failed to create annotation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _check_nodes_needing_splits(
        self,
        annotation_nodes: List[int],
        annotation_connections: List[Tuple[int, int]],
        entry_nodes: List[Any],  # Can contain int node IDs or str split_node_ids like "5_a"
        exit_nodes: List[Any],  # Can contain int node IDs or str split_node_ids like "5_a"
    ) -> Dict[int, Dict[str, Any]]:
        """
        Check which nodes in the annotation need splitting.

        A node needs splitting if:
        - It's in the annotation (entry, intermediate, or exit nodes)
        - It has outgoing connections that are NOT in the annotation

        Note: ALL nodes (entry, intermediate, and exit) are checked for splits.
        After splitting, nodes with all outgoing connections in the annotation
        are considered "covered" and can be collapsed.

        Args:
            annotation_nodes: List of node IDs in the annotation
            annotation_connections: List of (from, to) tuples in the annotation
            entry_nodes: List of entry node IDs
            exit_nodes: List of exit node IDs

        Returns:
            Dictionary mapping node_id -> {
                'in_annotation': list of connections in annotation,
                'not_in_annotation': list of connections NOT in annotation
            }
        """
        if not self.current_explorer:
            return {}

        # Get phenotype network to find all outgoing connections
        from explaneat.core.explaneat import ExplaNEAT

        explainer = ExplaNEAT(
            self.current_explorer.neat_genome, self.current_explorer.config
        )
        phenotype = explainer.get_phenotype_network()

        # Build set of annotation connections for fast lookup
        annotation_conn_set = set(
            tuple(conn) if isinstance(conn, (list, tuple)) else conn
            for conn in annotation_connections
        )

        # Build mapping of node -> all outgoing connections
        node_outgoing = {}
        for conn in phenotype.connections:
            if conn.enabled:
                from_node = conn.from_node
                to_node = conn.to_node
                if from_node not in node_outgoing:
                    node_outgoing[from_node] = []
                node_outgoing[from_node].append((from_node, to_node))

        # Check nodes for splits, but skip exit nodes (they're expected to have outgoing connections)
        nodes_needing_splits = {}
        exit_nodes_set = {str(nid) for nid in exit_nodes}

        for node_id in annotation_nodes:
            node_id_str = str(node_id)  # Convert to string for comparison
            
            # Skip exit nodes - they're expected to have outgoing connections outside the annotation
            if node_id_str in exit_nodes_set:
                continue
            
            # Skip input nodes (negative string IDs like "-1") - they shouldn't be split
            try:
                if int(node_id_str) < 0:
                    continue
            except ValueError:
                # If it's not a number (e.g., a split node), allow it
                pass
            
            # Get all outgoing connections for this node (using string ID)
            all_outgoing = node_outgoing.get(node_id_str, [])
            if not all_outgoing:
                continue  # No outgoing connections, no split needed

            # Separate connections into in-annotation and not-in-annotation
            in_annotation = [
                conn for conn in all_outgoing if conn in annotation_conn_set_str
            ]
            not_in_annotation = [
                conn for conn in all_outgoing if conn not in annotation_conn_set_str
            ]

            # If node has connections outside the annotation, it needs splitting
            if not_in_annotation:
                nodes_needing_splits[node_id_str] = {
                    "in_annotation": in_annotation,
                    "not_in_annotation": not_in_annotation,
                }

        return nodes_needing_splits

    def _create_automatic_splits(
        self,
        nodes_needing_splits: Dict[int, Dict[str, Any]],
        annotation_connections: List[Tuple[int, int]],
    ) -> Dict[str, List[str]]:
        """
        Create automatic node splits for nodes that need them.

        For each node, creates a dedicated split node for EACH outgoing connection.
        This ensures full splitting - each split node has exactly one outgoing connection.
        Multiple split nodes can be included in an annotation to recombine them.

        Args:
            nodes_needing_splits: Dictionary from _check_nodes_needing_splits (keys are string node IDs)
            annotation_connections: List of connections in the annotation (for reference)

        Returns:
            Dictionary mapping original_node_id (string) -> list of split_node_ids (strings like "5_a", "5_b")
            that have connections within the annotation (for use in the annotation)
        """
        if not self.current_explorer:
            raise ValueError("Genome must be selected")

        genome_id = str(self.current_explorer.genome_info.genome_id)
        # Automatically get or create the single explanation for this genome
        explanation = ExplanationManager.get_or_create_explanation(genome_id)
        explanation_id = explanation["id"]

        # Mapping: original_node_id -> list of split_node_ids with connections in annotation
        split_mapping = {}

        # Build set of annotation connections for fast lookup
        annotation_conn_set = set(
            tuple(conn) if isinstance(conn, (list, tuple)) else conn
            for conn in annotation_connections
        )

        for node_id, info in nodes_needing_splits.items():
            # Check existing splits for this node
            existing_split = NodeSplitManager.get_splits_for_node(
                genome_id, node_id, explanation_id
            )

            # Get all outgoing connections (both in and out of annotation)
            all_outgoing = info["in_annotation"] + info["not_in_annotation"]

            # Find used letters from existing split IDs
            used_letters = set()
            if existing_split:
                existing_split_mappings = existing_split.get("split_mappings", {})
                for split_id in existing_split_mappings.keys():
                    if isinstance(split_id, str) and split_id.startswith(f"{node_id}_"):
                        letter_part = split_id.split("_", 1)[1]
                        if len(letter_part) == 1 and letter_part.isalpha():
                            used_letters.add(letter_part.lower())

            # Track split nodes that have connections in the annotation
            annotation_split_ids = []

            # Build split_mappings: one split per outgoing connection
            # Convert node_id to string and connections to string tuples
            node_id_str = str(node_id)
            split_mappings: Dict[str, List[Tuple[str, str]]] = {}
            letters = "abcdefghijklmnopqrstuvwxyz"
            letter_idx = 0

            for conn in all_outgoing:
                # Find next available letter
                while letter_idx < len(letters) and letters[letter_idx] in used_letters:
                    letter_idx += 1
                
                if letter_idx >= len(letters):
                    raise ValueError(
                        f"Too many splits for node {node_id} (max 26 splits per node: a-z)"
                    )

                # Create split_node_id as string: "node_id_letter"
                letter = letters[letter_idx]
                split_node_id = f"{node_id_str}_{letter}"
                used_letters.add(letter)
                letter_idx += 1

                # Convert connection to string tuple
                conn_str = (str(conn[0]), str(conn[1]))
                
                # Add to split_mappings (each split has exactly one connection)
                split_mappings[split_node_id] = [conn_str]
                
                # Track if this split has a connection in the annotation
                if conn in annotation_conn_set:
                    annotation_split_ids.append(split_node_id)

            # Create complete split with all mappings at once
            try:
                NodeSplitManager.create_complete_split(
                    genome_id=genome_id,
                    original_node_id=node_id_str,
                    split_mappings=split_mappings,
                    explanation_id=explanation_id,
                    annotation_id=None,  # Will be set after annotation is created
                )
                
                print(
                    f"  ✓ Created complete split for node {node_id} "
                    f"with {len(split_mappings)} split nodes: {list(split_mappings.keys())}"
                )
                
                # Store mapping: original node -> list of split nodes in annotation
                if annotation_split_ids:
                    split_mapping[node_id] = annotation_split_ids
                    
            except Exception as e:
                print(
                    f"  ⚠️  Failed to create split for node {node_id}: {e}"
                )

        return split_mapping

    def guided_annotation(self):
        """Interactive guided process for creating annotations"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        print("\n📝 Guided Annotation Creation")
        print("=" * 70)
        print(
            "This will help you create an annotation by specifying start and end nodes."
        )
        print("The system will find the reachable subgraph between them.\n")

        # Step 1: Start nodes
        while True:
            start_input = input(
                "Step 1/4: Enter start nodes (comma-separated, e.g., '-2,-3,6'): "
            ).strip()
            if not start_input:
                print(
                    "❌ Start nodes cannot be empty. Try again or type 'cancel' to exit."
                )
                continue
            if start_input.lower() == "cancel":
                print("❌ Annotation creation cancelled.")
                return

            try:
                start_nodes = self.parse_node_list(start_input)
                if not start_nodes:
                    print("❌ No valid nodes found. Try again.")
                    continue
                print(f"✅ Start nodes: {start_nodes}")
                break
            except ValueError as e:
                print(f"❌ {e}. Try again or type 'cancel' to exit.")

        # Step 2: Find and show valid end nodes
        print(
            f"\n🔍 Finding valid end nodes reachable from start nodes {start_nodes}..."
        )
        if self.debug:
            print("📊 Debug information:")
        valid_end_nodes = SubgraphValidator.find_valid_end_nodes(
            self.current_explorer.neat_genome,
            start_nodes,
            config=self.current_explorer.config,
            debug=self.debug,
        )

        if not valid_end_nodes:
            print("❌ No valid end nodes found. All end nodes must be:")
            print("   1. Reachable from ALL start nodes")
            print("   2. Have all their inputs accounted for in the subgraph")
            print("\n💡 Try different start nodes or check the network structure.")
            return

        print(f"\n✅ Found {len(valid_end_nodes)} valid end node(s):")
        # Display in columns for readability
        for i, node in enumerate(valid_end_nodes):
            if i > 0 and i % 10 == 0:
                print()
            print(f"  {node:>6}", end="")
        print("\n")

        # Step 2: End nodes
        while True:
            end_input = input(
                "Step 2/4: Enter end nodes (space or comma-separated, e.g., '0' or '1321 452'): "
            ).strip()
            if not end_input:
                print(
                    "❌ End nodes cannot be empty. Try again or type 'cancel' to exit."
                )
                continue
            if end_input.lower() == "cancel":
                print("❌ Annotation creation cancelled.")
                return

            try:
                end_nodes = self.parse_node_list(end_input)
                if not end_nodes:
                    print("❌ No valid nodes found. Try again.")
                    continue

                # Validate that selected end nodes are in the valid list
                invalid_selected = [n for n in end_nodes if n not in valid_end_nodes]
                if invalid_selected:
                    print(
                        f"❌ Some selected end nodes are not valid: {invalid_selected}"
                    )
                    print(f"   Valid end nodes are: {valid_end_nodes}")
                    print(
                        "   End nodes must be reachable from ALL start nodes and have all inputs accounted for."
                    )
                    continue

                print(f"✅ End nodes: {end_nodes}")
                break
            except ValueError as e:
                print(f"❌ {e}. Try again or type 'cancel' to exit.")

        # Step 3: Find and validate subgraph
        print(
            f"\n🔍 Finding subgraph from start nodes {start_nodes} to end nodes {end_nodes}..."
        )
        subgraph_result = SubgraphValidator.find_reachable_subgraph(
            self.current_explorer.neat_genome,
            start_nodes,
            end_nodes,
            config=self.current_explorer.config,  # Pass config to access input/output keys
        )

        if not subgraph_result["is_valid"]:
            print(
                f"❌ Subgraph validation failed: {subgraph_result.get('error_message', 'Unknown error')}"
            )
            if subgraph_result.get("unreachable_ends"):
                print(
                    f"   Unreachable end nodes: {subgraph_result['unreachable_ends']}"
                )
            print(
                "\n💡 Tip: Make sure the end nodes are reachable from start nodes following the graph direction."
            )
            return

        nodes = subgraph_result["nodes"]
        connections = subgraph_result["connections"]

        print(
            f"✅ Found subgraph with {len(nodes)} nodes and {len(connections)} connections"
        )
        print(f"   Nodes: {nodes}")
        print(f"   Connections: {len(connections)}")
        if len(connections) <= 10:
            print("   Connection list:")
            for conn in connections:
                print(f"     {conn[0]} -> {conn[1]}")
        else:
            print("   Sample connections (first 5):")
            for conn in connections[:5]:
                print(f"     {conn[0]} -> {conn[1]}")
            print(f"     ... and {len(connections) - 5} more")

        # Step 3: Hypothesis
        print("\n" + "=" * 70)
        print("Step 3/4: Describe what this subgraph does (your hypothesis)")
        print("(You can enter multiple lines. Type 'END' on a new line when finished)")
        hypothesis_lines = []
        while True:
            line = input("Hypothesis> ").strip()
            if line.upper() == "END":
                break
            if line.lower() == "cancel":
                print("❌ Annotation creation cancelled.")
                return
            hypothesis_lines.append(line)

        hypothesis = "\n".join(hypothesis_lines).strip()
        if not hypothesis:
            print("❌ Hypothesis cannot be empty. Annotation creation cancelled.")
            return

        print(
            f"✅ Hypothesis: {hypothesis[:100]}..."
            if len(hypothesis) > 100
            else f"✅ Hypothesis: {hypothesis}"
        )

        # Step 4: Name (optional)
        print("\n" + "=" * 70)
        name_input = input(
            "Step 4/4: Enter annotation name (optional, press Enter to skip): "
        ).strip()
        name = name_input if name_input else None
        if name:
            print(f"✅ Name: {name}")
        else:
            print("✅ No name provided (annotation will be unnamed)")

        # Summary and confirmation
        print("\n" + "=" * 70)
        print("📋 Annotation Summary:")
        print(f"   Start nodes: {start_nodes}")
        print(f"   End nodes: {end_nodes}")
        print(f"   Subgraph: {len(nodes)} nodes, {len(connections)} connections")
        print(
            f"   Hypothesis: {hypothesis[:60]}..."
            if len(hypothesis) > 60
            else f"   Hypothesis: {hypothesis}"
        )
        print(f"   Name: {name or '(unnamed)'}")
        print("=" * 70)

        # Check for nodes that need splitting
        nodes_needing_splits = self._check_nodes_needing_splits(
            nodes, connections, start_nodes, end_nodes
        )

        if nodes_needing_splits:
            print("\n" + "=" * 70)
            print("⚠️  Node Splitting Required")
            print("=" * 70)
            print(
                f"Found {len(nodes_needing_splits)} node(s) with outgoing connections"
                " not fully covered by this annotation:"
            )
            for node_id, info in nodes_needing_splits.items():
                print(f"\n  Node {node_id}:")
                print(
                    f"    Outgoing connections in annotation: {len(info['in_annotation'])}"
                )
                print(
                    f"    Outgoing connections NOT in annotation: {len(info['not_in_annotation'])}"
                )
                print(f"    Missing connections: {info['not_in_annotation']}")

            print(
                "\nThese nodes will be fully split:"
                "\n  - Each outgoing connection gets its own dedicated split node"
                "\n  - Split nodes with connections in the annotation will be included"
                "\n  - This prevents the need for multiple rounds of splitting"
            )
            print("\n⚠️  Annotation creation cannot proceed without node splitting.")
            print("\n✓ Using genome's explanation (automatically created if needed)")

            split_confirm = (
                input("\nCreate automatic node splits? (yes/no): ").strip().lower()
            )
            if split_confirm not in ["yes", "y"]:
                print("❌ Annotation creation cancelled. Node splitting is required.")
                return

            print("\n🔧 Creating node splits (one split per outgoing connection)...")
            split_mapping = {}  # original_node_id -> list of annotation_split_node_ids
            try:
                # Create splits and get mapping
                split_mapping = self._create_automatic_splits(
                    nodes_needing_splits, connections
                )
                print("✅ Node splits created successfully!")

            except Exception as e:
                print(f"❌ Failed to create node splits: {e}")
                import traceback

                traceback.print_exc()
                print(
                    "\n❌ Annotation creation cancelled due to split creation failure."
                )
                return

        confirm = input("\nCreate this annotation? (yes/no): ").strip().lower()
        if confirm not in ["yes", "y"]:
            print("❌ Annotation creation cancelled.")
            return

        # Map original nodes to split nodes if splits exist
        # split_mapping: original_node_id -> list of split_node_ids with connections in annotation
        mapped_nodes = []
        mapped_connections = []
        mapped_entry_nodes = []
        mapped_exit_nodes = []

        if split_mapping:
            # Build mapping from connection to split node ID
            # For each connection, find which split node has that connection
            connection_to_split = {}  # (from_node, to_node) -> split_node_id
            
            # Get all splits to build connection mapping
            genome_id = str(self.current_explorer.genome_info.genome_id)
            explanation = ExplanationManager.get_or_create_explanation(genome_id)
            explanation_id = explanation["id"]
            
            for orig_node_id, split_node_ids in split_mapping.items():
                for split_node_id in split_node_ids:
                    split_conns = NodeSplitManager.get_split_node_connections(
                        genome_id, split_node_id
                    )
                    # split_conns are already string tuples
                    for conn in split_conns:
                        connection_to_split[conn] = split_node_id

            # Map nodes: include all split nodes that have connections in the annotation
            # Also include nodes that weren't split
            # Convert all node IDs to strings
            for node_id in nodes:
                node_id_str = str(node_id)
                if node_id in split_mapping:
                    # Include all split nodes for this original node (already strings)
                    mapped_nodes.extend(split_mapping[node_id])
                else:
                    # Node wasn't split, convert to string
                    mapped_nodes.append(node_id_str)

            # Map connections: use the split node that has this specific outgoing connection
            # For to_node, if it's split, all split nodes share incoming connections,
            # so we can use any split node of to_node that's in the annotation
            # Convert all node IDs to strings
            for from_node, to_node in connections:
                from_node_str = str(from_node)
                to_node_str = str(to_node)
                conn_key = (from_node_str, to_node_str)
                # Find the split node of from_node that has this connection
                mapped_from = connection_to_split.get(conn_key, from_node_str)
                
                # For to_node: if it's split, use the first split node in the annotation
                # (all splits share incoming connections, so any will work)
                if to_node in split_mapping:
                    to_splits = split_mapping[to_node]
                    mapped_to = to_splits[0] if to_splits else to_node_str
                else:
                    mapped_to = to_node_str
                
                mapped_connections.append((mapped_from, mapped_to))

            # Map entry nodes: include all relevant split nodes
            # Convert all node IDs to strings
            for node_id in start_nodes:
                node_id_str = str(node_id)
                if node_id in split_mapping:
                    mapped_entry_nodes.extend(split_mapping[node_id])  # Already strings
                else:
                    mapped_entry_nodes.append(node_id_str)

            # Map exit nodes: include all relevant split nodes
            # Convert all node IDs to strings
            for node_id in end_nodes:
                node_id_str = str(node_id)
                if node_id in split_mapping:
                    mapped_exit_nodes.extend(split_mapping[node_id])  # Already strings
                else:
                    mapped_exit_nodes.append(node_id_str)

            print(f"\n🔄 Mapped nodes to split nodes:")
            for orig_id, split_ids in split_mapping.items():
                print(f"   Node {orig_id} -> Splits {split_ids}")
        else:
            # No splits, convert all node IDs to strings
            mapped_nodes = [str(n) for n in nodes]
            mapped_connections = [(str(f), str(t)) for f, t in connections]
            mapped_entry_nodes = [str(n) for n in start_nodes]
            mapped_exit_nodes = [str(n) for n in end_nodes]

        # Create annotation with entry and exit nodes (using split nodes if applicable)
        try:
            annotation_dict = AnnotationManager.create_annotation(
                genome_id=self.current_explorer.genome_info.genome_id,
                nodes=mapped_nodes,
                connections=mapped_connections,
                hypothesis=hypothesis,
                entry_nodes=mapped_entry_nodes,  # Entry nodes (may be split nodes)
                exit_nodes=mapped_exit_nodes,  # Exit nodes (may be split nodes)
                name=name,
                validate_against_genome=False,  # Already validated
            )

            annotation_id = annotation_dict["id"]
            annotation_name = annotation_dict["name"] or "(unnamed)"

            print(f"\n✅ Annotation created successfully!")
            print(f"   ID: {annotation_id}")
            print(f"   Name: {annotation_name}")
            if split_mapping:
                print(f"   ✓ Annotation uses split nodes (not original nodes)")
            print(f"   View with: ann-show {annotation_id}")

        except Exception as e:
            print(f"❌ Failed to create annotation: {e}")
            import traceback

            traceback.print_exc()

    def list_annotations(self):
        """List all annotations for the current genome"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        annotations = AnnotationManager.get_annotations(
            self.current_explorer.genome_info.genome_id
        )

        if not annotations:
            print("📝 No annotations found for this genome.")
            return

        print(
            f"\n📝 Annotations for genome {self.current_explorer.genome_info.genome_id}:"
        )
        print("=" * 90)
        print(
            f"{'ID':<36} {'Name':<20} {'Nodes':<15} {'Connections':<12} {'Created':<20}"
        )
        print("-" * 90)

        for ann in annotations:
            name = ann.get("name") or "(unnamed)"
            if len(name) > 18:
                name = name[:15] + "..."
            nodes_str = f"{len(ann.get('subgraph_nodes', []))} nodes"
            conn_str = f"{len(ann.get('subgraph_connections', []))}"
            created_at = ann.get("created_at")
            if created_at:
                # created_at is an ISO format string from to_dict()
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, AttributeError):
                    created = created_at[:16] if len(created_at) > 16 else created_at
            else:
                created = "N/A"
            print(
                f"{str(ann.get('id')):<36} {name:<20} {nodes_str:<15} {conn_str:<12} {created:<20}"
            )

        print("-" * 90)
        print(f"Total: {len(annotations)} annotation(s)")

    def show_annotation(self, identifier: str):
        """
        Show details of a specific annotation.
        
        Supports multiple identifier types:
        - Index: "0", "1", "2", etc. (0-based index in the annotation list)
        - Name: exact name match (case-sensitive)
        - UUID: full UUID string
        
        Args:
            identifier: Annotation identifier (index, name, or UUID)
        """
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = self.current_explorer.genome_info.genome_id
        
        # Try to resolve the identifier
        annotation_dict = None
        
        # First, check if it's a numeric index
        try:
            index = int(identifier)
            annotations = AnnotationManager.get_annotations(genome_id)
            if 0 <= index < len(annotations):
                annotation_dict = annotations[index]
            else:
                print(f"❌ Annotation index {index} out of range. Use 0-{len(annotations)-1}.")
                return
        except ValueError:
            # Not a number, try UUID or name lookup
            # First try UUID lookup
            annotation_obj = AnnotationManager.get_annotation(identifier)
            if annotation_obj:
                # Check if it belongs to current genome
                if str(annotation_obj.genome_id) == str(genome_id):
                    annotation_dict = annotation_obj.to_dict()
                else:
                    print(f"❌ Annotation {identifier} belongs to a different genome.")
                    return
            else:
                # Try name lookup
                annotations = AnnotationManager.get_annotations(genome_id)
                matching = [ann for ann in annotations if ann.get("name") == identifier]
                if matching:
                    annotation_dict = matching[0]
                elif len(matching) > 1:
                    print(f"❌ Multiple annotations found with name '{identifier}'. Use UUID or index instead.")
                    return
                else:
                    print(f"❌ Annotation '{identifier}' not found (tried UUID and name lookup).")
                    print(f"   Use 'ann-list' to see available annotations.")
                    return

        if not annotation_dict:
            print(f"❌ Annotation {identifier} not found.")
            return

        # Display annotation details
        print(f"\n📝 Annotation Details:")
        print("=" * 70)
        print(f"ID:          {annotation_dict.get('id')}")
        print(f"Name:        {annotation_dict.get('name') or '(unnamed)'}")
        print(f"Genome ID:   {annotation_dict.get('genome_id')}")
        print(f"Created:     {annotation_dict.get('created_at', 'N/A')}")
        print(f"Updated:     {annotation_dict.get('updated_at', 'N/A')}")
        print(f"Connected:   {annotation_dict.get('is_connected', False)}")
        
        if annotation_dict.get('entry_nodes'):
            print(f"Entry Nodes: {annotation_dict.get('entry_nodes')}")
        if annotation_dict.get('exit_nodes'):
            print(f"Exit Nodes:  {annotation_dict.get('exit_nodes')}")
        
        print(f"\nHypothesis:")
        hypothesis = annotation_dict.get('hypothesis', '')
        if hypothesis:
            # Word wrap long hypotheses
            words = hypothesis.split()
            lines = []
            current_line = []
            current_length = 0
            for word in words:
                if current_length + len(word) + 1 > 70:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            if current_line:
                lines.append(' '.join(current_line))
            for line in lines:
                print(f"  {line}")
        else:
            print("  (no hypothesis)")
        
        print(f"\nSubgraph:")
        subgraph_nodes = annotation_dict.get('subgraph_nodes', [])
        subgraph_connections = annotation_dict.get('subgraph_connections', [])
        print(f"  Nodes ({len(subgraph_nodes)}): {subgraph_nodes}")
        print(f"  Connections ({len(subgraph_connections)}):")
        for conn in subgraph_connections[:10]:  # Show first 10
            print(f"    {conn[0]} -> {conn[1]}")
        if len(subgraph_connections) > 10:
            print(f"    ... and {len(subgraph_connections) - 10} more")

        evidence = annotation_dict.get('evidence')
        if evidence:
            print(f"\nEvidence:")
            print(
                f"  Analytical methods: {len(evidence.get('analytical_methods', []))}"
            )
            print(f"  Visualizations: {len(evidence.get('visualizations', []))}")
            print(f"  Counterfactuals: {len(evidence.get('counterfactuals', []))}")
            print(f"  Other evidence: {len(evidence.get('other_evidence', []))}")
        else:
            print(f"\nEvidence: None")
        
        # Show parent/children if applicable
        parent_id = annotation_dict.get('parent_annotation_id')
        if parent_id:
            print(f"\nParent Annotation: {parent_id}")
        
        explanation_id = annotation_dict.get('explanation_id')
        if explanation_id:
            print(f"Explanation ID: {explanation_id}")

    def delete_annotation(self, annotation_id: str):
        """Delete an annotation"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        deleted = AnnotationManager.delete_annotation(annotation_id)
        if deleted:
            print(f"✅ Deleted annotation {annotation_id}")
        else:
            print(f"❌ Annotation {annotation_id} not found.")

    # Explanation management methods
    def create_explanation(
        self, name: Optional[str] = None, description: Optional[str] = None
    ):
        """Create a new explanation for the current genome"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = str(self.current_explorer.genome_info.genome_id)
        explanation = ExplanationManager.create_explanation(
            genome_id, name, description
        )
        if explanation:
            print(f"✅ Created explanation: {explanation['id']}")
            print(f"   Name: {explanation.get('name') or '(unnamed)'}")
            # Note: Each genome has a single explanation, automatically managed
        else:
            print("❌ Failed to create explanation.")

    def list_explanations(self):
        """List all explanations for the current genome"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = str(self.current_explorer.genome_info.genome_id)
        explanations = ExplanationManager.get_explanations(genome_id)

        if not explanations:
            print("📝 No explanations found for this genome.")
            return

        print(f"\n📝 Explanations for genome {genome_id}:")
        print("=" * 90)
        print(
            f"{'ID':<36} {'Name':<20} {'Well-formed':<12} {'Coverage':<20} {'Created':<20}"
        )
        print("-" * 90)

        for exp in explanations:
            name = exp.get("name") or "(unnamed)"
            if len(name) > 18:
                name = name[:15] + "..."
            well_formed = "✓" if exp.get("is_well_formed") else "✗"
            struct_cov = exp.get("structural_coverage")
            comp_cov = exp.get("compositional_coverage")
            coverage_str = (
                f"{struct_cov:.2f}/{comp_cov:.2f}"
                if struct_cov is not None and comp_cov is not None
                else "N/A"
            )
            created_at = exp.get("created_at")
            if created_at:
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, AttributeError):
                    created = created_at[:16] if len(created_at) > 16 else created_at
            else:
                created = "N/A"
            print(
                f"{str(exp.get('id')):<36} {name:<20} {well_formed:<12} {coverage_str:<20} {created:<20}"
            )

        print("-" * 90)
        print(f"Total: {len(explanations)} explanation(s)")

    def select_explanation(self, explanation_id: str):
        """Select an explanation to work with (deprecated - each genome has a single explanation)"""
        print(
            "⚠️  Note: Each genome has a single explanation that is automatically managed."
        )
        print("   Explanation selection is no longer needed.")
        explanation = ExplanationManager.get_explanation(explanation_id)
        if explanation:
            print(f"   Explanation info: {explanation.get('name') or explanation_id}")
        else:
            print(f"❌ Explanation {explanation_id} not found.")

    def show_coverage_metrics(self):
        """Show coverage metrics for the genome's explanation"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = str(self.current_explorer.genome_info.genome_id)
        # Get or create the single explanation for this genome
        explanation = ExplanationManager.get_or_create_explanation(genome_id)
        explanation_id = explanation["id"]

        print(
            f"\n📊 Coverage Metrics for Genome Explanation: {explanation.get('name') or explanation_id}"
        )
        print("=" * 70)

        # Compute coverage if not cached
        if explanation.get("structural_coverage") is None:
            print("Computing coverage metrics...")
            ExplanationManager.compute_and_cache_coverage(explanation_id)
            explanation = ExplanationManager.get_explanation(explanation_id)

        struct_cov = explanation.get("structural_coverage", 0.0)
        comp_cov = explanation.get("compositional_coverage", 0.0)

        print(
            f"Structural Coverage (C_V^struct): {struct_cov:.4f} ({struct_cov*100:.2f}%)"
        )
        print(
            f"Compositional Coverage (C_V^comp): {comp_cov:.4f} ({comp_cov*100:.2f}%)"
        )
        print(
            f"Well-formed: {'✓ Yes' if explanation.get('is_well_formed') else '✗ No'}"
        )

    def show_annotation_hierarchy(self):
        """Show annotation hierarchy for the genome's explanation"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = str(self.current_explorer.genome_info.genome_id)
        annotations = ExplanationManager.get_explanation_annotations(
            genome_id=genome_id
        )
        if not annotations:
            print("📝 No annotations in this explanation.")
            return

        # Build hierarchy mapping
        annotations_by_id = {str(ann.get("id")): ann for ann in annotations}
        children_by_parent: Dict[str, List[Dict[str, Any]]] = {}

        for ann in annotations:
            parent_id = ann.get("parent_annotation_id")
            if parent_id:
                parent_id_str = str(parent_id)
                if parent_id_str not in children_by_parent:
                    children_by_parent[parent_id_str] = []
                children_by_parent[parent_id_str].append(ann)

        # Find root annotations
        root_annotations = [
            ann for ann in annotations if not ann.get("parent_annotation_id")
        ]

        def print_annotation_tree(ann, indent=0):
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            name = ann.get("name") or f"Annotation {str(ann.get('id'))[:8]}"
            ann_id_str = str(ann.get("id"))
            has_children = (
                ann_id_str in children_by_parent
                and len(children_by_parent[ann_id_str]) > 0
            )
            ann_type = "Composition" if has_children else "Leaf"
            print(f"{prefix}{name} ({ann_type})")
            if has_children:
                for child in children_by_parent[ann_id_str]:
                    print_annotation_tree(child, indent + 1)

        print(f"\n🌳 Annotation Hierarchy:")
        print("=" * 70)
        if root_annotations:
            for root in root_annotations:
                print_annotation_tree(root)
        else:
            print("No root annotations found (all annotations have parents).")

    def create_node_split(
        self,
        original_node_id: int,
        split_mappings: Dict[str, List[Tuple[int, int]]],
    ):
        """Create a complete node split with all split mappings.
        
        Args:
            original_node_id: ID of the node being split
            split_mappings: Dict mapping split_node_id (str) -> list of outgoing connections
                Example: {"5_a": [[5, 10]], "5_b": [[5, 20]]}
        """
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = str(self.current_explorer.genome_info.genome_id)
        # Automatically get or create the single explanation for this genome
        explanation = ExplanationManager.get_or_create_explanation(genome_id)
        explanation_id = explanation["id"]

        try:
            split = NodeSplitManager.create_complete_split(
                genome_id,
                original_node_id,
                split_mappings,
                explanation_id=explanation_id,
            )
            if split:
                print(f"✅ Created node split: {split['id']}")
                print(f"   Split nodes: {list(split_mappings.keys())}")
            else:
                print("❌ Failed to create node split.")
        except Exception as e:
            print(f"❌ Error creating node split: {e}")

    def list_node_splits(self):
        """List node splits for the genome's explanation"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = str(self.current_explorer.genome_info.genome_id)
        splits = NodeSplitManager.get_splits_for_explanation(genome_id=genome_id)

        if not splits:
            print("📝 No node splits found for this explanation.")
            return

        print(f"\n🔀 Node Splits for Explanation:")
        print("=" * 90)
        print(
            f"{'Original Node':<15} {'Split Nodes':<30} {'Total Splits':<15} {'ID':<36}"
        )
        print("-" * 90)

        for split in splits:
            orig_id = split.get("original_node_id")
            split_mappings = split.get("split_mappings", {})
            split_node_ids = list(split_mappings.keys())
            split_count = len(split_node_ids)
            # Show first few split IDs, truncate if many
            split_display = ", ".join(split_node_ids[:5])
            if len(split_node_ids) > 5:
                split_display += f" ... (+{len(split_node_ids) - 5} more)"
            print(
                f"{orig_id:<15} {split_display:<30} {split_count:<15} {str(split.get('id')):<36}"
            )

        print("-" * 90)
        print(f"Total: {len(splits)} node(s) split")

    def show_phenotype_with_splits(self):
        """Show phenotype network with splits applied"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = str(self.current_explorer.genome_info.genome_id)
        # Get or create the single explanation for this genome
        explanation = ExplanationManager.get_or_create_explanation(genome_id)
        explanation_id = explanation["id"]

        try:
            phenotype = ExplanationManager.get_phenotype_with_splits(explanation_id)
            print("\n🧬 Phenotype Network Structure (with splits)")
            print("=" * 70)
            print(f"\n📊 Summary:")
            print(f"  Nodes: {len(phenotype.nodes)}")
            print(f"  Connections: {len(phenotype.connections)}")
            print(f"  Input Nodes: {len(phenotype.input_node_ids)}")
            print(f"  Output Nodes: {len(phenotype.output_node_ids)}")
            if phenotype.metadata.get("has_splits"):
                print(f"  ✓ Splits applied")
        except Exception as e:
            print(f"❌ Error getting phenotype with splits: {e}")

    def create_direct_connections_annotation(self):
        """Create an annotation for all direct input-output connections where inputs have no other connections"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        genome_id = str(self.current_explorer.genome_info.genome_id)

        # Get phenotype network
        from explaneat.core.explaneat import ExplaNEAT

        explainer = ExplaNEAT(
            self.current_explorer.neat_genome, self.current_explorer.config
        )
        phenotype = explainer.get_phenotype_network()

        # Get input and output nodes
        input_nodes = set(phenotype.input_node_ids)
        output_nodes = set(phenotype.output_node_ids)

        # Build mapping of outgoing edges per input node
        input_outgoing_edges: Dict[int, List[Tuple[int, int]]] = {}
        for conn in phenotype.connections:
            if conn.enabled and conn.from_node in input_nodes:
                if conn.from_node not in input_outgoing_edges:
                    input_outgoing_edges[conn.from_node] = []
                input_outgoing_edges[conn.from_node].append(
                    (conn.from_node, conn.to_node)
                )

        # Find inputs that have ONLY direct connections to outputs (no other outgoing connections)
        qualifying_inputs = []
        direct_connection_edges = []

        for input_node in input_nodes:
            outgoing = input_outgoing_edges.get(input_node, [])

            # Check if all outgoing connections are direct to outputs
            if outgoing:
                all_direct_to_outputs = all(
                    to_node in output_nodes for _, to_node in outgoing
                )

                if all_direct_to_outputs:
                    qualifying_inputs.append(input_node)
                    direct_connection_edges.extend(outgoing)

        if not qualifying_inputs:
            print(
                "📝 No direct input-output connections found (all inputs have other connections or no direct connections)."
            )
            return

        # Get unique output nodes that receive direct connections
        receiving_outputs = set(to_node for _, to_node in direct_connection_edges)

        # Create annotation
        entry_nodes = qualifying_inputs
        exit_nodes = list(receiving_outputs)
        subgraph_nodes = list(set(qualifying_inputs) | receiving_outputs)
        subgraph_connections = direct_connection_edges

        name = "Direct Input-Output Connections"
        hypothesis = "Trivial direct connections from inputs to outputs that require no intermediate processing. These connections represent inputs that feed directly to outputs without any hidden layer processing."

        # Explanation will be automatically created/assigned by create_annotation
        try:
            annotation = AnnotationManager.create_annotation(
                genome_id=genome_id,
                nodes=subgraph_nodes,
                connections=subgraph_connections,
                hypothesis=hypothesis,
                entry_nodes=entry_nodes,
                exit_nodes=exit_nodes,
                name=name,
                explanation_id=None,  # Will be auto-assigned to genome's explanation
                validate_against_genome=False,  # Already validated against phenotype
            )

            print(f"\n✅ Created direct connections annotation:")
            print(f"   ID: {annotation['id']}")
            print(f"   Name: {annotation.get('name', '(unnamed)')}")
            print(f"   Input nodes: {len(qualifying_inputs)}")
            print(f"   Output nodes: {len(receiving_outputs)}")
            print(f"   Direct connections: {len(direct_connection_edges)}")
            print(
                f"   Explanation: {annotation.get('explanation_id', 'auto-assigned')}"
            )

        except Exception as e:
            print(f"❌ Failed to create direct connections annotation: {e}")
            import traceback

            traceback.print_exc()

    def interactive_mode(self):
        """Start interactive mode"""
        print("\n🧬 Genome Explorer CLI - Interactive Mode")
        print("=" * 50)
        print("Available commands:")
        print("  list [page] [page_size]  - List experiments (with pagination)")
        print("  select / s <experiment_id> - Select experiment")
        print("  summary                 - Show genome summary")
        print("  network                 - Show network structure")
        print("  network-interactive / ni - Show Pyvis interactive visualization")
        print(
            "  network-interactive-react / ni-re - Show React interactive visualization"
        )
        print("  genotype / gt           - Print genotype network structure")
        print("  phenotype / pt          - Print phenotype network structure")
        print("  training                - Plot training metrics")
        print("  ancestry [max_gen]      - Plot ancestry fitness")
        print("  evolution [max_gen]      - Plot evolution progression")
        print("  export [filename]       - Export genome data")
        print()
        print("📝 Annotation commands:")
        print(
            "  annotate                - Create annotation (guided interactive process)"
        )
        print("  annotations / ann-list   - List all annotations for current genome")
        print("  ann-show <index|name|uuid> - Show annotation details (e.g., ann-show 0, ann-show 1086, ann-show <uuid>)")
        print("  ann-delete <id>         - Delete annotation")
        print(
            "  create-direct-connections / create-direct-ann - Create annotation for direct input-output connections"
        )
        print()
        print("  help                    - Show this help")
        print("  quit                    - Exit")
        print()
        print("📄 Pagination commands (when viewing list):")
        print("  n, next                 - Next page")
        print("  p, prev                 - Previous page")
        print("  page <N>                - Go to page N")
        print()

        in_list_mode = False

        while True:
            try:
                command = input("genome-explorer> ").strip().split()
                if not command:
                    continue

                cmd = command[0].lower()

                # Handle pagination navigation when in list mode
                if in_list_mode and cmd in ["n", "next"]:
                    self.current_page += 1
                    experiments_df, actual_page = self.list_experiments(
                        page=self.current_page,
                        page_size=self.page_size,
                        interactive=True,
                    )
                    self.current_page = actual_page  # Sync with actual displayed page
                    if experiments_df.empty:
                        in_list_mode = False
                    continue
                elif in_list_mode and cmd in ["p", "prev", "previous"]:
                    if self.current_page > 1:
                        self.current_page -= 1
                    experiments_df, actual_page = self.list_experiments(
                        page=self.current_page,
                        page_size=self.page_size,
                        interactive=True,
                    )
                    self.current_page = actual_page  # Sync with actual displayed page
                    if experiments_df.empty:
                        in_list_mode = False
                    continue
                elif in_list_mode and cmd == "page" and len(command) > 1:
                    try:
                        page_num = int(command[1])
                        self.current_page = page_num
                        experiments_df, actual_page = self.list_experiments(
                            page=self.current_page,
                            page_size=self.page_size,
                            interactive=True,
                        )
                        self.current_page = (
                            actual_page  # Sync with actual displayed page
                        )
                        if experiments_df.empty:
                            in_list_mode = False
                    except ValueError:
                        print(f"❌ Invalid page number: {command[1]}")
                    continue
                elif in_list_mode and cmd == "q":
                    in_list_mode = False
                    continue

                # Regular commands
                if cmd == "quit" or cmd == "exit":
                    print("👋 Goodbye!")
                    break
                elif cmd == "help":
                    print(
                        "Available commands: list, select/s, summary, network, network-interactive/ni, network-interactive-react/ni-re, genotype/gt, phenotype/pt, training, ancestry, evolution, export, quit"
                    )
                    print(
                        "Annotation commands: annotate (guided), annotations/ann-list, ann-show <index|name|uuid>, ann-delete, create-direct-connections/create-direct-ann"
                    )
                    print("Explanation commands: exp-coverage, exp-hierarchy")
                    print(
                        "Node splitting commands: split-create, split-list, phenotype-splits"
                    )
                    print("Pagination: n/next, p/prev, page N (when viewing list)")
                elif cmd == "list":
                    # Reset to page 1 or use specified page
                    if len(command) > 1:
                        try:
                            self.current_page = int(command[1])
                        except ValueError:
                            print(f"❌ Invalid page number: {command[1]}")
                            continue
                    else:
                        self.current_page = 1

                    # Optional page size
                    if len(command) > 2:
                        try:
                            self.page_size = int(command[2])
                        except ValueError:
                            print(f"❌ Invalid page size: {command[2]}")
                            continue

                    experiments_df, actual_page = self.list_experiments(
                        page=self.current_page,
                        page_size=self.page_size,
                        interactive=True,
                    )
                    self.current_page = actual_page  # Sync with actual displayed page
                    if not experiments_df.empty:
                        in_list_mode = True
                elif cmd == "select" or cmd == "s":
                    in_list_mode = False
                    exp_id = command[1] if len(command) > 1 else None
                    self.select_experiment(exp_id)
                elif cmd == "summary":
                    in_list_mode = False
                    self.show_summary()
                elif cmd == "network":
                    in_list_mode = False
                    self.show_network()
                elif cmd in ["network-interactive", "ni"]:
                    in_list_mode = False
                    self.show_network(interactive=True, renderer="pyvis")
                elif cmd in ["network-interactive-react", "ni-re"]:
                    in_list_mode = False
                    self.show_network(interactive=True, renderer="react")
                elif cmd in ["genotype", "gt"]:
                    in_list_mode = False
                    self.show_genotype()
                elif cmd in ["phenotype", "pt"]:
                    in_list_mode = False
                    self.show_phenotype()
                elif cmd == "training":
                    in_list_mode = False
                    self.plot_training_metrics()
                elif cmd == "ancestry":
                    in_list_mode = False
                    max_gen = (
                        int(command[1]) if len(command) > 1 else None
                    )  # None = unlimited
                    self.plot_ancestry_fitness(max_gen)
                elif cmd == "evolution":
                    in_list_mode = False
                    max_gen = int(command[1]) if len(command) > 1 else 50
                    self.plot_evolution_progression(max_gen)
                elif cmd == "export":
                    in_list_mode = False
                    filename = command[1] if len(command) > 1 else None
                    self.export_data(filename)
                elif cmd == "annotate":
                    in_list_mode = False
                    self.guided_annotation()
                elif cmd in ["annotations", "ann-list"]:
                    in_list_mode = False
                    self.list_annotations()
                elif cmd == "ann-show":
                    in_list_mode = False
                    if len(command) < 2:
                        print("❌ Usage: ann-show <index|name|uuid>")
                        print("   Examples:")
                        print("     ann-show 0              # Show first annotation")
                        print("     ann-show 1086           # Show annotation by name")
                        print("     ann-show <uuid>         # Show annotation by UUID")
                        continue
                    self.show_annotation(command[1])
                elif cmd == "ann-delete":
                    in_list_mode = False
                    if len(command) < 2:
                        print("❌ Usage: ann-delete <annotation_id>")
                        continue
                    # Confirm deletion
                    confirm = (
                        input(
                            f"⚠️  Are you sure you want to delete annotation {command[1]}? (yes/no): "
                        )
                        .strip()
                        .lower()
                    )
                    if confirm in ["yes", "y"]:
                        self.delete_annotation(command[1])
                    else:
                        print("❌ Deletion cancelled.")
                elif cmd in ["create-direct-connections", "create-direct-ann"]:
                    in_list_mode = False
                    self.create_direct_connections_annotation()
                elif cmd == "exp-coverage":
                    in_list_mode = False
                    self.show_coverage_metrics()
                elif cmd == "exp-hierarchy":
                    in_list_mode = False
                    self.show_annotation_hierarchy()
                elif cmd == "split-list":
                    in_list_mode = False
                    self.list_node_splits()
                elif cmd == "phenotype-splits":
                    in_list_mode = False
                    self.show_phenotype_with_splits()
                else:
                    print(f"❌ Unknown command: {cmd}")
                    in_list_mode = False

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                in_list_mode = False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Genome Explorer CLI")
    parser.add_argument("--experiment-id", help="Experiment ID to load directly")
    parser.add_argument(
        "--list", action="store_true", help="List all experiments and exit"
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number for listing experiments (default: 1)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Number of experiments per page (default: 20)",
    )
    parser.add_argument("--summary", action="store_true", help="Show summary and exit")
    parser.add_argument("--network", action="store_true", help="Show network and exit")
    parser.add_argument(
        "--network-interactive",
        action="store_true",
        help="Show interactive network visualization and exit",
    )
    parser.add_argument(
        "--network-interactive-pyvis",
        action="store_true",
        help="Show legacy Pyvis interactive visualization and exit",
    )
    parser.add_argument(
        "--training", action="store_true", help="Plot training metrics and exit"
    )
    parser.add_argument(
        "--ancestry", type=int, help="Plot ancestry fitness (max generations)"
    )
    parser.add_argument(
        "--evolution", type=int, help="Plot evolution progression (max generations)"
    )
    parser.add_argument("--export", help="Export data to filename")
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive mode"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (shows detailed traversal information)",
    )

    args = parser.parse_args()

    cli = GenomeExplorerCLI(debug=args.debug)

    # Initialize database
    try:
        db.init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    # Handle command line arguments
    if args.list:
        cli.list_experiments(
            page=args.page, page_size=args.page_size, interactive=False
        )[
            0
        ]  # Just get the dataframe
        return

    if args.experiment_id:
        if not cli.select_experiment(args.experiment_id):
            sys.exit(1)
    else:
        # Interactive mode or need to select experiment
        if not args.interactive:
            print("Please specify --experiment-id or use --interactive mode")
            return

    # Execute commands
    if args.summary:
        cli.show_summary()
    if args.network:
        cli.show_network()
    if args.network_interactive:
        cli.show_network(interactive=True, renderer="react")
    if args.network_interactive_pyvis:
        cli.show_network(interactive=True, renderer="pyvis")
    if args.training:
        cli.plot_training_metrics()
    if args.ancestry:
        cli.plot_ancestry_fitness(args.ancestry)
    if args.evolution:
        cli.plot_evolution_progression(args.evolution)
    if args.export:
        cli.export_data(args.export)

    # Start interactive mode if requested or no specific commands
    if args.interactive or not any(
        [
            args.summary,
            args.network,
            args.network_interactive,
            args.training,
            args.ancestry,
            args.evolution,
            args.export,
        ]
    ):
        cli.interactive_mode()


if __name__ == "__main__":
    main()
