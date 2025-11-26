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
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

from explaneat.analysis.genome_explorer import GenomeExplorer
from explaneat.analysis.annotation_manager import AnnotationManager
from explaneat.analysis.subgraph_validator import SubgraphValidator
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
        hidden_nodes = [n for n in genotype.nodes if n.id not in genotype.input_node_ids and n.id not in genotype.output_node_ids]
        
        if input_nodes:
            print(f"\n📥 Input Nodes ({len(input_nodes)}):")
            for node in sorted(input_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}")
        
        if output_nodes:
            print(f"\n📤 Output Nodes ({len(output_nodes)}):")
            for node in sorted(output_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}")
        
        if hidden_nodes:
            print(f"\n⚪ Hidden Nodes ({len(hidden_nodes)}):")
            for node in sorted(hidden_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}")
        
        print(f"\n🔗 Connections ({len(genotype.connections)}):")
        enabled_connections = [c for c in genotype.connections if c.enabled]
        disabled_connections = [c for c in genotype.connections if not c.enabled]
        
        if enabled_connections:
            print(f"  ✅ Enabled ({len(enabled_connections)}):")
            for conn in sorted(enabled_connections, key=lambda c: (c.from_node, c.to_node)):
                print(f"    {conn.from_node:4d} → {conn.to_node:4d}: weight={conn.weight:8.4f}")
        
        if disabled_connections:
            print(f"  ❌ Disabled ({len(disabled_connections)}):")
            for conn in sorted(disabled_connections, key=lambda c: (c.from_node, c.to_node)):
                print(f"    {conn.from_node:4d} → {conn.to_node:4d}: weight={conn.weight:8.4f}")
        
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
        hidden_nodes = [n for n in phenotype.nodes if n.id not in phenotype.input_node_ids and n.id not in phenotype.output_node_ids]
        
        if input_nodes:
            print(f"\n📥 Input Nodes ({len(input_nodes)}):")
            for node in sorted(input_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}")
        
        if output_nodes:
            print(f"\n📤 Output Nodes ({len(output_nodes)}):")
            for node in sorted(output_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}")
        
        if hidden_nodes:
            print(f"\n⚪ Hidden Nodes ({len(hidden_nodes)}):")
            for node in sorted(hidden_nodes, key=lambda n: n.id):
                bias_str = f"{node.bias:.4f}" if node.bias is not None else "N/A"
                activation_str = node.activation or "N/A"
                print(f"  Node {node.id:4d}: bias={bias_str:>8}, activation={activation_str}")
        
        print(f"\n🔗 Active Connections ({len(phenotype.connections)}):")
        for conn in sorted(phenotype.connections, key=lambda c: (c.from_node, c.to_node)):
            print(f"  {conn.from_node:4d} → {conn.to_node:4d}: weight={conn.weight:8.4f}")
        
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
        """Parse comma-separated node list (e.g., '-2, -3, 6' -> [-2, -3, 6])"""
        try:
            # Strip quotes if present
            node_str = node_str.strip("'\"")
            nodes = [int(n.strip()) for n in node_str.split(",") if n.strip()]
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

            # Create annotation
            annotation_dict = AnnotationManager.create_annotation(
                genome_id=self.current_explorer.genome_info.genome_id,
                nodes=nodes,
                connections=connections,
                hypothesis=hypothesis,
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
                "Step 2/4: Enter end nodes (comma-separated, e.g., '0' or '1321,452'): "
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

        confirm = input("\nCreate this annotation? (yes/no): ").strip().lower()
        if confirm not in ["yes", "y"]:
            print("❌ Annotation creation cancelled.")
            return

        # Create annotation
        try:
            annotation_dict = AnnotationManager.create_annotation(
                genome_id=self.current_explorer.genome_info.genome_id,
                nodes=nodes,
                connections=connections,
                hypothesis=hypothesis,
                name=name,
                validate_against_genome=False,  # Already validated
            )

            annotation_id = annotation_dict["id"]
            annotation_name = annotation_dict["name"] or "(unnamed)"

            print(f"\n✅ Annotation created successfully!")
            print(f"   ID: {annotation_id}")
            print(f"   Name: {annotation_name}")
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

    def show_annotation(self, annotation_id: str):
        """Show details of a specific annotation"""
        if not self.current_explorer:
            print("❌ No genome loaded. Use 'select' command first.")
            return

        annotation = AnnotationManager.get_annotation(annotation_id)
        if not annotation:
            print(f"❌ Annotation {annotation_id} not found.")
            return

        print(f"\n📝 Annotation Details:")
        print("=" * 70)
        print(f"ID:          {annotation.id}")
        print(f"Name:        {annotation.name or '(unnamed)'}")
        print(f"Genome ID:   {annotation.genome_id}")
        print(f"Created:     {annotation.created_at}")
        print(f"Updated:     {annotation.updated_at}")
        print(f"Connected:   {annotation.is_connected}")
        print(f"\nHypothesis:")
        print(f"  {annotation.hypothesis}")
        print(f"\nSubgraph:")
        print(
            f"  Nodes ({len(annotation.subgraph_nodes)}): {annotation.subgraph_nodes}"
        )
        print(f"  Connections ({len(annotation.subgraph_connections)}):")
        for conn in annotation.subgraph_connections[:10]:  # Show first 10
            print(f"    {conn[0]} -> {conn[1]}")
        if len(annotation.subgraph_connections) > 10:
            print(f"    ... and {len(annotation.subgraph_connections) - 10} more")

        if annotation.evidence:
            print(f"\nEvidence:")
            evidence = annotation.evidence
            print(
                f"  Analytical methods: {len(evidence.get('analytical_methods', []))}"
            )
            print(f"  Visualizations: {len(evidence.get('visualizations', []))}")
            print(f"  Counterfactuals: {len(evidence.get('counterfactuals', []))}")
            print(f"  Other evidence: {len(evidence.get('other_evidence', []))}")
        else:
            print(f"\nEvidence: None")

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
        print("  network-interactive-react / ni-re - Show React interactive visualization")
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
        print("  ann-show <id>           - Show annotation details")
        print("  ann-delete <id>         - Delete annotation")
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
                        "Annotation commands: annotate (guided), annotations/ann-list, ann-show, ann-delete"
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
                        print("❌ Usage: ann-show <annotation_id>")
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
