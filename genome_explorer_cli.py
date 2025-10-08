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
from typing import Optional, List, Dict, Any
import pandas as pd

from explaneat.analysis.genome_explorer import GenomeExplorer
from explaneat.db import db

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GenomeExplorerCLI:
    """Interactive CLI for exploring genome data"""
    
    def __init__(self):
        self.current_explorer = None
        self.current_experiment_id = None
        
    def list_experiments(self, show_numbers: bool = True) -> pd.DataFrame:
        """List all experiments in the database

        Args:
            show_numbers: If True, show row numbers for easy selection
        """
        try:
            experiments_df = GenomeExplorer.list_experiments()
            if experiments_df.empty:
                print("No experiments found in database")
                return experiments_df

            # Sort by creation date (most recent first)
            experiments_df = experiments_df.sort_values('created_at', ascending=False).reset_index(drop=True)

            print("\n📊 Available Experiments:")
            print("=" * 90)
            if show_numbers:
                print(f"{'#':<4} {'Name':<30} {'Status':<10} {'Gens':<6} {'Best Fitness':<12} {'ID':<36}")
            else:
                print(f"{'ID':<36} {'Name':<25} {'Status':<10} {'Generations':<12} {'Best Fitness':<12}")
            print("-" * 90)

            for idx, exp in experiments_df.iterrows():
                fitness_str = f"{exp['best_fitness']:.3f}" if exp['best_fitness'] else "N/A"
                if show_numbers:
                    print(f"{idx:<4} {exp['name'][:29]:<30} {exp['status']:<10} {exp['generations']:<6} {fitness_str:<12} {exp['experiment_id']:<36}")
                else:
                    print(f"{exp['experiment_id']:<36} {exp['name'][:24]:<25} {exp['status']:<10} {exp['generations']:<12} {fitness_str:<12}")

            return experiments_df

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return pd.DataFrame()
    
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
                experiments_df = experiments_df.sort_values('created_at', ascending=False).reset_index(drop=True)

                selected_id = None

                # Try as number first
                if selector.isdigit():
                    idx = int(selector)
                    if 0 <= idx < len(experiments_df):
                        selected_id = experiments_df.iloc[idx]['experiment_id']
                        print(f"📍 Selected experiment #{idx}: {experiments_df.iloc[idx]['name']}")
                    else:
                        print(f"❌ Invalid experiment number: {idx} (valid range: 0-{len(experiments_df)-1})")
                        return False

                # Try special keywords
                elif selector.lower() in ['latest', 'best', 'recent']:
                    selected_id = experiments_df.iloc[0]['experiment_id']
                    print(f"📍 Selected latest experiment: {experiments_df.iloc[0]['name']}")

                # Try as full UUID
                elif selector in experiments_df['experiment_id'].values:
                    selected_id = selector
                    exp_name = experiments_df[experiments_df['experiment_id'] == selector].iloc[0]['name']
                    print(f"📍 Selected experiment: {exp_name}")

                # Try as name substring
                else:
                    matches = experiments_df[experiments_df['name'].str.contains(selector, case=False, na=False)]
                    if len(matches) == 1:
                        selected_id = matches.iloc[0]['experiment_id']
                        print(f"📍 Found matching experiment: {matches.iloc[0]['name']}")
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
                # Interactive selection
                experiments_df = self.list_experiments()
                if experiments_df.empty:
                    return False

                print(f"\n💡 Enter: number (0-{len(experiments_df)-1}), 'latest', name substring, or full UUID")
                print("   Or 'q' to quit")
                choice = input("> ").strip()

                if choice.lower() == 'q':
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
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"✅ Data exported to {filename}")
        else:
            print("📊 Data exported (use --filename to save to file)")
            print(f"   Contains {len(data)} sections")
    
    def interactive_mode(self):
        """Start interactive mode"""
        print("\n🧬 Genome Explorer CLI - Interactive Mode")
        print("=" * 50)
        print("Available commands:")
        print("  list                    - List all experiments")
        print("  select <experiment_id>   - Select experiment")
        print("  summary                 - Show genome summary")
        print("  network                 - Show network structure")
        print("  training                - Plot training metrics")
        print("  ancestry [max_gen]      - Plot ancestry fitness")
        print("  evolution [max_gen]      - Plot evolution progression")
        print("  export [filename]       - Export genome data")
        print("  help                    - Show this help")
        print("  quit                    - Exit")
        print()
        
        while True:
            try:
                command = input("genome-explorer> ").strip().split()
                if not command:
                    continue
                    
                cmd = command[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    print("👋 Goodbye!")
                    break
                elif cmd == 'help':
                    print("Available commands: list, select, summary, network, training, ancestry, evolution, export, quit")
                elif cmd == 'list':
                    self.list_experiments()
                elif cmd == 'select':
                    exp_id = command[1] if len(command) > 1 else None
                    self.select_experiment(exp_id)
                elif cmd == 'summary':
                    self.show_summary()
                elif cmd == 'network':
                    self.show_network()
                elif cmd == 'training':
                    self.plot_training_metrics()
                elif cmd == 'ancestry':
                    max_gen = int(command[1]) if len(command) > 1 else None  # None = unlimited
                    self.plot_ancestry_fitness(max_gen)
                elif cmd == 'evolution':
                    max_gen = int(command[1]) if len(command) > 1 else 50
                    self.plot_evolution_progression(max_gen)
                elif cmd == 'export':
                    filename = command[1] if len(command) > 1 else None
                    self.export_data(filename)
                else:
                    print(f"❌ Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Genome Explorer CLI")
    parser.add_argument("--experiment-id", help="Experiment ID to load directly")
    parser.add_argument("--list", action="store_true", help="List all experiments and exit")
    parser.add_argument("--summary", action="store_true", help="Show summary and exit")
    parser.add_argument("--network", action="store_true", help="Show network and exit")
    parser.add_argument("--training", action="store_true", help="Plot training metrics and exit")
    parser.add_argument("--ancestry", type=int, help="Plot ancestry fitness (max generations)")
    parser.add_argument("--evolution", type=int, help="Plot evolution progression (max generations)")
    parser.add_argument("--export", help="Export data to filename")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    cli = GenomeExplorerCLI()
    
    # Initialize database
    try:
        db.init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    
    # Handle command line arguments
    if args.list:
        cli.list_experiments()
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
    if args.training:
        cli.plot_training_metrics()
    if args.ancestry:
        cli.plot_ancestry_fitness(args.ancestry)
    if args.evolution:
        cli.plot_evolution_progression(args.evolution)
    if args.export:
        cli.export_data(args.export)
    
    # Start interactive mode if requested or no specific commands
    if args.interactive or not any([args.summary, args.network, args.training, args.ancestry, args.evolution, args.export]):
        cli.interactive_mode()


if __name__ == "__main__":
    main()

