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
from explaneat.db import db

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GenomeExplorerCLI:
    """Interactive CLI for exploring genome data"""
    
    def __init__(self):
        self.current_explorer = None
        self.current_experiment_id = None
        self.current_page = 1
        self.page_size = 20
        
    def list_experiments(self, show_numbers: bool = True, page: int = 1, page_size: Optional[int] = 20, interactive: bool = False) -> Tuple[pd.DataFrame, int]:
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
            experiments_df = experiments_df.sort_values('created_at', ascending=False).reset_index(drop=True)

            total_experiments = len(experiments_df)
            
            # Handle pagination: if page_size is None or 0, show all
            if page_size is None or page_size <= 0:
                page_df = experiments_df
                total_pages = 1
                start_idx = 0
                end_idx = total_experiments
            else:
                total_pages = (total_experiments + page_size - 1) // page_size  # Ceiling division
                
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
                print(f"{'#':<4} {'Name':<30} {'Status':<10} {'Gens':<6} {'Best Fitness':<12} {'ID':<36}")
            else:
                print(f"{'ID':<36} {'Name':<25} {'Status':<10} {'Generations':<12} {'Best Fitness':<12}")
            print("-" * 90)

            for idx, exp in page_df.iterrows():
                fitness_str = f"{exp['best_fitness']:.3f}" if exp['best_fitness'] else "N/A"
                if show_numbers:
                    # Use the original index from the full dataframe for selection
                    print(f"{idx:<4} {exp['name'][:29]:<30} {exp['status']:<10} {exp['generations']:<6} {fitness_str:<12} {exp['experiment_id']:<36}")
                else:
                    print(f"{exp['experiment_id']:<36} {exp['name'][:24]:<25} {exp['status']:<10} {exp['generations']:<12} {fitness_str:<12}")

            # Show pagination info
            print("-" * 90)
            if total_pages > 1:
                print(f"Page {page} of {total_pages} (Showing {start_idx + 1}-{end_idx} of {total_experiments} experiments)")
                if interactive:
                    print("💡 Navigation: 'n' (next), 'p' (prev), 'page N' (go to page), 'q' (quit)")
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
                # Interactive selection - show all experiments (no pagination)
                experiments_df, _ = self.list_experiments(page_size=None, interactive=False)
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
        print("  list [page] [page_size]  - List experiments (with pagination)")
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
                if in_list_mode and cmd in ['n', 'next']:
                    self.current_page += 1
                    experiments_df, actual_page = self.list_experiments(page=self.current_page, page_size=self.page_size, interactive=True)
                    self.current_page = actual_page  # Sync with actual displayed page
                    if experiments_df.empty:
                        in_list_mode = False
                    continue
                elif in_list_mode and cmd in ['p', 'prev', 'previous']:
                    if self.current_page > 1:
                        self.current_page -= 1
                    experiments_df, actual_page = self.list_experiments(page=self.current_page, page_size=self.page_size, interactive=True)
                    self.current_page = actual_page  # Sync with actual displayed page
                    if experiments_df.empty:
                        in_list_mode = False
                    continue
                elif in_list_mode and cmd == 'page' and len(command) > 1:
                    try:
                        page_num = int(command[1])
                        self.current_page = page_num
                        experiments_df, actual_page = self.list_experiments(page=self.current_page, page_size=self.page_size, interactive=True)
                        self.current_page = actual_page  # Sync with actual displayed page
                        if experiments_df.empty:
                            in_list_mode = False
                    except ValueError:
                        print(f"❌ Invalid page number: {command[1]}")
                    continue
                elif in_list_mode and cmd == 'q':
                    in_list_mode = False
                    continue
                
                # Regular commands
                if cmd == 'quit' or cmd == 'exit':
                    print("👋 Goodbye!")
                    break
                elif cmd == 'help':
                    print("Available commands: list, select, summary, network, training, ancestry, evolution, export, quit")
                    print("Pagination: n/next, p/prev, page N (when viewing list)")
                elif cmd == 'list':
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
                    
                    experiments_df, actual_page = self.list_experiments(page=self.current_page, page_size=self.page_size, interactive=True)
                    self.current_page = actual_page  # Sync with actual displayed page
                    if not experiments_df.empty:
                        in_list_mode = True
                elif cmd == 'select':
                    in_list_mode = False
                    exp_id = command[1] if len(command) > 1 else None
                    self.select_experiment(exp_id)
                elif cmd == 'summary':
                    in_list_mode = False
                    self.show_summary()
                elif cmd == 'network':
                    in_list_mode = False
                    self.show_network()
                elif cmd == 'training':
                    in_list_mode = False
                    self.plot_training_metrics()
                elif cmd == 'ancestry':
                    in_list_mode = False
                    max_gen = int(command[1]) if len(command) > 1 else None  # None = unlimited
                    self.plot_ancestry_fitness(max_gen)
                elif cmd == 'evolution':
                    in_list_mode = False
                    max_gen = int(command[1]) if len(command) > 1 else 50
                    self.plot_evolution_progression(max_gen)
                elif cmd == 'export':
                    in_list_mode = False
                    filename = command[1] if len(command) > 1 else None
                    self.export_data(filename)
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
    parser.add_argument("--list", action="store_true", help="List all experiments and exit")
    parser.add_argument("--page", type=int, default=1, help="Page number for listing experiments (default: 1)")
    parser.add_argument("--page-size", type=int, default=20, help="Number of experiments per page (default: 20)")
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
        cli.list_experiments(page=args.page, page_size=args.page_size, interactive=False)[0]  # Just get the dataframe
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

