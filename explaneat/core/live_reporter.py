"""
LiveReporter: Real-time terminal UI for tracking evolution progress

Displays a clean, updating status dashboard at the top of the terminal
showing current generation, fitness, and population statistics.
"""
import sys
import os
from datetime import datetime
from typing import Optional


class LiveReporter:
    """
    Reporter that displays evolution progress in a clean, updating terminal UI.

    Shows a fixed header with current generation stats that updates in place,
    while detailed logs scroll below.
    """

    def __init__(self, max_generations: int = None, compact: bool = True, sticky: bool = True):
        """
        Initialize live reporter.

        Args:
            max_generations: Total generations to run (for progress bar)
            compact: If True, use compact single-line mode. If False, multi-line dashboard.
            sticky: If True, keep status line at top by moving cursor
        """
        self.max_generations = max_generations
        self.compact = compact
        self.sticky = sticky
        self.current_generation = 0
        self.best_fitness = None
        self.mean_fitness = None
        self.population_size = 0
        self.num_species = 0
        self.start_time = datetime.now()
        self.generation_start_time = None
        self.status_printed = False

        # Terminal capabilities
        self.supports_ansi = self._check_ansi_support()

    def _check_ansi_support(self) -> bool:
        """Check if terminal supports ANSI escape codes."""
        # Check if we're in a real terminal (not a file/pipe)
        if not sys.stdout.isatty():
            return False

        # Check TERM environment variable
        term = os.environ.get('TERM', '')
        if term in ['dumb', '']:
            return False

        return True

    def _clear_lines(self, n: int):
        """Clear n lines above cursor."""
        if not self.supports_ansi:
            return

        for _ in range(n):
            # Move cursor up one line and clear it
            sys.stdout.write('\033[F\033[K')
        sys.stdout.flush()

    def _move_to_line_start(self):
        """Move cursor to start of current line."""
        if self.supports_ansi:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def _render_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Render a progress bar."""
        if total == 0:
            return '[' + '?' * width + ']'

        filled = int(width * current / total)
        bar = '█' * filled + '░' * (width - filled)
        percent = 100 * current / total
        return f'[{bar}] {percent:5.1f}%'

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f'{hours:02d}:{minutes:02d}:{secs:02d}'
        else:
            return f'{minutes:02d}:{secs:02d}'

    def _render_compact(self) -> str:
        """Render compact single-line status."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_str = self._format_time(elapsed)

        # Progress bar
        if self.max_generations:
            progress = self._render_progress_bar(self.current_generation, self.max_generations, width=20)
        else:
            progress = f'Gen {self.current_generation:3d}'

        # Fitness
        if self.best_fitness is not None:
            fitness_str = f'Best: {self.best_fitness:7.4f}'
            if self.mean_fitness is not None:
                fitness_str += f' | Mean: {self.mean_fitness:7.4f}'
        else:
            fitness_str = 'Best: -.----'

        # Population info
        pop_str = f'Pop: {self.population_size:3d} | Species: {self.num_species:2d}'

        # Combine
        status = f'⚡ {progress} | {fitness_str} | {pop_str} | ⏱️  {elapsed_str}'

        return status

    def _render_dashboard(self) -> str:
        """Render multi-line dashboard."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_str = self._format_time(elapsed)

        lines = []
        lines.append('=' * 80)
        lines.append('  🧬 NEAT EVOLUTION STATUS')
        lines.append('=' * 80)

        # Generation info
        gen_line = f'  Generation: {self.current_generation:3d}'
        if self.max_generations:
            gen_line += f' / {self.max_generations}'
            progress = self._render_progress_bar(self.current_generation, self.max_generations, width=30)
            gen_line += f'  {progress}'
        lines.append(gen_line)

        # Fitness
        if self.best_fitness is not None:
            lines.append(f'  Best Fitness:  {self.best_fitness:8.5f}')
            if self.mean_fitness is not None:
                lines.append(f'  Mean Fitness:  {self.mean_fitness:8.5f}')
        else:
            lines.append('  Best Fitness:  -.-----')

        # Population
        lines.append(f'  Population:    {self.population_size:3d} genomes in {self.num_species:2d} species')

        # Time
        lines.append(f'  Elapsed Time:  {elapsed_str}')

        lines.append('=' * 80)

        return '\n'.join(lines)

    def _clear_screen_to_status(self):
        """Clear from current position up to status line."""
        if not self.supports_ansi:
            return

        # Save cursor position
        sys.stdout.write('\033[s')
        # Move to top of screen
        sys.stdout.write('\033[H')
        # Clear from cursor to end of screen
        sys.stdout.write('\033[J')
        # Restore cursor position
        sys.stdout.write('\033[u')
        sys.stdout.flush()

    def update_display(self, force_clear=False):
        """Update the terminal display.

        Args:
            force_clear: If True, clear the screen before rendering
        """
        if self.sticky and self.supports_ansi:
            # Sticky mode: save position, move to top, render, restore
            sys.stdout.write('\033[s')  # Save cursor position
            sys.stdout.write('\033[H')  # Move to home (top-left)

            if self.compact:
                status = self._render_compact()
                sys.stdout.write('\033[K')  # Clear line
                sys.stdout.write(status)
            else:
                # Clear the dashboard area (9 lines)
                for _ in range(9):
                    sys.stdout.write('\033[K\n')
                sys.stdout.write('\033[H')  # Back to top
                dashboard = self._render_dashboard()
                sys.stdout.write(dashboard)

            sys.stdout.write('\033[u')  # Restore cursor position
            sys.stdout.flush()
        else:
            # Non-sticky mode: update in place
            if self.compact:
                # Clear previous line and render new status
                if self.status_printed and self.supports_ansi:
                    # Move to start of line and clear it
                    sys.stdout.write('\r\033[K')
                status = self._render_compact()
                sys.stdout.write(status)
                sys.stdout.flush()
                self.status_printed = True
            else:
                # Clear previous dashboard and render new one
                if self.status_printed and self.supports_ansi:
                    # Move cursor up and clear lines
                    self._clear_lines(9)  # Dashboard is 9 lines
                dashboard = self._render_dashboard()
                sys.stdout.write(dashboard + '\n')
                sys.stdout.flush()
                self.status_printed = True

    # NEAT Reporter Interface

    def start_generation(self, generation: int):
        """Called at start of generation."""
        self.current_generation = generation
        self.generation_start_time = datetime.now()
        self.update_display()

    def post_evaluate(self, config, population, species, best_genome):
        """Called after fitness evaluation."""
        # Update fitness stats
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        if fitnesses:
            self.best_fitness = max(fitnesses)
            self.mean_fitness = sum(fitnesses) / len(fitnesses)

        self.population_size = len(population)
        self.num_species = len(species.species)

        self.update_display()

    def end_generation(self, config, population, species_set):
        """Called at end of generation."""
        # Print newline to move to next line for logs
        if self.supports_ansi:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def pre_backprop(self, config, population, species):
        """Called before backpropagation."""
        pass

    def post_backprop(self, config, population, species):
        """Called after backpropagation."""
        pass

    def post_reproduction(self, config, population, species):
        """Called after reproduction."""
        pass

    def pre_reproduction(self, config, population, species):
        """Called before reproduction."""
        pass

    def info(self, msg: str):
        """Log info messages."""
        pass

    def species_stagnant(self, sid, species):
        """Handle stagnant species."""
        pass

    def found_solution(self, config, generation, best):
        """Handle when solution is found."""
        if self.compact:
            sys.stdout.write('\n')
        sys.stdout.write(f'\n🎉 Solution found at generation {generation}! Fitness: {best.fitness:.5f}\n')
        sys.stdout.flush()

    def complete_extinction(self):
        """Handle complete extinction."""
        if self.compact:
            sys.stdout.write('\n')
        sys.stdout.write('\n⚠️  Complete extinction occurred!\n')
        sys.stdout.flush()

    def start_experiment(self, config):
        """Called at start of experiment."""
        self.start_time = datetime.now()

    def end_experiment(self, config, population, species):
        """Called at end of experiment."""
        if self.compact:
            sys.stdout.write('\n')

        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_str = self._format_time(elapsed)

        sys.stdout.write(f'\n✅ Evolution complete! Total time: {elapsed_str}\n')
        sys.stdout.flush()


class QuietLogger:
    """
    Wrapper for standard logger that suppresses verbose output during evolution.

    Only shows important messages (warnings, errors, major milestones).
    """

    def __init__(self, logger, show_info: bool = False):
        """
        Initialize quiet logger.

        Args:
            logger: The underlying logger to wrap
            show_info: If True, still show info messages. If False, suppress them.
        """
        self.logger = logger
        self.show_info = show_info

    def debug(self, msg, *args, **kwargs):
        """Suppress debug messages."""
        pass

    def info(self, msg, *args, **kwargs):
        """Conditionally show info messages."""
        if self.show_info:
            self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Always show warnings."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Always show errors."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Always show critical messages."""
        self.logger.critical(msg, *args, **kwargs)
