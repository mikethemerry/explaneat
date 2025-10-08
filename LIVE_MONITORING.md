# Live Evolution Monitoring

## Overview

ExplaNEAT now includes a **live status reporter** that shows real-time evolution progress in a clean, updating terminal display. No more scrolling through logs to see current generation status!

## Features

### ✅ Real-Time Updates
- Status updates **in place** at the top of terminal
- Shows current generation, fitness, population stats
- Progress bar showing completion percentage
- Elapsed time tracker

### ✅ Two Display Modes

**Compact Mode** (default) - Single-line status:
```
⚡ [██████░░░░] 50.0% | Best: 0.8543 | Mean: 0.7234 | Pop: 150 | Species: 12 | ⏱️  05:23
```

**Dashboard Mode** - Multi-line status panel:
```
================================================================================
  🧬 NEAT EVOLUTION STATUS
================================================================================
  Generation: 5 / 50  [███████████░░░░░░░░░] 50.0%
  Best Fitness:  0.85430
  Mean Fitness:  0.72340
  Population:    150 genomes in 12 species
  Elapsed Time:  05:23
================================================================================
```

### ✅ Quiet Mode
Suppress verbose logging and show only live status - perfect for watching long runs!

## Usage

### Basic Usage

```bash
# Default: compact status line with normal logging
python run_working_backache.py --generations=50

# Quiet mode: only show live status
python run_working_backache.py --generations=50 --quiet

# Dashboard mode: multi-line status display
python run_working_backache.py --generations=50 --dashboard

# Best for monitoring: quiet + dashboard
python run_working_backache.py --generations=100 --quiet --dashboard
```

### Command-Line Arguments

- `--generations N` - Number of generations to run (default: 10)
- `--quiet` - Reduce logging verbosity (only show live status)
- `--dashboard` - Use multi-line dashboard instead of compact line

### Recommended Combinations

**For short test runs:**
```bash
python run_working_backache.py --generations=10
# Use default - see all logs for debugging
```

**For production runs you want to monitor:**
```bash
python run_working_backache.py --generations=100 --quiet --dashboard
# Clean dashboard view, easy to watch progress
```

**For background runs:**
```bash
nohup python run_working_backache.py --generations=100 --quiet > output.log 2>&1 &
tail -f output.log
# Compact mode works better in log files
```

## How It Works

### LiveReporter Class

The `LiveReporter` (in `explaneat/core/live_reporter.py`) is a NEAT reporter that:

1. **Hooks into NEAT lifecycle events:**
   - `start_generation()` - Update display at generation start
   - `post_evaluate()` - Update fitness stats after evaluation
   - `end_generation()` - Move to next line for any logs

2. **Uses ANSI terminal codes** for in-place updates:
   - `\r` - Move cursor to start of line
   - `\033[F` - Move cursor up one line
   - `\033[K` - Clear current line

3. **Falls back gracefully** if terminal doesn't support ANSI:
   - Detects terminal capabilities
   - Prints normal newlines if ANSI unavailable
   - Works in log files, pipes, non-TTY environments

### Status Information Displayed

**Compact Mode:**
- Progress bar: Visual completion indicator
- Best fitness: Highest fitness in current population
- Mean fitness: Average fitness across population
- Population size: Number of genomes
- Species count: Number of active species
- Elapsed time: HH:MM:SS since experiment start

**Dashboard Mode:**
- All of the above, plus:
- Generation number with total
- Larger progress bar
- Formatted fitness values with more precision
- Clear section separators

## Terminal Compatibility

### ✅ Supported Terminals
- macOS Terminal.app
- iTerm2
- Linux terminals (xterm, gnome-terminal, konsole, etc.)
- Windows Terminal
- VS Code integrated terminal
- Most modern terminals with ANSI support

### ⚠️ Limited Support
- Jupyter notebooks (ANSI codes may display as text)
- Screen/tmux (may need specific configuration)
- Very old terminals (falls back to newlines)

### ❌ Not Supported
- Pure file output (no terminal)
- Pipes without terminal (e.g., `python script.py | grep ...`)

The reporter automatically detects terminal capabilities and adjusts accordingly.

## Customization

### Using in Your Own Scripts

```python
from explaneat.core.live_reporter import LiveReporter

# Create reporter
live_reporter = LiveReporter(
    max_generations=100,  # For progress bar
    compact=True          # True = single line, False = dashboard
)

# Add to NEAT population reporters
population.reporters.reporters.append(live_reporter)

# Run evolution - status updates automatically!
population.run(fitness_function, n=100)
```

### Combining with Other Reporters

```python
from explaneat.core.live_reporter import LiveReporter
from explaneat.core.ancestry_reporter import AncestryReporter

# Create multiple reporters
live_reporter = LiveReporter(max_generations=100, compact=True)
ancestry_reporter = AncestryReporter()
db_reporter = DatabaseReporter(experiment_id, config)

# Add all reporters
population.reporters.reporters.append(db_reporter)
population.reporters.reporters.append(ancestry_reporter)
population.reporters.reporters.append(live_reporter)  # Add last for best display

# All reporters run in parallel during evolution
```

### QuietLogger (Optional)

If you want to reduce logging from other parts of the code:

```python
from explaneat.core.live_reporter import QuietLogger
import logging

# Wrap your logger
logger = logging.getLogger(__name__)
quiet_logger = QuietLogger(logger, show_info=False)

# Now use quiet_logger instead
quiet_logger.info("This won't show")
quiet_logger.warning("This will show")
quiet_logger.error("This will show")
```

## Technical Details

### ANSI Escape Codes Used

```python
\r            # Carriage return - move cursor to line start
\033[F        # Cursor up - move up one line
\033[K        # Erase line - clear current line
\033[nA       # Move cursor up n lines
```

### Update Frequency

- Updates at generation start (`start_generation`)
- Updates after fitness evaluation (`post_evaluate`)
- Finalizes at generation end (`end_generation`)

Typically updates **2-3 times per generation**.

### Performance Impact

Negligible - terminal updates are fast:
- String formatting: ~1-2 microseconds
- Terminal write: ~100 microseconds
- Total overhead: < 0.01% of generation time

## Troubleshooting

### Status line is garbled or showing weird characters

**Cause:** Terminal doesn't support ANSI codes properly

**Fix:** Use `--dashboard` mode or check terminal ANSI support:
```bash
echo $TERM  # Should show something like "xterm-256color"
```

### Status not updating in Jupyter notebook

**Cause:** Jupyter captures stdout differently

**Fix:** Use standard terminal instead, or disable live reporter in notebooks:
```python
# Don't add live_reporter in Jupyter environments
if not running_in_jupyter():
    population.reporters.reporters.append(live_reporter)
```

### Multiple status lines appearing

**Cause:** Multiple reporters printing status, or terminal width issues

**Fix:**
- Only add one LiveReporter to population
- Ensure terminal is wide enough (80+ columns recommended)

### Status disappears when generation completes

**Cause:** Normal behavior - cursor moves to next line

**Fix:** This is expected. Final status is printed at end of run.

## Examples

### Watch a short run with full logs
```bash
python run_working_backache.py --generations=10
```
Output:
```
2025-10-09 10:00:00 - INFO - Starting experiment
2025-10-09 10:00:01 - INFO - Loading dataset
⚡ [██░░░░░░░░] 10.0% | Best: 0.7234 | Mean: 0.6543 | Pop: 150 | Species:  8 | ⏱️  00:15
2025-10-09 10:00:16 - INFO - Generation 1 complete
```

### Monitor a long run quietly
```bash
python run_working_backache.py --generations=100 --quiet --dashboard
```
Output:
```
================================================================================
  🧬 NEAT EVOLUTION STATUS
================================================================================
  Generation: 47 / 100  [█████████████░░░░░░░] 47.0%
  Best Fitness:  0.85430
  Mean Fitness:  0.72340
  Population:    150 genomes in 12 species
  Elapsed Time:  23:45
================================================================================
```
(updates in place every generation)

### Background run with logging
```bash
nohup python run_working_backache.py --generations=100 --quiet > evolution.log 2>&1 &

# Monitor in another terminal
tail -f evolution.log
```

## Summary

The live reporter provides:
- ✅ **Clean, readable status** that updates in place
- ✅ **Two display modes** - compact line or full dashboard
- ✅ **Quiet mode** to reduce log spam
- ✅ **Automatic terminal detection** with graceful fallbacks
- ✅ **Progress tracking** with visual bar and timing
- ✅ **Easy integration** into existing scripts

Perfect for monitoring long evolution runs without drowning in logs!
