# Running ExplaNEAT Experiments

## Virtual Environment Setup

**IMPORTANT:** This project uses a virtual environment located at:
```bash
/Users/mike/dev/explaneat/explaneat-env
```

**Activate before running any experiments:**
```bash
source /Users/mike/dev/explaneat/explaneat-env/bin/activate
```

You should see `(explaneat-env)` in your prompt:
```
(explaneat-env) ➜  explaneat git:(main) ✗
```

## Running Experiments

### ✅ Correct Way - Use `python`

```bash
# Activate virtual environment first
source /Users/mike/dev/explaneat/explaneat-env/bin/activate

# Run with python (NOT ipython)
python run_working_backache.py --generations=50
```

### ❌ Incorrect Way - Don't use `ipython` with args

```bash
# This will fail with "Unrecognized alias" warning
ipython run_working_backache.py --generations=50
```

**Why?** IPython tries to parse `--generations` as an IPython flag, not as a script argument.

### Alternative: Use IPython's `%run` magic

If you want to use IPython interactively:

```bash
# Start IPython
ipython

# Inside IPython, use %run magic
%run run_working_backache.py --generations=50
```

Or pass arguments after `--`:

```bash
ipython -- run_working_backache.py --generations=50
```

## Available Scripts

### run_working_backache.py

Main experiment script with ancestry tracking.

**Arguments:**
- `--generations` (int, default=10): Number of generations to run
- `--quiet`: Reduce logging verbosity (only show live status updates)
- `--dashboard`: Use multi-line dashboard instead of compact single-line status

**Examples:**
```bash
# Short run (10 generations, default)
python run_working_backache.py

# Longer run with live status line
python run_working_backache.py --generations=50

# Quiet mode - less logging, only live status
python run_working_backache.py --generations=50 --quiet

# Dashboard mode - multi-line status display
python run_working_backache.py --generations=50 --dashboard

# Quiet + dashboard for clean monitoring
python run_working_backache.py --generations=100 --quiet --dashboard
```

**Output Modes:**

1. **Default** - All logs + compact status line:
   ```
   2025-10-09 10:00:00 - INFO - Starting generation 5
   ⚡ [██████░░░░] 50.0% | Best: 0.8543 | Mean: 0.7234 | Pop: 150 | Species: 12 | ⏱️  05:23
   2025-10-09 10:00:05 - INFO - Backprop complete
   ```

2. **Quiet mode** (`--quiet`) - Only status line:
   ```
   ⚡ [██████░░░░] 50.0% | Best: 0.8543 | Mean: 0.7234 | Pop: 150 | Species: 12 | ⏱️  05:23
   ```

3. **Dashboard mode** (`--dashboard`) - Multi-line dashboard:
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

### Database Management

```bash
# Initialize database
python -m explaneat db init

# Create new migration
python -m explaneat db revision "description"

# Apply migrations
python -m explaneat db upgrade

# Check current version
python -m explaneat db current
```

## Testing Ancestry Tracking

After running an experiment:

```python
from explaneat.analysis import GenomeExplorer, AncestryAnalyzer
from explaneat.db import db

# Initialize database
db.init_db()

# List experiments
experiments = GenomeExplorer.list_experiments()
print(experiments)

# Get experiment ID (use the UUID from the list)
experiment_id = "..."  # Copy from experiments list

# Load best genome
explorer = GenomeExplorer.load_best_genome(experiment_id)

# Get ancestry tree (NEW - now has parent data!)
ancestry_df = explorer.get_ancestry_tree(max_generations=10)
print(ancestry_df[['generation', 'fitness', 'parent1_id', 'parent2_id']])

# Analyze lineage
analyzer = AncestryAnalyzer(explorer.genome_id)
stats = analyzer.get_lineage_statistics()
print(stats)
```

## Common Issues

### "Unrecognized alias" Warning

**Symptom:**
```
[TerminalIPythonApp] WARNING | Unrecognized alias: 'generations', it will have no effect.
```

**Solution:** Use `python` instead of `ipython` to run scripts with arguments.

### "Attempting to work in a virtualenv" Warning

**Symptom:**
```
UserWarning: Attempting to work in a virtualenv. If you encounter problems,
please install IPython inside the virtualenv.
```

**This is just a warning** - you can ignore it. But if you want to fix it:

```bash
# Install IPython inside the virtual environment
source /Users/mike/dev/explaneat/explaneat-env/bin/activate
pip install ipython
```

### Database Connection Errors

**Symptom:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solution:** Ensure PostgreSQL is running:
```bash
# Check if PostgreSQL is running
psql -U postgres -c "SELECT version();"

# If not running, start it (macOS)
brew services start postgresql
```

### Missing Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'explaneat'
```

**Solution:** Activate virtual environment first:
```bash
source /Users/mike/dev/explaneat/explaneat-env/bin/activate
```

If still failing, install in editable mode:
```bash
pip install -e .
```

## Quick Reference

```bash
# Setup (once)
source /Users/mike/dev/explaneat/explaneat-env/bin/activate
python -m explaneat db init

# Run experiment
python run_working_backache.py --generations=50

# Check results in Python
python
>>> from explaneat.analysis import GenomeExplorer
>>> experiments = GenomeExplorer.list_experiments()
>>> print(experiments)

# Or use IPython for interactive analysis
ipython
In [1]: from explaneat.analysis import GenomeExplorer
In [2]: experiments = GenomeExplorer.list_experiments()
In [3]: print(experiments)
```

## Performance Tips

### Short Test Runs
```bash
# Quick test (10 generations, ~5 minutes)
python run_working_backache.py --generations=10
```

### Production Runs
```bash
# Full run with ancestry tracking (100 generations, ~1-2 hours)
python run_working_backache.py --generations=100

# Monitor with logging
python run_working_backache.py --generations=100 2>&1 | tee experiment.log
```

### Background Runs
```bash
# Run in background with nohup
nohup python run_working_backache.py --generations=100 > output.log 2>&1 &

# Check progress
tail -f output.log

# Find the process
ps aux | grep run_working_backache
```

## Environment Variables

Optional environment variables:

```bash
# Custom database URL
export DATABASE_URL="postgresql://user:pass@localhost/explaneat_dev"

# Run experiment
python run_working_backache.py --generations=50
```

## Notes for Developers

- **Always activate venv**: `source explaneat-env/bin/activate`
- **Use `python`, not `ipython`** for scripts with CLI arguments
- **Database**: Ensure PostgreSQL is running before experiments
- **Ancestry tracking**: Automatically enabled in `run_working_backache.py`
- **Check git status**: Script logs current git branch and commit SHA
