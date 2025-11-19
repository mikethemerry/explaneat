# ExplaNEAT Project Context for Claude

## Project Overview
ExplaNEAT is a NEAT (NeuroEvolution of Augmenting Topologies) implementation with backpropagation capabilities for explanatory AI research.

## Environment Setup Notes
- **Python Package Management**: Project uses `uv` for package management (fast Python package installer)
- **Running Scripts**: **ALWAYS use `uv run`** - this automatically uses the virtual environment without manual activation
- **Database Commands**: User needs to run alembic and database setup commands directly
- **PostgreSQL**: Project uses PostgreSQL for genome serialization and experiment tracking

## Key Commands to Suggest

**IMPORTANT FOR AGENTS**: Always use `uv run` to execute Python commands. This ensures the correct virtual environment is used automatically.

### Initial Setup
```bash
# Install dependencies with uv (user should run)
# First ensure uv is installed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install -e .
```

### Database Setup
```bash
# Create PostgreSQL database (user should run)
uv run python -m explaneat db create-db

# Or if you need to start completely fresh:
# 1. Drop existing database if it exists
psql -U postgres -c "DROP DATABASE IF EXISTS explaneat_dev;"
# 2. Remove any existing alembic directory
rm -rf alembic/
# 3. Create new database
uv run python -m explaneat db create-db

# Initialize Alembic (user should run)
uv run alembic init alembic

# Move the generated env.py to use our custom one (user should run)
mv alembic_env.py alembic/env.py

# Create initial migration (user should run)
uv run alembic revision --autogenerate -m "Initial database schema"

# Apply migrations (user should run)
uv run alembic upgrade head

# Alternative: Initialize database without migrations (user should run)
uv run python -m explaneat db init
```

### Using the CLI
```bash
# View all commands
uv run python -m explaneat --help

# Database commands
uv run python -m explaneat db --help
uv run python -m explaneat db init              # Create all tables
uv run python -m explaneat db drop              # Drop all tables
uv run python -m explaneat db revision "message" # Create new migration
uv run python -m explaneat db upgrade           # Apply migrations
uv run python -m explaneat db downgrade         # Revert last migration
uv run python -m explaneat db current           # Show current revision
uv run python -m explaneat db history           # Show migration history
```

### Running Scripts
```bash
# Run experiment scripts
uv run python run_working_backache.py --generations=50
uv run python run_simple_experiment.py
uv run python genome_explorer_cli.py --interactive

# Run examples
uv run python examples/basic_usage.py
```

### Running Tests
```bash
# Run tests (if pytest is configured)
uv run pytest

# Run specific test file
uv run pytest tests/test_db.py
```

### Database Connection
- Default connection string: `postgresql://localhost/explaneat_dev`
- Can be overridden with `DATABASE_URL` environment variable
- Database name: `explaneat_dev` (for development)

## Project Structure
- `/explaneat/core/` - Core NEAT implementation with backprop
- `/explaneat/experimenter/` - Experiment management
- `/explaneat/db/` - Database models and connection management
- Configuration files use both `.cfg` (NEAT) and `.json` (experiments)

## Recent Development
- Added SQLAlchemy models for genome serialization
- Database schema includes: experiments, populations, species, genomes, training_metrics, checkpoints, results
- Models follow Flask-SQLAlchemy patterns for familiarity

## Important Notes
- Do NOT run package installation commands directly - suggest them to the user
- Do NOT run database migration commands directly - suggest them to the user
- Focus on writing code and suggesting commands rather than executing system-level operations