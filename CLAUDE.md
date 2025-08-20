# ExplaNEAT Project Context for Claude

## Project Overview
ExplaNEAT is a NEAT (NeuroEvolution of Augmenting Topologies) implementation with backpropagation capabilities for explanatory AI research.

## Environment Setup Notes
- **Python Package Management**: User needs to run pip/conda commands directly
- **Database Commands**: User needs to run alembic and database setup commands directly
- **PostgreSQL**: Project uses PostgreSQL for genome serialization and experiment tracking

## Key Commands to Suggest

### Database Setup
```bash
# Install database dependencies (user should run)
pip install -r requirements.txt

# Create PostgreSQL database (user should run)
python -m explaneat db create-db

# Or if you need to start completely fresh:
# 1. Drop existing database if it exists
psql -U postgres -c "DROP DATABASE IF EXISTS explaneat_dev;"
# 2. Remove any existing alembic directory
rm -rf alembic/
# 3. Create new database
python -m explaneat db create-db

# Initialize Alembic (user should run)
alembic init alembic

# Move the generated env.py to use our custom one (user should run)
mv alembic_env.py alembic/env.py

# Create initial migration (user should run)
alembic revision --autogenerate -m "Initial database schema"

# Apply migrations (user should run)
alembic upgrade head

# Alternative: Initialize database without migrations (user should run)
python -m explaneat db init
```

### Using the CLI
```bash
# View all commands
python -m explaneat --help

# Database commands
python -m explaneat db --help
python -m explaneat db init              # Create all tables
python -m explaneat db drop              # Drop all tables
python -m explaneat db revision "message" # Create new migration
python -m explaneat db upgrade           # Apply migrations
python -m explaneat db downgrade         # Revert last migration
python -m explaneat db current           # Show current revision
python -m explaneat db history           # Show migration history
```

### Running Tests
```bash
# Run tests (if pytest is configured)
pytest

# Run specific test file
pytest tests/test_db.py
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