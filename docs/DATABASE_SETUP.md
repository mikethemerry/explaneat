# ExplaNEAT Database Setup Guide

This guide will walk you through setting up PostgreSQL database support for ExplaNEAT from scratch.

## Prerequisites

- PostgreSQL installed and running
- Python environment activated (using `uv`)
- Current working directory is the project root
- [uv](https://github.com/astral-sh/uv) installed (install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`)

## Fresh Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -e .
```

### 2. Create the Development Database

```bash
# Option A: Using the CLI (if you have psql installed)
uv run python -m explaneat db create-db

# Option B: Manual creation with psql
psql -U postgres -c "CREATE DATABASE explaneat_dev;"

# Option C: If using a different PostgreSQL user
psql -U your_username -d postgres -c "CREATE DATABASE explaneat_dev;"
```

### 3. Clean Up Any Previous Alembic Setup (if exists)

```bash
# Remove existing alembic directory if it exists
rm -rf alembic/

# Remove alembic_version table from database if it exists
psql -U postgres -d explaneat_dev -c "DROP TABLE IF EXISTS alembic_version;"
```

### 4. Initialize Alembic Fresh

```bash
# Initialize alembic
uv run alembic init alembic

# Replace the generated env.py with our custom one
cp alembic_env.py alembic/env.py
```

### 5. Create Initial Migration

```bash
# Generate the initial migration from our models
uv run alembic revision --autogenerate -m "Initial database schema"
```

### 6. Apply the Migration

```bash
# Apply migrations to create all tables
uv run alembic upgrade head
```

### 7. Verify Setup

```bash
# Check current revision
uv run alembic current

# List all tables (using psql)
psql -U postgres -d explaneat_dev -c "\dt"

# Or use Python to verify
uv run python -c "from explaneat.db import db; db.init_db(); print('Database connection successful!')"
```

## Environment Variables

You can override the default database URL by setting:

```bash
export DATABASE_URL=postgresql://username:password@localhost:5432/explaneat_dev
```

## Common Connection Strings

```bash
# Local development (no password)
postgresql://localhost/explaneat_dev

# With username
postgresql://postgres@localhost/explaneat_dev

# With username and password
postgresql://postgres:password@localhost/explaneat_dev

# Full format
postgresql://username:password@host:port/database_name
```

## Using the Database in Code

```python
from explaneat.db import db, Experiment, Population, Genome

# Initialize database connection
db.init_db()

# Use context manager for sessions
with db.session_scope() as session:
    # Create new experiment
    experiment = Experiment(
        experiment_sha="abc123",
        name="My NEAT Experiment",
        config_json={"pop_size": 150},
        neat_config_text="[NEAT]\nfitness_criterion = max\n..."
    )
    session.add(experiment)
    # Commit happens automatically when exiting the context
```

## CLI Commands Reference

```bash
# Database management (use uv run - automatically uses virtual environment)
uv run python -m explaneat db init              # Create all tables
uv run python -m explaneat db drop              # Drop all tables
uv run python -m explaneat db create-db         # Create PostgreSQL database

# Migration management  
uv run python -m explaneat db revision "message" # Create new migration
uv run python -m explaneat db upgrade           # Apply migrations
uv run python -m explaneat db downgrade         # Revert last migration
uv run python -m explaneat db current           # Show current revision
uv run python -m explaneat db history           # Show migration history
```

## Troubleshooting

### "Can't locate revision" Error
This happens when alembic history is out of sync. Fix by:
1. Dropping the alembic_version table
2. Removing alembic/versions directory
3. Starting fresh with steps 3-6 above

### Connection Refused
- Ensure PostgreSQL is running: `pg_ctl status` or `brew services list` (on macOS)
- Check PostgreSQL is listening on correct port: `psql -U postgres -c "SHOW port;"`

### Permission Denied
- Use correct PostgreSQL username
- Ensure user has CREATE DATABASE privileges
- Check pg_hba.conf for authentication settings