# NEAT Database Connector

The NEAT Database Connector provides a PostgreSQL persistence layer for ExplaNEAT experiments, allowing you to store complete genome data for all individuals across all generations.

## Features

- **Complete Genome Persistence**: Store full genome structure including nodes, connections, and fitness data
- **Experiment Tracking**: Manage multiple experiments with metadata and configuration
- **Generation Management**: Track populations across generations with statistics
- **Query Interface**: Retrieve best genomes, load specific generations, analyze lineages
- **Graceful Degradation**: Automatic disable if database unavailable or dependencies missing
- **Connection Pooling**: Efficient connection management with retry logic
- **Transaction Safety**: Rollback on failures, data integrity guarantees

## Quick Start

### 1. Install Dependencies

```bash
pip install psycopg2-binary
```

### 2. Configure Database Connection

Set environment variables:
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/explaneat"
```

Or set individual components:
```bash
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DB="explaneat"
export POSTGRES_USER="username"
export POSTGRES_PASSWORD="password"
```

### 3. Basic Usage

```python
from explaneat.database import NEATDatabaseConnector

# Initialize connector
with NEATDatabaseConnector() as db:
    # Initialize schema
    db.initialize_schema()
    
    # Create experiment
    config = {'pop_size': 50, 'fitness_threshold': 0.95}
    experiment_id = db.create_experiment("my_experiment", config)
    
    # Save generation (population should be list of (genome_id, genome) tuples)
    db.save_generation(experiment_id, generation=0, population=population)
    
    # Query results
    best_genomes = db.get_best_genomes(experiment_id, n=10)
    gen_data = db.load_generation(experiment_id, generation=0)
```

## Database Schema

The connector creates the following tables:

### experiments
- Tracks distinct experimental runs
- Stores configuration, metadata, timestamps

### generations  
- Population data per generation
- Fitness statistics, population size

### genomes
- Individual genome metadata
- Fitness scores, species assignment, parentage

### nodes
- Node genes for each genome
- Type, bias, activation functions, layer info

### connections
- Connection genes for each genome  
- Weights, enabled status, innovation numbers

### species
- Species information per generation
- Representatives, thresholds, member counts

## API Reference

### NEATDatabaseConnector

#### `__init__(connection_string=None, pool_size=5, enabled=True, retry_attempts=3, retry_delay=1.0)`

Initialize the database connector.

**Parameters:**
- `connection_string`: PostgreSQL connection string (uses env vars if None)
- `pool_size`: Connection pool size
- `enabled`: Enable/disable database operations
- `retry_attempts`: Number of retry attempts for failed operations  
- `retry_delay`: Delay between retries in seconds

#### `initialize_schema(drop_existing=False) -> bool`

Initialize database schema.

**Parameters:**
- `drop_existing`: Drop existing tables first

**Returns:** True if successful

#### `create_experiment(name, config, description=None) -> Optional[int]`

Create a new experiment.

**Parameters:**
- `name`: Unique experiment name
- `config`: Configuration dictionary
- `description`: Optional description

**Returns:** Experiment ID or None

#### `save_generation(experiment_id, generation, population, species_data=None) -> bool`

Save complete generation data.

**Parameters:**
- `experiment_id`: Experiment ID
- `generation`: Generation number  
- `population`: List of genomes or (genome_id, genome) tuples
- `species_data`: Optional species information

**Returns:** True if successful

#### `load_generation(experiment_id, generation) -> Optional[Dict]`

Load generation data including all genomes.

**Parameters:**
- `experiment_id`: Experiment ID
- `generation`: Generation number

**Returns:** Dictionary with generation info and genomes

#### `get_best_genomes(experiment_id, n=10) -> Optional[List[Dict]]`

Get best performing genomes across all generations.

**Parameters:**
- `experiment_id`: Experiment ID
- `n`: Number of genomes to return

**Returns:** List of genome dictionaries sorted by fitness

#### `get_experiment_info(experiment_id) -> Optional[Dict]`

Get experiment metadata.

**Parameters:**
- `experiment_id`: Experiment ID

**Returns:** Experiment information dictionary

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Complete PostgreSQL connection string | None |
| `POSTGRES_HOST` | PostgreSQL server host | localhost |
| `POSTGRES_PORT` | PostgreSQL server port | 5432 |
| `POSTGRES_DB` | Database name | explaneat |
| `POSTGRES_USER` | Database user | postgres |
| `POSTGRES_PASSWORD` | Database password | Required |

### Connection String Format

```
postgresql://username:password@hostname:port/database_name
```

## Error Handling

The connector implements graceful error handling:

- **Missing Dependencies**: Automatically disables if psycopg2 unavailable
- **Connection Failures**: Retry logic with exponential backoff
- **Transaction Safety**: Automatic rollback on partial failures
- **Graceful Degradation**: Returns None/False when disabled, doesn't break workflow

## Integration with NEAT

The connector works with standard NEAT-Python genome objects:

```python
# Standard NEAT genome structure expected
genome.nodes = {
    node_id: node_object,  # with .type, .bias, .response, .activation
    ...
}

genome.connections = {
    (in_node, out_node): connection_object,  # with .weight, .enabled
    ...
}

genome.fitness = 0.85  # fitness value
genome.key = genome_id  # unique identifier
```

## Performance Considerations

- **Bulk Operations**: Uses batch inserts for efficiency
- **Indexing**: Optimized indexes for common query patterns
- **Connection Pooling**: Reuses connections to reduce overhead
- **Async Support**: Optional asynchronous operations (future feature)

## Examples

See `examples/database_example.py` for a complete working example.

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check PostgreSQL is running and accessible
2. **Authentication Failed**: Verify username/password in connection string
3. **Schema Errors**: Ensure database exists and user has CREATE permissions
4. **Performance Issues**: Consider increasing connection pool size

### Logging

Enable debug logging to see connector operations:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## License

This module is part of the ExplaNEAT project and follows the same license terms.