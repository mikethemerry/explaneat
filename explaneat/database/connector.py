"""
PostgreSQL Database Connector for NEAT Module

Provides persistence layer for NEAT experiments, storing complete genome data
for all individuals across all generations.
"""

import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)


class NEATDatabaseConnector:
    """
    Database connector for persisting NEAT experiments and genome data to PostgreSQL.
    
    Provides methods for:
    - Creating and managing experiments
    - Storing and retrieving genome populations across generations
    - Querying fitness and lineage information
    - Managing database schema
    """
    
    def __init__(self, connection_string: str = None, pool_size: int = 5, 
                 enabled: bool = True, retry_attempts: int = 3, retry_delay: float = 1.0):
        """
        Initialize the database connector.
        
        Args:
            connection_string: PostgreSQL connection string. If None, uses environment variables.
            pool_size: Size of connection pool
            enabled: Whether database persistence is enabled
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
        """
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.pool = None
        
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not available. Database persistence disabled.")
            self.enabled = False
            return
        
        self.enabled = enabled
        if not self.enabled:
            logger.info("Database persistence disabled.")
            return
        
        # Get connection parameters
        self.connection_string = connection_string or self._get_connection_string()
        
        try:
            # Create connection pool
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=pool_size,
                dsn=self.connection_string
            )
            logger.info(f"Database connection pool created with {pool_size} connections")
        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            self.enabled = False
    
    def _get_connection_string(self) -> str:
        """Get connection string from environment variables."""
        # Default to environment variables if no connection string provided
        db_url = os.environ.get('DATABASE_URL')
        if db_url:
            return db_url
        
        # Build from individual components
        host = os.environ.get('POSTGRES_HOST', 'localhost')
        port = os.environ.get('POSTGRES_PORT', '5432')
        database = os.environ.get('POSTGRES_DB', 'explaneat')
        user = os.environ.get('POSTGRES_USER', 'postgres')
        password = os.environ.get('POSTGRES_PASSWORD', '')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    def _get_connection(self):
        """Get a connection from the pool."""
        if not self.enabled or not self.pool:
            return None
        return self.pool.getconn()
    
    def _put_connection(self, conn):
        """Return connection to the pool."""
        if self.pool and conn:
            self.pool.putconn(conn)
    
    def _execute_with_retry(self, operation_func, *args, **kwargs):
        """Execute database operation with retry logic."""
        if not self.enabled:
            return None
            
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Database operation failed (attempt {attempt + 1}): {e}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Database operation failed after {self.retry_attempts} attempts: {e}")
        
        # If all retries failed, return None or raise depending on operation
        return None
    
    def initialize_schema(self, drop_existing: bool = False) -> bool:
        """
        Initialize database schema.
        
        Args:
            drop_existing: Whether to drop existing tables first
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.info("Database not enabled, skipping schema initialization")
            return False
        
        def _init_schema():
            conn = self._get_connection()
            if not conn:
                return False
                
            try:
                with conn.cursor() as cursor:
                    if drop_existing:
                        logger.info("Dropping existing tables...")
                        cursor.execute("""
                            DROP TABLE IF EXISTS connections CASCADE;
                            DROP TABLE IF EXISTS nodes CASCADE;
                            DROP TABLE IF EXISTS genomes CASCADE;
                            DROP TABLE IF EXISTS species CASCADE;
                            DROP TABLE IF EXISTS generations CASCADE;
                            DROP TABLE IF EXISTS experiments CASCADE;
                        """)
                    
                    # Read and execute schema
                    schema_path = Path(__file__).parent / 'schema.sql'
                    with open(schema_path, 'r') as f:
                        schema_sql = f.read()
                    
                    cursor.execute(schema_sql)
                    conn.commit()
                    logger.info("Database schema initialized successfully")
                    return True
                    
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to initialize schema: {e}")
                raise
            finally:
                self._put_connection(conn)
        
        return self._execute_with_retry(_init_schema) is not False
    
    def create_experiment(self, name: str, config: dict, description: str = None) -> Optional[int]:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name (must be unique)
            config: Configuration dictionary
            description: Optional description
            
        Returns:
            Experiment ID if successful, None otherwise
        """
        if not self.enabled:
            return None
        
        def _create_experiment():
            conn = self._get_connection()
            if not conn:
                return None
                
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO experiments (name, config_json, description, seed)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (name, json.dumps(config), description, config.get('random_seed')))
                    
                    result = cursor.fetchone()
                    experiment_id = result[0] if result else None
                    conn.commit()
                    
                    logger.info(f"Created experiment '{name}' with ID {experiment_id}")
                    return experiment_id
                    
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to create experiment: {e}")
                raise
            finally:
                self._put_connection(conn)
        
        return self._execute_with_retry(_create_experiment)
    
    def save_generation(self, experiment_id: int, generation: int, 
                       population: List, species_data: Dict = None) -> bool:
        """
        Save complete generation data including all genomes.
        
        Args:
            experiment_id: Experiment ID
            generation: Generation number
            population: List of genome objects or (genome_id, genome) tuples
            species_data: Optional species information
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        def _save_generation():
            conn = self._get_connection()
            if not conn:
                return False
                
            try:
                with conn.cursor() as cursor:
                    # Calculate generation statistics
                    fitnesses = []
                    for item in population:
                        if isinstance(item, tuple):
                            genome = item[1]
                        else:
                            genome = item
                        if hasattr(genome, 'fitness') and genome.fitness is not None:
                            fitnesses.append(genome.fitness)
                    
                    best_fitness = max(fitnesses) if fitnesses else None
                    avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else None
                    stdev_fitness = None
                    if len(fitnesses) > 1:
                        variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
                        stdev_fitness = variance ** 0.5
                    
                    # Insert generation record
                    cursor.execute("""
                        INSERT INTO generations 
                        (experiment_id, generation_number, population_size, best_fitness, avg_fitness, stdev_fitness)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (experiment_id, generation_number) 
                        DO UPDATE SET 
                            population_size = EXCLUDED.population_size,
                            best_fitness = EXCLUDED.best_fitness,
                            avg_fitness = EXCLUDED.avg_fitness,
                            stdev_fitness = EXCLUDED.stdev_fitness
                        RETURNING id
                    """, (experiment_id, generation, len(population), best_fitness, avg_fitness, stdev_fitness))
                    
                    generation_db_id = cursor.fetchone()[0]
                    
                    # Insert genomes
                    genome_inserts = []
                    node_inserts = []
                    connection_inserts = []
                    
                    for item in population:
                        if isinstance(item, tuple):
                            genome_id, genome = item
                        else:
                            # Assume genome object has a key attribute
                            genome_id = getattr(genome, 'key', id(genome))
                            
                        # Prepare genome data
                        fitness = getattr(genome, 'fitness', None)
                        adjusted_fitness = getattr(genome, 'adjusted_fitness', None)
                        
                        genome_inserts.append((
                            generation_db_id, genome_id, None,  # species_id will be updated later
                            fitness, adjusted_fitness, None, None  # parent IDs not tracked yet
                        ))
                    
                    # Bulk insert genomes
                    cursor.executemany("""
                        INSERT INTO genomes 
                        (generation_id, genome_id, species_id, fitness, adjusted_fitness, parent1_id, parent2_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (generation_id, genome_id) 
                        DO UPDATE SET 
                            fitness = EXCLUDED.fitness,
                            adjusted_fitness = EXCLUDED.adjusted_fitness
                    """, genome_inserts)
                    
                    # Get genome table IDs for nodes and connections
                    cursor.execute("""
                        SELECT genome_id, id FROM genomes WHERE generation_id = %s
                    """, (generation_db_id,))
                    genome_id_map = dict(cursor.fetchall())
                    
                    # Prepare node and connection data
                    for item in population:
                        if isinstance(item, tuple):
                            genome_id, genome = item
                        else:
                            genome_id = getattr(genome, 'key', id(genome))
                        
                        genome_table_id = genome_id_map[genome_id]
                        
                        # Extract nodes from genome
                        if hasattr(genome, 'nodes'):
                            for node_id, node in genome.nodes.items():
                                node_inserts.append((
                                    genome_table_id, node_id,
                                    getattr(node, 'type', 'hidden'),
                                    getattr(node, 'bias', 0.0),
                                    getattr(node, 'response', 1.0),
                                    getattr(node, 'activation', 'sigmoid'),
                                    getattr(node, 'aggregation', 'sum')
                                ))
                        
                        # Extract connections from genome  
                        if hasattr(genome, 'connections'):
                            for key, connection in genome.connections.items():
                                # Handle different key formats
                                if isinstance(key, tuple):
                                    in_node, out_node = key
                                    innovation = getattr(connection, 'innovation', hash(key))
                                else:
                                    innovation = key
                                    in_node = getattr(connection, 'in_node_id', 0)
                                    out_node = getattr(connection, 'out_node_id', 0)
                                
                                connection_inserts.append((
                                    genome_table_id, innovation, in_node, out_node,
                                    getattr(connection, 'weight', 0.0),
                                    getattr(connection, 'enabled', True)
                                ))
                    
                    # Bulk insert nodes and connections
                    if node_inserts:
                        cursor.executemany("""
                            INSERT INTO nodes 
                            (genome_table_id, node_id, node_type, bias, response, activation_function, aggregation_function)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (genome_table_id, node_id) DO NOTHING
                        """, node_inserts)
                    
                    if connection_inserts:
                        cursor.executemany("""
                            INSERT INTO connections 
                            (genome_table_id, innovation_number, in_node, out_node, weight, enabled)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (genome_table_id, innovation_number) 
                            DO UPDATE SET 
                                weight = EXCLUDED.weight,
                                enabled = EXCLUDED.enabled
                        """, connection_inserts)
                    
                    conn.commit()
                    logger.info(f"Saved generation {generation} with {len(population)} genomes")
                    return True
                    
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to save generation: {e}")
                raise
            finally:
                self._put_connection(conn)
        
        return self._execute_with_retry(_save_generation) is not False
    
    def load_generation(self, experiment_id: int, generation: int) -> Optional[Dict]:
        """
        Load generation data including all genomes.
        
        Args:
            experiment_id: Experiment ID
            generation: Generation number
            
        Returns:
            Dictionary with generation data or None if not found
        """
        if not self.enabled:
            return None
        
        def _load_generation():
            conn = self._get_connection()
            if not conn:
                return None
                
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Get generation info
                    cursor.execute("""
                        SELECT * FROM generations 
                        WHERE experiment_id = %s AND generation_number = %s
                    """, (experiment_id, generation))
                    
                    gen_data = cursor.fetchone()
                    if not gen_data:
                        return None
                    
                    # Get genomes with nodes and connections
                    cursor.execute("""
                        SELECT g.*, 
                               array_agg(DISTINCT jsonb_build_object(
                                   'node_id', n.node_id,
                                   'type', n.node_type,
                                   'bias', n.bias,
                                   'response', n.response,
                                   'activation', n.activation_function,
                                   'aggregation', n.aggregation_function
                               )) FILTER (WHERE n.id IS NOT NULL) as nodes,
                               array_agg(DISTINCT jsonb_build_object(
                                   'innovation', c.innovation_number,
                                   'in_node', c.in_node,
                                   'out_node', c.out_node,
                                   'weight', c.weight,
                                   'enabled', c.enabled
                               )) FILTER (WHERE c.id IS NOT NULL) as connections
                        FROM genomes g
                        LEFT JOIN nodes n ON g.id = n.genome_table_id
                        LEFT JOIN connections c ON g.id = c.genome_table_id
                        WHERE g.generation_id = %s
                        GROUP BY g.id
                        ORDER BY g.fitness DESC
                    """, (gen_data['id'],))
                    
                    genomes = cursor.fetchall()
                    
                    return {
                        'generation_info': dict(gen_data),
                        'genomes': [dict(g) for g in genomes]
                    }
                    
            except Exception as e:
                logger.error(f"Failed to load generation: {e}")
                raise
            finally:
                self._put_connection(conn)
        
        return self._execute_with_retry(_load_generation)
    
    def get_best_genomes(self, experiment_id: int, n: int = 10) -> Optional[List[Dict]]:
        """
        Get the best performing genomes across all generations.
        
        Args:
            experiment_id: Experiment ID  
            n: Number of top genomes to return
            
        Returns:
            List of genome dictionaries sorted by fitness
        """
        if not self.enabled:
            return None
        
        def _get_best():
            conn = self._get_connection()
            if not conn:
                return None
                
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT g.*, gen.generation_number
                        FROM genomes g
                        JOIN generations gen ON g.generation_id = gen.id  
                        WHERE gen.experiment_id = %s AND g.fitness IS NOT NULL
                        ORDER BY g.fitness DESC
                        LIMIT %s
                    """, (experiment_id, n))
                    
                    return [dict(row) for row in cursor.fetchall()]
                    
            except Exception as e:
                logger.error(f"Failed to get best genomes: {e}")
                raise
            finally:
                self._put_connection(conn)
        
        return self._execute_with_retry(_get_best)
    
    def get_experiment_info(self, experiment_id: int) -> Optional[Dict]:
        """Get experiment information."""
        if not self.enabled:
            return None
        
        def _get_info():
            conn = self._get_connection()
            if not conn:
                return None
                
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM experiments WHERE id = %s", (experiment_id,))
                    result = cursor.fetchone()
                    return dict(result) if result else None
                    
            except Exception as e:
                logger.error(f"Failed to get experiment info: {e}")
                raise
            finally:
                self._put_connection(conn)
        
        return self._execute_with_retry(_get_info)
    
    def close(self):
        """Close database connections and cleanup."""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connections closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()