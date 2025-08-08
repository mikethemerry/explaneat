-- Schema for ExplaNEAT database persistence
-- PostgreSQL compatible schema

-- Experiments table - tracks distinct experimental runs
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    config_json JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    status VARCHAR(50) DEFAULT 'running',
    seed INTEGER,
    CONSTRAINT unique_experiment_name UNIQUE (name)
);

-- Generations table - tracks population data per generation
CREATE TABLE IF NOT EXISTS generations (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    generation_number INTEGER NOT NULL,
    population_size INTEGER NOT NULL,
    best_fitness FLOAT,
    avg_fitness FLOAT,
    stdev_fitness FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_experiment_generation UNIQUE (experiment_id, generation_number)
);

-- Species table - tracks species information per generation  
CREATE TABLE IF NOT EXISTS species (
    id SERIAL PRIMARY KEY,
    generation_id INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    species_id INTEGER NOT NULL,
    representative_genome_id INTEGER,
    threshold FLOAT,
    members_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_generation_species UNIQUE (generation_id, species_id)
);

-- Genomes table - tracks individual genome metadata
CREATE TABLE IF NOT EXISTS genomes (
    id SERIAL PRIMARY KEY,
    generation_id INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    genome_id INTEGER NOT NULL,
    species_id INTEGER,
    fitness FLOAT,
    adjusted_fitness FLOAT,
    parent1_id INTEGER,
    parent2_id INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_generation_genome UNIQUE (generation_id, genome_id)
);

-- Nodes table - tracks node genes for each genome
CREATE TABLE IF NOT EXISTS nodes (
    id SERIAL PRIMARY KEY,
    genome_table_id INTEGER NOT NULL REFERENCES genomes(id) ON DELETE CASCADE,
    node_id INTEGER NOT NULL,
    node_type VARCHAR(20) NOT NULL, -- 'input', 'output', 'hidden'
    bias FLOAT,
    response FLOAT DEFAULT 1.0,
    activation_function VARCHAR(50) DEFAULT 'sigmoid',
    aggregation_function VARCHAR(50) DEFAULT 'sum',
    layer INTEGER,
    x_position FLOAT,
    y_position FLOAT,
    CONSTRAINT unique_genome_node UNIQUE (genome_table_id, node_id)
);

-- Connections table - tracks connection genes for each genome  
CREATE TABLE IF NOT EXISTS connections (
    id SERIAL PRIMARY KEY,
    genome_table_id INTEGER NOT NULL REFERENCES genomes(id) ON DELETE CASCADE,
    innovation_number INTEGER NOT NULL,
    in_node INTEGER NOT NULL,
    out_node INTEGER NOT NULL,
    weight FLOAT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    CONSTRAINT unique_genome_connection UNIQUE (genome_table_id, innovation_number)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);

CREATE INDEX IF NOT EXISTS idx_generations_experiment_id ON generations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_generations_generation_number ON generations(generation_number);
CREATE INDEX IF NOT EXISTS idx_generations_best_fitness ON generations(best_fitness DESC);

CREATE INDEX IF NOT EXISTS idx_species_generation_id ON species(generation_id);
CREATE INDEX IF NOT EXISTS idx_species_species_id ON species(species_id);

CREATE INDEX IF NOT EXISTS idx_genomes_generation_id ON genomes(generation_id);
CREATE INDEX IF NOT EXISTS idx_genomes_genome_id ON genomes(genome_id);
CREATE INDEX IF NOT EXISTS idx_genomes_species_id ON genomes(species_id);
CREATE INDEX IF NOT EXISTS idx_genomes_fitness ON genomes(fitness DESC);
CREATE INDEX IF NOT EXISTS idx_genomes_parents ON genomes(parent1_id, parent2_id);

CREATE INDEX IF NOT EXISTS idx_nodes_genome_table_id ON nodes(genome_table_id);
CREATE INDEX IF NOT EXISTS idx_nodes_node_id ON nodes(node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_layer ON nodes(layer);

CREATE INDEX IF NOT EXISTS idx_connections_genome_table_id ON connections(genome_table_id);
CREATE INDEX IF NOT EXISTS idx_connections_innovation ON connections(innovation_number);
CREATE INDEX IF NOT EXISTS idx_connections_nodes ON connections(in_node, out_node);
CREATE INDEX IF NOT EXISTS idx_connections_enabled ON connections(enabled);