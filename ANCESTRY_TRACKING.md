# Ancestry Tracking in ExplaNEAT

## Overview

ExplaNEAT now includes full ancestry tracking for genomes throughout evolution. This enables you to:
- Trace the complete lineage of any genome back to generation 0
- Identify both parents for any child genome
- Analyze fitness progression through family lines
- Understand which genetic lineages were most successful
- Debug evolutionary dynamics

## How It Works

### The Two-Layer System

NEAT uses **innovation numbers** to track when genes (nodes and connections) were first introduced. However, innovation numbers alone don't tell you which genome is the parent of another.

ExplaNEAT implements ancestry tracking using:

1. **NEAT's `reproduction.ancestors` dictionary** (in-memory)
   - Maps genome ID → (parent1_id, parent2_id)
   - Updated during crossover in `DefaultReproduction.reproduce()`
   - Lives only during the current run

2. **Database parent relationships** (persistent)
   - `Genome.parent1_id` and `Genome.parent2_id` foreign keys
   - Self-referential relationships for tree traversal
   - Survives across runs for analysis

### The AncestryReporter

The `AncestryReporter` class bridges these two systems:

```python
from explaneat.core.ancestry_reporter import AncestryReporter

# Create reporter
ancestry_reporter = AncestryReporter()

# Link it to NEAT's reproduction object
ancestry_reporter.reproduction = population.reproduction
```

During evolution:
1. NEAT's reproduction creates children and records parents in `reproduction.ancestors`
2. After fitness evaluation, genomes are saved to database
3. AncestryReporter translates NEAT genome IDs → database UUIDs
4. Parent UUIDs are stored in `Genome.parent1_id` and `Genome.parent2_id`
5. Reporter registers the new genome for next generation's parent lookups

## Usage

### Basic Usage in run_working_backache.py

```python
from explaneat.core.ancestry_reporter import AncestryReporter

# Create ancestry reporter
ancestry_reporter = AncestryReporter()

# Create population
population = instantiate_population(config, X_train, y_train)

# Link reporter to reproduction
ancestry_reporter.reproduction = population.reproduction

# Create database reporter with ancestry tracking
db_reporter = DatabaseReporter(experiment_id, config,
                               ancestry_reporter=ancestry_reporter)

# Add both reporters
population.reporters.reporters.append(db_reporter)
population.reporters.reporters.append(ancestry_reporter)

# Run evolution - parents are now tracked automatically!
```

### Using DatabaseBackpropPopulation

```python
from explaneat.db.population import DatabaseBackpropPopulation
from explaneat.core.ancestry_reporter import AncestryReporter

# Create ancestry reporter
ancestry_reporter = AncestryReporter()

# Create population with ancestry tracking
population = DatabaseBackpropPopulation(
    config,
    x_train,
    y_train,
    experiment_name="My Experiment",
    ancestry_reporter=ancestry_reporter
)

# Run - parents tracked automatically
best_genome = population.run(fitness_function, n=100, nEpochs=5)
```

### Analyzing Ancestry

Use the `AncestryAnalyzer` to explore lineages:

```python
from explaneat.analysis import AncestryAnalyzer

# Get a genome's ID from the database
genome_id = "..."  # UUID from database

# Create analyzer
analyzer = AncestryAnalyzer(genome_id)

# Get full ancestry tree
ancestry_df = analyzer.get_ancestry_tree(max_generations=10)
print(ancestry_df[['generation', 'fitness', 'num_nodes', 'parent1_id', 'parent2_id']])

# Get lineage statistics
stats = analyzer.get_lineage_statistics()
print(f"Fitness improved from {stats['fitness_progression']['initial_fitness']:.3f}")
print(f"to {stats['fitness_progression']['final_fitness']:.3f}")

# Find common ancestor with another genome
other_genome_id = "..."
common = analyzer.find_common_ancestor(other_genome_id)
if common:
    print(f"Common ancestor at generation {common['generation']}")
    print(f"Distance: {common['distance_from_self']} generations")
```

### Visualizing Ancestry

```python
from explaneat.analysis import GenomeExplorer

explorer = GenomeExplorer(experiment_id)

# Get best genome from final generation
best_genome_id = explorer.get_best_genome_id()

# Trace its ancestry
analyzer = AncestryAnalyzer(best_genome_id)
ancestry_df = analyzer.get_ancestry_tree()

# Plot fitness progression through lineage
import matplotlib.pyplot as plt
plt.plot(ancestry_df['generation'], ancestry_df['fitness'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Through Lineage')
plt.show()
```

## Database Schema

The `Genome` table includes:

```python
class Genome(Base):
    id = Column(UUID, primary_key=True)  # Database ID
    genome_id = Column(Integer)          # NEAT's genome ID
    parent1_id = Column(UUID, ForeignKey('genomes.id'))
    parent2_id = Column(UUID, ForeignKey('genomes.id'))

    # Relationships for traversing ancestry tree
    parent1 = relationship('Genome', foreign_keys=[parent1_id])
    parent2 = relationship('Genome', foreign_keys=[parent2_id])
    children_as_parent1 = relationship('Genome',
                                       foreign_keys=[parent1_id],
                                       back_populates='parent1')
    children_as_parent2 = relationship('Genome',
                                       foreign_keys=[parent2_id],
                                       back_populates='parent2')
```

## Why Track Parents Explicitly?

### Innovation Numbers Are Not Enough

**What innovation numbers tell you:**
- Which genes (nodes/connections) exist in a genome
- When each gene was first introduced to the population
- Structural similarity between genomes

**What innovation numbers DON'T tell you:**
- Which specific genome is the parent of another
- Whether two genomes with same genes are siblings or unrelated
- Which parent contributed which genes during crossover

### Benefits of Explicit Parent Tracking

1. **Complete Lineage Tracing**
   - Follow any genome back to generation 0
   - Identify all ancestors and their contributions
   - Build full family trees

2. **Fitness Analysis**
   - Track fitness progression through lineages
   - Identify which families produced champions
   - Understand when fitness gains occurred

3. **Evolutionary Dynamics**
   - Measure genetic diversity
   - Detect bottlenecks (many descendants from few parents)
   - Analyze selection pressure effects

4. **Debugging & Research**
   - Understand why certain lineages succeeded
   - Reproduce specific evolutionary paths
   - Validate crossover and mutation effects

5. **Performance Optimization**
   - Fast SQL queries: "SELECT * FROM genomes WHERE parent1_id = ?"
   - No need to reconstruct ancestry from innovation numbers
   - Efficient common ancestor algorithms

## Technical Details

### ID Translation Flow

```
Generation N:
  NEAT genome IDs: 1, 2, 3, ...
  Database UUIDs: uuid-a, uuid-b, uuid-c, ...
  AncestryReporter stores: {1: uuid-a, 2: uuid-b, 3: uuid-c}

Reproduction creates Generation N+1:
  Child genome ID: 150
  reproduction.ancestors[150] = (5, 23)  # NEAT IDs

Save to database:
  1. AncestryReporter.get_parent_ids(150)
  2. Look up NEAT IDs 5 and 23 → UUIDs uuid-e and uuid-w
  3. Create Genome record with parent1_id=uuid-e, parent2_id=uuid-w
  4. Save and get new UUID: uuid-new
  5. Register: ancestry_reporter.register_genome(150, uuid-new)

Next generation can now look up parents of children of genome 150
```

### Generation 0 Handling

Generation 0 genomes have no parents:
- `parent1_id = None`
- `parent2_id = None`
- `reproduction.ancestors[genome_id] = tuple()`  # Empty tuple

### Elitism and Cloning

Elites carried forward:
- May have same NEAT genome ID across generations (implementation-dependent)
- AncestryReporter handles this by tracking generation-specific mappings
- Parents will be the elite's parents from previous generation

## Example Queries

### Find All Descendants of a Genome

```python
from explaneat.db import db, Genome

with db.session_scope() as session:
    parent_id = "..."  # UUID

    # Direct children
    children = session.query(Genome).filter(
        (Genome.parent1_id == parent_id) |
        (Genome.parent2_id == parent_id)
    ).all()

    print(f"Genome {parent_id} has {len(children)} direct children")
```

### Find Champions' Common Ancestor

```python
from explaneat.analysis import AncestryAnalyzer

# Get two best genomes from different experiments
genome1_id = "..."
genome2_id = "..."

analyzer1 = AncestryAnalyzer(genome1_id)
common = analyzer1.find_common_ancestor(genome2_id)

if common:
    print(f"These champions share a common ancestor!")
    print(f"Generation: {common['generation']}")
    print(f"Fitness: {common['fitness']:.3f}")
else:
    print("No common ancestor found (different initial populations)")
```

## Troubleshooting

### Parents showing as None

**Cause:** AncestryReporter not properly linked to reproduction

**Fix:**
```python
ancestry_reporter.reproduction = population.reproduction
# Do this AFTER creating population but BEFORE running evolution
```

### Missing parent IDs for some genomes

**Cause:** Reporter not registered with population reporters

**Fix:**
```python
population.reporters.reporters.append(ancestry_reporter)
```

### Database foreign key errors

**Cause:** Parent UUID not found in database (parent wasn't saved)

**Fix:** Ensure genomes are flushed before being referenced:
```python
session.add(genome_record)
session.flush()  # Get ID before it's used as parent
```

## Future Enhancements

Potential additions:
- Mutation history tracking (what mutations occurred to create this genome)
- Gene contribution tracking (which parent contributed which genes)
- Ancestry visualization tools (phylogenetic trees)
- Breeding analysis (most successful parent combinations)
- Diversity metrics based on ancestry

## Summary

Ancestry tracking in ExplaNEAT:
- ✅ Records both parents for every genome
- ✅ Persists to database for long-term analysis
- ✅ Integrates seamlessly with existing NEAT reproduction
- ✅ Enables powerful lineage analysis via `AncestryAnalyzer`
- ✅ Supports research into evolutionary dynamics
- ✅ Complements innovation numbers for complete genome history

**Innovation numbers** tell you what genes exist and when they appeared.
**Parent tracking** tells you how genomes are related and where they came from.
**Together** they provide complete evolutionary traceability.
