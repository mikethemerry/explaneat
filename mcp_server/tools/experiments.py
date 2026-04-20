"""Experiment and genome discovery tools for the MCP server."""

import json
from typing import Optional

from sqlalchemy import func

from explaneat.db.models import Experiment, Genome, Population

from mcp_server.server import get_db
from mcp_server.helpers import _to_uuid


def list_experiments(offset: int = 0, limit: int = 20) -> str:
    """List experiments sorted by creation date (most recent first).

    Returns experiment metadata including generation count, genome count, and best fitness.
    """
    db = get_db()
    with db.session_scope() as session:
        experiments = (
            session.query(Experiment)
            .order_by(Experiment.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        result = []
        for exp in experiments:
            generations = (
                session.query(Population)
                .filter(Population.experiment_id == exp.id)
                .count()
            )

            genome_count = (
                session.query(Genome)
                .join(Population)
                .filter(Population.experiment_id == exp.id)
                .count()
            )

            best_fitness_val = (
                session.query(func.max(Genome.fitness))
                .join(Population)
                .filter(Population.experiment_id == exp.id)
                .scalar()
            )

            result.append({
                "id": str(exp.id),
                "name": exp.name,
                "description": exp.description,
                "status": exp.status,
                "generations": generations,
                "genome_count": genome_count,
                "best_fitness": best_fitness_val,
                "dataset_id": str(exp.dataset_id) if exp.dataset_id else None,
                "split_id": str(exp.split_id) if exp.split_id else None,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            })

        return json.dumps(result, indent=2)


def get_experiment(experiment_id: str) -> str:
    """Get detailed information about a specific experiment.

    Returns experiment metadata including resolved config from config_json.
    """
    db = get_db()
    with db.session_scope() as session:
        exp = session.query(Experiment).filter_by(id=_to_uuid(experiment_id)).first()
        if not exp:
            return json.dumps({"error": f"Experiment not found: {experiment_id}"})

        resolved_config = None
        if exp.config_json:
            resolved_config = exp.config_json.get("resolved_config")

        result = {
            "id": str(exp.id),
            "name": exp.name,
            "description": exp.description,
            "status": exp.status,
            "dataset_id": str(exp.dataset_id) if exp.dataset_id else None,
            "split_id": str(exp.split_id) if exp.split_id else None,
            "config_template_id": str(exp.config_template_id) if exp.config_template_id else None,
            "resolved_config": resolved_config,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
        }

        return json.dumps(result, indent=2)


def get_best_genome(experiment_id: str) -> str:
    """Get the best genome (highest fitness) from an experiment.

    Returns genome metadata including fitness, size metrics, and generation.
    """
    db = get_db()
    with db.session_scope() as session:
        exp = session.query(Experiment).filter_by(id=_to_uuid(experiment_id)).first()
        if not exp:
            return json.dumps({"error": f"Experiment not found: {experiment_id}"})

        best = (
            session.query(Genome)
            .join(Population)
            .filter(
                Population.experiment_id == exp.id,
                Genome.fitness.isnot(None),
            )
            .order_by(Genome.fitness.desc())
            .first()
        )

        if not best:
            return json.dumps({"error": f"No genomes with fitness found in experiment {experiment_id}"})

        result = {
            "id": str(best.id),
            "neat_genome_id": best.genome_id,
            "fitness": best.fitness,
            "num_nodes": best.num_nodes,
            "num_connections": best.num_connections,
            "network_depth": best.network_depth,
            "network_width": best.network_width,
            "generation": best.population.generation,
            "experiment_id": str(experiment_id),
        }

        return json.dumps(result, indent=2)


def list_genomes(experiment_id: str, min_fitness: Optional[float] = None, offset: int = 0, limit: int = 20) -> str:
    """List genomes from an experiment with optional minimum fitness filter.

    Returns genome metadata sorted by fitness (highest first).
    """
    db = get_db()
    with db.session_scope() as session:
        query = (
            session.query(Genome)
            .join(Population)
            .filter(Population.experiment_id == _to_uuid(experiment_id))
        )

        if min_fitness is not None:
            query = query.filter(Genome.fitness >= min_fitness)

        genomes = (
            query.order_by(Genome.fitness.desc().nullslast())
            .offset(offset)
            .limit(limit)
            .all()
        )

        result = []
        for g in genomes:
            result.append({
                "id": str(g.id),
                "neat_genome_id": g.genome_id,
                "fitness": g.fitness,
                "num_nodes": g.num_nodes,
                "num_connections": g.num_connections,
                "generation": g.population.generation,
            })

        return json.dumps(result, indent=2)


def get_genome(genome_id: str) -> str:
    """Get detailed metadata about a specific genome.

    Returns full genome metadata including parent IDs, fitness, and network metrics.
    """
    db = get_db()
    with db.session_scope() as session:
        genome = session.query(Genome).filter_by(id=_to_uuid(genome_id)).first()
        if not genome:
            return json.dumps({"error": f"Genome not found: {genome_id}"})

        result = {
            "id": str(genome.id),
            "neat_genome_id": genome.genome_id,
            "fitness": genome.fitness,
            "adjusted_fitness": genome.adjusted_fitness,
            "num_nodes": genome.num_nodes,
            "num_connections": genome.num_connections,
            "num_enabled_connections": genome.num_enabled_connections,
            "network_depth": genome.network_depth,
            "network_width": genome.network_width,
            "parent1_id": str(genome.parent1_id) if genome.parent1_id else None,
            "parent2_id": str(genome.parent2_id) if genome.parent2_id else None,
            "generation": genome.population.generation,
            "experiment_id": str(genome.population.experiment_id),
        }

        return json.dumps(result, indent=2)


def register(mcp) -> None:
    """Register experiment and genome discovery tools with the MCP server."""
    mcp.tool()(list_experiments)
    mcp.tool()(get_experiment)
    mcp.tool()(get_best_genome)
    mcp.tool()(list_genomes)
    mcp.tool()(get_genome)
