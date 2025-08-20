"""Initial database schema

Revision ID: 3f302fe63174
Revises: 
Create Date: 2025-08-20 16:06:57.174223

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3f302fe63174'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create experiments table first (no dependencies)
    op.create_table('experiments',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('experiment_sha', sa.String(length=40), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('dataset_name', sa.String(length=255), nullable=True),
    sa.Column('dataset_version', sa.String(length=50), nullable=True),
    sa.Column('config_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('neat_config_text', sa.Text(), nullable=False),
    sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
    sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=True),
    sa.Column('git_commit_sha', sa.String(length=40), nullable=True),
    sa.Column('git_branch', sa.String(length=255), nullable=True),
    sa.Column('hardware_info', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('running', 'completed', 'failed', 'paused')", name='check_status'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_experiments_dataset_name'), 'experiments', ['dataset_name'], unique=False)
    op.create_index(op.f('ix_experiments_experiment_sha'), 'experiments', ['experiment_sha'], unique=False)
    op.create_index(op.f('ix_experiments_name'), 'experiments', ['name'], unique=False)
    
    # Create populations table (depends on experiments)
    op.create_table('populations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('experiment_id', sa.UUID(), nullable=False),
    sa.Column('generation', sa.Integer(), nullable=False),
    sa.Column('population_size', sa.Integer(), nullable=False),
    sa.Column('num_species', sa.Integer(), nullable=False),
    sa.Column('best_fitness', sa.Float(), nullable=True),
    sa.Column('mean_fitness', sa.Float(), nullable=True),
    sa.Column('stdev_fitness', sa.Float(), nullable=True),
    sa.Column('config_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('generation_time_seconds', sa.Float(), nullable=True),
    sa.Column('backprop_time_seconds', sa.Float(), nullable=True),
    sa.Column('improvement_count', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('experiment_id', 'generation', name='uq_experiment_generation')
    )
    op.create_index('idx_populations_fitness', 'populations', ['best_fitness'], unique=False)
    
    # Create genomes table (depends on populations, has self-referential FK)
    op.create_table('genomes',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('population_id', sa.UUID(), nullable=False),
    sa.Column('species_id', sa.UUID(), nullable=True),
    sa.Column('genome_id', sa.Integer(), nullable=False),
    sa.Column('fitness', sa.Float(), nullable=True),
    sa.Column('adjusted_fitness', sa.Float(), nullable=True),
    sa.Column('genome_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('parent1_id', sa.UUID(), nullable=True),
    sa.Column('parent2_id', sa.UUID(), nullable=True),
    sa.Column('mutation_history', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('network_depth', sa.Integer(), nullable=True),
    sa.Column('network_width', sa.Integer(), nullable=True),
    sa.Column('num_nodes', sa.Integer(), nullable=True),
    sa.Column('num_connections', sa.Integer(), nullable=True),
    sa.Column('num_enabled_connections', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['parent1_id'], ['genomes.id'], ),
    sa.ForeignKeyConstraint(['parent2_id'], ['genomes.id'], ),
    sa.ForeignKeyConstraint(['population_id'], ['populations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_genomes_parents', 'genomes', ['parent1_id', 'parent2_id'], unique=False)
    op.create_index(op.f('ix_genomes_fitness'), 'genomes', ['fitness'], unique=False)
    
    # Create species table (depends on populations and genomes)
    op.create_table('species',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('population_id', sa.UUID(), nullable=False),
    sa.Column('species_id', sa.Integer(), nullable=False),
    sa.Column('size', sa.Integer(), nullable=False),
    sa.Column('fitness_mean', sa.Float(), nullable=True),
    sa.Column('fitness_max', sa.Float(), nullable=True),
    sa.Column('fitness_min', sa.Float(), nullable=True),
    sa.Column('age', sa.Integer(), nullable=False),
    sa.Column('last_improved', sa.Integer(), nullable=False),
    sa.Column('representative_genome_id', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['population_id'], ['populations.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['representative_genome_id'], ['genomes.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_species_internal_id', 'species', ['population_id', 'species_id'], unique=False)
    
    # Add the FK from genomes to species now that species table exists
    op.create_foreign_key('fk_genomes_species', 'genomes', 'species', ['species_id'], ['id'], ondelete='SET NULL')
    
    # Create checkpoints table (depends on experiments and populations)
    op.create_table('checkpoints',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('experiment_id', sa.UUID(), nullable=False),
    sa.Column('population_id', sa.UUID(), nullable=False),
    sa.Column('generation', sa.Integer(), nullable=False),
    sa.Column('checkpoint_data', sa.LargeBinary(), nullable=False),
    sa.Column('file_path', sa.String(length=500), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['population_id'], ['populations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_checkpoints_generation', 'checkpoints', ['experiment_id', 'generation'], unique=False)
    
    # Create results table (depends on experiments, populations, and genomes)
    op.create_table('results',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('experiment_id', sa.UUID(), nullable=False),
    sa.Column('population_id', sa.UUID(), nullable=True),
    sa.Column('genome_id', sa.UUID(), nullable=True),
    sa.Column('measurement_type', sa.String(length=100), nullable=False),
    sa.Column('value', sa.Float(), nullable=False),
    sa.Column('iteration', sa.Integer(), nullable=True),
    sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['genome_id'], ['genomes.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['population_id'], ['populations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_results_created', 'results', ['created_at'], unique=False)
    op.create_index(op.f('ix_results_measurement_type'), 'results', ['measurement_type'], unique=False)
    
    # Create training_metrics table (depends on genomes and populations)
    op.create_table('training_metrics',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('genome_id', sa.UUID(), nullable=False),
    sa.Column('population_id', sa.UUID(), nullable=False),
    sa.Column('epoch', sa.Integer(), nullable=False),
    sa.Column('loss', sa.Float(), nullable=True),
    sa.Column('accuracy', sa.Float(), nullable=True),
    sa.Column('validation_loss', sa.Float(), nullable=True),
    sa.Column('validation_accuracy', sa.Float(), nullable=True),
    sa.Column('backprop_time_seconds', sa.Float(), nullable=True),
    sa.Column('additional_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['genome_id'], ['genomes.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['population_id'], ['populations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_metrics_epoch', 'training_metrics', ['genome_id', 'epoch'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_metrics_epoch', table_name='training_metrics')
    op.drop_table('training_metrics')
    op.drop_index(op.f('ix_results_measurement_type'), table_name='results')
    op.drop_index('idx_results_created', table_name='results')
    op.drop_table('results')
    op.drop_index('idx_checkpoints_generation', table_name='checkpoints')
    op.drop_table('checkpoints')
    op.drop_foreign_key('fk_genomes_species', 'genomes')
    op.drop_index('idx_species_internal_id', table_name='species')
    op.drop_table('species')
    op.drop_index(op.f('ix_genomes_fitness'), table_name='genomes')
    op.drop_index('idx_genomes_parents', table_name='genomes')
    op.drop_table('genomes')
    op.drop_index('idx_populations_fitness', table_name='populations')
    op.drop_table('populations')
    op.drop_index(op.f('ix_experiments_name'), table_name='experiments')
    op.drop_index(op.f('ix_experiments_experiment_sha'), table_name='experiments')
    op.drop_index(op.f('ix_experiments_dataset_name'), table_name='experiments')
    op.drop_table('experiments')