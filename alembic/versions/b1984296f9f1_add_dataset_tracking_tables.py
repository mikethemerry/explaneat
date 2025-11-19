"""add_dataset_tracking_tables

Revision ID: b1984296f9f1
Revises: dbf82e8c2095
Create Date: 2025-11-19 15:09:50.425556

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b1984296f9f1'
down_revision: Union[str, None] = 'dbf82e8c2095'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create datasets table
    op.create_table('datasets',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=True),
        sa.Column('source', sa.String(length=255), nullable=True),
        sa.Column('source_url', sa.String(length=500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('num_samples', sa.Integer(), nullable=True),
        sa.Column('num_features', sa.Integer(), nullable=True),
        sa.Column('num_classes', sa.Integer(), nullable=True),
        sa.Column('feature_names', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('feature_descriptions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('feature_types', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('target_name', sa.String(length=255), nullable=True),
        sa.Column('target_description', sa.Text(), nullable=True),
        sa.Column('class_names', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_datasets_name'), 'datasets', ['name'], unique=False)
    op.create_index('idx_datasets_name_version', 'datasets', ['name', 'version'], unique=False)
    
    # Create dataset_splits table
    op.create_table('dataset_splits',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('dataset_id', sa.UUID(), nullable=False),
        sa.Column('experiment_id', sa.UUID(), nullable=False),
        sa.Column('split_type', sa.String(length=50), nullable=False),
        sa.Column('test_size', sa.Float(), nullable=True),
        sa.Column('random_state', sa.Integer(), nullable=True),
        sa.Column('shuffle', sa.Boolean(), nullable=True),
        sa.Column('stratify', sa.Boolean(), nullable=True),
        sa.Column('train_indices', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('test_indices', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('validation_indices', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('scaler_type', sa.String(length=50), nullable=True),
        sa.Column('scaler_params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('preprocessing_steps', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('train_size', sa.Integer(), nullable=True),
        sa.Column('test_size_actual', sa.Integer(), nullable=True),
        sa.Column('validation_size', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_splits_experiment', 'dataset_splits', ['experiment_id'], unique=False)
    op.create_index('idx_splits_dataset', 'dataset_splits', ['dataset_id'], unique=False)
    
    # Add dataset_id and random_seed to experiments table
    op.add_column('experiments', sa.Column('dataset_id', sa.UUID(), nullable=True))
    op.add_column('experiments', sa.Column('random_seed', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_experiments_dataset_id', 'experiments', 'datasets', ['dataset_id'], ['id'], ondelete='SET NULL')
    op.create_index('idx_experiments_dataset_id', 'experiments', ['dataset_id'], unique=False)


def downgrade() -> None:
    # Remove columns from experiments table
    op.drop_index('idx_experiments_dataset_id', table_name='experiments')
    op.drop_constraint('fk_experiments_dataset_id', 'experiments', type_='foreignkey')
    op.drop_column('experiments', 'random_seed')
    op.drop_column('experiments', 'dataset_id')
    
    # Drop dataset_splits table
    op.drop_index('idx_splits_dataset', table_name='dataset_splits')
    op.drop_index('idx_splits_experiment', table_name='dataset_splits')
    op.drop_table('dataset_splits')
    
    # Drop datasets table
    op.drop_index('idx_datasets_name_version', table_name='datasets')
    op.drop_index(op.f('ix_datasets_name'), table_name='datasets')
    op.drop_table('datasets')
