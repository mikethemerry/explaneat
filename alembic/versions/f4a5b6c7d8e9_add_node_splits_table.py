"""add_node_splits_table

Revision ID: f4a5b6c7d8e9
Revises: f3a4b5c6d7e8
Create Date: 2025-01-21 10:03:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f4a5b6c7d8e9'
down_revision: Union[str, None] = 'f3a4b5c6d7e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create node_splits table
    op.create_table(
        'node_splits',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('genome_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('original_node_id', sa.Integer(), nullable=False),
        sa.Column('split_node_id', sa.Integer(), nullable=False),
        sa.Column(
            'outgoing_connections',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column('annotation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('explanation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['genome_id'], ['genomes.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['annotation_id'], ['annotations.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['explanation_id'], ['explanations.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    # Create indexes
    op.create_index('idx_node_splits_genome_original', 'node_splits', ['genome_id', 'original_node_id'], unique=False)
    op.create_index('idx_node_splits_genome_split', 'node_splits', ['genome_id', 'split_node_id'], unique=False)
    op.create_index('idx_node_splits_explanation', 'node_splits', ['explanation_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_node_splits_explanation', table_name='node_splits')
    op.drop_index('idx_node_splits_genome_split', table_name='node_splits')
    op.drop_index('idx_node_splits_genome_original', table_name='node_splits')
    op.drop_table('node_splits')

