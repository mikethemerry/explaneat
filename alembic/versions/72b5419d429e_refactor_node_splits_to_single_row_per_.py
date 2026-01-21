"""refactor_node_splits_to_single_row_per_node

Revision ID: 72b5419d429e
Revises: g5h6i7j8k9l0
Create Date: 2026-01-15 10:46:13.350315

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '72b5419d429e'
down_revision: Union[str, None] = 'g5h6i7j8k9l0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists using raw SQL (avoids inspector caching)."""
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :name)"
    ), {"name": table_name})
    return result.scalar()


def upgrade() -> None:
    """
    Refactor node_splits table to have one row per original_node_id.

    Old structure: Multiple rows per original_node_id, each with split_node_id and outgoing_connections
    New structure: Single row per original_node_id with split_mappings JSONB containing all splits

    split_mappings format: {"5_a": [[5, 10]], "5_b": [[5, 20]], "5_c": [[5, 30]]}
    """
    conn = op.get_bind()

    # Check if migration was partially run - if node_splits_old exists, skip rename
    if not table_exists(conn, 'node_splits_old'):
        # Rename old table only if it hasn't been renamed yet
        if table_exists(conn, 'node_splits'):
            op.rename_table('node_splits', 'node_splits_old')

    # Create new table structure (only if it doesn't exist)
    if not table_exists(conn, 'node_splits'):
        op.create_table(
            'node_splits',
            sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('genome_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('original_node_id', sa.Integer(), nullable=False),
            sa.Column(
                'split_mappings',
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
            ),  # Maps split_node_id -> list of outgoing connections: {"5_a": [[5, 10]], "5_b": [[5, 20]]}
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

    # Migrate data: consolidate multiple rows into single rows (only if old table exists)
    if table_exists(conn, 'node_splits_old'):
        op.execute("""
        INSERT INTO node_splits (
            id, genome_id, original_node_id, explanation_id, annotation_id,
            split_mappings, created_at, updated_at
        )
        SELECT
            gen_random_uuid(),
            genome_id,
            original_node_id,
            explanation_id,
            -- Use the first non-NULL annotation_id, or NULL if all are NULL
            (SELECT DISTINCT annotation_id FROM node_splits_old ns2
             WHERE ns2.genome_id = ns.genome_id
             AND ns2.original_node_id = ns.original_node_id
             AND ns2.explanation_id = ns.explanation_id
             AND ns2.annotation_id IS NOT NULL
             LIMIT 1) as annotation_id,
            jsonb_object_agg(
                split_node_id::text,
                outgoing_connections
            ) as split_mappings,
            MIN(created_at) as created_at,
            MAX(updated_at) as updated_at
        FROM node_splits_old ns
        GROUP BY genome_id, original_node_id, explanation_id;
        """)

    # Create indexes (only if node_splits table exists)
    if table_exists(conn, 'node_splits'):
        # Drop indexes first if they exist to handle partial migrations
        op.execute("DROP INDEX IF EXISTS idx_node_splits_genome_original")
        op.create_index(
            'idx_node_splits_genome_original',
            'node_splits',
            ['genome_id', 'original_node_id'],
            unique=False
        )
        op.execute("DROP INDEX IF EXISTS idx_node_splits_explanation")
        op.create_index(
            'idx_node_splits_explanation',
            'node_splits',
            ['explanation_id'],
            unique=False
        )
        # Add unique constraint: one split per original_node_id per explanation
        op.execute("DROP INDEX IF EXISTS idx_node_splits_unique")
        op.create_index(
            'idx_node_splits_unique',
            'node_splits',
            ['genome_id', 'original_node_id', 'explanation_id'],
            unique=True
        )

    # Drop old table (only if it exists)
    if table_exists(conn, 'node_splits_old'):
        op.drop_table('node_splits_old')


def downgrade() -> None:
    """
    Revert to old structure: multiple rows per original_node_id.
    Note: This will expand split_mappings back into multiple rows.
    """
    # Create old table structure
    op.create_table(
        'node_splits_old',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('genome_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('original_node_id', sa.Integer(), nullable=False),
        sa.Column('split_node_id', sa.String(length=50), nullable=False),
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
    
    # Expand split_mappings back into multiple rows
    op.execute("""
        INSERT INTO node_splits_old (
            id, genome_id, original_node_id, split_node_id, outgoing_connections,
            explanation_id, annotation_id, created_at, updated_at
        )
        SELECT 
            gen_random_uuid(),
            genome_id,
            original_node_id,
            split_node_id::text,
            outgoing_conns,
            explanation_id,
            annotation_id,
            created_at,
            updated_at
        FROM node_splits,
        LATERAL jsonb_each(split_mappings) AS split(split_node_id, outgoing_conns);
    """)
    
    # Drop new table and rename old
    op.drop_table('node_splits')
    op.rename_table('node_splits_old', 'node_splits')
    
    # Recreate old indexes
    op.create_index('idx_node_splits_genome_original', 'node_splits', ['genome_id', 'original_node_id'], unique=False)
    op.create_index('idx_node_splits_genome_split', 'node_splits', ['genome_id', 'split_node_id'], unique=False)
    op.create_index('idx_node_splits_explanation', 'node_splits', ['explanation_id'], unique=False)
