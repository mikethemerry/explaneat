"""add_explanations_table

Revision ID: f1a2b3c4d5e6
Revises: e0ec9cf2c6a6
Create Date: 2025-01-21 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e0ec9cf2c6a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create explanations table
    op.create_table(
        'explanations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('genome_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_well_formed', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('structural_coverage', sa.Float(), nullable=True),
        sa.Column('compositional_coverage', sa.Float(), nullable=True),
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
        sa.PrimaryKeyConstraint('id'),
    )
    # Create indexes
    op.create_index('idx_explanations_genome', 'explanations', ['genome_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_explanations_genome', table_name='explanations')
    op.drop_table('explanations')

