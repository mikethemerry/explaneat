"""Add the evidence log (provenance of every computed number) and explanation integrity settings

Revision ID: l5m6n7o8p9q0
Revises: k4l5m6n7o8p9
Create Date: 2026-11-11

Additive only: two new tables; no existing table or row changes. Explanations without an
``explanation_integrity`` row (all existing ones) are legacy: strict mode off.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'l5m6n7o8p9q0'
down_revision: Union[str, None] = 'k4l5m6n7o8p9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'evidence_log',
        sa.Column('id', sa.String(length=32), primary_key=True),
        sa.Column('genome_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('genomes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('tool', sa.String(length=64), nullable=False),
        sa.Column('arguments', postgresql.JSONB(), nullable=False),
        sa.Column('query', postgresql.JSONB(), nullable=False),
        sa.Column('split', sa.String(length=16)),
        sa.Column('n', sa.Integer()),
        sa.Column('n_split', sa.Integer()),
        sa.Column('sampled', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('rows_hash', sa.String(length=64)),
        sa.Column('op_seq', sa.Integer()),
        sa.Column('code_hash', sa.String(length=32)),
        sa.Column('values', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_evidence_log_genome_id', 'evidence_log', ['genome_id'])
    op.create_table(
        'explanation_integrity',
        sa.Column('genome_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('genomes.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('strict', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('explanation_integrity')
    op.drop_index('ix_evidence_log_genome_id', table_name='evidence_log')
    op.drop_table('evidence_log')
