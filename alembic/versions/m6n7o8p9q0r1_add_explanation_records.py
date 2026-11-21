"""Add explanation records (Claude-produced explanation reports) and their history

Revision ID: m6n7o8p9q0r1
Revises: l5m6n7o8p9q0
Create Date: 2026-11-22

Additive only: two new tables; no existing table or row changes. Until the records are imported
(``explaneat records import``) the code keeps reading the JSON files, so nothing changes before
then. ``record`` / ``report`` / ``mike_verification`` / ``payload`` are plain JSON (not JSONB) so
the key order of a record survives and ``explaneat records export`` reproduces its file.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'm6n7o8p9q0r1'
down_revision: Union[str, None] = 'l5m6n7o8p9q0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'explanation_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('label', sa.String(length=255), nullable=False, unique=True),
        sa.Column('study', sa.String(length=128), nullable=False),
        sa.Column('genome_id', sa.String(length=64), nullable=False),
        sa.Column('explanation_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('explanations.id', ondelete='SET NULL'), nullable=True),
        sa.Column('records_dir', sa.String(length=1024)),
        sa.Column('source_path', sa.String(length=1024), unique=True),
        sa.Column('record', sa.JSON(), nullable=False),
        sa.Column('report', sa.JSON()),
        sa.Column('mike_verification', sa.JSON()),
        sa.Column('content_hash', sa.String(length=64)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_explanation_records_genome_id', 'explanation_records', ['genome_id'])
    op.create_index('ix_explanation_records_explanation_id', 'explanation_records', ['explanation_id'])
    op.create_table(
        'explanation_record_events',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('record_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('explanation_records.id', ondelete='CASCADE'), nullable=False),
        sa.Column('kind', sa.String(length=32), nullable=False),
        sa.Column('entered_by', sa.String(length=32)),
        sa.Column('payload', sa.JSON()),
        sa.Column('previous', sa.JSON()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_explanation_record_events_record_id', 'explanation_record_events', ['record_id'])


def downgrade() -> None:
    op.drop_index('ix_explanation_record_events_record_id', table_name='explanation_record_events')
    op.drop_table('explanation_record_events')
    op.drop_index('ix_explanation_records_explanation_id', table_name='explanation_records')
    op.drop_index('ix_explanation_records_genome_id', table_name='explanation_records')
    op.drop_table('explanation_records')
