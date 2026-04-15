"""add dataset encoding fields

Add source_dataset_id and encoding_config columns to datasets table
to support prepared (one-hot encoded) datasets that link back to their
source dataset.

Revision ID: i2j3k4l5m6n7
Revises: h1i2j3k4l5m6
Create Date: 2026-04-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'i2j3k4l5m6n7'
down_revision: Union[str, None] = 'h1i2j3k4l5m6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('datasets', sa.Column('source_dataset_id', sa.UUID(), nullable=True))
    op.add_column('datasets', sa.Column('encoding_config', postgresql.JSONB(), nullable=True))
    op.create_foreign_key(
        'fk_datasets_source_dataset_id', 'datasets', 'datasets',
        ['source_dataset_id'], ['id'], ondelete='CASCADE'
    )


def downgrade() -> None:
    op.drop_constraint('fk_datasets_source_dataset_id', 'datasets', type_='foreignkey')
    op.drop_column('datasets', 'encoding_config')
    op.drop_column('datasets', 'source_dataset_id')
