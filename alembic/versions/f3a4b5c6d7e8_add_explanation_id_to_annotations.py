"""add_explanation_id_to_annotations

Revision ID: f3a4b5c6d7e8
Revises: f2a3b4c5d6e7
Create Date: 2025-01-21 10:02:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f3a4b5c6d7e8'
down_revision: Union[str, None] = 'f2a3b4c5d6e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add explanation_id column to annotations table
    op.add_column(
        'annotations',
        sa.Column(
            'explanation_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
        )
    )
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_annotations_explanation',
        'annotations',
        'explanations',
        ['explanation_id'],
        ['id'],
        ondelete='SET NULL'
    )
    # Add index
    op.create_index('idx_annotations_explanation', 'annotations', ['explanation_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_annotations_explanation', table_name='annotations')
    op.drop_constraint('fk_annotations_explanation', 'annotations', type_='foreignkey')
    op.drop_column('annotations', 'explanation_id')

