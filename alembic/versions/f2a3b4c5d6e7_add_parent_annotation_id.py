"""add_parent_annotation_id

Revision ID: f2a3b4c5d6e7
Revises: f1a2b3c4d5e6
Create Date: 2025-01-21 10:01:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f2a3b4c5d6e7'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add parent_annotation_id column to annotations table
    op.add_column(
        'annotations',
        sa.Column(
            'parent_annotation_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
        )
    )
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_annotations_parent',
        'annotations',
        'annotations',
        ['parent_annotation_id'],
        ['id'],
        ondelete='SET NULL'
    )
    # Add index for hierarchy queries
    op.create_index('idx_annotations_parent', 'annotations', ['parent_annotation_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_annotations_parent', table_name='annotations')
    op.drop_constraint('fk_annotations_parent', 'annotations', type_='foreignkey')
    op.drop_column('annotations', 'parent_annotation_id')

