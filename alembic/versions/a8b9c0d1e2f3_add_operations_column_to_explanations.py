"""add_operations_column_to_explanations

Revision ID: a8b9c0d1e2f3
Revises: 1c58b632a950
Create Date: 2026-01-22 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a8b9c0d1e2f3'
down_revision: Union[str, None] = '1c58b632a950'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add operations JSONB column to explanations table.

    This column stores the event stream of operations as a JSON array:
    [{"seq": 0, "type": "split_node", "params": {...}, "result": {...}, "created_at": "..."}]
    """
    op.add_column(
        'explanations',
        sa.Column(
            'operations',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default='[]',
        )
    )


def downgrade() -> None:
    """
    Remove operations column from explanations table.
    """
    op.drop_column('explanations', 'operations')
