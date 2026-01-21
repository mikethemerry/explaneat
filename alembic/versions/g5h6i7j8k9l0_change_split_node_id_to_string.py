"""change_split_node_id_to_string

Revision ID: g5h6i7j8k9l0
Revises: f4a5b6c7d8e9
Create Date: 2025-01-21 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'g5h6i7j8k9l0'
down_revision: Union[str, None] = 'f4a5b6c7d8e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Change split_node_id from Integer to String
    # This allows us to use alpha suffixes like "5_a", "5_b" directly
    op.alter_column(
        'node_splits',
        'split_node_id',
        type_=sa.String(length=50),
        existing_type=sa.Integer(),
        nullable=False,
    )


def downgrade() -> None:
    # Revert back to Integer (data loss warning - string IDs can't be converted back)
    op.alter_column(
        'node_splits',
        'split_node_id',
        type_=sa.Integer(),
        existing_type=sa.String(length=50),
        nullable=False,
    )
