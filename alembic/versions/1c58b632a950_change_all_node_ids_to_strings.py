"""change_all_node_ids_to_strings

Revision ID: 1c58b632a950
Revises: 72b5419d429e
Create Date: 2026-01-15 10:48:10.831323

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1c58b632a950'
down_revision: Union[str, None] = '72b5419d429e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Change original_node_id in node_splits table from Integer to String.
    
    This allows all node IDs (including original nodes and split nodes) to be strings,
    enabling split nodes to connect to other split nodes without type conversion.
    """
    # Change original_node_id from Integer to String
    op.alter_column(
        'node_splits',
        'original_node_id',
        type_=sa.String(length=50),
        existing_type=sa.Integer(),
        nullable=False,
        postgresql_using='original_node_id::text',  # Convert existing integers to strings
    )


def downgrade() -> None:
    """
    Revert original_node_id back to Integer.
    Note: This will fail if any string node IDs exist that can't be converted to integers.
    """
    op.alter_column(
        'node_splits',
        'original_node_id',
        type_=sa.Integer(),
        existing_type=sa.String(length=50),
        nullable=False,
        postgresql_using='original_node_id::integer',  # Attempt to convert strings back to integers
    )
