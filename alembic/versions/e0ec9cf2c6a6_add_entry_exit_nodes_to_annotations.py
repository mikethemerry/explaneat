"""add_entry_exit_nodes_to_annotations

Revision ID: e0ec9cf2c6a6
Revises: 8f6478941aa
Create Date: 2025-12-04 09:20:34.138060

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e0ec9cf2c6a6'
down_revision: Union[str, None] = '8f6478941aa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add entry_nodes and exit_nodes columns
    op.add_column(
        'annotations',
        sa.Column(
            'entry_nodes',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb")
        )
    )
    op.add_column(
        'annotations',
        sa.Column(
            'exit_nodes',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb")
        )
    )
    
    # Note: For existing annotations, entry_nodes and exit_nodes will be empty arrays.
    # These should be populated by inferring from the subgraph structure or by
    # re-creating the annotations with explicit entry/exit nodes.


def downgrade() -> None:
    op.drop_column('annotations', 'exit_nodes')
    op.drop_column('annotations', 'entry_nodes')
