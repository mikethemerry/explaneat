"""add_annotations_table

Revision ID: 8f6478941aa
Revises: b1984296f9f1
Create Date: 2025-01-20 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "8f6478941aa"
down_revision: Union[str, None] = "b1984296f9f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create annotations table
    op.create_table(
        "annotations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("genome_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "subgraph_nodes", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "subgraph_connections",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "is_connected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["genome_id"], ["genomes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # Create indexes
    op.create_index(
        "idx_annotations_genome", "annotations", ["genome_id"], unique=False
    )
    op.create_index(
        op.f("ix_annotations_genome_id"), "annotations", ["genome_id"], unique=False
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index(op.f("ix_annotations_genome_id"), table_name="annotations")
    op.drop_index("idx_annotations_genome", table_name="annotations")
    # Drop table
    op.drop_table("annotations")



