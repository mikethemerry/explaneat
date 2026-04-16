"""allow interrupted status on experiments

Revision ID: k4l5m6n7o8p9
Revises: j3k4l5m6n7o8
Create Date: 2026-04-17
"""
from typing import Sequence, Union
from alembic import op

revision: str = 'k4l5m6n7o8p9'
down_revision: Union[str, None] = 'j3k4l5m6n7o8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint('check_status', 'experiments', type_='check')
    op.create_check_constraint(
        'check_status',
        'experiments',
        "status IN ('running', 'completed', 'failed', 'paused', 'interrupted')",
    )


def downgrade() -> None:
    op.drop_constraint('check_status', 'experiments', type_='check')
    op.create_check_constraint(
        'check_status',
        'experiments',
        "status IN ('running', 'completed', 'failed', 'paused')",
    )
