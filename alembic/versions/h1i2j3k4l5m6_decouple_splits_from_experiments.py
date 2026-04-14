"""decouple dataset splits from experiments

Splits belong to datasets, not experiments. Move the FK so that
experiments reference a split rather than splits referencing an experiment.

Revision ID: h1i2j3k4l5m6
Revises: 1e22c8aa8775
Create Date: 2026-04-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'h1i2j3k4l5m6'
down_revision: Union[str, None] = '1e22c8aa8775'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add split_id to experiments and name to dataset_splits
    op.add_column('experiments', sa.Column('split_id', sa.UUID(), nullable=True))
    op.add_column('dataset_splits', sa.Column('name', sa.String(255), nullable=True))

    # 2. Migrate data: copy experiment_id from dataset_splits into experiments.split_id
    op.execute("""
        UPDATE experiments e
        SET split_id = ds.id
        FROM dataset_splits ds
        WHERE ds.experiment_id = e.id
    """)

    # 3. Add FK constraint on experiments.split_id
    op.create_foreign_key(
        'fk_experiments_split_id', 'experiments', 'dataset_splits',
        ['split_id'], ['id'], ondelete='SET NULL'
    )

    # 4. Drop old experiment_id column and its index from dataset_splits
    op.drop_index('idx_splits_experiment', table_name='dataset_splits')
    op.drop_constraint('dataset_splits_experiment_id_fkey', 'dataset_splits', type_='foreignkey')
    op.drop_column('dataset_splits', 'experiment_id')


def downgrade() -> None:
    # 1. Re-add experiment_id to dataset_splits
    op.add_column('dataset_splits', sa.Column('experiment_id', sa.UUID(), nullable=True))

    # 2. Migrate data back
    op.execute("""
        UPDATE dataset_splits ds
        SET experiment_id = e.id
        FROM experiments e
        WHERE e.split_id = ds.id
    """)

    # 3. Make experiment_id NOT NULL and add back FK + index
    op.alter_column('dataset_splits', 'experiment_id', nullable=False)
    op.create_foreign_key(
        'dataset_splits_experiment_id_fkey', 'dataset_splits', 'experiments',
        ['experiment_id'], ['id'], ondelete='CASCADE'
    )
    op.create_index('idx_splits_experiment', 'dataset_splits', ['experiment_id'])

    # 4. Drop the new columns
    op.drop_constraint('fk_experiments_split_id', 'experiments', type_='foreignkey')
    op.drop_column('experiments', 'split_id')
    op.drop_column('dataset_splits', 'name')
