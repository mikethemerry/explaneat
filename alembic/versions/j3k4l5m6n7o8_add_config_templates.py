"""add config templates

Revision ID: j3k4l5m6n7o8
Revises: i2j3k4l5m6n7
Create Date: 2026-04-16
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'j3k4l5m6n7o8'
down_revision: Union[str, None] = 'i2j3k4l5m6n7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEFAULT_CONFIG = {
    "training": {
        "population_size": 150,
        "n_generations": 10,
        "n_epochs_backprop": 5,
        "fitness_function": "bce",
    },
    "neat": {
        "bias_mutate_rate": 0.7,
        "bias_mutate_power": 0.5,
        "bias_replace_rate": 0.1,
        "weight_mutate_rate": 0.8,
        "weight_mutate_power": 0.5,
        "weight_replace_rate": 0.1,
        "enabled_mutate_rate": 0.01,
        "node_add_prob": 0.15,
        "node_delete_prob": 0.05,
        "conn_add_prob": 0.3,
        "conn_delete_prob": 0.1,
        "compatibility_threshold": 3.0,
        "compatibility_disjoint_coefficient": 1.0,
        "compatibility_weight_coefficient": 0.5,
        "max_stagnation": 15,
        "species_elitism": 2,
        "elitism": 2,
        "survival_threshold": 0.2,
    },
    "backprop": {
        "learning_rate": 1.5,
        "optimizer": "adadelta",
    },
}


def upgrade() -> None:
    op.create_table(
        'config_templates',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('config', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    op.add_column('experiments', sa.Column('config_template_id', sa.UUID(), nullable=True))
    op.create_foreign_key(
        'fk_experiments_config_template_id', 'experiments', 'config_templates',
        ['config_template_id'], ['id'], ondelete='SET NULL'
    )

    import json
    op.execute(
        f"INSERT INTO config_templates (id, name, description, config) "
        f"VALUES (gen_random_uuid(), 'Default', 'Default NEAT training configuration', "
        f"'{json.dumps(DEFAULT_CONFIG)}'::jsonb)"
    )


def downgrade() -> None:
    op.drop_constraint('fk_experiments_config_template_id', 'experiments', type_='foreignkey')
    op.drop_column('experiments', 'config_template_id')
    op.drop_table('config_templates')
