"""Add explicitly consented conversation training candidates."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003_training_contributions"
down_revision: Union[str, None] = "0002_compute_nodes_and_jobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "training_contributions",
        sa.Column("contribution_id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("source_endpoint", sa.String(length=500), nullable=False),
        sa.Column("model", sa.String(length=160), nullable=False),
        sa.Column("messages", sa.JSON(), nullable=False),
        sa.Column("consent_version", sa.String(length=40), nullable=False),
        sa.Column("consent_scope", sa.String(length=80), nullable=False),
        sa.Column("consent_at_epoch", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=24),
            server_default="pending_review",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending_review', 'eligible', 'frozen')",
            name="ck_training_contributions_status",
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("contribution_id"),
        sa.UniqueConstraint(
            "owner_user_id",
            "conversation_id",
            name="uq_training_contributions_owner_conversation",
        ),
    )
    op.create_index(
        "ix_training_contributions_owner_user_id",
        "training_contributions",
        ["owner_user_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_contributions_owner_user_id",
        table_name="training_contributions",
    )
    op.drop_table("training_contributions")
