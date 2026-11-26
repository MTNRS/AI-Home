"""Add reviewed, versioned frozen training datasets."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004_training_datasets"
down_revision: Union[str, None] = "0003_training_contributions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("training_contributions") as batch_op:
        batch_op.drop_constraint(
            "ck_training_contributions_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_training_contributions_status",
            "status IN ('pending_review', 'eligible', 'frozen', 'rejected')",
        )
    op.add_column(
        "training_contributions",
        sa.Column("reviewer_id", sa.String(length=80), nullable=True),
    )
    op.add_column(
        "training_contributions",
        sa.Column("reviewed_at_epoch", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "training_contributions",
        sa.Column("review_checklist", sa.JSON(), nullable=True),
    )
    op.create_table(
        "training_datasets",
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version", sa.String(length=80), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("records", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("dataset_id"),
        sa.UniqueConstraint(
            "dataset_version",
            name="uq_training_datasets_version",
        ),
    )


def downgrade() -> None:
    op.drop_table("training_datasets")
    op.drop_column("training_contributions", "review_checklist")
    op.drop_column("training_contributions", "reviewed_at_epoch")
    op.drop_column("training_contributions", "reviewer_id")
    with op.batch_alter_table("training_contributions") as batch_op:
        batch_op.drop_constraint(
            "ck_training_contributions_status",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_training_contributions_status",
            "status IN ('pending_review', 'eligible', 'frozen')",
        )

