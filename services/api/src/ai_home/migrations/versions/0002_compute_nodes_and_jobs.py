"""Add volunteer nodes and allowlisted compute jobs."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002_compute_nodes_and_jobs"
down_revision: Union[str, None] = "0001_identity_and_credits"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "compute_nodes",
        sa.Column("node_id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("credential_hash", sa.String(length=64), nullable=False),
        sa.Column("hardware", sa.JSON(), nullable=False),
        sa.Column("supported_job_types", sa.JSON(), nullable=False),
        sa.Column("last_heartbeat_epoch", sa.BigInteger(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("node_id"),
        sa.UniqueConstraint("credential_hash"),
    )
    op.create_index(
        "ix_compute_nodes_owner_user_id",
        "compute_nodes",
        ["owner_user_id"],
    )

    op.create_table(
        "compute_jobs",
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("job_type", sa.String(length=80), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("idempotency_key", sa.String(length=160), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("public_node_consent_at_epoch", sa.BigInteger(), nullable=False),
        sa.Column("public_node_consent_version", sa.String(length=40), nullable=False),
        sa.Column("assigned_node_id", sa.String(length=36), nullable=True),
        sa.Column("lease_token_hash", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_epoch", sa.BigInteger(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("failure_code", sa.String(length=80), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["assigned_node_id"],
            ["compute_nodes.node_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id"),
        sa.UniqueConstraint(
            "owner_user_id",
            "idempotency_key",
            name="uq_compute_jobs_owner_idempotency",
        ),
    )
    op.create_index("ix_compute_jobs_owner_user_id", "compute_jobs", ["owner_user_id"])
    op.create_index("ix_compute_jobs_status", "compute_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_compute_jobs_status", table_name="compute_jobs")
    op.drop_index("ix_compute_jobs_owner_user_id", table_name="compute_jobs")
    op.drop_table("compute_jobs")
    op.drop_index("ix_compute_nodes_owner_user_id", table_name="compute_nodes")
    op.drop_table("compute_nodes")

