"""Persist reserved and settled compute rewards without deleting ledger evidence."""
from alembic import op
import sqlalchemy as sa

revision = "0006_job_funding"
down_revision = "0005_supabase_managed_auth"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("job_funding",
        sa.Column("job_id", sa.String(36), primary_key=True),
        sa.Column("owner_account", sa.String(160), sa.ForeignKey("credit_accounts.account_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("worker_account", sa.String(160), sa.ForeignKey("credit_accounts.account_id", ondelete="RESTRICT"), nullable=True),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("amount > 0 AND amount <= 1000000", name="ck_job_funding_amount"),
        sa.CheckConstraint("status IN ('reserved', 'settled', 'refunded')", name="ck_job_funding_status"))


def downgrade():
    op.drop_table("job_funding")
