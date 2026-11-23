"""Create identity, auth-session, and credit-ledger tables."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_identity_and_credits"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_table(
        "auth_sessions",
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at_epoch", sa.BigInteger(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id"),
        sa.UniqueConstraint("token_hash", name="uq_auth_sessions_token_hash"),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])

    op.create_table(
        "credit_accounts",
        sa.Column("account_id", sa.String(length=160), nullable=False),
        sa.Column("account_type", sa.String(length=40), nullable=False),
        sa.Column("balance", sa.BigInteger(), nullable=False),
        sa.Column("allow_negative", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "allow_negative OR balance >= 0",
            name="ck_credit_accounts_nonnegative",
        ),
        sa.PrimaryKeyConstraint("account_id"),
    )
    op.create_table(
        "credit_transactions",
        sa.Column(
            "transaction_id",
            sa.BigInteger().with_variant(sa.Integer, "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("operation", sa.String(length=80), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("transaction_id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_table(
        "credit_postings",
        sa.Column("posting_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "transaction_id",
            sa.BigInteger().with_variant(sa.Integer, "sqlite"),
            nullable=False,
        ),
        sa.Column("account_id", sa.String(length=160), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("amount != 0", name="ck_credit_postings_nonzero"),
        sa.ForeignKeyConstraint(
            ["account_id"],
            ["credit_accounts.account_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["credit_transactions.transaction_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("posting_id"),
    )


def downgrade() -> None:
    op.drop_table("credit_postings")
    op.drop_table("credit_transactions")
    op.drop_table("credit_accounts")
    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("users")
