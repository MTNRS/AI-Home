"""Allow user profiles whose credentials are managed by Supabase Auth."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0005_supabase_managed_auth"
down_revision: Union[str, None] = "0004_training_datasets"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "password_hash",
            existing_type=sa.String(length=512),
            nullable=True,
        )


def downgrade() -> None:
    connection = op.get_bind()
    users_without_local_passwords = connection.scalar(
        sa.text("SELECT count(*) FROM users WHERE password_hash IS NULL")
    )
    if users_without_local_passwords:
        raise RuntimeError(
            "Cannot downgrade while Supabase-managed user profiles exist."
        )
    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "password_hash",
            existing_type=sa.String(length=512),
            nullable=False,
        )

