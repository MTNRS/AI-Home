"""Bound inference reservations and permit crash recovery."""
from alembic import op
import sqlalchemy as sa

revision = "0008_inference_expiry"
down_revision = "0007_inference_funding"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("inference_funding", sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))
    # Legacy reservations had no deadline; make them recoverable immediately.
    op.execute("UPDATE inference_funding SET expires_at = created_at")
    with op.batch_alter_table("inference_funding") as batch:
        batch.alter_column("expires_at", existing_type=sa.DateTime(timezone=True), nullable=False)


def downgrade():
    with op.batch_alter_table("inference_funding") as batch:
        batch.drop_column("expires_at")
