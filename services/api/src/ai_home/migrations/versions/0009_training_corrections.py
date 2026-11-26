"""Preserve correction provenance on independent consented candidates."""
from alembic import op
import sqlalchemy as sa

revision = "0009_training_corrections"
down_revision = "0008_inference_expiry"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("training_contributions", sa.Column("correction", sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table("training_contributions") as batch:
        batch.drop_column("correction")
