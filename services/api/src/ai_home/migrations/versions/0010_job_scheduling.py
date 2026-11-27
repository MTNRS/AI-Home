"""Optional gradient scheduling constraints; existing jobs stay unconstrained."""
from alembic import op
import sqlalchemy as sa

revision = "0010_job_scheduling"
down_revision = "0009_training_corrections"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("compute_jobs", sa.Column("scheduling", sa.JSON(none_as_null=True), nullable=True))


def downgrade():
    with op.batch_alter_table("compute_jobs") as batch:
        batch.drop_column("scheduling")
