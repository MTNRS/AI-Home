"""Funded job rewards using the existing balanced ledger in one transaction."""
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column

from ai_home.credits import Account, CreditLedger, InvalidPosting
from ai_home.database import Base


class JobFunding(Base):
    __tablename__ = "job_funding"
    __table_args__ = (
        CheckConstraint("amount > 0 AND amount <= 1000000", name="ck_job_funding_amount"),
        CheckConstraint("status IN ('reserved', 'settled', 'refunded')", name="ck_job_funding_status"),
    )
    # Keep accounting evidence even if the operational job/account is removed.
    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_account: Mapped[str] = mapped_column(ForeignKey("credit_accounts.account_id", ondelete="RESTRICT"))
    worker_account: Mapped[str | None] = mapped_column(ForeignKey("credit_accounts.account_id", ondelete="RESTRICT"))
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def user_account_id(user_id):
    return f"user:{user_id}"


def ensure_account(session, account_id, account_type):
    account = session.get(Account, account_id)
    if account is None:
        try:
            with session.begin_nested():
                session.add(Account(account_id=account_id, account_type=account_type,
                                    balance=0, allow_negative=False))
                session.flush()
        except IntegrityError:
            account = session.get(Account, account_id)
            if account is None:
                raise
        else:
            account = session.get(Account, account_id)
    if account.allow_negative:
        raise InvalidPosting("Job accounts must not allow negative balances")
    return account


def reserve_job(session: Session, job, amount):
    if type(amount) is not int or not 0 <= amount <= 1_000_000:
        raise ValueError("Invalid job reward")
    if amount == 0:
        return
    owner = user_account_id(job.owner_user_id)
    escrow = f"job:{job.job_id}"
    ensure_account(session, owner, "user")
    ensure_account(session, escrow, "job_escrow")
    CreditLedger.post_in_session(session, idempotency_key=f"job:{job.job_id}:reserve",
        operation="job_reserve", postings={owner: -amount, escrow: amount})
    session.add(JobFunding(job_id=job.job_id, owner_account=owner, amount=amount, status="reserved"))
    session.flush()


def settle_job(session: Session, job, worker_user_id):
    funding = session.scalar(select(JobFunding).where(JobFunding.job_id == job.job_id).with_for_update())
    if funding is None:
        return
    if funding.status != "reserved":
        raise InvalidPosting("Job funding is not reserved")
    worker = user_account_id(worker_user_id)
    ensure_account(session, worker, "user")
    CreditLedger.post_in_session(session, idempotency_key=f"job:{job.job_id}:settle",
        operation="job_settle", postings={f"job:{job.job_id}": -funding.amount, worker: funding.amount})
    funding.worker_account = worker
    funding.status = "settled"
    funding.resolved_at = datetime.now(timezone.utc)
    session.flush()


def refund_job(session: Session, job):
    funding = session.scalar(select(JobFunding).where(JobFunding.job_id == job.job_id).with_for_update())
    if funding is None or funding.status == "refunded":
        return
    if funding.status != "reserved":
        raise InvalidPosting("Settled jobs cannot be refunded")
    CreditLedger.post_in_session(session, idempotency_key=f"job:{job.job_id}:refund",
        operation="job_refund", postings={f"job:{job.job_id}": -funding.amount, funding.owner_account: funding.amount})
    funding.status = "refunded"
    funding.resolved_at = datetime.now(timezone.utc)
    session.flush()
