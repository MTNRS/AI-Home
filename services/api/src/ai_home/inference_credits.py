"""Transactional inference reservations. Callers own commit/rollback and execution."""
from datetime import datetime, timezone, timedelta
import re
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, func, select, update
from sqlalchemy.orm import Mapped, mapped_column

from .credits import CreditLedger, IdempotencyConflict, InvalidPosting
from .database import Base
from .job_credits import ensure_account, user_account_id


class InferenceFunding(Base):
    __tablename__ = "inference_funding"
    __table_args__ = (
        CheckConstraint("amount > 0 AND amount <= 1000000", name="ck_inference_funding_amount"),
        CheckConstraint("status IN ('reserved', 'settled', 'refunded')", name="ck_inference_funding_status"),
    )
    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_account: Mapped[str] = mapped_column(ForeignKey("credit_accounts.account_id", ondelete="RESTRICT"))
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def reserve_inference(session, user_id, key, request_hash, amount, *, lifetime_seconds=600):
    """Return (reservation, created); a duplicate never authorizes another execution."""
    if type(lifetime_seconds) is not int or not 30 <= lifetime_seconds <= 3600:
        raise ValueError("Reservation lifetime must be between 30 and 3600 seconds")
    if type(amount) is not int or not 1 <= amount <= 1_000_000:
        raise ValueError("Invalid inference price")
    if not isinstance(key, str) or not key.strip() or len(key) > 128:
        raise ValueError("Invalid inference idempotency key")
    if not isinstance(request_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", request_hash):
        raise ValueError("Invalid inference request hash")
    owner = user_account_id(user_id)
    request_id = str(uuid5(NAMESPACE_URL, owner + ":inference:" + key))
    existing = session.get(InferenceFunding, request_id)
    if existing is not None:
        if existing.request_hash != request_hash or existing.amount != amount or existing.owner_account != owner:
            raise IdempotencyConflict("Inference key reused with different request or price")
        return existing, False
    ensure_account(session, owner, "user")
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise InvalidPosting("Unsupported accounting database")
    created = session.execute(insert(InferenceFunding).values(
        request_id=request_id, owner_account=owner, request_hash=request_hash,
        amount=amount, status="reserved", expires_at=datetime.now(timezone.utc) + timedelta(seconds=lifetime_seconds)).on_conflict_do_nothing(index_elements=["request_id"]))
    if not created.rowcount:
        return reserve_inference(session, user_id, key, request_hash, amount, lifetime_seconds=lifetime_seconds)
    funding = session.get(InferenceFunding, request_id)
    escrow = "inference:" + request_id
    ensure_account(session, escrow, "inference_escrow")
    CreditLedger.post_in_session(session, idempotency_key=escrow + ":reserve",
                                operation="inference_reserve", postings={owner: -amount, escrow: amount})
    return funding, True


def resolve_inference(session, request_id, user_id, *, refund=False):
    """Exactly one terminal transition; ownership checked before any ledger action."""
    owner = user_account_id(user_id)
    funding = session.scalar(select(InferenceFunding).where(
        InferenceFunding.request_id == request_id, InferenceFunding.owner_account == owner))
    if funding is None:
        raise InvalidPosting("Inference reservation not found")
    terminal = "refunded" if refund else "settled"
    now = datetime.now(timezone.utc)
    conditions = [InferenceFunding.request_id == request_id, InferenceFunding.status == "reserved"]
    if not refund:
        conditions.append(InferenceFunding.expires_at > now)
    changed = session.execute(update(InferenceFunding).where(*conditions).values(
        status=terminal, resolved_at=now).execution_options(synchronize_session=False))
    if not changed.rowcount:
        session.refresh(funding)
        if funding.status != terminal:
            raise InvalidPosting("Inference reservation already resolved differently")
        return funding
    session.refresh(funding)
    destination = owner if refund else "service:foundation-inference"
    ensure_account(session, destination, "user" if refund else "service")
    escrow = "inference:" + request_id
    CreditLedger.post_in_session(session, idempotency_key=escrow + ":" + terminal,
                                operation="inference_refund" if refund else "inference_settle",
                                postings={escrow: -funding.amount, destination: funding.amount})
    session.flush()
    return funding


def refund_expired_inference(session, user_id, *, limit=100):
    """Lazy recovery after crashes; never refund another user's reservation."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Invalid reconciliation batch limit")
    rows = session.scalars(select(InferenceFunding).where(
        InferenceFunding.owner_account == user_account_id(user_id),
        InferenceFunding.status == "reserved", InferenceFunding.expires_at <= datetime.now(timezone.utc)
    ).order_by(InferenceFunding.expires_at, InferenceFunding.request_id).limit(limit)).all()
    recovered = 0
    for row in rows:
        try:
            resolve_inference(session, row.request_id, user_id, refund=True)
            recovered += 1
        except InvalidPosting:
            session.refresh(row)
            if row.status != "settled":
                raise
    return recovered


def refund_pending_inference(session, user_id):
    """Account closure: caller must first serialize admission on the user row."""
    rows = session.scalars(select(InferenceFunding).where(
        InferenceFunding.owner_account == user_account_id(user_id),
        InferenceFunding.status == "reserved").with_for_update()).all()
    for row in rows:
        resolve_inference(session, row.request_id, user_id, refund=True)
    return len(rows)
