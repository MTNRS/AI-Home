"""Transactional inference reservations. Callers own commit/rollback and execution."""
from datetime import datetime, timezone
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
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def reserve_inference(session, user_id, key, request_hash, amount):
    """Return (reservation, created); a duplicate never authorizes another execution."""
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
        amount=amount, status="reserved").on_conflict_do_nothing(index_elements=["request_id"]))
    if not created.rowcount:
        return reserve_inference(session, user_id, key, request_hash, amount)
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
    changed = session.execute(update(InferenceFunding).where(
        InferenceFunding.request_id == request_id, InferenceFunding.status == "reserved"
    ).values(status=terminal, resolved_at=datetime.now(timezone.utc)))
    if not changed.rowcount:
        session.refresh(funding)
        if funding.status != terminal:
            raise InvalidPosting("Inference reservation already resolved differently")
        return funding
    destination = owner if refund else "service:foundation-inference"
    ensure_account(session, destination, "user" if refund else "service")
    escrow = "inference:" + request_id
    CreditLedger.post_in_session(session, idempotency_key=escrow + ":" + terminal,
                                operation="inference_refund" if refund else "inference_settle",
                                postings={escrow: -funding.amount, destination: funding.amount})
    session.flush()
    return funding
