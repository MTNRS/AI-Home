"""Transactional, idempotent internal ledger primitives for AI@Home credits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class LedgerError(Exception):
    """Base class for expected ledger failures."""


class InvalidPosting(LedgerError):
    """Raised when a proposed ledger transaction is not balanced or valid."""


class AccountNotFound(LedgerError):
    """Raised when a posting references an unknown account."""


class InsufficientCredits(LedgerError):
    """Raised when a non-negative account would become overdrawn."""


class IdempotencyConflict(LedgerError):
    """Raised when an idempotency key is reused for a different transaction."""


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "credit_accounts"
    __table_args__ = (
        CheckConstraint(
            "allow_negative OR balance >= 0",
            name="ck_credit_accounts_nonnegative",
        ),
    )

    account_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    account_type: Mapped[str] = mapped_column(String(40), nullable=False)
    balance: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    allow_negative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class LedgerTransaction(Base):
    __tablename__ = "credit_transactions"

    transaction_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    operation: Mapped[str] = mapped_column(String(80), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class Posting(Base):
    __tablename__ = "credit_postings"
    __table_args__ = (
        CheckConstraint("amount != 0", name="ck_credit_postings_nonzero"),
    )

    posting_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("credit_transactions.transaction_id", ondelete="RESTRICT"),
        nullable=False,
    )
    account_id: Mapped[str] = mapped_column(
        ForeignKey("credit_accounts.account_id", ondelete="RESTRICT"),
        nullable=False,
    )
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)


@dataclass(frozen=True)
class LedgerReceipt:
    transaction_id: int
    idempotency_key: str
    operation: str


class CreditLedger:
    """Posts balanced transactions and maintains a transactionally updated balance."""

    def __init__(self, engine: Engine):
        self._engine = engine

    @classmethod
    def from_url(cls, database_url: str) -> CreditLedger:
        return cls(create_engine(database_url))

    def initialize(self) -> None:
        Base.metadata.create_all(self._engine)

    def create_account(
        self,
        account_id: str,
        account_type: str,
        *,
        allow_negative: bool = False,
    ) -> None:
        if (
            not account_id.strip()
            or len(account_id) > 160
            or not account_type.strip()
            or len(account_type) > 40
        ):
            raise InvalidPosting("Account ID or account type is invalid.")
        with Session(self._engine) as session, session.begin():
            if session.get(Account, account_id) is not None:
                raise InvalidPosting(f"Account already exists: {account_id}")
            session.add(
                Account(
                    account_id=account_id,
                    account_type=account_type,
                    balance=0,
                    allow_negative=allow_negative,
                )
            )

    def balance(self, account_id: str) -> int:
        with Session(self._engine) as session:
            account = session.get(Account, account_id)
            if account is None:
                raise AccountNotFound(account_id)
            return account.balance

    def post(
        self,
        *,
        idempotency_key: str,
        operation: str,
        postings: Mapping[str, int],
    ) -> LedgerReceipt:
        canonical_postings = self._validate(
            idempotency_key=idempotency_key,
            operation=operation,
            postings=postings,
        )
        request_hash = self._request_hash(operation, canonical_postings)

        try:
            with Session(self._engine) as session, session.begin():
                existing = session.scalar(
                    select(LedgerTransaction).where(
                        LedgerTransaction.idempotency_key == idempotency_key
                    )
                )
                if existing is not None:
                    return self._existing_receipt(existing, request_hash)

                transaction = LedgerTransaction(
                    idempotency_key=idempotency_key,
                    operation=operation,
                    request_hash=request_hash,
                )
                session.add(transaction)
                session.flush()

                for account_id, amount in canonical_postings:
                    result = session.execute(
                        update(Account)
                        .where(Account.account_id == account_id)
                        .where(
                            (Account.allow_negative.is_(True))
                            | (Account.balance + amount >= 0)
                        )
                        .values(balance=Account.balance + amount)
                    )
                    if result.rowcount != 1:
                        account = session.get(Account, account_id)
                        if account is None:
                            raise AccountNotFound(account_id)
                        raise InsufficientCredits(account_id)
                    session.add(
                        Posting(
                            transaction_id=transaction.transaction_id,
                            account_id=account_id,
                            amount=amount,
                        )
                    )

                session.flush()
                return LedgerReceipt(
                    transaction_id=transaction.transaction_id,
                    idempotency_key=idempotency_key,
                    operation=operation,
                )
        except IntegrityError:
            existing = self._find_transaction(idempotency_key)
            if existing is None:
                raise
            return self._existing_receipt(existing, request_hash)

    @staticmethod
    def _validate(
        *,
        idempotency_key: str,
        operation: str,
        postings: Mapping[str, int],
    ) -> tuple[tuple[str, int], ...]:
        if not idempotency_key.strip() or not operation.strip():
            raise InvalidPosting("Idempotency key and operation are required.")
        if len(idempotency_key) > 200:
            raise InvalidPosting("Idempotency key exceeds 200 characters.")
        if len(operation) > 80:
            raise InvalidPosting("Operation exceeds 80 characters.")
        if len(postings) < 2:
            raise InvalidPosting("A ledger transaction needs at least two accounts.")

        canonical: list[tuple[str, int]] = []
        for account_id, amount in postings.items():
            if not account_id.strip() or len(account_id) > 160:
                raise InvalidPosting("Account ID is empty or too long.")
            if isinstance(amount, bool) or not isinstance(amount, int) or amount == 0:
                raise InvalidPosting("Posting amounts must be non-zero integers.")
            canonical.append((account_id, amount))

        canonical.sort()
        if sum(amount for _, amount in canonical) != 0:
            raise InvalidPosting("Ledger postings must sum to zero.")
        return tuple(canonical)

    @staticmethod
    def _request_hash(operation: str, postings: tuple[tuple[str, int], ...]) -> str:
        serialized = json.dumps(
            {"operation": operation, "postings": postings},
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _find_transaction(self, idempotency_key: str) -> LedgerTransaction | None:
        with Session(self._engine) as session:
            return session.scalar(
                select(LedgerTransaction).where(
                    LedgerTransaction.idempotency_key == idempotency_key
                )
            )

    @staticmethod
    def _existing_receipt(
        transaction: LedgerTransaction,
        request_hash: str,
    ) -> LedgerReceipt:
        if transaction.request_hash != request_hash:
            raise IdempotencyConflict(
                f"Idempotency key already used: {transaction.idempotency_key}"
            )
        return LedgerReceipt(
            transaction_id=transaction.transaction_id,
            idempotency_key=transaction.idempotency_key,
            operation=transaction.operation,
        )
