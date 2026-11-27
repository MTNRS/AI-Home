import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from ai_home.credits import (
    AccountNotFound,
    CreditLedger,
    IdempotencyConflict,
    InsufficientCredits,
    InvalidPosting,
)


@pytest.fixture
def ledger():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    service = CreditLedger(engine)
    service.initialize()
    service.create_account("system:issuance", "system", allow_negative=True)
    service.create_account("user:alice:available", "user_available")
    service.create_account("user:alice:reserved", "user_reserved")
    service.create_account("node:one:available", "node_available")
    try:
        yield service
    finally:
        engine.dispose()


def issue(ledger: CreditLedger, amount: int = 100) -> None:
    ledger.post(
        idempotency_key="issue:alice:initial",
        operation="issue",
        postings={
            "system:issuance": -amount,
            "user:alice:available": amount,
        },
    )


def test_reserve_settle_and_refund_preserve_balances(ledger: CreditLedger):
    issue(ledger)

    ledger.post(
        idempotency_key="reserve:job-1",
        operation="reserve",
        postings={
            "user:alice:available": -60,
            "user:alice:reserved": 60,
        },
    )
    ledger.post(
        idempotency_key="settle:job-1",
        operation="settle",
        postings={
            "user:alice:reserved": -40,
            "node:one:available": 40,
        },
    )
    ledger.post(
        idempotency_key="refund:job-1",
        operation="refund",
        postings={
            "user:alice:reserved": -20,
            "user:alice:available": 20,
        },
    )

    assert ledger.balance("user:alice:available") == 60
    assert ledger.balance("user:alice:reserved") == 0
    assert ledger.balance("node:one:available") == 40
    assert ledger.balance("system:issuance") == -100


def test_same_idempotency_key_does_not_apply_transaction_twice(ledger: CreditLedger):
    issue(ledger)
    postings = {
        "user:alice:available": -25,
        "user:alice:reserved": 25,
    }

    first = ledger.post(
        idempotency_key="reserve:job-repeat",
        operation="reserve",
        postings=postings,
    )
    repeated = ledger.post(
        idempotency_key="reserve:job-repeat",
        operation="reserve",
        postings=postings,
    )

    assert repeated == first
    assert ledger.balance("user:alice:available") == 75
    assert ledger.balance("user:alice:reserved") == 25


def test_idempotency_key_cannot_be_reused_for_different_postings(ledger: CreditLedger):
    issue(ledger)
    ledger.post(
        idempotency_key="reserve:job-conflict",
        operation="reserve",
        postings={
            "user:alice:available": -20,
            "user:alice:reserved": 20,
        },
    )

    with pytest.raises(IdempotencyConflict):
        ledger.post(
            idempotency_key="reserve:job-conflict",
            operation="reserve",
            postings={
                "user:alice:available": -30,
                "user:alice:reserved": 30,
            },
        )

    assert ledger.balance("user:alice:available") == 80
    assert ledger.balance("user:alice:reserved") == 20


def test_failed_debit_rolls_back_all_postings(ledger: CreditLedger):
    issue(ledger)
    with pytest.raises(InsufficientCredits):
        ledger.post(
            idempotency_key="reserve:too-large",
            operation="reserve",
            postings={
                "node:one:available": 20,
                "user:alice:available": -10,
                "user:alice:reserved": -10,
            },
        )

    assert ledger.balance("node:one:available") == 0
    assert ledger.balance("user:alice:available") == 100
    assert ledger.balance("user:alice:reserved") == 0


def test_unbalanced_transaction_is_rejected(ledger: CreditLedger):
    with pytest.raises(InvalidPosting):
        ledger.post(
            idempotency_key="invalid:unbalanced",
            operation="transfer",
            postings={
                "user:alice:available": -10,
                "node:one:available": 9,
            },
        )


def test_unknown_account_rolls_back_transaction(ledger: CreditLedger):
    issue(ledger)
    with pytest.raises(AccountNotFound):
        ledger.post(
            idempotency_key="reserve:unknown",
            operation="reserve",
            postings={
                "user:alice:available": -10,
                "z:missing:account": 10,
            },
        )
    assert ledger.balance("user:alice:available") == 100

