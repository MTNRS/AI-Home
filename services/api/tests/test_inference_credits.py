from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from ai_home.database import Base
from ai_home.credits import Account, CreditLedger, InsufficientCredits, IdempotencyConflict, InvalidPosting
from ai_home.inference_credits import InferenceFunding, reserve_inference, resolve_inference


@pytest.fixture
def engine(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "credits.db"))
    Base.metadata.create_all(engine)
    ledger = CreditLedger(engine)
    ledger.create_account("issuance", "test", allow_negative=True)
    ledger.create_account("user:alice", "user")
    ledger.create_account("user:bob", "user")
    ledger.post(idempotency_key="fund", operation="test", postings={"issuance": -100, "user:alice": 100})
    yield engine
    engine.dispose()


def test_reserve_settle_retry_ownership_and_conservation(engine):
    with Session(engine) as session, session.begin():
        funding, created = reserve_inference(session, "alice", "request", "a" * 64, 7)
        key = funding.request_id
        assert created
    with Session(engine) as session, session.begin():
        assert not reserve_inference(session, "alice", "request", "a" * 64, 7)[1]
        with pytest.raises(IdempotencyConflict):
            reserve_inference(session, "alice", "request", "b" * 64, 7)
        with pytest.raises(InvalidPosting):
            resolve_inference(session, key, "bob")
        resolve_inference(session, key, "alice")
        resolve_inference(session, key, "alice")
    ledger = CreditLedger(engine)
    assert ledger.balance("user:alice") == 93
    assert ledger.balance("service:foundation-inference") == 7
    with Session(engine) as session, session.begin():
        with pytest.raises(InvalidPosting):
            resolve_inference(session, key, "alice", refund=True)
        assert session.scalar(select(func.sum(Account.balance))) == 0


def test_refund_and_insufficient_reservation_rollback(engine):
    with pytest.raises(InsufficientCredits):
        with Session(engine) as session, session.begin():
            reserve_inference(session, "bob", "poor", "a" * 64, 7)
    with Session(engine) as session, session.begin():
        assert session.scalar(select(func.count()).select_from(InferenceFunding)) == 0
        funding, _ = reserve_inference(session, "alice", "refund", "a" * 64, 100)
        key = funding.request_id
    with Session(engine) as session, session.begin():
        resolve_inference(session, key, "alice", refund=True)
        resolve_inference(session, key, "alice", refund=True)
    assert CreditLedger(engine).balance("user:alice") == 100


def test_resolution_failure_rolls_back_status_and_postings(engine, monkeypatch):
    with Session(engine) as session, session.begin():
        funding, _ = reserve_inference(session, "alice", "rollback", "a" * 64, 7)
        key = funding.request_id
    def fail(*args, **kwargs):
        raise RuntimeError("injected ledger failure")
    with monkeypatch.context() as patch:
        patch.setattr(CreditLedger, "post_in_session", fail)
        with pytest.raises(RuntimeError):
            with Session(engine) as session, session.begin():
                resolve_inference(session, key, "alice")
    with Session(engine) as session, session.begin():
        assert session.get(InferenceFunding, key).status == "reserved"
        resolve_inference(session, key, "alice", refund=True)
    assert CreditLedger(engine).balance("user:alice") == 100


def test_concurrent_duplicate_reservation_and_settlement(engine):
    def reserve(_):
        with Session(engine) as session, session.begin():
            row, created = reserve_inference(session, "alice", "same", "a" * 64, 7)
            return row.request_id, created
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, range(2)))
    assert sum(created for _, created in results) == 1
    assert results[0][0] == results[1][0]
    def settle(_):
        with Session(engine) as session, session.begin():
            resolve_inference(session, results[0][0], "alice")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(settle, range(2)))
    assert CreditLedger(engine).balance("service:foundation-inference") == 7
