"""Operator-only internal credit grants; requires direct database access."""
import argparse

from sqlalchemy.orm import Session

from ai_home.config import Settings
from ai_home.credits import Account, CreditLedger, InvalidPosting
from ai_home.database import build_engine
from ai_home.job_credits import ensure_account, user_account_id
from ai_home.models import User


def grant_credits(session, *, user_id, amount, idempotency_key, reason):
    if type(amount) is not int or not 1 <= amount <= 1_000_000:
        raise ValueError("Grant amount must be between 1 and 1000000")
    if not reason.strip() or len(reason) > 60 or len(idempotency_key) > 160 or not idempotency_key.strip():
        raise ValueError("A short audit reason and idempotency key are required")
    if session.get(User, user_id) is None:
        raise ValueError("User does not exist")
    system = session.get(Account, "system:issuance")
    if system is None:
        system = Account(account_id="system:issuance", account_type="issuance", balance=0, allow_negative=True)
        session.add(system)
    elif not system.allow_negative or system.account_type != "issuance":
        raise InvalidPosting("Unexpected issuance account configuration")
    target = user_account_id(user_id)
    ensure_account(session, target, "user")
    return CreditLedger.post_in_session(session, idempotency_key=f"operator-grant:{idempotency_key}",
        operation=f"operator_grant:{reason.strip()}", postings={"system:issuance": -amount, target: amount})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--amount", required=True, type=int)
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    engine = build_engine(Settings().database_url)
    try:
        with Session(engine) as session, session.begin():
            receipt = grant_credits(session, user_id=args.user_id, amount=args.amount,
                idempotency_key=args.idempotency_key, reason=args.reason)
        print(f"Internal credit grant recorded as transaction {receipt.transaction_id}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
