"""Password verification and opaque bearer-session management."""

import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_home.models import AuthSession, User

_PASSWORD_HASHER = PasswordHasher()
_DUMMY_PASSWORD_HASH = _PASSWORD_HASHER.hash(secrets.token_urlsafe(32))


class InvalidCredentials(Exception):
    """Raised when an email/password pair is not valid."""


class InvalidSession(Exception):
    """Raised when a bearer session is unknown, revoked, or expired."""


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str


@dataclass(frozen=True)
class IssuedSession:
    access_token: str
    expires_at_epoch: int


def normalize_email(email: str) -> str:
    return email.strip().casefold()


def create_user(session: Session, *, email: str, password: str) -> User:
    user = User(
        user_id=str(uuid4()),
        email=normalize_email(email),
        password_hash=_PASSWORD_HASHER.hash(password),
    )
    session.add(user)
    session.flush()
    return user


def authenticate_user(session: Session, *, email: str, password: str) -> User:
    user = session.scalar(select(User).where(User.email == normalize_email(email)))
    valid_password = verify_password(
        user.password_hash if user is not None else _DUMMY_PASSWORD_HASH,
        password,
    )
    if user is None or not valid_password:
        raise InvalidCredentials
    return user


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (InvalidHashError, VerificationError, VerifyMismatchError):
        return False


def issue_session(
    session: Session,
    *,
    user: User,
    lifetime_seconds: int,
) -> IssuedSession:
    access_token = secrets.token_urlsafe(32)
    expires_at_epoch = int(time.time()) + lifetime_seconds
    session.add(
        AuthSession(
            session_id=str(uuid4()),
            user_id=user.user_id,
            token_hash=_token_hash(access_token),
            expires_at_epoch=expires_at_epoch,
        )
    )
    session.flush()
    return IssuedSession(access_token, expires_at_epoch)


def resolve_session(session: Session, access_token: str) -> AuthenticatedUser:
    token_digest = _token_hash(access_token)
    auth_session = session.scalar(
        select(AuthSession).where(
            AuthSession.token_hash == token_digest,
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at_epoch > int(time.time()),
        )
    )
    if auth_session is None:
        raise InvalidSession

    user = session.get(User, auth_session.user_id)
    if user is None:
        raise InvalidSession
    return AuthenticatedUser(user_id=user.user_id, email=user.email)


def revoke_session(session: Session, access_token: str) -> None:
    auth_session = session.scalar(
        select(AuthSession).where(
            AuthSession.token_hash == _token_hash(access_token),
            AuthSession.revoked_at.is_(None),
        )
    )
    if auth_session is None:
        raise InvalidSession
    auth_session.revoked_at = datetime.now(timezone.utc)
    session.flush()


def _token_hash(access_token: str) -> str:
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()
