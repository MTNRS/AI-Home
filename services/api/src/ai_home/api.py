"""HTTP API for initial identity and account-isolation flows."""

from collections.abc import Generator

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ai_home.auth import (
    AuthenticatedUser,
    InvalidCredentials,
    InvalidSession,
    authenticate_user,
    create_user,
    issue_session,
    resolve_session,
    revoke_session,
)
from ai_home.config import Settings
from ai_home.database import build_engine
from ai_home.models import User
from ai_home.schemas import (
    LoginRequest,
    RegisterRequest,
    SessionResponse,
    UserResponse,
)

_BEARER = HTTPBearer(auto_error=False)


def get_db(request: Request) -> Generator[Session, None, None]:
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as session:
        yield session


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER),
    session: Session = Depends(get_db),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return resolve_session(session, credentials.credentials)
    except InvalidSession:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
) -> FastAPI:
    application_settings = settings or Settings()
    database_engine = engine or build_engine(application_settings.database_url)
    session_factory = sessionmaker(
        bind=database_engine,
        autoflush=False,
        expire_on_commit=False,
    )

    app = FastAPI(title="AI@Home API", version="0.1.0")
    app.state.settings = application_settings
    app.state.engine = database_engine
    app.state.session_factory = session_factory

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/api/v1/auth/register",
        response_model=UserResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def register(
        payload: RegisterRequest,
        session: Session = Depends(get_db),
    ) -> User:
        try:
            with session.begin():
                user = create_user(
                    session,
                    email=str(payload.email),
                    password=payload.password.get_secret_value(),
                )
        except IntegrityError as error:
            if not _is_email_unique_violation(error):
                raise
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            ) from None
        return user

    @app.post("/api/v1/auth/login", response_model=SessionResponse)
    def login(
        payload: LoginRequest,
        response: Response,
        session: Session = Depends(get_db),
    ) -> SessionResponse:
        try:
            with session.begin():
                user = authenticate_user(
                    session,
                    email=str(payload.email),
                    password=payload.password.get_secret_value(),
                )
                issued = issue_session(
                    session,
                    user=user,
                    lifetime_seconds=application_settings.session_lifetime_seconds,
                )
        except InvalidCredentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        response.headers["Cache-Control"] = "no-store"
        return SessionResponse(
            access_token=issued.access_token,
            expires_at_epoch=issued.expires_at_epoch,
        )

    @app.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(
        credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER),
        _current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> Response:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            revoke_session(session, credentials.credentials)
            session.commit()
        except InvalidSession:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/v1/users/me", response_model=UserResponse)
    def get_me(
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> User:
        user = session.get(User, current_user.user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    return app


app = create_app()


def _is_email_unique_violation(error: IntegrityError) -> bool:
    diagnostic = getattr(error.orig, "diag", None)
    if getattr(diagnostic, "constraint_name", None) == "uq_users_email":
        return True
    return "UNIQUE constraint failed: users.email" in str(error.orig)
