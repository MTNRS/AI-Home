"""HTTP API for initial identity and account-isolation flows."""

from collections.abc import Generator
import logging

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
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
    verify_password,
)
from ai_home.config import Settings
from ai_home.database import build_engine
from ai_home.jobs import (
    ClaimedJob,
    IdempotencyConflict as JobIdempotencyConflict,
    InvalidJobResult,
    JobNotFound,
    LeaseExpired,
    NodeNotFound,
    authenticate_node,
    claim_job,
    create_math_sum_job,
    find_existing_math_sum_job,
    heartbeat_node,
    is_job_idempotency_violation,
    list_owner_jobs,
    register_node,
    revoke_node,
    submit_job_result,
)
from ai_home.models import AuthSession, ComputeJob, ComputeNode, User
from ai_home.schemas import (
    DeleteAccountRequest,
    ClaimedJobResponse,
    CreateMathSumJobRequest,
    JobResponse,
    LoginRequest,
    NodeCredentialResponse,
    NodeResponse,
    RegisterNodeRequest,
    RegisterRequest,
    SessionResponse,
    SubmitJobResultRequest,
    UserResponse,
)

_BEARER = HTTPBearer(auto_error=False)
_LOGGER = logging.getLogger(__name__)


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


def get_current_node(
    credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER),
    session: Session = Depends(get_db),
) -> ComputeNode:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Node authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return authenticate_node(session, credentials.credentials)
    except NodeNotFound:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked node credential.",
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

    @app.middleware("http")
    async def disable_private_api_caching(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/v1/"):
            response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def readiness(session: Session = Depends(get_db)) -> dict[str, str]:
        try:
            session.execute(text("SELECT 1"))
        except SQLAlchemyError:
            _LOGGER.exception("Database readiness check failed.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database unavailable.",
            ) from None
        return {"status": "ready", "database": "ok"}

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

    @app.get("/api/v1/users/me/export", response_model=UserResponse)
    def export_me(
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

    @app.delete("/api/v1/users/me", status_code=status.HTTP_204_NO_CONTENT)
    def delete_me(
        payload: DeleteAccountRequest,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> Response:
        user = session.get(User, current_user.user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not verify_password(
            user.password_hash,
            payload.password.get_secret_value(),
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Password confirmation failed.",
            )
        session.execute(
            delete(AuthSession).where(AuthSession.user_id == current_user.user_id)
        )
        session.execute(
            update(ComputeJob)
            .where(
                ComputeJob.assigned_node_id.in_(
                    select(ComputeNode.node_id).where(
                        ComputeNode.owner_user_id == current_user.user_id
                    )
                ),
                ComputeJob.status == "leased",
            )
            .values(
                status="queued",
                assigned_node_id=None,
                lease_token_hash=None,
                lease_expires_epoch=None,
            )
        )
        session.execute(
            delete(ComputeJob).where(ComputeJob.owner_user_id == current_user.user_id)
        )
        session.execute(
            delete(ComputeNode).where(ComputeNode.owner_user_id == current_user.user_id)
        )
        session.delete(user)
        session.commit()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post(
        "/api/v1/nodes",
        response_model=NodeCredentialResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_node(
        payload: RegisterNodeRequest,
        response: Response,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> NodeCredentialResponse:
        registered = register_node(
            session,
            owner_user_id=current_user.user_id,
            name=payload.name,
            hardware=payload.hardware,
        )
        session.commit()
        response.headers["Cache-Control"] = "no-store"
        return NodeCredentialResponse(
            node_id=registered.node.node_id,
            node_token=registered.node_token,
            supported_job_types=registered.node.supported_job_types,
        )

    @app.get("/api/v1/nodes", response_model=list[NodeResponse])
    def get_nodes(
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> list[ComputeNode]:
        return list(
            session.scalars(
                select(ComputeNode)
                .where(
                    ComputeNode.owner_user_id == current_user.user_id,
                    ComputeNode.revoked_at.is_(None),
                )
                .order_by(ComputeNode.created_at.desc())
                .limit(100)
            )
        )

    @app.delete("/api/v1/nodes/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_node(
        node_id: str,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> Response:
        node = session.get(ComputeNode, node_id)
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Node not found.",
            )
        try:
            revoke_node(session, node, current_user.user_id)
            session.commit()
        except NodeNotFound:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Node not found.",
            ) from None
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/v1/node/heartbeat")
    def node_heartbeat(
        current_node: ComputeNode = Depends(get_current_node),
        session: Session = Depends(get_db),
    ) -> dict[str, int]:
        last_heartbeat = heartbeat_node(session, current_node)
        session.commit()
        return {"last_heartbeat_epoch": last_heartbeat}

    @app.post(
        "/api/v1/jobs",
        response_model=JobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_job(
        payload: CreateMathSumJobRequest,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> ComputeJob:
        try:
            job = create_math_sum_job(
                session,
                owner_user_id=current_user.user_id,
                idempotency_key=payload.idempotency_key,
                values=payload.values,
                allow_public_node_processing=payload.allow_public_node_processing,
            )
            session.commit()
        except JobIdempotencyConflict:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Idempotency key already used for a different job.",
            ) from None
        except IntegrityError as error:
            session.rollback()
            if not is_job_idempotency_violation(error):
                raise
            try:
                job = find_existing_math_sum_job(
                    session,
                    owner_user_id=current_user.user_id,
                    idempotency_key=payload.idempotency_key,
                    values=payload.values,
                )
            except JobIdempotencyConflict:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Idempotency key already used for a different job.",
                ) from None
            if job is None:
                raise
        return job

    @app.get("/api/v1/jobs", response_model=list[JobResponse])
    def get_jobs(
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> list[ComputeJob]:
        return list_owner_jobs(session, current_user.user_id)

    @app.post(
        "/api/v1/node/jobs/claim",
        response_model=ClaimedJobResponse,
    )
    def get_node_job(
        current_node: ComputeNode = Depends(get_current_node),
        session: Session = Depends(get_db),
    ) -> ClaimedJobResponse | Response:
        heartbeat_node(session, current_node)
        try:
            claim = claim_job(session, current_node)
        except NodeNotFound:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Node is inactive or revoked.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        if claim is None:
            session.commit()
            return Response(
                status_code=status.HTTP_204_NO_CONTENT,
                headers={"Cache-Control": "private, no-store"},
            )
        session.commit()
        return _claimed_job_response(claim)

    @app.post(
        "/api/v1/node/jobs/{job_id}/result",
        response_model=JobResponse,
    )
    def submit_node_job_result(
        job_id: str,
        payload: SubmitJobResultRequest,
        current_node: ComputeNode = Depends(get_current_node),
        session: Session = Depends(get_db),
    ) -> ComputeJob:
        try:
            job = submit_job_result(
                session,
                node=current_node,
                job_id=job_id,
                lease_token=payload.lease_token.get_secret_value(),
                result=payload.result,
            )
            session.commit()
        except JobNotFound:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job or lease not found.",
            ) from None
        except LeaseExpired:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job lease expired.",
            ) from None
        except InvalidJobResult:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Job result failed validation.",
            ) from None
        return job

    return app


app = create_app()


def _is_email_unique_violation(error: IntegrityError) -> bool:
    diagnostic = getattr(error.orig, "diag", None)
    if getattr(diagnostic, "constraint_name", None) == "uq_users_email":
        return True
    return "UNIQUE constraint failed: users.email" in str(error.orig)


def _claimed_job_response(claim: ClaimedJob) -> ClaimedJobResponse:
    return ClaimedJobResponse(
        job_id=claim.job.job_id,
        job_type="math.sum.v1",
        payload=claim.job.payload,
        lease_token=claim.lease_token,
        lease_expires_epoch=claim.job.lease_expires_epoch or 0,
    )
