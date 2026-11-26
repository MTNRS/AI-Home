"""HTTP API for initial identity and account-isolation flows."""

from collections.abc import Generator
import hashlib
import hmac
import json
import logging
import time
import threading
from typing import Literal
import uuid

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ai_home.auth import (
    AuthenticatedUser,
    InvalidCredentials,
    InvalidSession,
    authenticate_user,
    create_user,
    create_supabase_user,
    issue_session,
    issue_supabase_session,
    resolve_session,
    revoke_session,
    verify_password,
)
from ai_home.config import Settings
from ai_home.credits import Account, InsufficientCredits, LedgerTransaction, Posting, IdempotencyConflict
from ai_home.inference_credits import InferenceFunding, reserve_inference, resolve_inference, refund_expired_inference, refund_pending_inference
from ai_home.job_credits import JobFunding, refund_job, user_account_id
from ai_home.database import build_engine
from ai_home.inference import chat_completion
from ai_home.jobs import (
    ClaimedJob,
    IdempotencyConflict as JobIdempotencyConflict,
    InvalidJobResult,
    JobNotFound,
    LeaseExpired,
    NodeNotFound,
    authenticate_node,
    claim_job,
    cancel_job,
    create_math_sum_job,
    create_tokenize_job,
    create_gradient_job,
    find_existing_math_sum_job,
    heartbeat_node,
    is_job_idempotency_violation,
    list_owner_jobs,
    register_node,
    revoke_node,
    submit_job_result,
)
from ai_home.models import (
    AuthSession,
    ComputeJob,
    ComputeNode,
    TrainingContribution,
    TrainingDataset,
    User,
)
from ai_home.privacy import contains_common_secret
from ai_home.schemas import (
    DeleteAccountRequest,
    ClaimedJobResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FoundationGeneration,
    CreateMathSumJobRequest,
    CreateTokenizeJobRequest,
    CreateGradientJobRequest,
    JobResponse,
    LoginRequest,
    NodeCredentialResponse,
    NodeResponse,
    NodeResources,
    RegisterNodeRequest,
    RegisterRequest,
    SessionResponse,
    SubmitJobResultRequest,
    TrainingContributionExportResponse,
    TrainingContributionReviewResponse,
    TrainingContributionRequest,
    TrainingContributionResponse,
    TrainingDatasetExportResponse,
    TrainingDatasetResponse,
    TrainingReviewRequest,
    TrainingReviewResponse,
    FreezeTrainingDatasetRequest,
    UserResponse,
)
from ai_home.supabase_auth import (
    InvalidSupabaseToken,
    SupabaseAuth,
    SupabaseAuthError,
    SupabaseAuthUnavailable,
)

_BEARER = HTTPBearer(auto_error=False)
_LOGGER = logging.getLogger(__name__)


def get_db(request: Request) -> Generator[Session, None, None]:
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as session:
        yield session


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER),
    session: Session = Depends(get_db),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    supabase_auth: SupabaseAuth | None = request.app.state.supabase_auth
    if supabase_auth is not None:
        try:
            local_session = resolve_session(session, credentials.credentials)
            identity = supabase_auth.verify_access_token(credentials.credentials)
        except (InvalidSession, InvalidSupabaseToken):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        except SupabaseAuthUnavailable:
            _LOGGER.exception("Supabase signing-key verification failed.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Supabase authentication is temporarily unavailable.",
            ) from None
        if local_session.user_id != identity.user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = session.get(User, identity.user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return AuthenticatedUser(user_id=identity.user_id, email=identity.email)
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


def require_training_reviewer(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER),
) -> None:
    settings: Settings = request.app.state.settings
    expected_token = settings.training_review_token.get_secret_value()
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training review is not configured for this server.",
        )
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(credentials.credentials, expected_token)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Training reviewer authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    supabase_auth: SupabaseAuth | None = None,
) -> FastAPI:
    application_settings = settings or Settings()
    database_engine = engine or build_engine(application_settings.database_url)
    session_factory = sessionmaker(
        bind=database_engine,
        autoflush=False,
        expire_on_commit=False,
    )

    app = FastAPI(title="AI@Home API", version="0.1.0")
    foundation_inference_lock = threading.Lock()
    app.state.settings = application_settings
    app.state.engine = database_engine
    app.state.session_factory = session_factory
    if supabase_auth is not None:
        app.state.supabase_auth = supabase_auth
    elif application_settings.supabase_url is None:
        app.state.supabase_auth = None
    elif application_settings.supabase_jwks_url is None:
        raise RuntimeError("The configured Supabase project has no JWKS URL.")
    else:
        app.state.supabase_auth = SupabaseAuth(
            project_url=application_settings.supabase_url,
            publishable_key=(
                application_settings.supabase_publishable_key.get_secret_value()
            ),
            secret_key=application_settings.supabase_secret_key.get_secret_value(),
            jwks_url=application_settings.supabase_jwks_url,
        )

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
        supabase_client: SupabaseAuth | None = app.state.supabase_auth
        if supabase_client is not None:
            try:
                identity = supabase_client.sign_up(
                    email=str(payload.email),
                    password=payload.password.get_secret_value(),
                )
                with session.begin():
                    user = create_supabase_user(
                        session,
                        user_id=identity.user_id,
                        email=identity.email,
                    )
            except SupabaseAuthError as error:
                if error.status_code in {400, 409, 422} and any(
                    marker in error.code
                    for marker in (
                        "already registered",
                        "already exists",
                        "user_already",
                        "email_exists",
                    )
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="An account with this email already exists.",
                    ) from None
                _LOGGER.warning("Supabase rejected account registration.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Supabase could not create the account.",
                ) from None
            except SupabaseAuthUnavailable:
                _LOGGER.exception("Supabase account registration failed.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Supabase Auth is temporarily unavailable.",
                ) from None
            except IntegrityError:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="An account with this email already exists.",
                ) from None
            return user
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
        supabase_client: SupabaseAuth | None = app.state.supabase_auth
        if supabase_client is not None:
            try:
                issued = supabase_client.sign_in(
                    email=str(payload.email),
                    password=payload.password.get_secret_value(),
                )
                with session.begin():
                    user = create_supabase_user(
                        session,
                        user_id=issued.identity.user_id,
                        email=issued.identity.email,
                    )
                    issue_supabase_session(
                        session,
                        user=user,
                        access_token=issued.access_token,
                        expires_at_epoch=issued.expires_at_epoch,
                    )
            except SupabaseAuthError as error:
                if error.status_code in {400, 401, 403}:
                    if (
                        "email_not_confirmed" in error.code
                        or "email not confirmed" in error.code
                    ):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=(
                                "Confirma tu correo electrónico antes de iniciar sesión."
                            ),
                        ) from None
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid email or password.",
                        headers={"WWW-Authenticate": "Bearer"},
                    ) from None
                _LOGGER.warning("Supabase rejected account login.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Supabase Auth is temporarily unavailable.",
                ) from None
            except SupabaseAuthUnavailable:
                _LOGGER.exception("Supabase account login failed.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Supabase Auth is temporarily unavailable.",
                ) from None
            except IntegrityError:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The Supabase account conflicts with a local profile.",
                ) from None
            response.headers["Cache-Control"] = "no-store"
            return SessionResponse(
                user_id=user.user_id,
                access_token=issued.access_token,
                expires_at_epoch=issued.expires_at_epoch,
            )
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
            user_id=user.user_id,
            access_token=issued.access_token,
            expires_at_epoch=issued.expires_at_epoch,
        )

    @app.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(
        request: Request,
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
        supabase_client: SupabaseAuth | None = request.app.state.supabase_auth
        if supabase_client is not None:
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
            try:
                supabase_client.sign_out(credentials.credentials)
            except SupabaseAuthError:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Could not end the Supabase session.",
                ) from None
            except SupabaseAuthUnavailable:
                _LOGGER.exception("Supabase logout failed.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Supabase Auth is temporarily unavailable.",
                ) from None
            return Response(status_code=status.HTTP_204_NO_CONTENT)
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

    @app.get("/api/v1/models")
    def list_approved_models(request: Request, _user: AuthenticatedUser = Depends(get_current_user)):
        settings = request.app.state.settings
        if settings.chat_backend != "foundation" or settings.foundation_registry is None:
            return {"models": []}
        try:
            from ai_home.foundation.registry import ModelRegistry
            return {"models": ModelRegistry(settings.foundation_registry).catalog()}
        except Exception:
            raise HTTPException(status_code=503, detail="Model catalog unavailable.") from None

    def generation_options(payload, settings):
        options = (payload.generation or FoundationGeneration()).model_dump()
        maximum = options["max_new_tokens"]
        if maximum is not None and maximum > settings.foundation_max_new_tokens:
            raise HTTPException(status_code=422, detail="Requested generation exceeds the server token limit.")
        options["max_new_tokens"] = maximum if maximum is not None else settings.foundation_max_new_tokens
        return options

    @app.get("/api/v1/credits/inference-price")
    def inference_price(request: Request, _user: AuthenticatedUser = Depends(get_current_user)):
        price = request.app.state.settings.foundation_inference_price
        return {"credits_per_completion": price, "paid_streaming_available": True,
                "billing_unit": "successfully_computed_completion"}

    def reserve_chat_credit(payload, request, user):
        price = request.app.state.settings.foundation_inference_price
        if not price:
            return None
        if payload.credit_request_id is None or payload.credit_limit < price:
            raise HTTPException(status_code=402, detail="Explicit credit request ID and sufficient credit limit required.")
        with request.app.state.session_factory() as session, session.begin():
            refund_expired_inference(session, user.user_id)
        digest = hashlib.sha256(payload.model_dump_json().encode()).hexdigest()
        try:
            with request.app.state.session_factory() as session, session.begin():
                # A no-op write serializes reservation admission with account deletion
                # on SQLite as well as PostgreSQL; authentication alone can race deletion.
                admitted = session.execute(update(User).where(User.user_id == user.user_id).values(user_id=user.user_id).returning(User.user_id)).scalar_one_or_none()
                if admitted is None:
                    raise HTTPException(status_code=401, detail="Account no longer available.")
                funding, created = reserve_inference(session, user.user_id, str(payload.credit_request_id), digest, price,
                                                      lifetime_seconds=request.app.state.settings.foundation_reservation_seconds)
                if not created:
                    raise HTTPException(status_code=409, detail="Credit request already processed or in progress; inference was not repeated.")
                return funding.request_id
        except InsufficientCredits:
            raise HTTPException(status_code=402, detail="Insufficient compute credits for inference.") from None
        except IdempotencyConflict:
            raise HTTPException(status_code=409, detail="Credit request ID conflicts with an earlier request.") from None

    def finish_chat_credit(request, user, reservation, *, refund=False):
        if reservation is not None:
            with request.app.state.session_factory() as session, session.begin():
                resolve_inference(session, reservation, user.user_id, refund=refund)

    @app.post("/api/v1/chat/stream")
    def stream_foundation_chat(payload: ChatCompletionRequest, request: Request,
                               _user: AuthenticatedUser = Depends(get_current_user)):
        settings = request.app.state.settings
        if settings.chat_backend != "foundation":
            raise HTTPException(status_code=503, detail="Streaming requires the Foundation backend.")
        if settings.foundation_inference_price and (
                payload.credit_request_id is None or payload.credit_limit < settings.foundation_inference_price):
            raise HTTPException(status_code=402, detail="Explicit credit request ID and sufficient credit limit required.")
        if settings.foundation_checkpoint is None and settings.foundation_registry is None:
            raise HTTPException(status_code=503, detail="Foundation checkpoint is not configured.")
        if payload.model_version is not None and settings.foundation_registry is None:
            raise HTTPException(status_code=503, detail="Model selection requires a registry.")
        if payload.use_knowledge and settings.knowledge_index is None:
            raise HTTPException(status_code=503, detail="Knowledge retrieval is not configured.")
        options = generation_options(payload, settings)
        from ai_home.foundation.streaming import InferenceStream

        def work(cancel, emit):
            from ai_home.foundation.checkpoint import load_checkpoint
            from ai_home.foundation.inference import stream_text
            from ai_home.foundation.training import select_device
            checkpoint, version = settings.foundation_checkpoint, None
            if settings.foundation_registry is not None:
                if checkpoint is not None:
                    raise ValueError("Ambiguous checkpoint configuration")
                from ai_home.foundation.registry import ModelRegistry
                version, checkpoint = ModelRegistry(settings.foundation_registry).resolve(
                    str(payload.model_version) if payload.model_version else None)
            model, tokenizer, _ = load_checkpoint(checkpoint, select_device(settings.foundation_device),
                                                  max_bytes=settings.foundation_memory_budget_bytes)
            if cancel.is_set():
                return
            messages = [message.model_dump() for message in payload.messages]
            references = []
            if payload.use_knowledge:
                from ai_home.knowledge import prepare_knowledge_context
                messages, references = prepare_knowledge_context(messages, model, tokenizer, settings.knowledge_index)
            emit({"type": "metadata", "model_version": version, "references": references, "generation": options})
            started = time.perf_counter()
            chunks, characters, nonempty = 0, 0, False
            for content in stream_text(model, tokenizer, messages, cancel=cancel,
                                       **options):
                if cancel.is_set():
                    return
                chunks += 1
                characters += len(content)
                nonempty = nonempty or bool(content.strip())
                emit({"type": "delta", "content": content})
            if cancel.is_set():
                return
            if not nonempty:
                raise RuntimeError("No generated text")
            emit({"type": "done", "chunks": chunks, "characters": characters,
                  "generation_seconds": time.perf_counter() - started})

        def accounted_work(cancel, emit):
            reservation, settled = None, False
            if cancel.is_set():
                return
            try:
                reservation = reserve_chat_credit(payload, request, _user)
                def accounted_emit(event):
                    nonlocal settled
                    if event["type"] == "done":
                        if cancel.is_set():
                            return
                        finish_chat_credit(request, _user, reservation)
                        settled = True
                        event = {**event, "charged_credits": settings.foundation_inference_price,
                                 "credit_request_id": str(payload.credit_request_id) if payload.credit_request_id else None}
                    elif event["type"] == "metadata":
                        event = {**event, "reserved_credits": settings.foundation_inference_price}
                    emit(event)
                work(cancel, accounted_emit)
            except HTTPException as error:
                emit({"type": "error", "code": "insufficient_credits" if error.status_code == 402 else "credit_request_conflict"})
            finally:
                if reservation is not None and not settled:
                    finish_chat_credit(request, _user, reservation, refund=True)

        stream = InferenceStream(accounted_work, foundation_inference_lock)
        return StreamingResponse(stream.events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post(
        "/api/v1/chat/completions",
        response_model=ChatCompletionResponse,
    )
    def create_chat_completion(
        payload: ChatCompletionRequest,
        request: Request,
        current_user: AuthenticatedUser = Depends(get_current_user),
    ) -> ChatCompletionResponse:
        settings: Settings = request.app.state.settings
        if payload.generation is not None and settings.chat_backend != "foundation":
            raise HTTPException(status_code=422, detail="Generation controls require the Foundation backend.")
        if payload.model_version is not None and (settings.chat_backend != "foundation" or settings.foundation_registry is None):
            raise HTTPException(status_code=503, detail="Approved model selection requires a Foundation registry.")
        if payload.use_knowledge and (settings.chat_backend != "foundation" or settings.knowledge_index is None):
            raise HTTPException(status_code=503, detail="Foundation knowledge retrieval is not configured.")
        if settings.chat_backend == "foundation":
            if settings.foundation_checkpoint is None and settings.foundation_registry is None:
                raise HTTPException(status_code=503, detail="Foundation checkpoint is not configured.")
            options = generation_options(payload, settings)
            # One active local generation per process; no prompts or KV caches are shared.
            if not foundation_inference_lock.acquire(blocking=False):
                raise HTTPException(status_code=503, detail="Foundation inference is busy.",
                                    headers={"Retry-After": "1"})
            reservation = None
            try:
                reservation = reserve_chat_credit(payload, request, current_user)
                from ai_home.foundation.inference import complete

                selected_version = None
                checkpoint = settings.foundation_checkpoint
                if settings.foundation_registry is not None:
                    if checkpoint is not None:
                        raise ValueError("Configure a registry or a direct checkpoint, not both")
                    from ai_home.foundation.registry import ModelRegistry
                    selected_version, checkpoint = ModelRegistry(settings.foundation_registry).resolve(
                        str(payload.model_version) if payload.model_version is not None else None)
                references = []
                completion = complete
                extra = {}
                if payload.use_knowledge:
                    from ai_home.knowledge import complete_with_knowledge
                    completion = complete_with_knowledge
                    extra["index_path"] = settings.knowledge_index
                result = completion(
                    [message.model_dump() for message in payload.messages],
                    checkpoint,
                    device=settings.foundation_device,
                    max_bytes=settings.foundation_memory_budget_bytes,
                    **options,
                    **extra,
                )
                if payload.use_knowledge:
                    answer, references = result
                else:
                    answer = result
                finish_chat_credit(request, current_user, reservation)
            except HTTPException:
                finish_chat_credit(request, current_user, reservation, refund=True)
                raise
            except Exception:
                try:
                    finish_chat_credit(request, current_user, reservation, refund=True)
                except Exception:
                    logging.exception("Inference refund failed; durable reservation requires reconciliation")
                # Do not return filesystem paths, checkpoint contents or conversation data.
                raise HTTPException(status_code=503,
                                    detail="Foundation inference failed; verify the checkpoint and runtime.") from None
            finally:
                foundation_inference_lock.release()
            return ChatCompletionResponse(charged_credits=settings.foundation_inference_price,
                                          credit_request_id=payload.credit_request_id, generation=options, model_version=selected_version, references=references, choices=[{
                "index": 0, "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }])
        if settings.chat_provider_url is None or settings.chat_provider_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat inference is not configured for this server.",
            )
        try:
            answer = chat_completion(
                settings.chat_provider_url,
                settings.chat_provider_model,
                [message.model_dump() for message in payload.messages],
                api_key=settings.chat_provider_api_key.get_secret_value(),
                timeout_seconds=settings.chat_provider_timeout_seconds,
            )
        except (httpx.HTTPError, ValueError):
            _LOGGER.exception(
                "Configured chat provider request failed for user %s.",
                current_user.user_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Configured chat provider failed to complete the request.",
            ) from None
        return ChatCompletionResponse(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ]
        )

    @app.post(
        "/api/v1/training/conversations",
        response_model=TrainingContributionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_training_contribution(
        payload: TrainingContributionRequest,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> TrainingContribution:
        messages = [message.model_dump() for message in payload.messages]
        if _contains_secret(messages):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "The conversation appears to contain a credential or secret; "
                    "remove it before submitting."
                ),
            )
        contribution = TrainingContribution(
            contribution_id=str(uuid.uuid4()),
            owner_user_id=current_user.user_id,
            conversation_id=str(payload.conversation_id),
            provider=payload.provider,
            source_endpoint=payload.source_endpoint,
            model=payload.model.strip(),
            messages=messages,
            consent_version=payload.consent_version,
            consent_scope=payload.consent_scope,
            consent_at_epoch=int(time.time()),
            status="pending_review",
        )
        session.add(contribution)
        try:
            session.commit()
        except IntegrityError as error:
            session.rollback()
            if not _is_training_conversation_unique_violation(error):
                raise
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This conversation already has a training contribution.",
            ) from None
        return contribution

    @app.get(
        "/api/v1/training/conversations",
        response_model=list[TrainingContributionResponse],
    )
    def get_training_contributions(
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> list[TrainingContribution]:
        return list(
            session.scalars(
                select(TrainingContribution)
                .where(TrainingContribution.owner_user_id == current_user.user_id)
                .order_by(
                    TrainingContribution.consent_at_epoch.desc(),
                    TrainingContribution.contribution_id.desc(),
                )
                .offset(offset)
                .limit(limit)
            )
        )

    @app.get(
        "/api/v1/training/conversations/{contribution_id}",
        response_model=TrainingContributionExportResponse,
    )
    def get_training_contribution(
        contribution_id: str,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> TrainingContribution:
        contribution = session.scalar(
            select(TrainingContribution).where(
                TrainingContribution.contribution_id == contribution_id,
                TrainingContribution.owner_user_id == current_user.user_id,
            )
        )
        if contribution is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training contribution not found.",
            )
        return contribution

    @app.delete(
        "/api/v1/training/conversations/{contribution_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_training_contribution(
        contribution_id: str,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> Response:
        contribution = session.scalar(
            select(TrainingContribution).where(
                TrainingContribution.contribution_id == contribution_id,
                TrainingContribution.owner_user_id == current_user.user_id,
            ).with_for_update()
        )
        if contribution is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training contribution not found.",
            )
        if contribution.status == "frozen":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This contribution is already part of a frozen dataset; "
                    "its influence on published model weights cannot be undone."
                ),
            )
        session.delete(contribution)
        session.commit()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get(
        "/api/v1/admin/training/conversations",
        response_model=list[TrainingContributionReviewResponse],
    )
    def list_training_contributions_for_review(
        status_filter: Literal["pending_review", "eligible"] = Query(
            default="pending_review",
            alias="status",
        ),
        limit: int = Query(default=20, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
        _reviewer: None = Depends(require_training_reviewer),
        session: Session = Depends(get_db),
    ) -> list[TrainingContribution]:
        return list(
            session.scalars(
                select(TrainingContribution)
                .where(TrainingContribution.status == status_filter)
                .order_by(
                    TrainingContribution.consent_at_epoch,
                    TrainingContribution.contribution_id,
                )
                .offset(offset)
                .limit(limit)
            )
        )

    @app.post(
        "/api/v1/admin/training/conversations/{contribution_id}/review",
        response_model=TrainingReviewResponse,
    )
    def review_training_contribution(
        contribution_id: str,
        payload: TrainingReviewRequest,
        request: Request,
        _reviewer: None = Depends(require_training_reviewer),
        session: Session = Depends(get_db),
    ) -> TrainingReviewResponse:
        contribution = session.scalar(
            select(TrainingContribution)
            .where(TrainingContribution.contribution_id == contribution_id)
            .with_for_update()
        )
        if contribution is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training contribution not found.",
            )
        if contribution.status != "pending_review":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only pending training contributions can be reviewed.",
            )
        reviewed_at_epoch = int(time.time())
        reviewer_id = request.app.state.settings.training_reviewer_id.strip()
        contribution.reviewer_id = reviewer_id
        contribution.reviewed_at_epoch = reviewed_at_epoch
        review_checklist = {
            "provider_terms_verified": payload.provider_terms_verified,
            "sensitive_data_reviewed": payload.sensitive_data_reviewed,
            "reuse_rights_verified": payload.reuse_rights_verified,
        }
        contribution.review_checklist = review_checklist
        if payload.approve:
            contribution.status = "eligible"
        else:
            contribution.status = "rejected"
            contribution.messages = []
        session.commit()
        return TrainingReviewResponse(
            contribution_id=contribution.contribution_id,
            status=contribution.status,
            reviewer_id=reviewer_id,
            reviewed_at_epoch=reviewed_at_epoch,
            review_checklist=review_checklist,
        )

    @app.post(
        "/api/v1/admin/training/datasets",
        response_model=TrainingDatasetResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def freeze_training_dataset(
        payload: FreezeTrainingDatasetRequest,
        _reviewer: None = Depends(require_training_reviewer),
        session: Session = Depends(get_db),
    ) -> TrainingDataset:
        contribution_ids = sorted(str(value) for value in payload.contribution_ids)
        contributions = list(
            session.scalars(
                select(TrainingContribution)
                .where(TrainingContribution.contribution_id.in_(contribution_ids))
                .order_by(TrainingContribution.contribution_id)
                .with_for_update()
            )
        )
        if len(contributions) != len(contribution_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more training contributions were not found.",
            )
        if any(
            item.status != "eligible"
            or item.reviewer_id is None
            or item.reviewed_at_epoch is None
            or item.review_checklist is None
            or not all(item.review_checklist.values())
            for item in contributions
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Datasets may include only human-reviewed eligible contributions.",
            )

        records = [
            {
                "contribution_id": item.contribution_id,
                "provider": item.provider,
                "source_endpoint": item.source_endpoint,
                "model": item.model,
                "consent_version": item.consent_version,
                "consent_scope": item.consent_scope,
                "consent_at_epoch": item.consent_at_epoch,
                "reviewer_id": item.reviewer_id,
                "reviewed_at_epoch": item.reviewed_at_epoch,
                "review_checklist": item.review_checklist,
                "messages": item.messages,
            }
            for item in contributions
        ]
        manifest = {
            "dataset_version": payload.dataset_version,
            "records": records,
        }
        manifest_sha256 = hashlib.sha256(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        dataset = TrainingDataset(
            dataset_id=str(uuid.uuid4()),
            dataset_version=payload.dataset_version,
            manifest_sha256=manifest_sha256,
            item_count=len(records),
            records=records,
        )
        session.add(dataset)
        for contribution in contributions:
            contribution.status = "frozen"
        try:
            session.commit()
        except IntegrityError as error:
            session.rollback()
            if not _is_dataset_version_violation(error):
                raise
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A dataset with this version already exists.",
            ) from None
        return dataset

    @app.get(
        "/api/v1/admin/training/datasets",
        response_model=list[TrainingDatasetResponse],
    )
    def list_training_datasets(
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        _reviewer: None = Depends(require_training_reviewer),
        session: Session = Depends(get_db),
    ) -> list[TrainingDataset]:
        return list(
            session.scalars(
                select(TrainingDataset)
                .order_by(TrainingDataset.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
        )

    @app.get(
        "/api/v1/admin/training/datasets/{dataset_id}",
        response_model=TrainingDatasetExportResponse,
    )
    def get_training_dataset(
        dataset_id: str,
        _reviewer: None = Depends(require_training_reviewer),
        session: Session = Depends(get_db),
    ) -> TrainingDataset:
        dataset = session.get(TrainingDataset, dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training dataset not found.",
            )
        return dataset

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
        request: Request,
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
        supabase_client: SupabaseAuth | None = request.app.state.supabase_auth
        if supabase_client is not None:
            try:
                confirmation = supabase_client.sign_in(
                    email=user.email,
                    password=payload.password.get_secret_value(),
                )
            except SupabaseAuthError:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Password confirmation failed.",
                ) from None
            except SupabaseAuthUnavailable:
                _LOGGER.exception("Supabase account deletion confirmation failed.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Supabase Auth is temporarily unavailable.",
                ) from None
            if confirmation.identity.user_id != current_user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Password confirmation failed.",
                )
            try:
                supabase_client.delete_user(current_user.user_id)
            except SupabaseAuthUnavailable as error:
                if not supabase_client.secret_key:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Supabase account deletion is not configured.",
                    ) from None
                _LOGGER.exception("Supabase account deletion failed.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Supabase account deletion failed.",
                ) from error
            except SupabaseAuthError:
                _LOGGER.exception("Supabase account deletion was rejected.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Supabase account deletion failed.",
                ) from None
        elif user.password_hash is None or not verify_password(
            user.password_hash,
            payload.password.get_secret_value(),
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Password confirmation failed.",
            )
        admitted = session.execute(update(User).where(User.user_id == current_user.user_id).values(user_id=current_user.user_id).returning(User.user_id)).scalar_one_or_none()
        if admitted is None:
            raise HTTPException(status_code=401, detail="Account no longer available.")
        refund_pending_inference(session, current_user.user_id)
        session.execute(
            delete(AuthSession).where(AuthSession.user_id == current_user.user_id)
        )
        for owned_job in session.scalars(select(ComputeJob).where(
                ComputeJob.owner_user_id == current_user.user_id,
                ComputeJob.status.in_(("queued", "leased"))).with_for_update()):
            refund_job(session, owned_job)
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
        session.execute(
            delete(TrainingContribution).where(
                TrainingContribution.owner_user_id == current_user.user_id
            )
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
            supported_job_types=payload.supported_job_types,
            foundation_torch_version=payload.foundation_torch_version,
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
        payload: NodeResources | None = None,
        current_node: ComputeNode = Depends(get_current_node),
        session: Session = Depends(get_db),
    ) -> dict[str, int]:
        last_heartbeat = heartbeat_node(session, current_node, resources=payload)
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

    @app.post("/api/v1/jobs/foundation/tokenize", response_model=JobResponse, status_code=201)
    def create_foundation_tokenize_job(
        payload: CreateTokenizeJobRequest,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> ComputeJob:
        arguments = dict(owner_user_id=current_user.user_id, idempotency_key=payload.idempotency_key,
                         payload=payload.payload, allow_public_node_processing=payload.allow_public_node_processing,
                         reward_credits=payload.reward_credits)
        try:
            try:
                job = create_tokenize_job(session, **arguments)
                session.commit()
            except IntegrityError as error:
                session.rollback()
                if not is_job_idempotency_violation(error):
                    raise
                job = create_tokenize_job(session, **arguments)
                session.commit()
        except JobIdempotencyConflict:
            session.rollback()
            raise HTTPException(status_code=409, detail="Idempotency key already used for a different job.") from None
        except InsufficientCredits:
            session.rollback()
            raise HTTPException(status_code=402, detail="Insufficient compute credits for this reward.") from None
        return job

    @app.post("/api/v1/jobs/foundation/gradient", response_model=JobResponse, status_code=201)
    def create_foundation_gradient_job(
        payload: CreateGradientJobRequest, request: Request,
        current_user: AuthenticatedUser = Depends(get_current_user),
        session: Session = Depends(get_db),
    ) -> ComputeJob:
        if not request.app.state.settings.foundation_distributed_training_enabled:
            raise HTTPException(status_code=503, detail="Experimental gradient jobs are disabled.")
        arguments = dict(owner_user_id=current_user.user_id, idempotency_key=payload.idempotency_key,
                         payload=payload.payload, allow_public_node_processing=payload.allow_public_node_processing,
                         reward_credits=payload.reward_credits)
        try:
            try:
                job = create_gradient_job(session, **arguments)
                session.commit()
            except IntegrityError as error:
                session.rollback()
                if not is_job_idempotency_violation(error):
                    raise
                job = create_gradient_job(session, **arguments)
                session.commit()
        except JobIdempotencyConflict:
            session.rollback()
            raise HTTPException(status_code=409, detail="Idempotency key already used for a different job.") from None
        except (ValueError, ImportError):
            session.rollback()
            raise HTTPException(status_code=422, detail="Invalid or unsupported gradient task.") from None
        except InsufficientCredits:
            session.rollback()
            raise HTTPException(status_code=402, detail="Insufficient compute credits for this reward.") from None
        return job

    @app.get("/api/v1/credits")
    def get_own_credits(current_user: AuthenticatedUser = Depends(get_current_user),
                        session: Session = Depends(get_db)) -> dict[str, int]:
        refund_expired_inference(session, current_user.user_id)
        session.commit()
        account_id = user_account_id(current_user.user_id)
        account = session.get(Account, account_id)
        reserved = session.scalar(select(func.sum(JobFunding.amount)).where(
            JobFunding.owner_account == account_id, JobFunding.status == "reserved")) or 0
        reserved += session.scalar(select(func.sum(InferenceFunding.amount)).where(
            InferenceFunding.owner_account == account_id, InferenceFunding.status == "reserved")) or 0
        earned = session.scalar(select(func.sum(JobFunding.amount)).where(
            JobFunding.worker_account == account_id, JobFunding.owner_account != account_id,
            JobFunding.status == "settled")) or 0
        return {"available": account.balance if account else 0, "reserved": reserved, "earned": earned}

    @app.get("/api/v1/credits/transactions")
    def get_own_credit_transactions(current_user: AuthenticatedUser = Depends(get_current_user),
                                    session: Session = Depends(get_db)) -> list[dict]:
        rows = session.execute(select(LedgerTransaction, Posting.amount).join(Posting).where(
            Posting.account_id == user_account_id(current_user.user_id)).order_by(
                LedgerTransaction.transaction_id.desc()).limit(100))
        return [{"transaction_id": entry.transaction_id, "operation": entry.operation,
                 "amount": amount, "created_at": entry.created_at} for entry, amount in rows]

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_owned_job(job_id: str, current_user: AuthenticatedUser = Depends(get_current_user),
                         session: Session = Depends(get_db)):
        try:
            job = cancel_job(session, job_id, current_user.user_id)
            session.commit()
            return job
        except JobNotFound:
            session.rollback()
            raise HTTPException(status_code=404, detail="Job not found.") from None
        except ValueError:
            session.rollback()
            raise HTTPException(status_code=409, detail="Only pending jobs can be cancelled.") from None

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
        request: Request,
        current_node: ComputeNode = Depends(get_current_node),
        session: Session = Depends(get_db),
    ) -> ClaimedJobResponse | Response:
        heartbeat_node(session, current_node)
        try:
            claim = claim_job(session, current_node,
                              allow_foundation_training=request.app.state.settings.foundation_distributed_training_enabled)
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
        job_type=claim.job.job_type,
        payload=claim.job.payload,
        lease_token=claim.lease_token,
        lease_expires_epoch=claim.job.lease_expires_epoch or 0,
        reward_credits=claim.reward_credits,
    )


def _contains_secret(messages: list[dict[str, str]]) -> bool:
    return any(
        contains_common_secret(message["content"])
        for message in messages
    )


def _is_training_conversation_unique_violation(error: IntegrityError) -> bool:
    diagnostic = getattr(error.orig, "diag", None)
    if (
        getattr(diagnostic, "constraint_name", None)
        == "uq_training_contributions_owner_conversation"
    ):
        return True
    return (
        "UNIQUE constraint failed: "
        "training_contributions.owner_user_id, "
        "training_contributions.conversation_id"
    ) in str(error.orig)


def _is_dataset_version_violation(error: IntegrityError) -> bool:
    diagnostic = getattr(error.orig, "diag", None)
    if getattr(diagnostic, "constraint_name", None) == "uq_training_datasets_version":
        return True
    return (
        "UNIQUE constraint failed: training_datasets.dataset_version"
        in str(error.orig)
    )

