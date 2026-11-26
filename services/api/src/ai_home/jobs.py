"""Persistent, allowlisted compute jobs and lease-based node scheduling."""

import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_home.models import ComputeJob, ComputeNode
from ai_home.schemas import HardwareReport

SUPPORTED_JOB_TYPES = ("math.sum.v1",)
MAX_JOB_ATTEMPTS = 3
NODE_HEARTBEAT_TIMEOUT_SECONDS = 120
JOB_LEASE_SECONDS = 120
PUBLIC_NODE_CONSENT_VERSION = "volunteer-compute-v1"


class JobError(Exception):
    """Base class for expected node and job errors."""


class NodeNotFound(JobError):
    """Raised when a node credential or ID is unknown or revoked."""


class IdempotencyConflict(JobError):
    """Raised when a job idempotency key is reused for different content."""


class JobNotFound(JobError):
    """Raised when a job is not visible to its owner or assigned node."""


class LeaseExpired(JobError):
    """Raised when a claimed job no longer has a valid lease."""


class InvalidJobResult(JobError):
    """Raised when the reported result does not match server validation."""


@dataclass(frozen=True)
class RegisteredNode:
    node: ComputeNode
    node_token: str


@dataclass(frozen=True)
class ClaimedJob:
    job: ComputeJob
    lease_token: str


def register_node(
    session: Session,
    *,
    owner_user_id: str,
    name: str,
    hardware: HardwareReport,
) -> RegisteredNode:
    if not name.strip():
        raise ValueError("Node name cannot be empty.")
    node_token = secrets.token_urlsafe(32)
    node = ComputeNode(
        node_id=str(uuid4()),
        owner_user_id=owner_user_id,
        name=name.strip(),
        credential_hash=_token_hash(node_token),
        hardware=hardware.model_dump(mode="json"),
        supported_job_types=list(SUPPORTED_JOB_TYPES),
        last_heartbeat_epoch=int(time.time()),
    )
    session.add(node)
    session.flush()
    return RegisteredNode(node=node, node_token=node_token)


def authenticate_node(session: Session, node_token: str) -> ComputeNode:
    node = session.scalar(
        select(ComputeNode).where(
            ComputeNode.credential_hash == _token_hash(node_token),
            ComputeNode.revoked_at.is_(None),
        )
    )
    if node is None:
        raise NodeNotFound
    return node


def heartbeat_node(session: Session, node: ComputeNode) -> int:
    now = int(time.time())
    node.last_heartbeat_epoch = now
    session.flush()
    return now


def revoke_node(session: Session, node: ComputeNode, owner_user_id: str) -> None:
    owned_node = session.scalar(
        select(ComputeNode).where(
            ComputeNode.node_id == node.node_id,
            ComputeNode.owner_user_id == owner_user_id,
            ComputeNode.revoked_at.is_(None),
        )
    )
    if owned_node is None:
        raise NodeNotFound
    owned_node.revoked_at = datetime.now(timezone.utc)
    session.execute(
        update(ComputeJob)
        .where(
            ComputeJob.assigned_node_id == owned_node.node_id,
            ComputeJob.status == "leased",
        )
        .values(
            status="queued",
            assigned_node_id=None,
            lease_token_hash=None,
            lease_expires_epoch=None,
        )
    )
    session.flush()


def create_math_sum_job(
    session: Session,
    *,
    owner_user_id: str,
    idempotency_key: str,
    values: list[int],
    allow_public_node_processing: bool,
) -> ComputeJob:
    if not allow_public_node_processing:
        raise ValueError("Public node processing requires explicit consent.")
    job_type = "math.sum.v1"
    payload = {"values": values}
    request_hash = _request_hash(job_type, payload)

    existing = session.scalar(
        select(ComputeJob).where(
            ComputeJob.owner_user_id == owner_user_id,
            ComputeJob.idempotency_key == idempotency_key,
        )
    )
    if existing is not None:
        if existing.request_hash != request_hash:
            raise IdempotencyConflict
        return existing

    job = ComputeJob(
        job_id=str(uuid4()),
        owner_user_id=owner_user_id,
        job_type=job_type,
        payload=payload,
        status="queued",
        idempotency_key=idempotency_key,
        request_hash=request_hash,
        public_node_consent_at_epoch=int(time.time()),
        public_node_consent_version=PUBLIC_NODE_CONSENT_VERSION,
        attempt_count=0,
    )
    session.add(job)
    session.flush()
    return job


def find_existing_math_sum_job(
    session: Session,
    *,
    owner_user_id: str,
    idempotency_key: str,
    values: list[int],
) -> ComputeJob | None:
    job = session.scalar(
        select(ComputeJob).where(
            ComputeJob.owner_user_id == owner_user_id,
            ComputeJob.idempotency_key == idempotency_key,
        )
    )
    if job is not None and job.request_hash != _request_hash(
        "math.sum.v1",
        {"values": values},
    ):
        raise IdempotencyConflict
    return job


def is_job_idempotency_violation(error: IntegrityError) -> bool:
    diagnostic = getattr(error.orig, "diag", None)
    if (
        getattr(diagnostic, "constraint_name", None)
        == "uq_compute_jobs_owner_idempotency"
    ):
        return True
    return (
        "UNIQUE constraint failed: compute_jobs.owner_user_id, "
        "compute_jobs.idempotency_key"
    ) in str(error.orig)


def list_owner_jobs(session: Session, owner_user_id: str) -> list[ComputeJob]:
    return list(
        session.scalars(
            select(ComputeJob)
            .where(ComputeJob.owner_user_id == owner_user_id)
            .order_by(ComputeJob.created_at.desc(), ComputeJob.job_id.desc())
            .limit(100)
        )
    )


def claim_job(session: Session, node: ComputeNode) -> ClaimedJob | None:
    now = int(time.time())
    if node.revoked_at is not None or (
        node.last_heartbeat_epoch < now - NODE_HEARTBEAT_TIMEOUT_SECONDS
    ):
        raise NodeNotFound

    session.execute(
        update(ComputeJob)
        .where(
            ComputeJob.status == "leased",
            ComputeJob.lease_expires_epoch <= now,
            ComputeJob.attempt_count >= MAX_JOB_ATTEMPTS,
        )
        .values(
            status="failed",
            assigned_node_id=None,
            lease_token_hash=None,
            lease_expires_epoch=None,
            failure_code="lease_expired",
        )
    )
    session.execute(
        update(ComputeJob)
        .where(
            ComputeJob.status == "leased",
            ComputeJob.lease_expires_epoch <= now,
            ComputeJob.attempt_count < MAX_JOB_ATTEMPTS,
        )
        .values(
            status="queued",
            assigned_node_id=None,
            lease_token_hash=None,
            lease_expires_epoch=None,
        )
    )

    job = session.scalar(
        select(ComputeJob)
        .where(
            ComputeJob.status == "queued",
            ComputeJob.job_type.in_(node.supported_job_types),
        )
        .order_by(ComputeJob.created_at, ComputeJob.job_id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if job is None:
        return None

    lease_token = secrets.token_urlsafe(32)
    job.status = "leased"
    job.assigned_node_id = node.node_id
    job.lease_token_hash = _token_hash(lease_token)
    job.lease_expires_epoch = now + JOB_LEASE_SECONDS
    job.attempt_count += 1
    job.failure_code = None
    session.flush()
    return ClaimedJob(job=job, lease_token=lease_token)


def submit_job_result(
    session: Session,
    *,
    node: ComputeNode,
    job_id: str,
    lease_token: str,
    result: dict[str, int],
) -> ComputeJob:
    job = session.scalar(
        select(ComputeJob).where(
            ComputeJob.job_id == job_id,
            ComputeJob.assigned_node_id == node.node_id,
            ComputeJob.status == "leased",
        )
    )
    if job is None or job.lease_token_hash != _token_hash(lease_token):
        raise JobNotFound
    if job.lease_expires_epoch is None or job.lease_expires_epoch <= int(time.time()):
        raise LeaseExpired
    if job.job_type != "math.sum.v1" or set(result) != {"sum"}:
        raise InvalidJobResult
    expected = sum(job.payload["values"])
    if result["sum"] != expected:
        raise InvalidJobResult

    job.status = "completed"
    job.result = result
    job.assigned_node_id = None
    job.lease_token_hash = None
    job.lease_expires_epoch = None
    session.flush()
    return job


def _request_hash(job_type: str, payload: dict) -> str:
    body = json.dumps(
        {"job_type": job_type, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

