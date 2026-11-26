"""Validated API request and response models."""

from datetime import datetime
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ai_home.providers import PROVIDER_DEFAULTS
from ai_home.privacy import contains_common_secret
from ai_home.urls import normalize_base_url


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: SecretStr = Field(min_length=12, max_length=128)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: SecretStr = Field(min_length=12, max_length=128)


class DeleteAccountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: SecretStr = Field(min_length=12, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    email: EmailStr
    created_at: datetime


class SessionResponse(BaseModel):
    user_id: UUID | None = None
    access_token: str
    token_type: Annotated[str, Field(pattern="^bearer$")] = "bearer"
    expires_at_epoch: int


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: StrictStr = Field(min_length=1, max_length=8192)


class FoundationGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0, le=5, allow_inf_nan=False)
    top_k: int = Field(default=40, ge=0, le=100_000)
    top_p: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.0, ge=1, le=10, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)


class ChatCompletionRequest(BaseModel):
    credit_request_id: UUID | None = None
    credit_limit: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    generation: FoundationGeneration | None = None
    use_knowledge: StrictBool = False
    model_version: UUID | None = None
    model_config = ConfigDict(extra="forbid", protected_namespaces=("model_validate", "model_dump"))

    messages: list[ChatMessage] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_conversation(self) -> "ChatCompletionRequest":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("A chat request must contain a user message.")
        if sum(len(message.content) for message in self.messages) > 32_000:
            raise ValueError("The total chat content must not exceed 32,000 characters.")
        return self


class TrainingConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: StrictStr = Field(min_length=1, max_length=8192)


class TrainingContributionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    provider: Literal[
        "openai",
        "anthropic",
        "gemini",
        "openai-compatible",
    ]
    source_endpoint: StrictStr = Field(min_length=8, max_length=500)
    model: StrictStr = Field(min_length=1, max_length=160)
    messages: list[TrainingConversationMessage] = Field(min_length=2, max_length=50)
    consent: StrictBool
    consent_version: Literal["training_consent_v1"]
    consent_scope: Literal["conversation_for_aihome_training_candidates"]
    provider_terms_allow_reuse: StrictBool
    content_reviewed_for_sensitive_data: StrictBool

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("A provider model is required.")
        return normalized

    @model_validator(mode="after")
    def validate_training_contribution(self) -> "TrainingContributionRequest":
        if not self.consent:
            raise ValueError("Explicit training consent is required.")
        if not self.provider_terms_allow_reuse:
            raise ValueError("Provider terms must allow reuse for training.")
        if not self.content_reviewed_for_sensitive_data:
            raise ValueError("The conversation must be reviewed before submission.")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("A conversation must contain a user message.")
        if not any(message.role == "assistant" for message in self.messages):
            raise ValueError("A conversation must contain an assistant message.")
        if sum(len(message.content) for message in self.messages) > 32_000:
            raise ValueError("The conversation is too large to submit.")
        normalized_endpoint = normalize_base_url(self.source_endpoint)
        if contains_common_secret(normalized_endpoint):
            raise ValueError("The provider endpoint must not contain credentials.")
        if (
            self.provider != "openai-compatible"
            and normalized_endpoint != PROVIDER_DEFAULTS[self.provider][0]
        ):
            raise ValueError("The provider endpoint does not match its provenance.")
        self.source_endpoint = normalized_endpoint
        return self


class TrainingContributionResponse(BaseModel):
    contribution_id: str
    conversation_id: UUID
    provider: str
    source_endpoint: str
    model: str
    consent_version: str
    consent_scope: str
    consent_at_epoch: int
    status: Literal["pending_review", "eligible", "frozen", "rejected"]


class TrainingContributionExportResponse(TrainingContributionResponse):
    messages: list[TrainingConversationMessage]


class TrainingReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve: StrictBool
    provider_terms_verified: StrictBool = False
    sensitive_data_reviewed: StrictBool = False
    reuse_rights_verified: StrictBool = False

    @model_validator(mode="after")
    def require_review_checklist_for_approval(self) -> "TrainingReviewRequest":
        if self.approve and not all(
            (
                self.provider_terms_verified,
                self.sensitive_data_reviewed,
                self.reuse_rights_verified,
            )
        ):
            raise ValueError(
                "Provider terms, sensitive data and reuse rights must be reviewed."
            )
        return self


class TrainingReviewResponse(BaseModel):
    contribution_id: str
    status: Literal["eligible", "rejected"]
    reviewer_id: str
    reviewed_at_epoch: int
    review_checklist: dict[str, bool]


class TrainingContributionReviewResponse(
    TrainingContributionExportResponse
):
    owner_user_id: str


class FreezeTrainingDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: StrictStr = Field(min_length=1, max_length=80)
    contribution_ids: list[UUID] = Field(min_length=1, max_length=100)

    @field_validator("dataset_version")
    @classmethod
    def dataset_version_must_be_safe(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", normalized):
            raise ValueError(
                "Dataset version may contain only letters, numbers, '.', '_' and '-'."
            )
        return normalized

    @model_validator(mode="after")
    def require_unique_contributions(self) -> "FreezeTrainingDatasetRequest":
        if len(self.contribution_ids) != len(set(self.contribution_ids)):
            raise ValueError("Dataset contribution identifiers must be unique.")
        return self


class TrainingDatasetResponse(BaseModel):
    dataset_id: str
    dataset_version: str
    manifest_sha256: str
    item_count: int
    created_at: datetime


class FrozenTrainingRecord(BaseModel):
    contribution_id: str
    provider: str
    source_endpoint: str
    model: str
    consent_version: str
    consent_scope: str
    consent_at_epoch: int
    reviewer_id: str
    reviewed_at_epoch: int
    review_checklist: dict[str, bool]
    messages: list[TrainingConversationMessage]


class TrainingDatasetExportResponse(TrainingDatasetResponse):
    records: list[FrozenTrainingRecord]


class ChatCompletionMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoiceResponse(BaseModel):
    index: int
    message: ChatCompletionMessageResponse
    finish_reason: Literal["stop"]


class KnowledgeReference(BaseModel):
    id: str
    source: str
    url: str
    license: str
    date: str
    language: str
    document_sha256: str
    offset: int
    text: str
    score: float
    index_version: str


class ChatCompletionResponse(BaseModel):
    charged_credits: int = 0
    credit_request_id: UUID | None = None
    generation: FoundationGeneration | None = None
    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))
    model_version: UUID | None = None
    references: list[KnowledgeReference] = Field(default_factory=list)
    choices: list[ChatCompletionChoiceResponse]


class GPUDevice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_index: int = Field(ge=0, le=31)
    name: str = Field(min_length=1, max_length=128)
    memory_total_bytes: int = Field(gt=0, le=2**63 - 1)


class HardwareReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = Field(min_length=1, max_length=80)
    cpu_model: str = Field(min_length=1, max_length=128)
    logical_cores: int = Field(ge=1, le=4096)
    ram_total_bytes: int = Field(gt=0, le=2**63 - 1)
    gpus: list[GPUDevice] = Field(default_factory=list, max_length=32)
    foundation_torch_version: StrictStr | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def require_unique_gpu_indices(self) -> "HardwareReport":
        indices = [gpu.device_index for gpu in self.gpus]
        if len(indices) != len(set(indices)):
            raise ValueError("GPU device indices must be unique.")
        return self


class RegisterNodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    hardware: HardwareReport
    supported_job_types: list[Literal["math.sum.v1", "foundation.tokenize.v1", "foundation.gradient.v1"]] = Field(
        default_factory=lambda: ["math.sum.v1"], min_length=1, max_length=3)
    foundation_torch_version: StrictStr | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def require_gradient_runtime(self):
        if "foundation.gradient.v1" in self.supported_job_types and not self.foundation_torch_version:
            raise ValueError("Gradient workers must declare their PyTorch runtime")
        return self

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Node name cannot be blank.")
        return normalized


class NodeCredentialResponse(BaseModel):
    node_id: str
    node_token: str
    supported_job_types: list[str]


class NodeResponse(BaseModel):
    node_id: str
    name: str
    hardware: HardwareReport
    supported_job_types: list[str]
    last_heartbeat_epoch: int
    created_at: datetime


class CreateMathSumJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[StrictInt] = Field(min_length=1, max_length=10_000)
    allow_public_node_processing: bool
    idempotency_key: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def require_public_node_consent(self) -> "CreateMathSumJobRequest":
        if not self.allow_public_node_processing:
            raise ValueError(
                "This job requires explicit consent to share its input with "
                "volunteer nodes."
            )
        if any(isinstance(value, bool) or abs(value) > 10**12 for value in self.values):
            raise ValueError("Values must be integers between -10^12 and 10^12.")
        return self


class MathSumJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[StrictInt] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_values(self) -> "MathSumJobPayload":
        if any(isinstance(value, bool) or abs(value) > 10**12 for value in self.values):
            raise ValueError("Values must be integers between -10^12 and 10^12.")
        return self


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    result: dict | None
    failure_code: str | None
    attempt_count: int
    created_at: datetime


class ClaimedJobResponse(BaseModel):
    job_id: str
    job_type: Literal["math.sum.v1", "foundation.tokenize.v1", "foundation.gradient.v1"]
    payload: dict
    lease_token: str
    lease_expires_epoch: int
    reward_credits: int = 0


class SubmitJobResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_token: SecretStr = Field(min_length=32, max_length=128)
    result: dict[str, StrictInt | StrictStr] = Field(max_length=3)

    @model_validator(mode="after")
    def bounded_result(self):
        if any(isinstance(v, str) and len(v) > 600_000 for v in self.result.values()):
            raise ValueError("Result exceeds artifact budget")
        if "sum" in self.result and type(self.result["sum"]) is not int:
            raise ValueError("Sum must be an integer")
        return self


class CreateTokenizeJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: dict
    allow_public_node_processing: StrictBool
    idempotency_key: str = Field(min_length=1, max_length=160)
    reward_credits: StrictInt = Field(default=0, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_public_payload(self):
        from ai_home.foundation.tasks import TokenizePayload
        if not self.allow_public_node_processing:
            raise ValueError("Public node processing requires explicit consent")
        self.payload = TokenizePayload.model_validate(self.payload).model_dump()
        return self


class CreateGradientJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: dict
    allow_public_node_processing: StrictBool
    idempotency_key: str = Field(min_length=1, max_length=160)
    reward_credits: StrictInt = Field(default=0, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def require_consent(self):
        if not self.allow_public_node_processing:
            raise ValueError("Public node processing requires explicit consent")
        return self

