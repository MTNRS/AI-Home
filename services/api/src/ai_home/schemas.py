"""Validated API request and response models."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


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
    user_id: str
    email: EmailStr
    created_at: datetime


class SessionResponse(BaseModel):
    access_token: str
    token_type: Annotated[str, Field(pattern="^bearer$")] = "bearer"
    expires_at_epoch: int


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: StrictStr = Field(min_length=1, max_length=8192)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_conversation(self) -> "ChatCompletionRequest":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("A chat request must contain a user message.")
        if sum(len(message.content) for message in self.messages) > 32_000:
            raise ValueError("The total chat content must not exceed 32,000 characters.")
        return self


class ChatCompletionMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoiceResponse(BaseModel):
    index: int
    message: ChatCompletionMessageResponse
    finish_reason: Literal["stop"]


class ChatCompletionResponse(BaseModel):
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
    job_type: Literal["math.sum.v1"]
    payload: dict[str, list[int]]
    lease_token: str
    lease_expires_epoch: int


class SubmitJobResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_token: SecretStr = Field(min_length=32, max_length=128)
    result: dict[str, StrictInt]
