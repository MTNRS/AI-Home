"""Validated API request and response models."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: SecretStr = Field(min_length=12, max_length=128)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: SecretStr = Field(min_length=12, max_length=128)


class UserResponse(BaseModel):
    user_id: str
    email: EmailStr
    created_at: datetime


class SessionResponse(BaseModel):
    access_token: str
    token_type: Annotated[str, Field(pattern="^bearer$")] = "bearer"
    expires_at_epoch: int
