"""Supabase Auth REST client and access-token verification."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any
from uuid import UUID

import httpx
import jwt
from jwt.exceptions import PyJWTError


class SupabaseAuthError(Exception):
    def __init__(self, status_code: int, code: str):
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class SupabaseAuthUnavailable(Exception):
    """Raised when Supabase cannot complete an authentication operation."""


class InvalidSupabaseToken(Exception):
    """Raised when a Supabase JWT is invalid or does not identify a user."""


@dataclass(frozen=True)
class SupabaseIdentity:
    user_id: str
    email: str


@dataclass(frozen=True)
class SupabaseSession:
    access_token: str
    expires_at_epoch: int
    identity: SupabaseIdentity


class SupabaseAuth:
    _ALLOWED_ALGORITHMS = {"ES256", "RS256"}
    _JWKS_CACHE_SECONDS = 300

    def __init__(
        self,
        *,
        project_url: str,
        publishable_key: str,
        jwks_url: str,
        secret_key: str = "",
        timeout_seconds: float = 20,
        transport: httpx.BaseTransport | None = None,
    ):
        self.project_url = project_url.rstrip("/")
        self.auth_url = f"{self.project_url}/auth/v1/"
        self.issuer = f"{self.project_url}/auth/v1"
        self.publishable_key = publishable_key
        self.secret_key = secret_key
        self.jwks_url = jwks_url
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self._jwks: dict[str, dict[str, Any]] = {}
        self._jwks_expires_at = 0.0
        self._jwks_lock = threading.Lock()

    def sign_up(self, *, email: str, password: str) -> SupabaseIdentity:
        body = self._request(
            "POST",
            "signup",
            payload={"email": email, "password": password},
        )
        user = body.get("user", body)
        if not isinstance(user, dict):
            raise SupabaseAuthUnavailable("Supabase returned an invalid user.")
        return self._identity(user)

    def sign_in(self, *, email: str, password: str) -> SupabaseSession:
        body = self._request(
            "POST",
            "token?grant_type=password",
            payload={"email": email, "password": password},
        )
        access_token = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise SupabaseAuthUnavailable("Supabase returned an invalid session.")
        identity_data = body.get("user")
        if not isinstance(identity_data, dict):
            raise SupabaseAuthUnavailable("Supabase returned an invalid user.")
        expires_at = body.get("expires_at")
        if not isinstance(expires_at, int):
            expires_in = body.get("expires_in")
            if not isinstance(expires_in, int) or expires_in <= 0:
                raise SupabaseAuthUnavailable("Supabase returned an invalid session.")
            expires_at = int(time.time()) + expires_in
        return SupabaseSession(
            access_token=access_token,
            expires_at_epoch=expires_at,
            identity=self._identity(identity_data),
        )

    def sign_out(self, access_token: str) -> None:
        self._request("POST", "logout", access_token=access_token)

    def verify_access_token(self, access_token: str) -> SupabaseIdentity:
        try:
            header = jwt.get_unverified_header(access_token)
        except PyJWTError as error:
            raise InvalidSupabaseToken from error
        algorithm = header.get("alg")
        key_id = header.get("kid")
        if algorithm not in self._ALLOWED_ALGORITHMS or not isinstance(key_id, str):
            raise InvalidSupabaseToken

        try:
            jwk = self._find_signing_key(key_id)
            if jwk is None:
                jwk = self._find_signing_key(key_id, refresh=True)
            if jwk is None or jwk.get("alg") not in {None, algorithm}:
                raise InvalidSupabaseToken
            signing_key = jwt.PyJWK.from_dict(jwk, algorithm=algorithm).key
            claims = jwt.decode(
                access_token,
                signing_key,
                algorithms=[algorithm],
                audience="authenticated",
                issuer=self.issuer,
                options={"require": ["aud", "exp", "iss", "sub"]},
            )
        except InvalidSupabaseToken:
            raise
        except PyJWTError as error:
            raise InvalidSupabaseToken from error

        if claims.get("role") != "authenticated" or claims.get("is_anonymous") is True:
            raise InvalidSupabaseToken
        identity_data = {"id": claims.get("sub"), "email": claims.get("email")}
        try:
            return self._identity(identity_data)
        except SupabaseAuthUnavailable as error:
            raise InvalidSupabaseToken from error

    def delete_user(self, user_id: str) -> None:
        if not self.secret_key:
            raise SupabaseAuthUnavailable(
                "Supabase account deletion requires a server-side secret key."
            )
        normalized_id = str(UUID(user_id))
        self._request(
            "DELETE",
            f"admin/users/{normalized_id}",
            api_key=self.secret_key,
            access_token=self.secret_key,
        )

    def _find_signing_key(
        self,
        key_id: str,
        *,
        refresh: bool = False,
    ) -> dict[str, Any] | None:
        keys = self._load_jwks(force=refresh)
        return next((key for key in keys if key.get("kid") == key_id), None)

    def _load_jwks(self, *, force: bool = False) -> list[dict[str, Any]]:
        with self._jwks_lock:
            now = time.monotonic()
            if not force and now < self._jwks_expires_at:
                return list(self._jwks.values())
            try:
                with httpx.Client(
                    timeout=self.timeout_seconds,
                    transport=self.transport,
                ) as client:
                    response = client.get(self.jwks_url)
                    response.raise_for_status()
                    body = response.json()
            except (httpx.HTTPError, ValueError) as error:
                raise SupabaseAuthUnavailable(
                    "Could not retrieve Supabase signing keys."
                ) from error
            keys = body.get("keys") if isinstance(body, dict) else None
            if not isinstance(keys, list) or not all(
                isinstance(key, dict) and isinstance(key.get("kid"), str)
                for key in keys
            ):
                raise SupabaseAuthUnavailable(
                    "Supabase returned invalid signing keys."
                )
            self._jwks = {key["kid"]: key for key in keys}
            self._jwks_expires_at = now + self._JWKS_CACHE_SECONDS
            return list(self._jwks.values())

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, str] | None = None,
        access_token: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        headers = {"apikey": api_key or self.publishable_key}
        if access_token is not None:
            headers["Authorization"] = f"Bearer {access_token}"
        try:
            with httpx.Client(
                base_url=self.auth_url,
                headers=headers,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, json=payload)
        except httpx.HTTPError as error:
            raise SupabaseAuthUnavailable(
                "Could not connect to Supabase Auth."
            ) from error

        try:
            body = response.json() if response.content else {}
        except ValueError as error:
            if response.is_success:
                raise SupabaseAuthUnavailable(
                    "Supabase Auth returned an invalid response."
                ) from error
            body = {}
        if not response.is_success:
            code = ""
            if isinstance(body, dict):
                for field in ("error_code", "code", "msg", "message", "error"):
                    value = body.get(field)
                    if isinstance(value, str):
                        code = value
                        break
            raise SupabaseAuthError(response.status_code, code.casefold())
        if not isinstance(body, dict):
            raise SupabaseAuthUnavailable("Supabase Auth returned an invalid response.")
        return body

    @staticmethod
    def _identity(user: dict[str, Any]) -> SupabaseIdentity:
        user_id = user.get("id")
        email = user.get("email")
        if not isinstance(user_id, str) or not isinstance(email, str) or not email:
            raise SupabaseAuthUnavailable("Supabase returned an invalid user.")
        try:
            user_id = str(UUID(user_id))
        except ValueError as error:
            raise SupabaseAuthUnavailable(
                "Supabase returned an invalid user ID."
            ) from error
        return SupabaseIdentity(user_id=user_id, email=email)

