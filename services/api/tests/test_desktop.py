import json

import httpx
import pytest

from ai_home.desktop import (
    chat_completion,
    load_settings,
    login_to_server,
    normalize_base_url,
    save_settings,
)


def test_base_url_requires_https_outside_local_machine():
    assert normalize_base_url("http://localhost:8000/") == "http://localhost:8000"
    assert normalize_base_url("https://community.example/api/") == (
        "https://community.example/api"
    )

    with pytest.raises(ValueError, match="Usa HTTPS"):
        normalize_base_url("http://community.example")
    with pytest.raises(ValueError, match="credenciales"):
        normalize_base_url("https://alice:secret@community.example")


def test_desktop_login_uses_api_and_validates_session_response():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/auth/login"
        assert json.loads(request.read()) == {
            "email": "alice@example.com",
            "password": "example-password",
        }
        return httpx.Response(
            200,
            json={
                "access_token": "a-secure-session-token",
                "token_type": "bearer",
                "expires_at_epoch": 2_000_000_000,
            },
        )

    session = login_to_server(
        "http://localhost:8000",
        "alice@example.com",
        "example-password",
        transport=httpx.MockTransport(handle),
    )

    assert session.access_token == "a-secure-session-token"


def test_chat_completion_sends_chat_and_optional_api_key():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer provider-secret"
        assert json.loads(request.read())["model"] == "test-model"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hello from the model."}}]},
        )

    answer = chat_completion(
        "http://localhost:11434/v1",
        "test-model",
        [{"role": "user", "content": "Hello"}],
        api_key="provider-secret",
        transport=httpx.MockTransport(handle),
    )

    assert answer == "Hello from the model."


def test_chat_completion_rejects_malformed_provider_response():
    answer = httpx.Response(200, json={"choices": []})
    transport = httpx.MockTransport(lambda _request: answer)

    with pytest.raises(ValueError, match="respuesta"):
        chat_completion(
            "http://localhost:11434/v1",
            "test-model",
            [{"role": "user", "content": "Hello"}],
            transport=transport,
        )


def test_settings_round_trip_stores_only_non_secret_preferences(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("ai_home.desktop.sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    settings = {
        "server_url": "https://community.example",
        "email": "alice@example.com",
        "chat_endpoint": "http://localhost:11434/v1",
        "chat_model": "test-model",
    }

    save_settings(settings)

    assert load_settings() == settings
    assert "password" not in load_settings()
