import json

import httpx
import pytest

from ai_home.desktop import (
    _SPHERE_EMOTION_COLORS,
    _blend_sphere_color,
    _sphere_emotion_for_text,
    _sphere_pointer_angles,
    chat_completion,
    delete_provider_api_key,
    get_training_candidate,
    get_provider_api_key,
    load_settings,
    login_to_server,
    list_training_candidates,
    normalize_base_url,
    save_provider_api_key,
    save_settings,
    server_chat_completion,
    submit_training_candidate,
    withdraw_training_candidate,
)
from ai_home.providers import provider_completion


def test_sphere_orientation_tracks_pointer_with_a_wide_smooth_range():
    assert _sphere_pointer_angles(0, 0, 60) == (0, 0)

    left, level = _sphere_pointer_angles(-80, 0, 60)
    right, _ = _sphere_pointer_angles(80, 0, 60)
    _, above = _sphere_pointer_angles(0, -80, 60)
    _, below = _sphere_pointer_angles(0, 80, 60)

    assert left < 0 < right
    assert above > 0 > below
    assert 0.35 < right < 0.6
    assert level == 0
    assert _sphere_pointer_angles(10_000, -10_000, 60) == (0.75, 0.5)
    assert _sphere_pointer_angles(-10_000, 10_000, 60) == (-0.75, -0.5)


@pytest.mark.parametrize(
    ("text", "emotion"),
    [
        ("¡Me alegra que hayas venido!", "alegria"),
        ("Lo siento mucho por lo ocurrido.", "tristeza"),
        ("Entiendo que estés frustrado.", "enfado"),
        ("Es normal sentir miedo ante eso.", "miedo"),
        ("Respira con calma; todo está tranquilo.", "calma"),
    ],
)
def test_sphere_detects_emotional_cues_in_agent_responses(text, emotion):
    assert _sphere_emotion_for_text(text) == emotion
    assert _SPHERE_EMOTION_COLORS[emotion].startswith("#")


def test_sphere_stays_neutral_without_emotional_cues():
    assert _sphere_emotion_for_text("La respuesta es 42.") is None


def test_sphere_blush_tint_is_soft_and_clamped():
    assert _blend_sphere_color("#ffffff", "#000000", 0.28) == "#b8b8b8"
    assert _blend_sphere_color("#ffffff", "#000000", -1) == "#ffffff"
    assert _blend_sphere_color("#ffffff", "#000000", 2) == "#000000"


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


def test_desktop_login_displays_the_api_authentication_error():
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            403,
            json={"detail": "Confirma tu correo electrónico antes de iniciar sesión."},
        )
    )

    with pytest.raises(RuntimeError, match="Confirma tu correo electrónico"):
        login_to_server(
            "http://localhost:8000",
            "alice@example.com",
            "example-password",
            transport=transport,
        )


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


def test_server_chat_completion_sends_account_token_and_conversation():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer account-session-token"
        assert json.loads(request.read()) == {
            "messages": [{"role": "user", "content": "Hola"}]
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hola también."}}]},
        )

    answer = server_chat_completion(
        "https://community.example",
        "account-session-token",
        [{"role": "user", "content": "Hola"}],
        transport=httpx.MockTransport(handle),
    )

    assert answer == "Hola también."


def test_settings_round_trip_stores_only_non_secret_preferences(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("ai_home.desktop.sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    settings = {
        "server_url": "https://community.example",
        "email": "alice@example.com",
        "chat_endpoint": "http://localhost:9000/v1",
        "chat_model": "test-model",
    }

    save_settings(settings)

    assert load_settings() == settings
    assert "password" not in load_settings()


def test_load_settings_removes_legacy_ollama_configuration(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("ai_home.desktop.sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    settings_path = tmp_path / "ai-home" / "desktop.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "server_url": "https://community.example",
                "chat_provider": "openai-compatible",
                "chat_endpoint": "http://localhost:11434/v1",
                "chat_model": "llama3.2",
                "chat_model_openai-compatible": "llama3.2",
                "email": "alice@example.com",
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings()

    assert settings == {
        "server_url": "https://community.example",
        "chat_provider": "ai-home-local",
        "email": "alice@example.com",
    }
    assert json.loads(settings_path.read_text(encoding="utf-8")) == settings


def test_load_settings_selects_local_model_after_old_endpoint_was_removed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("ai_home.desktop.sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    settings_path = tmp_path / "ai-home" / "desktop.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "chat_provider": "openai-compatible",
                "email": "alice@example.com",
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings()

    assert settings == {
        "chat_provider": "ai-home-local",
        "email": "alice@example.com",
    }


def test_local_provider_routes_to_the_first_party_model(monkeypatch):
    received_messages = []

    def complete(messages):
        received_messages.extend(messages)
        return "Respuesta del modelo local."

    monkeypatch.setattr("ai_home.local_llm.complete_locally", complete)

    answer = provider_completion(
        "ai-home-local",
        "",
        [{"role": "user", "content": "Hola"}],
    )

    assert answer == "Respuesta del modelo local."
    assert received_messages == [{"role": "user", "content": "Hola"}]


def test_local_provider_reports_missing_trained_checkpoint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("APPDATA", str(tmp_path))

    with pytest.raises(RuntimeError, match="todavía no está entrenado"):
        provider_completion(
            "ai-home-local",
            "",
            [{"role": "user", "content": "Hola"}],
        )


def test_provider_keys_use_keyring_and_are_partitioned_by_provider_and_endpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    stored: dict[tuple[str, str], str] = {}
    monkeypatch.setattr("ai_home.desktop._require_os_keyring", lambda: None)

    def set_password(service: str, account: str, password: str) -> None:
        stored[(service, account)] = password

    def get_password(service: str, account: str) -> str | None:
        return stored.get((service, account))

    def delete_password(service: str, account: str) -> None:
        del stored[(service, account)]

    monkeypatch.setattr("ai_home.desktop.keyring.set_password", set_password)
    monkeypatch.setattr("ai_home.desktop.keyring.get_password", get_password)
    monkeypatch.setattr("ai_home.desktop.keyring.delete_password", delete_password)

    save_provider_api_key(
        "openai-compatible",
        "https://provider.example/v1",
        "provider-secret",
    )
    save_provider_api_key(
        "openai-compatible",
        "https://other.example/v1",
        "other-secret",
    )

    assert get_provider_api_key(
        "openai-compatible",
        "https://provider.example/v1",
    ) == "provider-secret"
    assert get_provider_api_key(
        "openai-compatible",
        "https://other.example/v1",
    ) == "other-secret"
    assert all("provider.example" not in account for _, account in stored)
    assert delete_provider_api_key(
        "openai-compatible",
        "https://provider.example/v1",
    )
    assert get_provider_api_key(
        "openai-compatible",
        "https://provider.example/v1",
    ) == ""
    assert len(stored) == 1


def test_provider_keyring_failure_does_not_fall_back_to_plaintext(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("ai_home.desktop.settings_path", lambda: tmp_path / "settings.json")

    def unavailable(*_args, **_kwargs):
        raise keyring.errors.NoKeyringError("secure store unavailable")

    import keyring

    monkeypatch.setattr("ai_home.desktop.keyring.set_password", unavailable)
    monkeypatch.setattr("ai_home.desktop.keyring.get_password", unavailable)
    monkeypatch.setattr("ai_home.desktop._require_os_keyring", lambda: None)

    with pytest.raises(RuntimeError, match="almacén seguro"):
        save_provider_api_key("openai", "", "provider-secret")
    with pytest.raises(RuntimeError, match="almacén seguro"):
        get_provider_api_key("openai", "")

    assert not (tmp_path / "settings.json").exists()


def test_provider_keys_refuse_non_os_keyring_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    class InsecureBackend:
        pass

    calls = []
    monkeypatch.setattr(
        "ai_home.desktop.keyring.get_keyring",
        lambda: InsecureBackend(),
    )
    monkeypatch.setattr(
        "ai_home.desktop.keyring.set_password",
        lambda *_args: calls.append("stored"),
    )

    with pytest.raises(RuntimeError, match="almacén seguro"):
        save_provider_api_key("openai", "", "provider-secret")

    assert calls == []


def test_training_candidate_client_sends_opt_in_data_without_api_key():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/training/conversations"
        assert request.headers["Authorization"] == "Bearer account-token"
        body = json.loads(request.read())
        assert body == {
            "conversation_id": "d24b0b20-5ebc-48af-a3c8-000000000001",
            "provider": "openai",
            "source_endpoint": "https://api.openai.com/v1",
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "A public question."},
                {"role": "assistant", "content": "A public answer."},
            ],
            "consent": True,
            "consent_version": "training_consent_v1",
            "consent_scope": "conversation_for_aihome_training_candidates",
            "provider_terms_allow_reuse": True,
            "content_reviewed_for_sensitive_data": True,
        }
        assert "api_key" not in body
        return httpx.Response(
            201,
            json={
                "contribution_id": "4d5871c5-4766-4813-b8b5-000000000001",
                "conversation_id": body["conversation_id"],
                "provider": "openai",
                "source_endpoint": "https://api.openai.com/v1",
                "model": "gpt-test",
                "consent_version": "training_consent_v1",
                "consent_scope": "conversation_for_aihome_training_candidates",
                "consent_at_epoch": 1_700_000_000,
                "status": "pending_review",
            },
        )

    result = submit_training_candidate(
        "https://community.example",
        "account-token",
        "d24b0b20-5ebc-48af-a3c8-000000000001",
        "openai",
        "https://api.openai.com/v1",
        "gpt-test",
        [
            {"role": "user", "content": "A public question."},
            {"role": "assistant", "content": "A public answer."},
        ],
        reviewed_for_submission=True,
        transport=httpx.MockTransport(handle),
    )

    assert result.status == "pending_review"


def test_training_candidate_management_client_lists_and_withdraws():
    contribution_id = "4d5871c5-4766-4813-b8b5-000000000001"

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer account-token"
        if request.method == "GET":
            if request.url.path == "/api/v1/training/conversations":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "contribution_id": contribution_id,
                            "conversation_id": (
                                "d24b0b20-5ebc-48af-a3c8-000000000001"
                            ),
                            "provider": "openai",
                            "source_endpoint": "https://api.openai.com/v1",
                            "model": "gpt-test",
                            "consent_version": "training_consent_v1",
                            "consent_scope": (
                                "conversation_for_aihome_training_candidates"
                            ),
                            "consent_at_epoch": 1_700_000_000,
                            "status": "pending_review",
                        }
                    ],
                )
            assert request.url.path == (
                f"/api/v1/training/conversations/{contribution_id}"
            )
            return httpx.Response(
                200,
                json={
                    "contribution_id": contribution_id,
                    "conversation_id": "d24b0b20-5ebc-48af-a3c8-000000000001",
                    "provider": "openai",
                    "source_endpoint": "https://api.openai.com/v1",
                    "model": "gpt-test",
                    "consent_version": "training_consent_v1",
                    "consent_scope": "conversation_for_aihome_training_candidates",
                    "consent_at_epoch": 1_700_000_000,
                    "status": "pending_review",
                    "messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Answer"},
                    ],
                },
            )
        assert request.method == "DELETE"
        assert request.url.path == (
            f"/api/v1/training/conversations/{contribution_id}"
        )
        return httpx.Response(204)

    transport = httpx.MockTransport(handle)
    contributions = list_training_candidates(
        "https://community.example",
        "account-token",
        transport=transport,
    )
    contribution = get_training_candidate(
        "https://community.example",
        "account-token",
        contribution_id,
        transport=transport,
    )
    withdraw_training_candidate(
        "https://community.example",
        "account-token",
        contribution_id,
        transport=transport,
    )

    assert contributions[0].model == "gpt-test"
    assert contribution.messages[0].content == "Question"

