import json

import httpx
import pytest

from ai_home.providers import PROVIDER_DEFAULTS, provider_completion


def test_compatible_provider_has_no_preselected_external_model():
    assert PROVIDER_DEFAULTS["openai-compatible"] == (
        "",
        "",
    )


def test_compatible_provider_sends_standard_chat_request():
    def handle(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.read()) == {
            "model": "custom-model",
            "messages": [{"role": "user", "content": "Hola"}],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "¡Hola!"}}]},
        )

    answer = provider_completion(
        "openai-compatible",
        "custom-model",
        [{"role": "user", "content": "Hola"}],
        endpoint_url="http://localhost:1234/v1",
        transport=httpx.MockTransport(handle),
    )

    assert answer == "¡Hola!"


def test_openai_adapter_uses_fixed_official_endpoint_and_authorization_header():
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer openai-secret"
        assert "openai-secret" not in str(request.url)
        assert json.loads(request.read()) == {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OpenAI response"}}]},
        )

    answer = provider_completion(
        "openai",
        "gpt-test",
        [{"role": "user", "content": "Hello"}],
        api_key="openai-secret",
        endpoint_url="https://attacker.example",
        transport=httpx.MockTransport(handle),
    )

    assert answer == "OpenAI response"


def test_anthropic_adapter_translates_system_messages_and_reads_text_blocks():
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "anthropic-secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert json.loads(request.read()) == {
            "model": "claude-test",
            "max_tokens": 4096,
            "system": "Be concise.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " there"},
                ]
            },
        )

    answer = provider_completion(
        "anthropic",
        "claude-test",
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        api_key="anthropic-secret",
        transport=httpx.MockTransport(handle),
    )

    assert answer == "Hello there"


def test_gemini_adapter_uses_api_key_header():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == (
            "/v1beta/models/gemini-test:generateContent"
        )
        assert request.headers["x-goog-api-key"] == "gemini-secret"
        assert "gemini-secret" not in str(request.url)
        assert json.loads(request.read()) == {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Hi"}]},
            ],
            "systemInstruction": {"parts": [{"text": "Be concise."}]},
        }
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemini response"}],
                        }
                    }
                ]
            },
        )

    answer = provider_completion(
        "gemini",
        "gemini-test",
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        api_key="gemini-secret",
        transport=httpx.MockTransport(handle),
    )

    assert answer == "Gemini response"


def test_openai_compatible_adapter_requires_secure_non_local_endpoint():
    with pytest.raises(ValueError, match="HTTPS"):
        provider_completion(
            "openai-compatible",
            "test-model",
            [{"role": "user", "content": "Hello"}],
            endpoint_url="http://provider.example/v1",
        )

    with pytest.raises(ValueError, match="credenciales"):
        provider_completion(
            "openai-compatible",
            "test-model",
            [{"role": "user", "content": "Hello"}],
            endpoint_url="https://provider.example/v1/sk-" + "x" * 24,
        )


def test_native_adapter_requires_api_key():
    with pytest.raises(ValueError, match="clave API"):
        provider_completion(
            "gemini",
            "gemini-test",
            [{"role": "user", "content": "Hello"}],
        )


@pytest.mark.parametrize(
    ("provider", "body"),
    [
        ("anthropic", {"content": [{"type": "image"}]}),
        ("gemini", {"candidates": []}),
    ],
)
def test_native_adapters_reject_malformed_responses(provider, body):
    with pytest.raises(ValueError, match="respuesta|contenido"):
        provider_completion(
            provider,
            "test-model",
            [{"role": "user", "content": "Hello"}],
            api_key="provider-secret",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json=body)
            ),
        )
