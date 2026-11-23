"""Direct desktop adapters for external chat providers."""

from __future__ import annotations

from typing import Literal
from urllib.parse import quote

import httpx

from ai_home.inference import extract_chat_content
from ai_home.privacy import contains_common_secret
from ai_home.urls import normalize_base_url

ProviderId = Literal["openai", "anthropic", "gemini", "openai-compatible"]

PROVIDER_LABELS: dict[ProviderId, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "openai-compatible": "OpenAI-compatible / local",
}
PROVIDER_DEFAULTS: dict[ProviderId, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "anthropic": ("https://api.anthropic.com", "claude-3-5-haiku-latest"),
    "gemini": (
        "https://generativelanguage.googleapis.com",
        "gemini-2.0-flash",
    ),
    "openai-compatible": ("", ""),
}
_OFFICIAL_ENDPOINTS: dict[Literal["openai", "anthropic", "gemini"], str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}


def provider_completion(
    provider: ProviderId,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str = "",
    endpoint_url: str | None = None,
    timeout_seconds: float = 180,
    transport: httpx.BaseTransport | None = None,
) -> str:
    if provider not in PROVIDER_LABELS:
        raise ValueError("Selecciona un proveedor compatible.")
    if not model.strip():
        raise ValueError("Introduce el nombre del modelo.")
    if not messages or not any(message["role"] == "user" for message in messages):
        raise ValueError("Escribe un mensaje antes de iniciar la conversación.")
    if provider != "openai-compatible" and not api_key.strip():
        raise ValueError("Añade la clave API del proveedor seleccionado.")

    if provider == "openai-compatible":
        if endpoint_url is None:
            raise ValueError("Introduce la URL del endpoint compatible.")
        endpoint = normalize_base_url(endpoint_url)
        if contains_common_secret(endpoint):
            raise ValueError("La URL del endpoint no debe incluir credenciales.")
        return _openai_compatible_completion(
            endpoint,
            model,
            messages,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )

    endpoint = _OFFICIAL_ENDPOINTS[provider]
    if provider == "openai":
        return _openai_compatible_completion(
            endpoint,
            model,
            messages,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    if provider == "anthropic":
        return _anthropic_completion(
            endpoint,
            model,
            messages,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    return _gemini_completion(
        endpoint,
        model,
        messages,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def _client(
    endpoint: str,
    *,
    headers: dict[str, str],
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> httpx.Client:
    return httpx.Client(
        base_url=endpoint,
        headers=headers,
        timeout=timeout_seconds,
        transport=transport,
    )


def _openai_compatible_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with _client(
        endpoint,
        headers=headers,
        timeout_seconds=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "chat/completions",
            json={"model": model.strip(), "messages": messages},
        )
        response.raise_for_status()
        return extract_chat_content(response.json())


def _anthropic_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> str:
    system_messages = [
        message["content"] for message in messages if message["role"] == "system"
    ]
    request_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message["role"] != "system"
    ]
    body: dict[str, object] = {
        "model": model.strip(),
        "max_tokens": 4096,
        "messages": request_messages,
    }
    if system_messages:
        body["system"] = "\n\n".join(system_messages)
    with _client(
        endpoint,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout_seconds=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post("/v1/messages", json=body)
        response.raise_for_status()
        return _anthropic_text(response.json())


def _anthropic_text(body: object) -> str:
    if not isinstance(body, dict) or not isinstance(body.get("content"), list):
        raise ValueError("El proveedor de chat devolvió una respuesta no válida.")
    text_blocks = [
        block["text"]
        for block in body["content"]
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    if not text_blocks:
        raise ValueError("El proveedor de chat devolvió contenido que no es texto.")
    return "".join(text_blocks)


def _gemini_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> str:
    system_messages = [
        message["content"] for message in messages if message["role"] == "system"
    ]
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
        if message["role"] != "system"
    ]
    body: dict[str, object] = {"contents": contents}
    if system_messages:
        body["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_messages)}]
        }
    with _client(
        endpoint,
        headers={"x-goog-api-key": api_key},
        timeout_seconds=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            f"/v1beta/models/{quote(model.strip(), safe='')}:generateContent",
            json=body,
        )
        response.raise_for_status()
        return _gemini_text(response.json())


def _gemini_text(body: object) -> str:
    if not isinstance(body, dict) or not isinstance(body.get("candidates"), list):
        raise ValueError("El proveedor de chat devolvió una respuesta no válida.")
    candidates = body["candidates"]
    if not candidates or not isinstance(candidates[0], dict):
        raise ValueError("El proveedor de chat devolvió una respuesta no válida.")
    content = candidates[0].get("content")
    if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
        raise ValueError("El proveedor de chat devolvió una respuesta no válida.")
    text_parts = [
        part["text"]
        for part in content["parts"]
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    if not text_parts:
        raise ValueError("El proveedor de chat devolvió contenido que no es texto.")
    return "".join(text_parts)
