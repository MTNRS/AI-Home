"""Client for configured OpenAI-compatible chat providers."""

import httpx

from ai_home.urls import normalize_provider_url


def chat_completion(
    endpoint_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str = "",
    timeout_seconds: float = 180,
    transport: httpx.BaseTransport | None = None,
) -> str:
    if not model.strip():
        raise ValueError("Introduce el nombre del modelo.")
    if not messages or not any(message["role"] == "user" for message in messages):
        raise ValueError("Escribe un mensaje antes de iniciar la conversación.")

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(
        base_url=normalize_provider_url(endpoint_url),
        headers=headers,
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "chat/completions",
            json={"model": model.strip(), "messages": messages},
        )
        response.raise_for_status()
        return extract_chat_content(response.json())


def extract_chat_content(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("El proveedor de chat devolvió una respuesta no válida.")
    try:
        choices = body["choices"]
        if not isinstance(choices, list) or not choices or not isinstance(
            choices[0],
            dict,
        ):
            raise ValueError
        message = choices[0]["message"]
        if not isinstance(message, dict):
            raise ValueError
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("El proveedor de chat devolvió una respuesta no válida.") from error
    except ValueError as error:
        raise ValueError("El proveedor de chat devolvió una respuesta no válida.") from error
    if not isinstance(content, str):
        raise ValueError("El proveedor de chat devolvió contenido que no es texto.")
    return content
