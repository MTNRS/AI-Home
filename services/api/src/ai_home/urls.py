"""Client URL validation shared by desktop and node clients."""

from urllib.parse import urlsplit, urlunsplit


def normalize_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Introduce una URL HTTP(S) válida, sin credenciales.")
    if parsed.scheme != "https" and parsed.hostname.casefold() not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("Usa HTTPS para servidores fuera de este equipo.")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def normalize_provider_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Introduce una URL de proveedor HTTP(S) válida.")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )
