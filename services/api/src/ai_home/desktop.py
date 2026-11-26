"""Desktop client for OpenAI-compatible chat providers and volunteer nodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import os
import queue
import re
import threading
from pathlib import Path
import sys
from typing import TypeGuard, TypeVar
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import keyring

from ai_home.chat_credits import credit_authorization, validate_credit_receipt
from ai_home.chat_stream import GenerationCancelled, stream_server_chat
from ai_home.inference import chat_completion, extract_chat_content
from ai_home.memory import (
    MemoryNote,
    delete_memory,
    load_memories,
    messages_with_memory_context,
    profile_memory_vault,
    retrieve_memories,
    save_memory,
)
from ai_home.node_worker import NodeAgent, register_node_with_server
from ai_home.providers import (
    PROVIDER_DEFAULTS,
    PROVIDER_LABELS,
    ChatProviderId,
    ProviderId,
    TrainingProviderId,
    FoundationChatResult,
    foundation_chat_result,
    provider_completion,
)
from ai_home.privacy import contains_common_secret
from ai_home.schemas import (
    KnowledgeReference,
    FoundationGeneration,
    NodeCredentialResponse,
    SessionResponse,
    TrainingContributionExportResponse,
    TrainingContributionResponse,
)
from ai_home.urls import normalize_base_url

KEYRING_SERVICE = "ai-home-desktop"
_PROVIDER_ID_BY_LABEL = {
    label: provider_id for provider_id, label in PROVIDER_LABELS.items()
}
_OS_KEYRING_MODULES = {
    "keyring.backends.Windows",
    "keyring.backends.macOS",
    "keyring.backends.SecretService",
    "keyring.backends.kwallet",
}
T = TypeVar("T")

_SPHERE_EMOTION_COLORS = {
    "alegria": "#f0b5a8",
    "tristeza": "#a9c8e8",
    "enfado": "#e9a29e",
    "miedo": "#c5b2df",
    "calma": "#a8cfb4",
}
_SPHERE_EMOTION_CUES = {
    "alegria": ("feliz", "alegr", "encantad", "ilusion", "genial", "maravillos"),
    "tristeza": ("triste", "lo siento", "lamento", "pena", "dolor"),
    "enfado": ("enfad", "molest", "frustr", "rabia", "furia"),
    "miedo": ("miedo", "temor", "asust", "preocup", "nervios"),
    "calma": ("tranquil", "calma", "seren", "relaj"),
}


def _sphere_pointer_angles(
    pointer_dx: float,
    pointer_dy: float,
    radius: float,
) -> tuple[float, float]:
    depth = max(radius * 2.8, 1)
    yaw = max(-0.75, min(0.75, math.atan2(pointer_dx, depth)))
    pitch = max(-0.5, min(0.5, -math.atan2(pointer_dy, depth)))
    return yaw, pitch


def _sphere_emotion_for_text(text: str) -> str | None:
    normalized = text.casefold()
    matches = (
        (normalized.find(cue), emotion)
        for emotion, cues in _SPHERE_EMOTION_CUES.items()
        for cue in cues
        if normalized.find(cue) >= 0
    )
    first_match = min(matches, default=None)
    return first_match[1] if first_match is not None else None


def _blend_sphere_color(base: str, tint: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    channels = (
        round(int(base[index : index + 2], 16) * (1 - amount))
        + round(int(tint[index : index + 2], 16) * amount)
        for index in (1, 3, 5)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _is_provider_id(value: str) -> TypeGuard[ChatProviderId]:
    return value in PROVIDER_LABELS


def _is_training_provider(value: str) -> TypeGuard[TrainingProviderId]:
    return value in {"openai", "anthropic", "gemini", "openai-compatible", "ai-home-foundation"}


def _provider_keyring_account(provider: ChatProviderId, endpoint: str) -> str:
    if provider in {"ai-home-local", "ai-home-foundation"}:
        return "local-model"
    if provider == "openai-compatible":
        normalized_endpoint = normalize_base_url(endpoint)
    else:
        normalized_endpoint = PROVIDER_DEFAULTS[provider][0]
    endpoint_hash = hashlib.sha256(normalized_endpoint.encode("utf-8")).hexdigest()
    return f"provider:{provider}:{endpoint_hash}"


def _require_os_keyring() -> None:
    try:
        backend = keyring.get_keyring()
    except keyring.errors.KeyringError as error:
        raise RuntimeError(
            "No se pudo acceder al almacén seguro del sistema operativo."
        ) from error
    if type(backend).__module__ not in _OS_KEYRING_MODULES:
        raise RuntimeError(
            "No se detectó un almacén seguro del sistema operativo compatible. "
            "No se guardó ni se leyó ninguna clave."
        )


def get_provider_api_key(provider: ChatProviderId, endpoint: str) -> str:
    _require_os_keyring()
    try:
        return (
            keyring.get_password(
                KEYRING_SERVICE,
                _provider_keyring_account(provider, endpoint),
            )
            or ""
        )
    except keyring.errors.KeyringError as error:
        raise RuntimeError(
            "No se pudo acceder al almacén seguro del sistema. "
            "No se guardó ni se usó la clave."
        ) from error


def save_provider_api_key(
    provider: ChatProviderId,
    endpoint: str,
    api_key: str,
) -> None:
    if not api_key.strip():
        raise ValueError("Introduce una clave API antes de guardarla.")
    _require_os_keyring()
    try:
        keyring.set_password(
            KEYRING_SERVICE,
            _provider_keyring_account(provider, endpoint),
            api_key,
        )
    except keyring.errors.KeyringError as error:
        raise RuntimeError(
            "No se pudo guardar la clave en el almacén seguro del sistema."
        ) from error


def delete_provider_api_key(provider: ChatProviderId, endpoint: str) -> bool:
    _require_os_keyring()
    try:
        keyring.delete_password(
            KEYRING_SERVICE,
            _provider_keyring_account(provider, endpoint),
        )
    except keyring.errors.PasswordDeleteError:
        return False
    except keyring.errors.KeyringError as error:
        raise RuntimeError(
            "No se pudo acceder al almacén seguro del sistema."
        ) from error
    return True


def submit_training_candidate(
    server_url: str,
    user_token: str,
    conversation_id: str,
    provider: TrainingProviderId,
    source_endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    reviewed_for_submission: bool,
    timeout_seconds: float = 20,
    transport: httpx.BaseTransport | None = None,
) -> TrainingContributionResponse:
    with httpx.Client(
        base_url=normalize_base_url(server_url),
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "/api/v1/training/conversations",
            json={
                "conversation_id": conversation_id,
                "provider": provider,
                "source_endpoint": source_endpoint,
                "model": model,
                "messages": messages,
                "consent": True,
                "consent_version": "training_consent_v1",
                "consent_scope": "conversation_for_aihome_training_candidates",
                "provider_terms_allow_reuse": reviewed_for_submission,
                "content_reviewed_for_sensitive_data": reviewed_for_submission,
            },
        )
        response.raise_for_status()
        return TrainingContributionResponse.model_validate(response.json())


def list_training_candidates(
    server_url: str,
    user_token: str,
    *,
    limit: int = 50,
    offset: int = 0,
    timeout_seconds: float = 20,
    transport: httpx.BaseTransport | None = None,
) -> list[TrainingContributionResponse]:
    with httpx.Client(
        base_url=normalize_base_url(server_url),
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.get(
            "/api/v1/training/conversations",
            params={"limit": limit, "offset": offset},
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, list):
        raise ValueError("El servidor devolvió una lista de aportaciones no válida.")
    return [TrainingContributionResponse.model_validate(item) for item in body]


def submit_training_correction(server_url, user_token, contribution_id, message_index, replacement,
                               *, consent, reviewed, conversation_id, transport=None):
    from ai_home.schemas import TrainingCorrectionRequest
    payload = TrainingCorrectionRequest(conversation_id=conversation_id, message_index=message_index,
        replacement=replacement, consent=consent, consent_version="training_consent_v1",
        consent_scope="conversation_for_aihome_training_candidates", provider_terms_allow_reuse=reviewed,
        content_reviewed_for_sensitive_data=reviewed)
    if not consent or not reviewed:
        raise ValueError("Renueva consentimiento y revisión antes de enviar la corrección.")
    with httpx.Client(base_url=normalize_base_url(server_url), timeout=20, transport=transport,
                      headers={"Authorization": f"Bearer {user_token}"}) as client:
        response = client.post(f"/api/v1/training/conversations/{contribution_id}/corrections",
                               json=payload.model_dump(mode="json"))
        response.raise_for_status()
        return TrainingContributionResponse.model_validate(response.json())


def get_training_candidate(
    server_url: str,
    user_token: str,
    contribution_id: str,
    *,
    timeout_seconds: float = 20,
    transport: httpx.BaseTransport | None = None,
) -> TrainingContributionExportResponse:
    with httpx.Client(
        base_url=normalize_base_url(server_url),
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.get(
            f"/api/v1/training/conversations/{contribution_id}"
        )
        response.raise_for_status()
        return TrainingContributionExportResponse.model_validate(response.json())


def withdraw_training_candidate(
    server_url: str,
    user_token: str,
    contribution_id: str,
    *,
    timeout_seconds: float = 20,
    transport: httpx.BaseTransport | None = None,
) -> None:
    with httpx.Client(
        base_url=normalize_base_url(server_url),
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.delete(
            f"/api/v1/training/conversations/{contribution_id}"
        )
        response.raise_for_status()


def login_to_server(
    server_url: str,
    email: str,
    password: str,
    *,
    timeout_seconds: float = 20,
    transport: httpx.BaseTransport | None = None,
) -> SessionResponse:
    base_url = normalize_base_url(server_url)
    with httpx.Client(
        base_url=base_url,
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        _raise_for_auth_response(response)
        return SessionResponse.model_validate(response.json())


def register_account(
    server_url: str,
    email: str,
    password: str,
    *,
    timeout_seconds: float = 20,
    transport: httpx.BaseTransport | None = None,
) -> None:
    base_url = normalize_base_url(server_url)
    with httpx.Client(
        base_url=base_url,
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )
        _raise_for_auth_response(response)


def _raise_for_auth_response(response: httpx.Response) -> None:
    if not response.is_error:
        return
    try:
        body = response.json()
    except ValueError:
        body = {}
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        raise RuntimeError(body["detail"])
    response.raise_for_status()


@dataclass(frozen=True)
class ServerChatResult:
    answer: str
    references: tuple[KnowledgeReference, ...] = ()
    model_version: str | None = None
    charged_credits: int = 0


def server_stream_result(server_url, user_token, messages, **options):
    result = stream_server_chat(server_url, user_token, messages, **options)
    return ServerChatResult(result["answer"], tuple(KnowledgeReference.model_validate(r)
                            for r in result["references"]), result["model_version"], result.get("charged_credits", 0))


def server_chat_completion(server_url, user_token, messages, **options) -> str:
    return server_chat_result(server_url, user_token, messages, **options).answer


def server_chat_result(
    server_url: str,
    user_token: str,
    messages: list[dict[str, str]],
    *,
    credit_limit: int = 0,
    credit_request_id: str | None = None,
    generation: dict | None = None,
    use_knowledge: bool = False,
    model_version: str | None = None,
    timeout_seconds: float = 180,
    transport: httpx.BaseTransport | None = None,
) -> ServerChatResult:
    authorization = credit_authorization(credit_limit, credit_request_id)
    with httpx.Client(
        base_url=normalize_base_url(server_url),
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={"messages": messages, **authorization,
                  **({"generation": FoundationGeneration.model_validate(generation).model_dump(exclude_none=True)} if generation is not None else {}),
                  **({"use_knowledge": True} if use_knowledge else {}),
                  **({"model_version": str(UUID(model_version))} if model_version else {})},
        )
        response.raise_for_status()
        data = response.json()
        references = tuple(KnowledgeReference.model_validate(item) for item in data.get("references", []))
        actual_version = str(UUID(data["model_version"])) if data.get("model_version") else None
        if model_version is not None and actual_version != str(UUID(model_version)):
            raise ValueError("El servidor respondió con una versión distinta de la solicitada.")
        return ServerChatResult(extract_chat_content(data), references, actual_version, validate_credit_receipt(data, authorization))


DEFAULT_SERVER_VERSION = "Versión activa del servidor"


def server_model_versions(server_url, user_token, *, transport=None):
    with httpx.Client(base_url=normalize_base_url(server_url), timeout=20,
                      headers={"Authorization": f"Bearer {user_token}"}, transport=transport) as client:
        response = client.get("/api/v1/models")
        response.raise_for_status()
        models = response.json()["models"]
        return [str(UUID(model["id"])) for model in models]


def server_inference_price(server_url, user_token, *, transport=None):
    with httpx.Client(base_url=normalize_base_url(server_url), timeout=20,
                      headers={"Authorization": f"Bearer {user_token}"}, transport=transport) as client:
        response = client.get("/api/v1/credits/inference-price")
        response.raise_for_status()
        price = response.json()["credits_per_completion"]
        credit_authorization(price)
        return price


def server_credits(server_url, user_token, *, transport=None):
    with httpx.Client(base_url=normalize_base_url(server_url), timeout=20,
                      headers={"Authorization": f"Bearer {user_token}"}, transport=transport) as client:
        balance_response = client.get("/api/v1/credits")
        balance_response.raise_for_status()
        transactions_response = client.get("/api/v1/credits/transactions")
        transactions_response.raise_for_status()
    balance, transactions = balance_response.json(), transactions_response.json()
    if not isinstance(balance, dict) or any(type(balance.get(key)) is not int or balance[key] < 0
                                          for key in ("available", "reserved", "earned")):
        raise ValueError("El servidor devolvió un saldo de créditos inválido.")
    if not isinstance(transactions, list) or len(transactions) > 100:
        raise ValueError("El servidor devolvió movimientos inválidos.")
    for row in transactions:
        if (not isinstance(row, dict) or type(row.get("amount")) is not int
                or type(row.get("transaction_id")) is not int
                or not isinstance(row.get("operation"), str) or not isinstance(row.get("created_at"), str)):
            raise ValueError("El servidor devolvió un movimiento inválido.")
    return balance, transactions


def format_credits(balance, transactions):
    labels = {"job_reserve": "Reserva para trabajo", "job_settle": "Trabajo verificado",
              "job_refund": "Reembolso de trabajo"}
    lines = [f"Disponibles: {balance['available']}", f"Reservados: {balance['reserved']}",
             f"Ganados por trabajo para otras cuentas: {balance['earned']}", "",
             "Créditos computacionales internos; no son dinero ni criptomonedas.",
             "Saldo y movimientos consultados por separado; pueden cambiar entre consultas.",
             "Últimos movimientos (hasta 100, más recientes primero):", ""]
    if not transactions:
        lines.append("No hay movimientos registrados.")
    for row in transactions:
        label = labels.get(row["operation"], row["operation"])
        if row["operation"].startswith("operator_grant:"):
            label = "Asignación del operador: " + row["operation"].split(":", 1)[1]
        lines.append(f"{row['created_at']} | {row['amount']:+d} | {label} | ID {row['transaction_id']}")
    return "\n".join(lines)


def parse_generation_settings(values):
    parsed = {}
    for key in ("max_new_tokens", "top_k", "seed"):
        value = values[key].strip()
        if key == "max_new_tokens" and not value:
            continue
        parsed[key] = int(value)
    for key in ("temperature", "top_p", "repetition_penalty"):
        parsed[key] = float(values[key].strip())
    return FoundationGeneration.model_validate(parsed).model_dump(exclude_none=True)


def format_knowledge_references(turns) -> str:
    lines = ["Fragmentos enviados al modelo; no garantizan que su respuesta sea correcta.",
             "Conocimiento recuperado, separado de los pesos y de la memoria personal.", ""]
    if not turns:
        lines.append("Todavía no se ha utilizado recuperación en esta conversación.")
    for message_index, references in turns:
        lines.append(f"Respuesta en el mensaje {message_index + 1}")
        if not references:
            lines.append("No se encontraron fragmentos relevantes.")
        for number, reference in enumerate(references, 1):
            lines.extend([f"[{number}] {reference.source}", reference.url,
                          f"Licencia: {reference.license} · Fecha: {reference.date} · Idioma: {reference.language}",
                          f"Documento: {reference.document_sha256} · Posición: {reference.offset}",
                          f"Índice: {reference.index_version}", reference.text, ""])
    return "\n".join(lines)


def settings_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "ai-home" / "desktop.json"


def _is_legacy_ollama_endpoint(value: str) -> bool:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.hostname is not None
        and parsed.hostname.casefold() in {"localhost", "127.0.0.1", "::1"}
        and port == 11434
    )


def load_settings() -> dict[str, str]:
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read desktop settings at {path}.") from error
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in data.items()
    ):
        raise RuntimeError(f"Desktop settings at {path} have an invalid format.")
    if _is_legacy_ollama_endpoint(data.get("chat_endpoint", "")):
        data.pop("chat_endpoint", None)
        data.pop("chat_model", None)
        data.pop("chat_model_openai-compatible", None)
        data["chat_provider"] = "ai-home-local"
        save_settings(data)
    elif (
        data.get("chat_provider") == "openai-compatible"
        and not data.get("chat_endpoint", "").strip()
    ):
        data["chat_provider"] = "ai-home-local"
        save_settings(data)
    return data


def save_settings(settings: dict[str, str]) -> None:
    path = settings_path()
    temporary_path = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(settings, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if os.name != "nt":
            temporary_path.chmod(0o600)
        temporary_path.replace(path)
    except OSError as error:
        raise RuntimeError(f"Could not save desktop settings at {path}.") from error


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class AIHomeDesktop(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("AI@Home")
            self.geometry("900x700")
            self.minsize(720, 560)
            self.settings = load_settings()
            self.events: queue.Queue[tuple[str, object]] = queue.Queue()
            self.session_token: str | None = None
            self.session_server_url: str | None = None
            self.guest_session_active = False
            self.memory_vault = profile_memory_vault()
            self.private_windows = []
            self.messages: list[dict[str, str]] = []
            self.knowledge_references = []
            self.conversation_used_knowledge = False
            self.pinned_server_version = None
            self.pinned_foundation_checkpoint = None
            self.conversation_id = str(uuid4())
            self.conversation_source: tuple[str, str, str] | None = None
            self.generation_settings = None
            self.chat_credit_limit = 0
            self.chat_cancel = None
            self.chat_request_active = False
            self.training_submission_active = False
            self.training_candidate_submitted = False
            self.node_registration_active = False
            self.node_stop: threading.Event | None = None
            self.node_thread: threading.Thread | None = None
            self.sphere_yaw = 0.0
            self.sphere_pitch = 0.0
            self.sphere_emotion: str | None = None
            self.status_var = tk.StringVar(value="Listo para empezar")
            self._build_ui(ttk)
            self.protocol("WM_DELETE_WINDOW", self._close)
            self.after(100, self._drain_events)
            self.after(24, self._orient_sphere_to_pointer)

        def _build_ui(self, ttk) -> None:
            header = ttk.Frame(self, padding=(18, 14, 18, 8))
            header.pack(fill="x")
            ttk.Label(
                header,
                text="AI@Home",
                font=("Segoe UI", 18, "bold"),
            ).pack(anchor="w")
            ttk.Label(
                header,
                text="Tu espacio para conversar con IA y colaborar, si quieres.",
            ).pack(anchor="w", pady=(2, 0))

            self.page_stack = ttk.Notebook(self)
            style = ttk.Style(self)
            style.layout("AIHome.TNotebook.Tab", [])
            style.configure("AIHome.TNotebook", borderwidth=0)
            self.page_stack.configure(style="AIHome.TNotebook")
            self.page_stack.pack(fill="both", expand=True)
            chat_tab = ttk.Frame(self.page_stack)
            provider_tab = ttk.Frame(self.page_stack)
            account_tab = ttk.Frame(self.page_stack)
            node_tab = ttk.Frame(self.page_stack)
            self.page_stack.add(chat_tab, text="Chat")
            self.page_stack.add(provider_tab, text="Conectar IA")
            self.page_stack.add(account_tab, text="Cuenta y aportes")
            self.page_stack.add(node_tab, text="Colaborar")
            self._build_chat_tab(ttk, chat_tab)
            self._build_provider_tab(ttk, self._scrollable_page(provider_tab))
            self._build_account_tab(ttk, self._scrollable_page(account_tab))
            self._build_node_tab(ttk, self._scrollable_page(node_tab))
            nav = ttk.Frame(header)
            nav.pack(side="right", anchor="n")
            ttk.Button(
                nav,
                text="Chat",
                command=lambda: self._show_page(chat_tab),
            ).pack(side="left", padx=3)
            ttk.Button(
                nav,
                text="Nueva conversación",
                command=self._clear_chat,
            ).pack(side="left", padx=3)
            ttk.Button(
                nav,
                text="Memoria local",
                command=self._manage_memories,
            ).pack(side="left", padx=3)
            ttk.Button(
                nav,
                text="Conectar IA",
                command=lambda: self._show_page(provider_tab),
            ).pack(side="left", padx=3)
            ttk.Button(
                nav,
                text="Cuenta",
                command=lambda: self._show_page(account_tab),
            ).pack(side="left", padx=3)
            ttk.Button(
                nav,
                text="Colaborar",
                command=lambda: self._show_page(node_tab),
            ).pack(side="left", padx=3)
            ttk.Label(
                self,
                textvariable=self.status_var,
                anchor="w",
                padding=(16, 3),
                foreground="#62676b",
            ).pack(side="bottom", fill="x", padx=8, pady=(0, 3), before=self.page_stack)

        def _scrollable_page(self, parent):
            canvas = tk.Canvas(parent, highlightthickness=0, background="#eeeeee", height=400)
            scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            canvas.configure(yscrollcommand=scrollbar.set)
            content = ttk.Frame(canvas, padding=20)
            item = canvas.create_window(0, 0, window=content, anchor="nw")
            content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))

            def resize(event):
                canvas.itemconfigure(item, width=event.width)
                def wrap(widget):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Label) and int(child.cget("wraplength") or 0):
                            child.configure(wraplength=max(160, event.width - 80))
                        wrap(child)
                wrap(content)
            canvas.bind("<Configure>", resize)

            def wheel(event):
                if self.page_stack.select() == str(parent) and not isinstance(event.widget, (tk.Text, tk.Listbox, ttk.Combobox)):
                    canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            self.bind("<MouseWheel>", wheel, add="+")
            return content

        def _show_page(self, page) -> None:
            self.page_stack.select(page)

        def _layout_home(self, _event=None) -> None:
            canvas = self.face_canvas
            width = max(canvas.winfo_width(), 1)
            center_x = width / 2
            center_y = 76
            orb_radius = min(62, width * 0.14)
            canvas.delete("face")

            canvas.create_oval(
                center_x - orb_radius * 0.9,
                center_y + orb_radius * 0.72,
                center_x + orb_radius * 0.95,
                center_y + orb_radius * 1.05,
                fill="#d9d9d9",
                outline="#d9d9d9",
                tags=("face", "head"),
            )
            for layer in range(40, -1, -1):
                ratio = (40 - layer) / 40
                shade = round(211 + ratio * 44)
                light_x = -0.10 + 0.23 * math.cos(self.sphere_yaw)
                red = min(255, shade + round(ratio * 1))
                green = min(255, shade + round(ratio * 2))
                blue = min(255, shade + round(ratio * 4))
                color = f"#{red:02x}{green:02x}{blue:02x}"
                inset = layer * orb_radius / 48
                highlight_x = center_x + orb_radius * light_x * ratio
                highlight_y = center_y + orb_radius * (
                    -0.10 + 0.16 * math.sin(self.sphere_pitch)
                ) * ratio
                canvas.create_oval(
                    highlight_x - orb_radius + inset,
                    highlight_y - orb_radius + inset,
                    highlight_x + orb_radius - inset,
                    highlight_y + orb_radius - inset,
                    fill=color,
                    outline=color,
                    tags=("face", "head"),
                )

            emotion_color = _SPHERE_EMOTION_COLORS.get(self.sphere_emotion)
            if emotion_color is not None:
                blush_color = _blend_sphere_color("#f5f2ef", emotion_color, 0.28)
                for cheek_x in (-0.48, 0.48):
                    cheek_y = 0.30
                    cheek_z = math.sqrt(1 - cheek_x**2 - cheek_y**2)
                    rotated_x = (
                        cheek_x * math.cos(self.sphere_yaw)
                        + cheek_z * math.sin(self.sphere_yaw)
                    )
                    rotated_z = (
                        -cheek_x * math.sin(self.sphere_yaw)
                        + cheek_z * math.cos(self.sphere_yaw)
                    )
                    projected_y = (
                        cheek_y * math.cos(self.sphere_pitch)
                        - rotated_z * math.sin(self.sphere_pitch)
                    )
                    cheek_depth = (
                        cheek_y * math.sin(self.sphere_pitch)
                        + rotated_z * math.cos(self.sphere_pitch)
                    )
                    if cheek_depth <= 0.04:
                        continue
                    perspective = min(1.0, cheek_depth / cheek_z)
                    cheek_center_x = center_x + rotated_x * orb_radius
                    cheek_center_y = center_y + projected_y * orb_radius
                    cheek_width = orb_radius * 0.22 * perspective
                    cheek_height = orb_radius * 0.13 * perspective
                    canvas.create_oval(
                        cheek_center_x - cheek_width,
                        cheek_center_y - cheek_height,
                        cheek_center_x + cheek_width,
                        cheek_center_y + cheek_height,
                        fill=blush_color,
                        outline=blush_color,
                        tags=("face", "blush"),
                    )

            for eye_x, eye_scale in (
                (-0.34, 0.26),
                (0.25, 0.30),
            ):
                eye_y = -0.02
                eye_z = math.sqrt(1 - eye_x**2 - eye_y**2)
                rotated_x = (
                    eye_x * math.cos(self.sphere_yaw)
                    + eye_z * math.sin(self.sphere_yaw)
                )
                rotated_z = (
                    -eye_x * math.sin(self.sphere_yaw)
                    + eye_z * math.cos(self.sphere_yaw)
                )
                projected_y = (
                    eye_y * math.cos(self.sphere_pitch)
                    - rotated_z * math.sin(self.sphere_pitch)
                )
                eye_depth = (
                    eye_y * math.sin(self.sphere_pitch)
                    + rotated_z * math.cos(self.sphere_pitch)
                )
                if eye_depth <= 0.04:
                    continue
                projected_x = center_x + rotated_x * orb_radius
                projected_eye_y = center_y + projected_y * orb_radius
                perspective = min(1.0, eye_depth / eye_z)
                eye_width = orb_radius * eye_scale * perspective
                eye_height = eye_width * 1.26
                canvas.create_oval(
                    projected_x - eye_width / 2,
                    projected_eye_y - eye_height / 2,
                    projected_x + eye_width / 2,
                    projected_eye_y + eye_height / 2,
                    fill="#76533c",
                    outline="#76533c",
                    tags=("face", "head"),
                )
                pupil = canvas.create_oval(
                    projected_x - eye_width * 0.32,
                    projected_eye_y - eye_height * 0.40,
                    projected_x + eye_width * 0.32,
                    projected_eye_y + eye_height * 0.40,
                    fill="#241b17",
                    outline="#241b17",
                    tags=("face", "head"),
                )
                glint = canvas.create_oval(
                    projected_x - eye_width * 0.24,
                    projected_eye_y - eye_height * 0.31,
                    projected_x - eye_width * 0.04,
                    projected_eye_y - eye_height * 0.10,
                    fill="#ffffff",
                    outline="#ffffff",
                    tags=("face", "head"),
                )
        def _orient_sphere_to_pointer(self) -> None:
            if not self.winfo_exists():
                return
            if self.face_canvas.winfo_ismapped():
                pointer_x, pointer_y = self.winfo_pointerxy()
                center_x = self.face_canvas.winfo_rootx() + (
                    self.face_canvas.winfo_width() / 2
                )
                center_y = self.face_canvas.winfo_rooty() + 76
                target_yaw, target_pitch = _sphere_pointer_angles(
                    pointer_x - center_x,
                    pointer_y - center_y,
                    min(62, self.face_canvas.winfo_width() * 0.14),
                )
                next_yaw = self.sphere_yaw + (target_yaw - self.sphere_yaw) * 0.27
                next_pitch = self.sphere_pitch + (
                    target_pitch - self.sphere_pitch
                ) * 0.27
                if (
                    abs(next_yaw - self.sphere_yaw) > 0.001
                    or abs(next_pitch - self.sphere_pitch) > 0.001
                ):
                    self.sphere_yaw = next_yaw
                    self.sphere_pitch = next_pitch
                    self._layout_home()
            self.after(24, self._orient_sphere_to_pointer)

        def _chat_canvas_resized(self, event) -> None:
            previous_bottom = self.chat_messages_canvas.canvasy(
                self.chat_messages_canvas.winfo_height()
            )
            self._render_chat_canvas()
            bounds = self.chat_messages_canvas.bbox("all")
            if bounds and previous_bottom >= bounds[3] - 80:
                self.chat_messages_canvas.yview_moveto(1)

        def _render_chat_canvas(self) -> None:
            canvas = self.chat_messages_canvas
            width = max(canvas.winfo_width(), 1)
            viewport_height = max(canvas.winfo_height(), 1)
            graph_height = max(viewport_height, len(self.messages) * 92 + 140)
            canvas.delete("all")

            notes = self.memory_notes
            points: dict[str, tuple[float, float]] = {}
            for index, note in enumerate(notes):
                digest = hashlib.sha256(note.id.encode("ascii")).digest()
                points[note.id] = (
                    30 + digest[0] / 255 * max(width - 60, 1),
                    30
                    + (index + 0.5) * max(graph_height - 60, 1) / max(len(notes), 1),
                )

            drawn_edges: set[tuple[str, str]] = set()
            for note in notes:
                for linked_id in note.links:
                    if linked_id not in points:
                        continue
                    edge = tuple(sorted((note.id, linked_id)))
                    if edge in drawn_edges:
                        continue
                    drawn_edges.add(edge)
                    connected = (
                        note.id in self.active_memory_ids
                        or linked_id in self.active_memory_ids
                    )
                    start_x, start_y = points[note.id]
                    end_x, end_y = points[linked_id]
                    canvas.create_line(
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                        fill="#c6a960" if connected else "#dce0dc",
                        width=2 if connected else 1,
                        tags=("memory_graph",),
                    )

            if not notes and not self.messages:
                canvas.create_text(
                    width / 2,
                    viewport_height / 2,
                    text="Tu grafo personal aparecerá aquí\nAñade recuerdos desde «Memoria local»",
                    justify="center",
                    fill="#a0a5a1",
                    font=("Segoe UI", 10),
                    tags=("memory_graph",),
                )

            for note in notes:
                node_x, node_y = points[note.id]
                active = note.id in self.active_memory_ids
                node_color = "#bd9140" if active else "#cbd2cc"
                if active:
                    canvas.create_oval(
                        node_x - 12,
                        node_y - 12,
                        node_x + 12,
                        node_y + 12,
                        outline="#dfc98e",
                        width=2,
                        tags=("memory_graph",),
                    )
                canvas.create_oval(
                    node_x - (6 if active else 4),
                    node_y - (6 if active else 4),
                    node_x + (6 if active else 4),
                    node_y + (6 if active else 4),
                    fill=node_color,
                    outline=node_color,
                    tags=("memory_graph", f"memory:{note.id}"),
                )
                canvas.create_text(
                    node_x + 10,
                    node_y - 2,
                    text=note.title,
                    anchor="w",
                    fill="#8a938c" if active else "#c0c6c1",
                    font=("Segoe UI", 8),
                    tags=("memory_graph", f"memory:{note.id}"),
                )
            margin = 30
            text_width = max(160, min(width * 0.67, width - margin * 2))
            y = 16
            for message in self.messages:
                is_user = message["role"] == "user"
                speaker = "Tú" if is_user else "AI@Home"
                anchor = "ne" if is_user else "nw"
                x = width - margin if is_user else margin
                canvas.create_text(
                    x,
                    y,
                    text=speaker,
                    anchor=anchor,
                    fill="#777777",
                    font=("Segoe UI", 9, "bold"),
                    tags=("chat_text",),
                )
                y += 17
                text_item = canvas.create_text(
                    x,
                    y,
                    text=message["content"],
                    width=text_width,
                    anchor=anchor,
                    justify="right" if is_user else "left",
                    fill="#262626",
                    font=("Segoe UI", 10),
                    tags=("chat_text",),
                )
                bounds = canvas.bbox(text_item)
                y = (bounds[3] if bounds else y + 24) + 22

            content_height = max(graph_height, y)
            canvas.configure(scrollregion=(0, 0, width, content_height))
            canvas.tag_raise("chat_text")

        def _memory_graph_clicked(self, event) -> None:
            item = self.chat_messages_canvas.find_withtag("current")
            if not item:
                return
            for tag in self.chat_messages_canvas.gettags(item[0]):
                if tag.startswith("memory:"):
                    note_id = tag.partition(":")[2]
                    note = next(
                        (candidate for candidate in self.memory_notes if candidate.id == note_id),
                        None,
                    )
                    if note is not None:
                        self._show_memory(note)
                    return

        def _show_memory(self, note: MemoryNote) -> None:
            messagebox.showinfo(note.title, note.content, parent=self)

        def _reload_memories(self) -> None:
            self.memory_notes = load_memories(self.memory_vault)
            self.active_memory_ids.intersection_update(
                note.id for note in self.memory_notes
            )
            self._render_chat_canvas()

        def _manage_memories(self) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Memoria personal local")
            window.geometry("700x440")
            window.transient(self)
            ttk.Label(
                window,
                text=f"Recuerdos guardados solo en este equipo: {self.memory_vault}",
                wraplength=660,
            ).pack(anchor="w", padx=12, pady=(12, 8))
            body = ttk.Frame(window, padding=(12, 0, 12, 8))
            body.pack(fill="both", expand=True)
            memories_list = tk.Listbox(
                body,
                width=30,
                exportselection=False,
                activestyle="dotbox",
            )
            memories_list.pack(side="left", fill="y")
            details = tk.Text(body, wrap="word", state="disabled")
            details.pack(side="left", fill="both", expand=True, padx=(10, 0))
            actions = ttk.Frame(window, padding=(12, 0, 12, 12))
            actions.pack(fill="x")

            def refresh(selected_id: str | None = None) -> None:
                self._reload_memories()
                memories_list.delete(0, "end")
                for index, note in enumerate(self.memory_notes):
                    memories_list.insert("end", note.title)
                    if note.id == selected_id:
                        memories_list.selection_set(index)
                        memories_list.activate(index)
                show_selected()

            def show_selected(_event=None) -> None:
                selected = memories_list.curselection()
                details.configure(state="normal")
                details.delete("1.0", "end")
                if selected and selected[0] < len(self.memory_notes):
                    note = self.memory_notes[selected[0]]
                    linked = [
                        candidate.title
                        for candidate in self.memory_notes
                        if candidate.id in note.links
                    ]
                    details.insert(
                        "end",
                        f"{note.title}\n\n{note.content}",
                    )
                    if linked:
                        details.insert("end", "\n\nConectado con: " + ", ".join(linked))
                details.configure(state="disabled")

            def add_note() -> None:
                self._add_memory_dialog(window, lambda note: refresh(note.id))

            def remove_note() -> None:
                selected = memories_list.curselection()
                if not selected:
                    return
                note = self.memory_notes[selected[0]]
                if not messagebox.askyesno(
                    "Eliminar recuerdo",
                    f"¿Eliminar «{note.title}» del equipo?",
                    parent=window,
                ):
                    return
                delete_memory(note.id, vault_path=self.memory_vault)
                refresh()

            memories_list.bind("<<ListboxSelect>>", show_selected)
            ttk.Button(
                actions,
                text="Añadir recuerdo",
                command=add_note,
            ).pack(side="left")
            ttk.Button(
                actions,
                text="Eliminar seleccionado",
                command=remove_note,
            ).pack(side="left", padx=8)
            ttk.Button(
                actions,
                text="Cerrar",
                command=window.destroy,
            ).pack(side="right")
            refresh()

        def _add_memory_dialog(self, parent, on_saved) -> None:
            window = tk.Toplevel(parent)
            self.private_windows.append(window)
            window.title("Añadir recuerdo")
            window.geometry("540x500")
            window.transient(parent)
            window.grab_set()
            content = ttk.Frame(window, padding=14)
            content.pack(fill="both", expand=True)
            ttk.Label(content, text="Título").pack(anchor="w")
            title = ttk.Entry(content)
            title.pack(fill="x", pady=(3, 10))
            ttk.Label(content, text="Recuerdo").pack(anchor="w")
            body = tk.Text(content, height=8, wrap="word")
            body.pack(fill="both", expand=True, pady=(3, 10))
            ttk.Label(content, text="Conectar con recuerdos existentes (opcional)").pack(
                anchor="w"
            )
            links = tk.Listbox(
                content,
                height=6,
                selectmode="multiple",
                exportselection=False,
            )
            links.pack(fill="x", pady=(3, 10))
            existing = list(self.memory_notes)
            for note in existing:
                links.insert("end", note.title)

            def save_note() -> None:
                selected_links = tuple(
                    existing[index].id for index in links.curselection()
                )
                try:
                    note = save_memory(
                        title.get(),
                        body.get("1.0", "end"),
                        links=selected_links,
                        vault_path=self.memory_vault,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    messagebox.showerror("Memoria local", str(error), parent=window)
                    return
                on_saved(note)
                window.destroy()

            actions = ttk.Frame(content)
            actions.pack(fill="x")
            ttk.Button(actions, text="Guardar en este equipo", command=save_note).pack(
                side="left"
            )
            ttk.Button(actions, text="Cancelar", command=window.destroy).pack(
                side="right"
            )

        def _scroll_chat(self, event):
            if getattr(event, "num", None) == 4:
                direction = -1
            elif getattr(event, "num", None) == 5:
                direction = 1
            else:
                direction = -int(event.delta / 120)
            if direction:
                self.chat_messages_canvas.yview_scroll(direction, "units")
            return "break"

        def _build_account_tab(self, ttk, parent) -> None:
            ttk.Label(
                parent,
                text=(
                    "Puedes usar AI@Home como invitado. Inicia sesión solo "
                    "para las funciones de la comunidad."
                ),
                wraplength=760,
            ).pack(anchor="w", pady=(0, 10))
            self.account_status_var = tk.StringVar(
                value="Sin cuenta iniciada. Puedes continuar como invitado."
            )
            ttk.Label(
                parent,
                textvariable=self.account_status_var,
                wraplength=760,
            ).pack(anchor="w", pady=(0, 8))
            account = ttk.LabelFrame(parent, text="Cuenta AI@Home", padding=12)
            account.pack(fill="x", pady=(0, 12))
            self.server_var = tk.StringVar(
                value=self.settings.get("server_url", "http://localhost:8000")
            )
            self.email_var = tk.StringVar(value=self.settings.get("email", ""))
            self.password_var = tk.StringVar()
            self._entry_row(ttk, account, 0, "Servidor", self.server_var)
            self._entry_row(ttk, account, 1, "Correo", self.email_var)
            self._entry_row(
                ttk,
                account,
                2,
                "Contraseña",
                self.password_var,
                show="*",
            )
            ttk.Button(account, text="Entrar", command=self._login).grid(
                row=0, column=2, padx=6, pady=3
            )
            ttk.Button(
                account,
                text="Crear cuenta",
                command=self._register_account,
            ).grid(
                row=1, column=2, padx=6, pady=3
            )
            ttk.Button(account, text="Cerrar sesión", command=self._logout).grid(
                row=2, column=2, padx=6, pady=3
            )
            ttk.Button(account, text="Créditos y movimientos", command=self._show_credits).grid(
                row=4, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 3)
            )
            account.columnconfigure(1, weight=1)
            ttk.Button(
                account,
                text="Continuar como invitado",
                command=self._start_guest_session,
            ).grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(8, 3))

            contributions = ttk.LabelFrame(
                parent,
                text="Aportar una conversación (opcional)",
                padding=12,
            )
            contributions.pack(fill="x")
            ttk.Label(
                contributions,
                text=(
                    "Solo se envía si lo eliges. El contenido se guarda para "
                    "revisión por el operador de tu servidor; no se entrena "
                    "automáticamente y puedes retirarlo antes de congelarse."
                ),
                wraplength=760,
            ).pack(anchor="w", pady=(0, 8))
            self.training_consent_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                contributions,
                text="Quiero aportar esta conversación como candidata.",
                variable=self.training_consent_var,
            ).pack(anchor="w", pady=2)
            ttk.Label(
                contributions,
                text=(
                    "Antes, comprueba que no hay datos sensibles y que el "
                    "proveedor permite compartir los mensajes."
                ),
                wraplength=760,
            ).pack(anchor="w", pady=(6, 0))
            self.training_review_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                contributions,
                text="He revisado los mensajes y puedo compartirlos.",
                variable=self.training_review_var,
            ).pack(anchor="w", pady=2)
            actions = ttk.Frame(contributions)
            actions.pack(anchor="w", pady=(8, 0))
            ttk.Button(
                actions,
                text="Enviar para revisión",
                command=self._submit_training_candidate,
            ).pack(side="left", padx=(0, 8))
            ttk.Button(
                actions,
                text="Ver o retirar mis aportes",
                command=self._manage_training_candidates,
            ).pack(side="left")

        def _entry_row(
            self,
            ttk,
            parent,
            row: int,
            label: str,
            variable: tk.StringVar,
            *,
            show: str | None = None,
        ) -> None:
            ttk.Label(parent, text=label).grid(
                row=row,
                column=0,
                sticky="w",
                padx=6,
                pady=3,
            )
            ttk.Entry(parent, textvariable=variable, show=show).grid(
                row=row,
                column=1,
                sticky="ew",
                padx=6,
                pady=3,
            )

        def _build_chat_tab(self, ttk, parent) -> None:
            parent.configure(style="TFrame")
            self.face_canvas = tk.Canvas(
                parent,
                height=154,
                background="#eeeeee",
                highlightthickness=0,
            )
            self.face_canvas.pack(fill="x")
            self.face_canvas.bind("<Configure>", self._layout_home)
            memory_controls = ttk.Frame(parent)
            memory_controls.pack(fill="x", padx=28, pady=(0, 3))
            memory_notice = ttk.Label(
                parent,
                text=(
                    "La memoria local se usa automáticamente; solo los recuerdos "
                    "relevantes se envían al proveedor. No entrena el modelo."
                ),
                foreground="#666666", wraplength=650,
            )
            memory_notice.pack(fill="x", padx=28, before=memory_controls)
            parent.bind("<Configure>", lambda event: memory_notice.configure(wraplength=max(200, event.width - 60)))
            ttk.Button(memory_controls, text="Fuentes recuperadas",
                       command=self._show_knowledge_references).pack(side="right")
            self.chat_preview_var = tk.StringVar(value="")
            self.chat_preview = tk.Text(parent, height=4, wrap="word", state="disabled")
            self.chat_preview.pack(fill="x", padx=28, pady=(0, 4))
            self.chat_preview_var.trace_add("write", lambda *_: self._render_stream_preview())
            ttk.Button(memory_controls, text="Detener generación",
                       command=self._stop_generation).pack(side="right", padx=6)
            conversation = ttk.Frame(parent)
            conversation.pack(fill="both", expand=True, padx=24, pady=(0, 8))
            self.chat_messages_canvas = tk.Canvas(
                conversation, height=1,
                background="#eeeeee",
                highlightthickness=0,
            )
            self.chat_messages_canvas.pack(side="left", fill="both", expand=True)
            scrollbar = ttk.Scrollbar(
                conversation,
                orient="vertical",
                command=self.chat_messages_canvas.yview,
            )
            scrollbar.pack(side="right", fill="y"            )
            self.chat_messages_canvas.configure(yscrollcommand=scrollbar.set)
            self.memory_notes = load_memories(self.memory_vault)
            self.active_memory_ids: set[str] = set()
            self.chat_message_items: list[tuple[int, int, str]] = []
            self.chat_messages_canvas.bind(
                "<Configure>",
                self._chat_canvas_resized,
            )
            self.chat_messages_canvas.bind("<MouseWheel>", self._scroll_chat)
            self.chat_messages_canvas.bind("<Button-4>", self._scroll_chat)
            self.chat_messages_canvas.bind("<Button-5>", self._scroll_chat)
            self.chat_messages_canvas.bind("<Button-1>", self._memory_graph_clicked)
            self._render_chat_canvas()

            composer = tk.Frame(
                parent,
                background="#ffffff",
                padx=8,
                pady=4,
            )
            composer.pack(fill="x", padx=24, pady=(0, 18))
            ttk.Label(
                composer,
                text="✦",
                foreground="#747474",
                background="#ffffff",
                font=("Segoe UI Symbol", 13),
            ).pack(side="left", padx=(5, 8))
            self.prompt = tk.Text(
                composer,
                width=1,
                height=1,
                wrap="word",
                background="#ffffff",
                foreground="#303030",
                insertbackground="#303030",
                font=("Segoe UI", 11),
                padx=2,
                pady=7,
                relief="flat",
                borderwidth=0,
            )
            self.prompt_placeholder = "Pregúntale algo a tu IA..."
            self.prompt_is_placeholder = True
            self.prompt.insert("1.0", self.prompt_placeholder)
            self.prompt.configure(foreground="#999999")
            self.prompt.pack(side="left", fill="both", expand=True)
            self.prompt.bind("<FocusIn>", self._prompt_focus_in)
            self.prompt.bind("<FocusOut>", self._prompt_focus_out)
            tk.Button(
                composer,
                text="↑",
                command=self._send_message,
                background="#f0f0f0",
                foreground="#3d3d3d",
                activebackground="#e2e2e2",
                activeforeground="#222222",
                relief="flat",
                borderwidth=0,
                font=("Segoe UI", 13, "bold"),
                width=3,
                cursor="hand2",
            ).pack(side="right", padx=(8, 2))
            self.prompt.bind("<Return>", self._send_from_enter)

        def _send_from_enter(self, _event):
            self._send_message()
            return "break"

        def _prompt_focus_in(self, _event) -> None:
            if self.prompt_is_placeholder:
                self.prompt.delete("1.0", "end")
                self.prompt.configure(foreground="#303030")
                self.prompt_is_placeholder = False

        def _prompt_focus_out(self, _event) -> None:
            if not self.prompt.get("1.0", "end").strip():
                self.prompt.insert("1.0", self.prompt_placeholder)
                self.prompt.configure(foreground="#999999")
                self.prompt_is_placeholder = True

        def _scroll_chat(self, event):
            if getattr(event, "num", None) == 4:
                direction = -1
            elif getattr(event, "num", None) == 5:
                direction = 1
            else:
                direction = -int(event.delta / 120)
            if direction:
                self.chat_messages_canvas.yview_scroll(direction, "units")
            return "break"

        def _build_provider_tab(self, ttk, parent) -> None:
            ttk.Label(
                parent,
                text="Modelo de lenguaje",
                font=("Segoe UI", 14, "bold"),
            ).pack(anchor="w")
            ttk.Label(
                parent,
                text=(
                    "AI@Home usa por defecto su modelo local experimental, "
                    "entrenado desde cero. Hasta que exista un checkpoint "
                    "entrenado, el chat local no podrá responder. Los proveedores "
                    "externos son opcionales."
                ),
                wraplength=760,
            ).pack(anchor="w", pady=(2, 12))

            configured_provider = self.settings.get(
                "chat_provider",
                "ai-home-local",
            )
            if not _is_provider_id(configured_provider):
                configured_provider = "ai-home-local"
            self.provider_id: ChatProviderId = configured_provider
            self.provider_var = tk.StringVar(value=PROVIDER_LABELS[self.provider_id])
            self.provider_var.trace_add("write", self._provider_changed)
            default_endpoint, default_model = PROVIDER_DEFAULTS[self.provider_id]
            endpoint_setting = (
                "chat_endpoint"
                if self.provider_id == "openai-compatible"
                else f"chat_endpoint_{self.provider_id}"
            )
            model_setting = f"chat_model_{self.provider_id}"
            self.endpoint_var = tk.StringVar(
                value=self.settings.get(
                    endpoint_setting,
                    self.settings.get("chat_endpoint", default_endpoint)
                    if self.provider_id == "openai-compatible"
                    else default_endpoint,
                )
            )
            self.model_var = tk.StringVar(
                value=self.settings.get(
                    model_setting,
                    self.settings.get("chat_model", default_model)
                    if self.provider_id == "openai-compatible"
                    else default_model,
                )
            )
            self.api_key_var = tk.StringVar()

            provider = ttk.LabelFrame(parent, text="Proveedor y modelo", padding=12)
            provider.pack(fill="x", pady=(0, 10))
            ttk.Label(provider, text="Proveedor").grid(
                row=0, column=0, sticky="w", padx=6, pady=4
            )
            ttk.Combobox(
                provider,
                textvariable=self.provider_var,
                values=list(_PROVIDER_ID_BY_LABEL),
                state="readonly",
            ).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
            ttk.Label(provider, text="Modelo").grid(
                row=1, column=0, sticky="w", padx=6, pady=4
            )
            self.provider_model_entry = ttk.Entry(
                provider,
                textvariable=self.model_var,
            )
            self.provider_model_entry.grid(
                row=1, column=1, sticky="ew", padx=6, pady=4
            )
            ttk.Label(provider, text="Endpoint personalizado").grid(
                row=2, column=0, sticky="w", padx=6, pady=4
            )
            self.provider_endpoint_entry = ttk.Entry(
                provider,
                textvariable=self.endpoint_var,
            )
            self.provider_endpoint_entry.grid(
                row=2, column=1, sticky="ew", padx=6, pady=4
            )
            if self.provider_id != "openai-compatible":
                self.provider_endpoint_entry.configure(state="disabled")
            if self.provider_id == "ai-home-local":
                self.provider_model_entry.configure(state="disabled")
            provider.columnconfigure(1, weight=1)

            credentials = ttk.LabelFrame(parent, text="Clave API", padding=12)
            credentials.pack(fill="x", pady=(0, 10))
            ttk.Label(
                credentials,
                text=(
                    "El modelo local no necesita clave. Solo se usa para "
                    "proveedores externos."
                ),
                wraplength=760,
            ).pack(anchor="w", pady=(0, 6))
            ttk.Entry(
                credentials,
                textvariable=self.api_key_var,
                show="*",
            ).pack(fill="x", pady=(0, 8))
            key_actions = ttk.Frame(credentials)
            key_actions.pack(anchor="w")
            ttk.Button(
                key_actions,
                text="Guardar clave en este equipo",
                command=self._save_provider_key,
            ).pack(side="left", padx=(0, 8))
            ttk.Button(
                key_actions,
                text="Eliminar clave guardada",
                command=self._delete_provider_key,
            ).pack(side="left")

            routing = ttk.LabelFrame(parent, text="Dónde se procesa el chat", padding=12)
            routing.pack(fill="x")
            ttk.Label(
                routing,
                text=(
                    "Por defecto, el chat va directamente de este equipo al "
                    "proveedor seleccionado. El proveedor procesa tus mensajes "
                    "según sus propias condiciones."
                ),
                wraplength=760,
            ).pack(anchor="w")
            self.server_version_var = tk.StringVar(value=DEFAULT_SERVER_VERSION)
            self.server_version_combo = ttk.Combobox(
                routing, textvariable=self.server_version_var, state="readonly",
                values=[DEFAULT_SERVER_VERSION], width=44,
            )
            self.server_version_combo.pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Actualizar versiones aprobadas",
                       command=self._refresh_server_versions).pack(anchor="w", pady=(4, 0))
            ttk.Button(routing, text="Administrar registro local de modelos",
                       command=self._model_registry_dialog).pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Explorar conocimiento local",
                       command=self._knowledge_dialog).pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Crear versión de conocimiento",
                       command=self._build_knowledge_dialog).pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Entrenar modelo local",
                       command=self._run_training_dialog).pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Ejecutar evaluación local",
                       command=self._run_evaluation_dialog).pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Ejecutar evolución local",
                       command=self._run_evolution_dialog).pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Ver evaluación o evolución",
                       command=self._experiment_report_dialog).pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Inspeccionar checkpoint local",
                       command=self._inspect_checkpoint_dialog).pack(anchor="w", pady=(6, 0))
            ttk.Button(routing, text="Ajustes de generación Foundation",
                       command=self._generation_dialog).pack(anchor="w", pady=(6, 0))
            self.stream_enabled_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(routing, text="Texto progresivo (servidor Foundation)",
                            variable=self.stream_enabled_var).pack(anchor="w", pady=(6, 0))
            self.use_knowledge_var = tk.BooleanVar(value=False)
            self.use_server_ai_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                routing,
                text="Usar el proveedor configurado por mi servidor AI@Home",
                variable=self.use_server_ai_var,
            ).pack(anchor="w", pady=(8, 0))
            ttk.Checkbutton(
                routing,
                text="Recuperar fuentes públicas revisadas (requiere Foundation en el servidor)",
                variable=self.use_knowledge_var,
            ).pack(anchor="w", pady=(6, 0))
            ttk.Label(
                routing,
                text="La clave nunca se envía al servidor AI@Home.",
                foreground="#52606d",
            ).pack(anchor="w", pady=(6, 0))

        def _model_registry_dialog(self) -> None:
            directory = filedialog.askdirectory(parent=self, title="Registro local de modelos")
            if not directory:
                return
            if not (Path(directory) / "registry.sqlite3").is_file():
                messagebox.showerror("Modelos", "La carpeta no contiene un registro de modelos.", parent=self)
                return
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Modelos: aprobación, activación y rollback local")
            window.geometry("900x650")
            frame = ttk.Frame(window, padding=14)
            frame.pack(fill="both", expand=True)
            ttk.Label(frame, text=f"Registro local: {directory}", wraplength=850).pack(anchor="w")
            ttk.Label(frame, text="Afecta al modelo predeterminado de los procesos que usan este registro.\n"
                      "El chat con versión fijada conserva su selección. Rollback: elija una versión anterior aprobada y actívela.",
                      wraplength=850).pack(anchor="w", pady=8)
            version = tk.StringVar()
            selector = ttk.Combobox(frame, textvariable=version, state="readonly", width=65)
            selector.pack(fill="x")
            operator = tk.StringVar()
            reason = tk.StringVar()
            for label, variable in (("Operador (queda en auditoría)", operator), ("Motivo de activación / rollback", reason)):
                ttk.Label(frame, text=label).pack(anchor="w", pady=(6, 0))
                ttk.Entry(frame, textvariable=variable).pack(fill="x")
            actions = ttk.Frame(frame)
            actions.pack(fill="x", pady=10)
            status = tk.StringVar(value="Cargando registro…")
            ttk.Label(frame, textvariable=status, wraplength=850).pack(anchor="w")
            report_frame = ttk.Frame(frame)
            report_frame.pack(fill="both", expand=True, pady=(8, 0))
            scroll = ttk.Scrollbar(report_frame)
            scroll.pack(side="right", fill="y")
            content = tk.Text(report_frame, wrap="word", state="disabled", yscrollcommand=scroll.set)
            content.pack(fill="both", expand=True)
            scroll.configure(command=content.yview)
            busy = False
            buttons = []
            def run(action):
                nonlocal busy
                if busy:
                    return
                selected, who, why = version.get(), operator.get().strip(), reason.get().strip()
                if action != "refresh" and not selected:
                    messagebox.showerror("Modelos", "Seleccione una versión.", parent=window)
                    return
                if action in {"approve", "activate"} and (not who or (action == "activate" and not why)):
                    messagebox.showerror("Modelos", "Indique operador y, para activar, el motivo.", parent=window)
                    return
                busy = True
                selector.configure(state="disabled")
                for button in buttons:
                    button.configure(state="disabled")
                status.set("Comprobando registro y pesos…")
                def finish():
                    nonlocal busy
                    busy = False
                    selector.configure(state="readonly")
                    for button in buttons:
                        button.configure(state="normal")
                def failure():
                    status.set("Operación rechazada o fallida; revise el error y actualice el registro.")
                    finish()
                def operation():
                    from ai_home.foundation.registry import ModelRegistry
                    registry = ModelRegistry(directory)
                    if action == "approve":
                        registry.approve(selected, operator=who)
                    elif action == "activate":
                        registry.activate(selected, operator=who, reason=why)
                    return {"candidates": registry.list(), "approved_catalog": registry.catalog(),
                            "history": registry.history(),
                            "selected": registry.inspect(selected) if selected else None}
                def display(report):
                    selector.configure(values=[row["id"] for row in report["candidates"]])
                    content.configure(state="normal")
                    content.delete("1.0", "end")
                    content.insert("end", json.dumps(report, ensure_ascii=False, indent=2))
                    content.configure(state="disabled")
                    status.set("Registro actualizado. Aprobación: umbral 100%, procedencia y ausencia de regresiones obligatorias.")
                    finish()
                self._background(operation, display, failure, is_current=lambda: bool(window.winfo_exists()))
            for label, action in (("Actualizar", "refresh"), ("Ver evaluación", "inspect"),
                                  ("Aprobar candidato", "approve"), ("Activar / rollback", "activate")):
                button = ttk.Button(actions, text=label, command=lambda value=action: run(value))
                button.pack(side="left", padx=(0, 8))
                buttons.append(button)
            run("refresh")

        def _build_knowledge_dialog(self) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Crear versión de conocimiento")
            window.geometry("800x440")
            frame = ttk.Frame(window, padding=16)
            frame.pack(fill="both", expand=True)
            fields = [tk.StringVar() for _ in range(3)]
            inputs = []
            for row, label in enumerate(("Dataset autorizado JSON", "Nuevo índice JSON", "Checkpoint opcional (neuronal)")):
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=8)
                entry = ttk.Entry(frame, textvariable=fields[row], width=50)
                entry.grid(row=row, column=1, sticky="ew")
                def choose(index=row):
                    path = (filedialog.asksaveasfilename(parent=window, defaultextension=".json")
                            if index == 1 else filedialog.askopenfilename(parent=window))
                    if path:
                        fields[index].set(path)
                picker = ttk.Button(frame, text="Elegir", command=choose)
                picker.grid(row=row, column=2, padx=6)
                inputs.extend((entry, picker))
            frame.columnconfigure(1, weight=1)
            device = tk.StringVar(value="cpu")
            device_box = ttk.Combobox(frame, textvariable=device, values=("cpu", "cuda"), state="readonly", width=10)
            ttk.Label(frame, text="Dispositivo neuronal").grid(row=3, column=0, sticky="w")
            device_box.grid(row=3, column=1, sticky="w")
            ttk.Label(frame, text="Sin checkpoint: TF-IDF. Solo datos públicos revisados con permiso de recuperación.\n"
                      "Cada actualización crea otro archivo; no activa el índice del servidor ni entrena el modelo.",
                      wraplength=740).grid(row=4, column=0, columnspan=3, sticky="w", pady=14)
            status = tk.StringVar(value="Seleccione el dataset y una ruta nueva para el índice.")
            ttk.Label(frame, textvariable=status, wraplength=740).grid(row=5, column=0, columnspan=3, sticky="w")
            cancellation = threading.Event()
            busy = False
            def finish():
                nonlocal busy
                busy = False
                for widget in inputs:
                    widget.configure(state="normal")
                device_box.configure(state="readonly")
                start.configure(state="normal")
                stop.configure(state="disabled")
            def failure():
                status.set("No se publicó el índice; revise el error y los permisos del dataset.")
                finish()
            def run():
                nonlocal busy
                if busy:
                    return
                dataset, output, checkpoint = [field.get().strip() for field in fields]
                if not dataset or not output:
                    messagebox.showerror("Conocimiento", "Seleccione dataset e índice de salida.", parent=window)
                    return
                selected_device = device.get()
                cancellation.clear()
                busy = True
                for widget in (*inputs, device_box, start):
                    widget.configure(state="disabled")
                stop.configure(state="normal")
                status.set("Validando e indexando… Cancelar detiene antes de publicar, entre fragmentos.")
                def operation():
                    from ai_home.knowledge import build_from_dataset
                    return build_from_dataset(dataset, output, embedding_checkpoint=checkpoint or None,
                                              device=selected_device, cancel=cancellation)
                def success(report):
                    status.set(f"Índice creado: {report['output']}\nVersión: {report['index_version']}\nDocumentos: {report['documents']}")
                    finish()
                self._background(operation, success, failure, is_current=lambda: bool(window.winfo_exists()))
            start = ttk.Button(frame, text="Crear índice", command=run)
            start.grid(row=6, column=1, sticky="w", pady=16)
            stop = ttk.Button(frame, text="Cancelar", command=cancellation.set, state="disabled")
            stop.grid(row=6, column=2)
            window.bind("<Destroy>", lambda event: cancellation.set() if event.widget == window else None)

        def _knowledge_dialog(self) -> None:
            path = filedialog.askopenfilename(parent=self, title="Índice de conocimiento revisado",
                                             filetypes=[("Índice JSON", "*.json")])
            if not path:
                return
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Conocimiento: índice local y fuentes")
            window.geometry("820x580")
            embedding_controls = ttk.Frame(window, padding=12)
            embedding_controls.pack(fill="x")
            ttk.Label(embedding_controls, text="Checkpoint para índice neuronal (opcional)").grid(row=0, column=0, columnspan=3, sticky="w")
            embedding_path = tk.StringVar()
            embedding_entry = ttk.Entry(embedding_controls, textvariable=embedding_path)
            embedding_entry.grid(row=1, column=0, sticky="ew")
            embedding_controls.columnconfigure(0, weight=1)
            def choose_embedding():
                selected = filedialog.askopenfilename(parent=window, title="Checkpoint de embeddings",
                                                      filetypes=[("Foundation", "*.pt")])
                if selected:
                    embedding_path.set(selected)
            embedding_button = ttk.Button(embedding_controls, text="Elegir checkpoint", command=choose_embedding)
            embedding_button.grid(row=1, column=1, padx=6)
            embedding_device = tk.StringVar(value="cpu")
            device_box = ttk.Combobox(embedding_controls, textvariable=embedding_device,
                                     values=("cpu", "cuda"), state="readonly", width=8)
            device_box.grid(row=1, column=2)
            ttk.Label(embedding_controls, text="TF-IDF no requiere checkpoint. El índice neuronal exige el usado al crearlo.",
                      wraplength=760).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
            controls = ttk.Frame(window, padding=12)
            controls.pack(fill="x")
            query = ttk.Entry(controls)
            query.pack(side="left", fill="x", expand=True)
            scrollbar = ttk.Scrollbar(window)
            scrollbar.pack(side="right", fill="y")
            content = tk.Text(window, wrap="word", state="disabled", yscrollcommand=scrollbar.set)
            content.pack(fill="both", expand=True, padx=12, pady=(0, 12))
            scrollbar.configure(command=content.yview)
            busy = False

            def search():
                nonlocal busy
                if busy:
                    return
                text = query.get().strip()
                if len(text.encode("utf-8")) > 32000:
                    messagebox.showerror("Conocimiento", "La consulta supera el límite de 32.000 bytes UTF-8.", parent=window)
                    return
                checkpoint_path = embedding_path.get().strip() or None
                device = embedding_device.get()
                busy = True
                button.configure(state="disabled")
                for widget in (embedding_entry, embedding_button, device_box):
                    widget.configure(state="disabled")

                def inspect():
                    from ai_home.knowledge import inspect_knowledge
                    return inspect_knowledge(path, text, embedding_checkpoint=checkpoint_path, device=device)

                def finish():
                    nonlocal busy
                    busy = False
                    button.configure(state="normal")
                    embedding_entry.configure(state="normal")
                    embedding_button.configure(state="normal")
                    device_box.configure(state="readonly")

                def display(report):
                    content.configure(state="normal")
                    content.delete("1.0", "end")
                    content.insert("end", "Recuperación local de fragmentos. No genera respuestas ni entrena pesos.\n"
                                   "La consulta no se envía a Internet.\n\n")
                    content.insert("end", json.dumps(report, indent=2, ensure_ascii=False))
                    if text and not report["results"]:
                        content.insert("end", "\n\nNo se encontraron fragmentos relevantes.")
                    content.configure(state="disabled")
                    finish()

                self._background(inspect, display, finish, is_current=lambda: bool(window.winfo_exists()))

            button = ttk.Button(controls, text="Buscar / actualizar", command=search)
            button.pack(side="right", padx=(8, 0))
            query.bind("<Return>", lambda _: search())
            search()

        def _run_evolution_dialog(self) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Experimento de evolución Foundation")
            window.geometry("900x650")
            frame = ttk.Frame(window, padding=14)
            frame.pack(fill="both", expand=True)
            names = ("Checkpoint padre", "Suite JSON", "Validación JSON", "Replay JSON", "Datos nuevos JSON",
                     "Entrenamiento JSON", "Carpeta nueva")
            fields = [tk.StringVar() for _ in names]
            for row, (label, variable) in enumerate(zip(names, fields)):
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
                ttk.Entry(frame, textvariable=variable, width=60).grid(row=row, column=1, sticky="ew")
                def choose(index=row, value=variable):
                    path = (filedialog.asksaveasfilename(parent=window, title="Nueva carpeta de experimento")
                            if index == 6 else filedialog.askopenfilename(parent=window))
                    if path:
                        value.set(path)
                ttk.Button(frame, text="Elegir", command=choose).grid(row=row, column=2, padx=5)
            frame.columnconfigure(1, weight=1)
            reviewed = tk.BooleanVar(value=False)
            ttk.Checkbutton(frame, text="Los datos nuevos son una exportación SFT revisada", variable=reviewed).grid(row=7, column=1, sticky="w")
            device = tk.StringVar(value="cpu")
            multiplier = tk.StringVar(value="0.5")
            tokens = tk.StringVar(value="64")
            for row, label, value in ((8, "Dispositivo", device), (9, "Multiplicador learning rate", multiplier), (10, "Tokens por prueba", tokens)):
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
                widget = (ttk.Combobox(frame, textvariable=value, values=("cpu", "cuda"), state="readonly", width=12)
                          if row == 8 else ttk.Entry(frame, textvariable=value, width=12))
                widget.grid(row=row, column=1, sticky="w")
            status = tk.StringVar(value="Entrena control y experimento, compara y propone una decisión. No activa modelos.")
            ttk.Label(frame, textvariable=status, wraplength=820).grid(row=12, column=0, columnspan=3, sticky="w", pady=12)
            cancellation = threading.Event()
            events = queue.Queue()
            def stop():
                cancellation.set()
                status.set("Parada solicitada; esperando al paso o evaluación en curso…")
            def start():
                if any(getattr(self, name, None) is not None and getattr(self, name).is_alive()
                       for name in ("training_worker", "evaluation_worker", "evolution_worker")):
                    status.set("Ya hay un trabajo Foundation local en curso.")
                    return
                try:
                    paths = [field.get().strip() for field in fields]
                    rate, count = float(multiplier.get()), int(tokens.get())
                    if not all(paths) or not math.isfinite(rate) or not .1 <= rate <= 4 or rate == 1 or not 1 <= count <= 4096:
                        raise ValueError("Completa las rutas; multiplicador 0.1–4 distinto de 1 y tokens 1–4096.")
                except ValueError as error:
                    status.set(str(error))
                    return
                selected_device, selected_review = device.get(), reviewed.get()
                cancellation.clear()
                self.evolution_cancel = cancellation
                start_button.configure(state="disabled")
                status.set("Preparando experimento…")
                def execute():
                    try:
                        from ai_home.foundation.evolution_job import run_evolution
                        result = run_evolution(*paths, reviewed_export=selected_review, device=selected_device,
                                               multiplier=rate, max_new_tokens=count, cancel=cancellation,
                                               on_progress=lambda item: events.put(("progress", item)))
                        events.put(("done", result))
                    except Exception as error:
                        events.put(("error", str(error)))
                self.evolution_worker = threading.Thread(target=execute, daemon=True)
                self.evolution_worker.start()
            def poll():
                if not window.winfo_exists():
                    return
                while not events.empty():
                    kind, value = events.get_nowait()
                    if kind == "progress":
                        detail = (f"paso {value['step']}, pérdida {value['train_loss']:.4f}" if "step" in value
                                  else f"pruebas {value['completed']}/{value['total']}" if "completed" in value else "")
                        status.set(f"{value['stage']} · {detail}")
                    elif kind == "done":
                        status.set(f"{value['decision']}: {value['reason']}. Sin promoción automática.")
                        start_button.configure(state="normal")
                    else:
                        status.set(value)
                        start_button.configure(state="normal")
                window.after(100, poll)
            start_button = ttk.Button(frame, text="Ejecutar experimento", command=start)
            start_button.grid(row=11, column=1, sticky="w", pady=12)
            ttk.Button(frame, text="Parar", command=stop).grid(row=11, column=2)
            window.bind("<Destroy>", lambda event: cancellation.set() if event.widget is window else None)
            poll()

        def _run_training_dialog(self) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Entrenar Foundation")
            window.geometry("850x580")
            frame = ttk.Frame(window, padding=14)
            frame.pack(fill="both", expand=True)
            names = ("Dataset JSON", "Modelo JSON (opcional)", "Entrenamiento JSON", "Carpeta nueva", "Checkpoint de origen (opcional)", "Validación JSON (opcional)")
            values = {name: tk.StringVar() for name in names}
            for row, (name, value) in enumerate(values.items()):
                ttk.Label(frame, text=name).grid(row=row, column=0, sticky="w", pady=6)
                ttk.Entry(frame, textvariable=value, width=50).grid(row=row, column=1, sticky="ew")
                def choose(label=name, variable=value):
                    if label == "Carpeta nueva":
                        path = filedialog.asksaveasfilename(parent=window, title="Nombre de carpeta nueva para el entrenamiento")
                    else:
                        path = filedialog.askopenfilename(parent=window)
                    if path:
                        variable.set(path)
                ttk.Button(frame, text="Elegir", command=choose).grid(row=row, column=2, padx=5)
            frame.columnconfigure(1, weight=1)
            source_type = tk.StringVar(value="Dataset público")
            mode = tk.StringVar(value="Desde cero")
            ttk.Label(frame, text="Fuente").grid(row=6, column=0, sticky="w")
            ttk.Combobox(frame, textvariable=source_type, values=("Dataset público", "Exportación revisada SFT"), state="readonly", width=30).grid(row=6, column=1, sticky="w")
            ttk.Label(frame, text="Inicio").grid(row=7, column=0, sticky="w")
            ttk.Combobox(frame, textvariable=mode, values=("Desde cero", "Reanudar exactamente", "Padre con optimizador nuevo"), state="readonly", width=30).grid(row=7, column=1, sticky="w")
            device = tk.StringVar(value="cpu")
            ttk.Label(frame, text="Dispositivo").grid(row=8, column=0, sticky="w")
            ttk.Combobox(frame, textvariable=device, values=("cpu", "cuda"), state="readonly", width=12).grid(row=8, column=1, sticky="w")
            status = tk.StringVar(value="Datos autorizados o exportación SFT revisada. El candidato no se activa ni se distribuye.")
            ttk.Label(frame, textvariable=status, wraplength=730).grid(row=10, column=0, columnspan=3, sticky="w", pady=12)
            cancellation = threading.Event()
            events = queue.Queue()
            def stop():
                cancellation.set()
                status.set("Parada solicitada; guardando al terminar el paso en curso…")
            def start():
                if any(getattr(self, name, None) is not None and getattr(self, name).is_alive()
                       for name in ("training_worker", "evaluation_worker", "evolution_worker")):
                    status.set("Ya hay un entrenamiento o evaluación local en curso.")
                    return
                paths = [value.get().strip() for value in values.values()]
                if any(not paths[index] for index in (0, 2, 3)):
                    status.set("Selecciona dataset, configuración de entrenamiento y una carpeta nueva.")
                    return
                selected_mode, selected_source = mode.get(), source_type.get()
                if (selected_mode == "Desde cero") != (not paths[4]):
                    status.set("Selecciona un checkpoint para reanudar o continuar desde un padre; déjalo vacío para empezar de cero.")
                    return
                selected_device = device.get()
                cancellation.clear()
                self.training_cancel = cancellation
                start_button.configure(state="disabled")
                status.set("Preparando datos y entrenamiento…")
                def execute():
                    try:
                        from ai_home.foundation.training_job import run_training
                        result = run_training(paths[0], paths[1] or None, paths[2], paths[3],
                                              device=selected_device,
                                              resume=paths[4] if selected_mode == "Reanudar exactamente" else None,
                                              initial_checkpoint=paths[4] if selected_mode == "Padre con optimizador nuevo" else None,
                                              reviewed_export=selected_source == "Exportación revisada SFT",
                                              validation_dataset=paths[5] or None,
                                              cancel=cancellation, on_step=lambda item: events.put(("progress", dict(item))))
                        events.put(("done", result))
                    except Exception as error:
                        events.put(("error", str(error)))
                self.training_worker = threading.Thread(target=execute, daemon=True)
                self.training_worker.start()
            def poll():
                if not window.winfo_exists():
                    return
                while not events.empty():
                    kind, value = events.get_nowait()
                    if kind == "progress":
                        status.set(f"Paso {value['step']} — pérdida {value['train_loss']:.4f} — "
                                   f"{value.get('tokens_per_second', 0):.1f} tokens/s")
                    elif kind == "done":
                        status.set(f"{value['stop_reason']}. Paso {value['step']}. Checkpoint: {value['checkpoint']}")
                        start_button.configure(state="normal")
                    else:
                        status.set(value)
                        start_button.configure(state="normal")
                window.after(100, poll)
            start_button = ttk.Button(frame, text="Entrenar", command=start)
            start_button.grid(row=9, column=1, sticky="w", pady=12)
            ttk.Button(frame, text="Parar y guardar", command=stop).grid(row=9, column=2)
            window.bind("<Destroy>", lambda event: cancellation.set() if event.widget is window else None)
            poll()

        def _run_evaluation_dialog(self) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Ejecutar evaluación Foundation")
            window.geometry("740x420")
            frame = ttk.Frame(window, padding=14)
            frame.pack(fill="both", expand=True)
            values = {name: tk.StringVar() for name in ("Checkpoint", "Suite JSON", "Informe nuevo")}
            for row, (name, value) in enumerate(values.items()):
                ttk.Label(frame, text=name).grid(row=row, column=0, sticky="w", pady=6)
                ttk.Entry(frame, textvariable=value, width=58).grid(row=row, column=1, sticky="ew")
                def choose(label=name, variable=value):
                    if label == "Informe nuevo":
                        path = filedialog.asksaveasfilename(parent=window, defaultextension=".json")
                    else:
                        path = filedialog.askopenfilename(parent=window)
                    if path:
                        variable.set(path)
                ttk.Button(frame, text="Elegir", command=choose).grid(row=row, column=2, padx=5)
            frame.columnconfigure(1, weight=1)
            device = tk.StringVar(value="cpu")
            budget = tk.StringVar(value="64")
            ttk.Label(frame, text="Dispositivo").grid(row=3, column=0, sticky="w")
            ttk.Combobox(frame, textvariable=device, values=("cpu", "cuda"), state="readonly", width=12).grid(row=3, column=1, sticky="w")
            ttk.Label(frame, text="Tokens por prueba").grid(row=4, column=0, sticky="w")
            ttk.Entry(frame, textvariable=budget, width=12).grid(row=4, column=1, sticky="w")
            status = tk.StringVar(value="Se ejecutará inferencia real. El informe no activa ni aprueba el modelo.")
            ttk.Label(frame, textvariable=status, wraplength=660).grid(row=6, column=0, columnspan=3, sticky="w", pady=14)
            cancellation = threading.Event()
            events = queue.Queue()
            def stop():
                cancellation.set()
                status.set("Cancelación solicitada; esperando al trabajo en curso…")
            def start():
                if any(getattr(self, name, None) is not None and getattr(self, name).is_alive()
                       for name in ("training_worker", "evaluation_worker", "evolution_worker")):
                    status.set("Ya hay un entrenamiento o evaluación local en curso.")
                    return
                try:
                    count = int(budget.get())
                    if not 1 <= count <= 4096 or any(not value.get().strip() for value in values.values()):
                        raise ValueError("Completa las rutas y elige entre 1 y 4096 tokens.")
                except ValueError as error:
                    status.set(str(error))
                    return
                paths = [value.get().strip() for value in values.values()]
                selected_device = device.get()
                cancellation.clear()
                start_button.configure(state="disabled")
                status.set("Cargando checkpoint y ejecutando pruebas…")
                def execute():
                    try:
                        from ai_home.foundation.evaluation_job import run_evaluation
                        report = run_evaluation(*paths, device=selected_device, max_new_tokens=count,
                                                cancel=cancellation, on_probe=lambda item: events.put(("progress", item)))
                        events.put(("done", report))
                    except Exception as error:
                        events.put(("error", str(error)))
                self.evaluation_worker = threading.Thread(target=execute, daemon=True)
                self.evaluation_worker.start()
            def poll():
                if not window.winfo_exists():
                    return
                while not events.empty():
                    kind, value = events.get_nowait()
                    if kind == "progress":
                        status.set(f"Pruebas completadas: {value['completed']}/{value['total']}")
                    elif kind == "done":
                        passed = sum(row["passed"] for row in value["results"])
                        status.set(f"Informe guardado. Resultado: {passed}/{len(value['results'])} pruebas superadas.")
                        start_button.configure(state="normal")
                    else:
                        status.set(value)
                        start_button.configure(state="normal")
                window.after(100, poll)
            start_button = ttk.Button(frame, text="Evaluar y guardar", command=start)
            start_button.grid(row=5, column=1, sticky="w", pady=12)
            ttk.Button(frame, text="Cancelar", command=stop).grid(row=5, column=2)
            window.bind("<Destroy>", lambda event: cancellation.set() if event.widget is window else None)
            poll()

        def _experiment_report_dialog(self) -> None:
            path = filedialog.askopenfilename(parent=self, title="Informe de evaluación o evolución",
                                             filetypes=[("Informe JSON", "*.json")])
            if not path:
                return
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Foundation: evaluación y evolución")
            window.geometry("820x580")
            ttk.Label(window, text="Informe local: no verifica por sí solo la ejecución, los pesos ni la aprobación.",
                      wraplength=760).pack(padx=12, pady=10)
            summary = tk.StringVar(value="Cargando informe…")
            ttk.Label(window, textvariable=summary, wraplength=760).pack(padx=12, pady=6)
            frame = ttk.Frame(window)
            frame.pack(fill="both", expand=True, padx=12, pady=12)
            scrollbar = ttk.Scrollbar(frame)
            scrollbar.pack(side="right", fill="y")
            content = tk.Text(frame, wrap="word", state="disabled", yscrollcommand=scrollbar.set)
            content.pack(fill="both", expand=True)
            scrollbar.configure(command=content.yview)
            def inspect():
                from ai_home.foundation.reports import read_report
                return read_report(path)
            def display(document):
                report = document["report"]
                if report["format"] == "ai-home-evaluation-v1":
                    passed = sum(row["passed"] for row in report["results"])
                    summary.set(f"Evaluación: {passed}/{len(report['results'])} pruebas superadas. "
                                f"Versión de suite: {report.get('suite_version', 'sin especificar')}")
                else:
                    summary.set(f"Evolución: {report['decision']}. {report['reason']} — Sin promoción automática.")
                content.configure(state="normal")
                content.insert("end", json.dumps(document, indent=2, ensure_ascii=False))
                content.configure(state="disabled")
            self._background(inspect, display, is_current=lambda: bool(window.winfo_exists()))

        def _inspect_checkpoint_dialog(self) -> None:
            path = filedialog.askopenfilename(parent=self, title="Checkpoint Foundation",
                                             filetypes=[("PyTorch checkpoint", "*.pt"), ("Todos", "*.*")])
            if not path:
                return
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Foundation: arquitectura y entrenamiento")
            window.geometry("820x580")
            notebook = ttk.Notebook(window)
            notebook.pack(fill="both", expand=True, padx=12, pady=12)
            views = {}
            for name in ("Modelo", "Entrenamiento"):
                frame = ttk.Frame(notebook)
                notebook.add(frame, text=name)
                scrollbar = ttk.Scrollbar(frame)
                scrollbar.pack(side="right", fill="y")
                content = tk.Text(frame, wrap="word", state="disabled", yscrollcommand=scrollbar.set)
                content.pack(fill="both", expand=True)
                scrollbar.configure(command=content.yview)
                views[name] = content
            ttk.Label(window, text="Inspección local. Los datos del checkpoint no certifican capacidades ni aprobación.",
                      wraplength=760).pack(padx=12, pady=(0, 12))

            def inspect():
                from ai_home.foundation.inspection import inspect_checkpoint
                return inspect_checkpoint(path)

            def display(report):
                training_keys = {"training_config", "metrics", "validation", "embedding_delta_l2",
                                 "training_device", "step", "training_complete", "stop_reason",
                                 "training_document_count", "validation_document_count"}
                sections = {"Modelo": {k: v for k, v in report.items() if k not in training_keys},
                            "Entrenamiento": {k: v for k, v in report.items() if k in training_keys}}
                for name, value in sections.items():
                    views[name].configure(state="normal")
                    views[name].insert("end", json.dumps(value, indent=2, ensure_ascii=False))
                    views[name].configure(state="disabled")
                self.status_var.set("Checkpoint inspeccionado; no se ha cambiado el modelo de chat.")

            self._background(inspect, display, is_current=lambda: bool(window.winfo_exists()))

        def _generation_dialog(self) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Generación Foundation del servidor")
            window.transient(self)
            frame = ttk.Frame(window, padding=16)
            frame.pack(fill="both", expand=True)
            custom = tk.BooleanVar(value=self.generation_settings is not None)
            ttk.Checkbutton(frame, text="Usar ajustes personalizados", variable=custom).grid(
                row=0, column=0, columnspan=2, sticky="w")
            defaults = FoundationGeneration().model_dump()
            selected = self.generation_settings or defaults
            labels = [("temperature", "Temperatura (0 = determinista)"),
                      ("top_k", "Top-k (0 = sin filtro)"), ("top_p", "Top-p"),
                      ("repetition_penalty", "Penalización de repetición"),
                      ("seed", "Semilla"), ("max_new_tokens", "Máximo de tokens (vacío = servidor)")]
            fields = {}
            for row, (key, label) in enumerate(labels, 1):
                value = selected.get(key, defaults[key])
                fields[key] = tk.StringVar(value="" if value is None else str(value))
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
                ttk.Entry(frame, textvariable=fields[key], width=18).grid(row=row, column=1, pady=4)
            ttk.Label(frame, text="Se aplican a la próxima petición al servidor Foundation.\n"
                      "El servidor mantiene su propio límite de tokens.").grid(row=7, column=0, columnspan=2, pady=8)

            def save():
                try:
                    settings = parse_generation_settings({key: value.get() for key, value in fields.items()}) if custom.get() else None
                except (ValueError, TypeError):
                    messagebox.showerror("Generación", "Revisa los números y los límites de los ajustes.", parent=window)
                    return
                self.generation_settings = settings
                self.status_var.set("Ajustes personalizados de generación." if settings is not None else "Generación con valores del servidor.")
                window.destroy()

            ttk.Button(frame, text="Aplicar", command=save).grid(row=8, column=0, columnspan=2)

        def _show_credits(self) -> None:
            if self.session_token is None or self.session_server_url is None:
                messagebox.showerror("AI@Home", "Inicia sesión para consultar tus créditos.", parent=self)
                return
            server_url, token = self.session_server_url, self.session_token
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Créditos de mi cuenta")
            window.geometry("800x620")
            controls = ttk.Frame(window, padding=12)
            controls.pack(fill="x")
            price_label = tk.StringVar(value="Precio del servidor: pendiente de consulta")
            ttk.Label(controls, textvariable=price_label).pack(anchor="w")
            ttk.Label(controls, text="Máximo de créditos por respuesta (0 = no autorizar gasto):").pack(anchor="w")
            limit = tk.StringVar(value=str(self.chat_credit_limit))
            ttk.Entry(controls, textvariable=limit, width=12).pack(anchor="w")
            def apply_limit():
                try:
                    value = int(limit.get())
                    credit_authorization(value)
                except ValueError as error:
                    messagebox.showerror("AI@Home", str(error), parent=window)
                    return
                self.chat_credit_limit = value
                self.status_var.set(f"Límite autorizado: {value} créditos por respuesta del servidor.")
            ttk.Button(controls, text="Aplicar límite a próximas respuestas", command=apply_limit).pack(anchor="w", pady=6)
            ttk.Label(controls, text="El límite es por respuesta, no un presupuesto total. Se cobra el cálculo completado, aunque se pierda la conexión. Una nueva solicitud puede generar otro cobro. El límite se borra al cambiar de cuenta.", wraplength=740).pack(anchor="w")
            self._background(lambda: server_inference_price(server_url, token),
                             lambda price: price_label.set(f"Precio actual: {price} créditos por respuesta completada"),
                             is_current=lambda: bool(window.winfo_exists()))
            content = tk.Text(window, wrap="word", state="disabled")
            content.pack(fill="both", expand=True, padx=12, pady=12)

            def refresh():
                refresh_button.configure(state="disabled")

                def success(result):
                    content.configure(state="normal")
                    content.delete("1.0", "end")
                    content.insert("end", format_credits(*result))
                    content.configure(state="disabled")
                    refresh_button.configure(state="normal")

                self._background(lambda: server_credits(server_url, token), success,
                                 lambda: refresh_button.configure(state="normal"),
                                 is_current=lambda: bool(window.winfo_exists()))

            refresh_button = ttk.Button(window, text="Actualizar", command=refresh)
            refresh_button.pack(pady=(0, 12))
            refresh()

        def _refresh_server_versions(self) -> None:
            if self.session_token is None or self.session_server_url is None:
                messagebox.showerror("AI@Home", "Inicia sesión para consultar las versiones.", parent=self)
                return
            server_url, token = self.session_server_url, self.session_token

            def on_success(versions):
                self.server_version_combo.configure(values=[DEFAULT_SERVER_VERSION, *versions])
                self.status_var.set(f"{len(versions)} versiones aprobadas disponibles.")

            self._background(lambda: server_model_versions(server_url, token), on_success)

        def _provider_changed(self, *_args) -> None:
            provider_id = _PROVIDER_ID_BY_LABEL.get(self.provider_var.get())
            if provider_id is None or provider_id == self.provider_id:
                return
            self.provider_id = provider_id
            endpoint, model = PROVIDER_DEFAULTS[provider_id]
            endpoint_setting = (
                "chat_endpoint"
                if provider_id == "openai-compatible"
                else f"chat_endpoint_{provider_id}"
            )
            model_setting = f"chat_model_{provider_id}"
            self.endpoint_var.set(self.settings.get(endpoint_setting, endpoint))
            self.model_var.set(self.settings.get(model_setting, model))
            self.provider_endpoint_entry.configure(
                state="normal" if provider_id == "openai-compatible" else "disabled"
            )
            self.provider_model_entry.configure(
                state="disabled" if provider_id == "ai-home-local" else "normal"
            )

        def _selected_provider(self) -> tuple[ChatProviderId, str]:
            provider = self.provider_id
            if not _is_provider_id(provider):
                raise ValueError("Selecciona un proveedor compatible.")
            if provider in {"ai-home-local", "ai-home-foundation"}:
                return provider, ""
            if provider == "openai-compatible":
                configured_endpoint = self.endpoint_var.get().strip()
                if not configured_endpoint:
                    raise ValueError(
                        "Configura un proveedor en «Conectar IA» antes de enviar "
                        "mensajes."
                    )
                endpoint = normalize_base_url(configured_endpoint)
                if contains_common_secret(endpoint):
                    raise ValueError(
                        "La URL del endpoint no debe incluir credenciales."
                    )
            else:
                endpoint = PROVIDER_DEFAULTS[provider][0]
            return provider, endpoint

        def _save_provider_key(self) -> None:
            try:
                provider, endpoint = self._selected_provider()
                if provider in {"ai-home-local", "ai-home-foundation"}:
                    raise ValueError("El modelo local no necesita una clave API.")
                save_provider_api_key(
                    provider,
                    endpoint,
                    self.api_key_var.get(),
                )
            except (ValueError, RuntimeError) as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return
            self.api_key_var.set("")
            messagebox.showinfo(
                "AI@Home",
                "Clave guardada en el almacén seguro del sistema.",
                parent=self,
            )

        def _delete_provider_key(self) -> None:
            try:
                provider, endpoint = self._selected_provider()
                deleted = delete_provider_api_key(provider, endpoint)
            except (ValueError, RuntimeError) as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return
            self.api_key_var.set("")
            messagebox.showinfo(
                "AI@Home",
                (
                    "Clave eliminada."
                    if deleted
                    else "No había una clave guardada para este proveedor."
                ),
                parent=self,
            )

        def _build_node_tab(self, ttk, parent) -> None:
            ttk.Label(
                parent,
                text=(
                    "Aportar cómputo es opcional e independiente del uso de IA. "
                    "El nodo solo ejecuta trabajos tipados permitidos por el servidor. "
                    "No envíes datos personales ni secretos en trabajos compartidos."
                ),
                wraplength=800,
            ).pack(anchor="w", pady=6)
            self.consent_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                parent,
                text="Acepto registrar este equipo y ejecutar trabajos comunitarios.",
                variable=self.consent_var,
            ).pack(anchor="w", pady=8)
            ttk.Button(
                parent,
                text="Iniciar contribución",
                command=self._start_contribution,
            ).pack(anchor="w", pady=4)
            ttk.Button(
                parent,
                text="Detener contribución local",
                command=self._stop_contribution,
            ).pack(anchor="w", pady=4)
            ttk.Button(
                parent,
                text="Revocar este nodo",
                command=self._revoke_node,
            ).pack(anchor="w", pady=4)

        def _background(
            self,
            operation: Callable[[], T],
            on_success: Callable[[T], None],
            on_failure: Callable[[], None] | None = None,
            *,
            is_current: Callable[[], bool] | None = None,
        ) -> None:
            self.status_var.set("Procesando…")
            session_token = self.session_token
            valid = lambda: self.session_token == session_token and (is_current is None or is_current())

            def execute() -> None:
                try:
                    result = operation()
                except Exception as error:
                    self.events.put(("error", (error, on_failure, valid)))
                else:
                    self.events.put(("success", (on_success, result, valid)))

            threading.Thread(target=execute, daemon=True).start()

        def _drain_events(self) -> None:
            while True:
                try:
                    kind, payload = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "error":
                    error, on_failure, valid = payload
                    if not valid():
                        continue
                    if on_failure is not None:
                        on_failure()
                    if isinstance(error, GenerationCancelled):
                        self.status_var.set("Generación cancelada; respuesta parcial descartada.")
                    else:
                        self.status_var.set("Error")
                        messagebox.showerror("AI@Home", str(error), parent=self)
                elif kind == "chat_delta":
                    conversation_id, cancellation, content = payload
                    if self.conversation_id == conversation_id and self.chat_cancel is cancellation and not cancellation.is_set():
                        self.chat_preview_var.set((self.chat_preview_var.get() + content)[-2000:])
                elif kind == "success":
                    callback, result, valid = payload
                    if not valid():
                        continue
                    try:
                        callback(result)
                    except Exception as error:
                        self.status_var.set("Error")
                        messagebox.showerror("AI@Home", str(error), parent=self)
                elif kind == "node_status":
                    self.status_var.set(str(payload))
            self.after(100, self._drain_events)

        def _credentials(self) -> tuple[str, str, str]:
            return (
                normalize_base_url(self.server_var.get()),
                self.email_var.get().strip(),
                self.password_var.get(),
            )

        def _login(self) -> None:
            if self.session_token is not None:
                messagebox.showerror(
                    "AI@Home",
                    "Cierra la sesión actual antes de iniciar otra.",
                    parent=self,
                )
                return
            try:
                server_url, email, password = self._credentials()
                if not email or not password:
                    raise ValueError("Introduce tu correo y contraseña.")
            except ValueError as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return

            def on_success(session: SessionResponse) -> None:
                if session.user_id is None:
                    raise ValueError("Actualiza el servidor: falta la identidad de la cuenta.")
                self._switch_memory_profile(server_url, str(session.user_id))
                updated_settings = {
                    **self.settings,
                    "server_url": server_url,
                    "email": email,
                }
                save_settings(updated_settings)
                self.settings = updated_settings
                self.session_token = session.access_token
                self.session_server_url = server_url
                self.guest_session_active = False
                self.password_var.set("")
                self.account_status_var.set(f"Sesión iniciada como {email}.")
                self.status_var.set(f"Sesión iniciada: {email}")

            self._background(
                lambda: login_to_server(server_url, email, password),
                on_success,
            )

        def _register_account(self) -> None:
            try:
                server_url, email, password = self._credentials()
                if not email or len(password) < 12:
                    raise ValueError(
                        "Introduce un correo y una contraseña de al menos "
                        "12 caracteres."
                    )
            except ValueError as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return

            def on_success(_result) -> None:
                self.status_var.set("Cuenta creada; revisa tu correo si se solicita.")
                messagebox.showinfo(
                    "AI@Home",
                    "Cuenta creada. Si Supabase solicita confirmar el correo, "
                    "hazlo antes de iniciar sesión.",
                    parent=self,
                )

            self._background(
                lambda: register_account(server_url, email, password),
                on_success,
            )

        def _start_guest_session(self) -> None:
            if self.session_token is not None:
                messagebox.showinfo(
                    "AI@Home",
                    "Ya has iniciado sesión con tu cuenta. Cierra esa sesión "
                    "antes de continuar como invitado.",
                    parent=self,
                )
                return
            self.guest_session_active = True
            self.account_status_var.set(
                "Sesión de invitado activa: chat y recuerdos locales, sin cuenta. "
                "Las funciones de la comunidad requieren iniciar sesión."
            )
            self.status_var.set("Sesión de invitado iniciada.")

        def _logout(self) -> None:
            if self.session_token is None:
                if self.guest_session_active:
                    self.guest_session_active = False
                    self._switch_memory_profile()
                    self.account_status_var.set(
                        "Sin sesión activa. Puedes continuar como invitado."
                    )
                    self.status_var.set("Sesión de invitado cerrada.")
                else:
                    self.status_var.set("No hay una sesión activa.")
                return
            token = self.session_token
            server_url = self.session_server_url
            if server_url is None:
                self.status_var.set("No hay un servidor asociado a la sesión.")
                return

            def logout_request() -> None:
                with httpx.Client(base_url=server_url, timeout=20) as client:
                    response = client.post(
                        "/api/v1/auth/logout",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    if response.status_code == 401:
                        return
                    response.raise_for_status()

            def on_success(_result) -> None:
                self._switch_memory_profile()
                self.session_token = None
                self.session_server_url = None
                self.account_status_var.set(
                    "Sesión cerrada. Puedes continuar como invitado."
                )
                self.status_var.set("Sesión cerrada.")

            self._background(
                logout_request,
                on_success,
            )

        def _send_message(self) -> None:
            if self.chat_request_active:
                return
            if self.prompt_is_placeholder:
                return
            prompt = self.prompt.get("1.0", "end").strip()
            if not prompt:
                return
            messages = [
                *self.messages,
                {"role": "user", "content": prompt},
            ]
            request_cancel = threading.Event()
            request_conversation_id = self.conversation_id
            selected_memories = retrieve_memories(prompt, self.memory_notes)
            request_messages = messages_with_memory_context(
                messages,
                selected_memories,
            )
            try:
                use_server = self.use_server_ai_var.get()
                use_stream = self.stream_enabled_var.get()
                credit_limit = self.chat_credit_limit
                credit_request_id = str(uuid4()) if credit_limit else None
                generation = dict(self.generation_settings) if self.generation_settings is not None else None
                if generation is not None and not use_server:
                    raise ValueError("Los ajustes personalizados requieren el servidor Foundation.")
                if use_stream and not use_server:
                    raise ValueError("El texto progresivo requiere activar el servidor Foundation.")
                use_knowledge = self.use_knowledge_var.get()
                if use_knowledge and not use_server:
                    raise ValueError("La recuperación de fuentes requiere activar el servidor AI@Home.")
                if use_server:
                    if self.session_token is None or self.session_server_url is None:
                        raise ValueError(
                            "Inicia sesión antes de usar el proveedor del servidor."
                        )
                    server_url = self.session_server_url
                    user_token = self.session_token
                    selected_version = self.server_version_var.get()
                    requested_version = None if selected_version == DEFAULT_SERVER_VERSION else str(UUID(selected_version))
                    if self.pinned_server_version is not None:
                        if requested_version not in (None, self.pinned_server_version):
                            raise ValueError("Limpia la conversación antes de cambiar de versión.")
                        requested_version = self.pinned_server_version
                    model = f"AI@Home:{requested_version}" if requested_version else "AI@Home"
                    selected_source = (
                        "ai-home-server",
                        model,
                        server_url,
                    )
                    if (
                        self.conversation_source is not None
                        and self.conversation_source != selected_source
                    ):
                        raise ValueError(
                            "Mantén el mismo proveedor y modelo durante una "
                            "conversación, o límpiala antes de cambiar."
                        )
                    self.api_key_var.set("")
                    if use_stream:
                        operation = lambda: server_stream_result(
                            server_url, user_token, request_messages,
                            use_knowledge=use_knowledge, model_version=requested_version, generation=generation,
                            credit_limit=credit_limit, credit_request_id=credit_request_id,
                            cancel=request_cancel,
                            on_delta=lambda content: self.events.put(("chat_delta", (
                                request_conversation_id, request_cancel, content))),
                        )
                    else:
                        operation = lambda: server_chat_result(
                            server_url, user_token, request_messages,
                            use_knowledge=use_knowledge, model_version=requested_version, generation=generation,
                            credit_limit=credit_limit, credit_request_id=credit_request_id,
                        )
                else:
                    provider_id, provider_url = self._selected_provider()
                    model = self.model_var.get().strip()
                    if not model:
                        raise ValueError("Introduce el nombre del modelo.")
                    selected_source = (provider_id, model, provider_url)
                    if (
                        self.conversation_source is not None
                        and self.conversation_source != selected_source
                    ):
                        raise ValueError(
                            "Mantén el mismo proveedor y modelo durante una "
                            "conversación, o límpiala antes de cambiar."
                        )
                    entered_api_key = self.api_key_var.get()
                    self.api_key_var.set("")
                    self.settings["chat_provider"] = provider_id
                    if provider_id == "openai-compatible":
                        self.settings["chat_endpoint"] = provider_url
                    self.settings[f"chat_model_{provider_id}"] = model
                    if provider_id == "openai-compatible":
                        self.settings["chat_model"] = model
                    save_settings(self.settings)
                    provider_api_key = entered_api_key
                    if not provider_api_key and provider_id not in {"ai-home-local", "ai-home-foundation"}:
                        provider_api_key = get_provider_api_key(
                            provider_id,
                            provider_url,
                        )
                    operation = lambda: provider_completion(
                        provider_id,
                        model,
                        request_messages,
                        api_key=provider_api_key,
                        endpoint_url=provider_url,
                    )
                    if provider_id == "ai-home-foundation":
                        expected_checkpoint = self.pinned_foundation_checkpoint
                        operation = lambda: foundation_chat_result(model, request_messages,
                                                                    expected_checkpoint=expected_checkpoint)
            except ValueError as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return
            except RuntimeError as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return

            if self.conversation_source is None:
                self.conversation_source = selected_source
            self.prompt.delete("1.0", "end")
            self.chat_request_active = True
            self.chat_cancel = request_cancel if use_stream else None
            self.chat_preview_var.set("")
            self.messages.append({"role": "user", "content": prompt})
            self._append_chat("Tú", prompt)
            messages = list(self.messages)

            def on_failure():
                if self.conversation_id == request_conversation_id:
                    self.chat_request_active = False
                    self.chat_cancel = None
                    self.chat_preview_var.set("")

            def on_success(result: str | ServerChatResult | FoundationChatResult) -> None:
                if self.conversation_id != request_conversation_id:
                    return
                if request_cancel.is_set():
                    on_failure()
                    self.status_var.set("Generación cancelada; respuesta parcial descartada.")
                    return
                on_failure()
                answer = result.answer if isinstance(result, (ServerChatResult, FoundationChatResult)) else result
                if isinstance(result, FoundationChatResult):
                    self.pinned_foundation_checkpoint = result.checkpoint_sha256
                if isinstance(result, ServerChatResult) and result.model_version is not None:
                    self.pinned_server_version = result.model_version
                    self.server_version_var.set(result.model_version)
                    self.conversation_source = ("ai-home-server", f"AI@Home:{result.model_version}", server_url)
                if use_knowledge:
                    self.conversation_used_knowledge = True
                    references = result.references if isinstance(result, ServerChatResult) else ()
                    self.knowledge_references.append((len(self.messages), references))
                self.chat_request_active = False
                self.sphere_emotion = _sphere_emotion_for_text(answer)
                self._layout_home()
                self.messages.append({"role": "assistant", "content": answer})
                self.active_memory_ids.update(
                    note.id for note in selected_memories
                )
                self._append_chat("IA", answer)
                suffix = f" · {len(references)} fuentes recuperadas" if use_knowledge else ""
                charge = f" · {result.charged_credits} créditos" if isinstance(result, ServerChatResult) else ""
                self.status_var.set(f"Respuesta de {model}{suffix}{charge}")

            self._background(
                operation,
                on_success,
                on_failure,
                is_current=lambda: self.conversation_id == request_conversation_id,
            )

        def _append_chat(self, speaker: str, content: str) -> None:
            self._render_chat_canvas()
            self.chat_messages_canvas.yview_moveto(1.0)

        def _switch_memory_profile(self, server_url=None, user_id=None) -> None:
            vault = profile_memory_vault(server_url, user_id)
            notes = load_memories(vault)
            for window in self.private_windows:
                if window.winfo_exists():
                    window.destroy()
            self.private_windows.clear()
            self.memory_vault = vault
            self.memory_notes = notes
            self._clear_chat()
            self.prompt.delete("1.0", "end")
            self.prompt_is_placeholder = False
            self.api_key_var.set("")
            self.use_knowledge_var.set(False)
            self.stream_enabled_var.set(False)
            self.generation_settings = None
            self.chat_credit_limit = 0
            self.server_version_combo.configure(values=[DEFAULT_SERVER_VERSION])

        def _show_knowledge_references(self) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Fuentes recuperadas de esta conversación")
            window.geometry("760x500")
            content = tk.Text(window, wrap="word")
            content.pack(fill="both", expand=True, padx=12, pady=12)
            content.insert("end", format_knowledge_references(self.knowledge_references))
            content.configure(state="disabled")

        def _render_stream_preview(self) -> None:
            self.chat_preview.configure(state="normal")
            self.chat_preview.delete("1.0", "end")
            self.chat_preview.insert("end", self.chat_preview_var.get())
            self.chat_preview.configure(state="disabled")
            self.chat_preview.yview_moveto(1.0)

        def _stop_generation(self) -> None:
            if self.chat_cancel is not None:
                self.chat_cancel.set()
                self.chat_preview_var.set("")
                self.status_var.set("Cancelando generación…")

        def _clear_chat(self) -> None:
            self._stop_generation()
            self.chat_cancel = None
            self.chat_preview_var.set("")
            self.pinned_server_version = None
            self.pinned_foundation_checkpoint = None
            self.server_version_var.set(DEFAULT_SERVER_VERSION)
            self.knowledge_references.clear()
            self.conversation_used_knowledge = False
            self.chat_request_active = False
            self.training_submission_active = False
            self.messages.clear()
            self.conversation_id = str(uuid4())
            self.conversation_source = None
            self.training_candidate_submitted = False
            self.training_consent_var.set(False)
            self.training_review_var.set(False)
            self.active_memory_ids.clear()
            self._render_chat_canvas()
            self.chat_messages_canvas.yview_moveto(0)

        def _submit_training_candidate(self) -> None:
            if self.conversation_used_knowledge:
                messagebox.showerror(
                    "AI@Home",
                    "Esta conversación usó fuentes recuperadas. Su envío para entrenamiento "
                    "requiere conservar y revisar las licencias de las fuentes; inicia otra conversación.",
                    parent=self,
                )
                return
            if self.training_submission_active:
                return
            if self.training_candidate_submitted:
                messagebox.showerror(
                    "AI@Home",
                    "Esta conversación ya fue enviada. Inicia una conversación "
                    "nueva para crear otra aportación.",
                    parent=self,
                )
                return
            if not self.training_consent_var.get():
                messagebox.showerror(
                    "AI@Home",
                    "Activa el consentimiento explícito para esta conversación.",
                    parent=self,
                )
                return
            if not self.training_review_var.get():
                messagebox.showerror(
                    "AI@Home",
                    "Revisa el contenido y las condiciones del proveedor antes "
                    "de enviarlo.",
                    parent=self,
                )
                return
            if self.conversation_source is None:
                messagebox.showerror(
                    "AI@Home",
                    "Inicia y completa una conversación antes de aportarla.",
                    parent=self,
                )
                return
            source_provider, model, source_endpoint = self.conversation_source
            if source_provider == "ai-home-server":
                messagebox.showerror(
                    "AI@Home",
                    "No se puede aportar una conversación del proxy de instancia "
                    "porque aún no expone la procedencia exacta del proveedor.",
                    parent=self,
                )
                return
            if not _is_training_provider(source_provider):
                messagebox.showerror(
                    "AI@Home",
                    "La conversación necesita una procedencia compatible y verificable.",
                    parent=self,
                )
                return
            if self.session_token is None or self.session_server_url is None:
                messagebox.showerror(
                    "AI@Home",
                    "Inicia sesión en el servidor AI@Home que recibirá la aportación.",
                    parent=self,
                )
                return
            if self.chat_request_active:
                messagebox.showerror(
                    "AI@Home",
                    "Espera a que termine la respuesta antes de aportar la conversación.",
                    parent=self,
                )
                return
            provider: TrainingProviderId = source_provider
            if provider == "ai-home-foundation":
                if not self.pinned_foundation_checkpoint:
                    messagebox.showerror("AI@Home", "Falta la identidad del checkpoint cargado.", parent=self)
                    return
                model = self.pinned_foundation_checkpoint
                source_endpoint = f"ai-home-foundation://sha256/{model}"
            messages = [
                message
                for message in self.messages
                if message["role"] in {"user", "assistant"}
            ]
            if not any(item["role"] == "user" for item in messages) or not any(
                item["role"] == "assistant" for item in messages
            ):
                messagebox.showerror(
                    "AI@Home",
                    "La conversación necesita al menos una pregunta y una respuesta.",
                    parent=self,
                )
                return
            server_url = self.session_server_url
            user_token = self.session_token
            conversation_id = self.conversation_id
            self.training_submission_active = True

            def on_success(contribution: TrainingContributionResponse) -> None:
                if self.conversation_id != conversation_id:
                    return
                self.training_submission_active = False
                self.training_candidate_submitted = True
                self.training_consent_var.set(False)
                self.training_review_var.set(False)
                self.status_var.set("Aportación pendiente de revisión humana.")
                messagebox.showinfo(
                    "AI@Home",
                    "Aportación recibida como candidata. No se usará para entrenar "
                    "hasta superar la revisión y los controles de procedencia. "
                    f"Puedes retirarla mientras no se congele un dataset.\n\n"
                    f"ID: {contribution.contribution_id}",
                    parent=self,
                )

            self._background(
                lambda: submit_training_candidate(
                    server_url,
                    user_token,
                    conversation_id,
                    provider,
                    source_endpoint,
                    model,
                    messages,
                    reviewed_for_submission=True,
                ),
                on_success,
                lambda: setattr(self, "training_submission_active", False) if self.conversation_id == conversation_id else None,
                is_current=lambda: self.conversation_id == conversation_id,
            )

        def _correction_dialog(self, contribution, server_url, user_token) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Corregir respuesta: nueva aportación")
            window.geometry("800x620")
            frame = ttk.Frame(window, padding=14)
            frame.pack(fill="both", expand=True)
            indices = [i for i, message in enumerate(contribution.messages) if message.role == "assistant"]
            selected = tk.StringVar(value=str(indices[-1]))
            ttk.Label(frame, text="Respuesta del asistente (índice del mensaje)").pack(anchor="w")
            selector = ttk.Combobox(frame, textvariable=selected, values=[str(i) for i in indices], state="readonly")
            selector.pack(fill="x")
            original = tk.Text(frame, height=7, wrap="word", state="disabled")
            original.pack(fill="both", expand=True, pady=8)
            replacement = tk.Text(frame, height=7, wrap="word")
            ttk.Label(frame, text="Respuesta corregida").pack(anchor="w")
            replacement.pack(fill="both", expand=True)
            consent, reviewed = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
            ttk.Checkbutton(frame, text="Consiento enviar el prefijo y esta corrección como nuevo candidato.", variable=consent).pack(anchor="w", pady=(10, 0))
            ttk.Checkbutton(frame, text="He revisado datos sensibles y derechos de reutilización.", variable=reviewed).pack(anchor="w")
            ttk.Label(frame, text="El original no se modifica. Se excluyen los turnos posteriores; requiere nueva revisión humana.",
                      wraplength=750).pack(anchor="w", pady=8)
            def show_original(*args):
                index = int(selected.get())
                original.configure(state="normal")
                original.delete("1.0", "end")
                original.insert("end", "\n\n".join(f"{message.role}: {message.content}"
                    for message in contribution.messages[:index + 1]))
                original.configure(state="disabled")
                replacement.delete("1.0", "end")
                replacement.insert("end", contribution.messages[index].content)
                consent.set(False); reviewed.set(False)
            selector.bind("<<ComboboxSelected>>", show_original)
            show_original()
            candidate_id = str(uuid4())
            busy = False
            def send():
                nonlocal busy
                if busy:
                    return
                if not consent.get() or not reviewed.get():
                    messagebox.showerror("Corrección", "Confirma consentimiento y revisión.", parent=window)
                    return
                index, text = int(selected.get()), replacement.get("1.0", "end-1c")
                if not text.strip() or text == contribution.messages[index].content or len(text) > 8192:
                    messagebox.showerror("Corrección", "Escribe una corrección distinta de hasta 8192 caracteres.", parent=window)
                    return
                busy = True
                button.configure(state="disabled")
                def failure():
                    nonlocal busy
                    busy = False
                    button.configure(state="normal")
                def success(result):
                    self.status_var.set(f"Corrección pendiente de revisión: {result.contribution_id}")
                    window.destroy()
                self._background(lambda: submit_training_correction(server_url, user_token, contribution.contribution_id,
                    index, text, consent=True, reviewed=True, conversation_id=candidate_id), success, failure,
                    is_current=lambda: bool(window.winfo_exists()))
            button = ttk.Button(frame, text="Enviar nueva corrección", command=send)
            button.pack(anchor="e", pady=8)

        def _manage_training_candidates(self) -> None:
            if self.session_token is None or self.session_server_url is None:
                messagebox.showerror(
                    "AI@Home",
                    "Inicia sesión para consultar o retirar tus aportaciones.",
                    parent=self,
                )
                return
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Mis aportaciones para entrenamiento")
            window.geometry("760x360")
            listing = tk.Listbox(window, height=12)
            listing.pack(fill="both", expand=True, padx=10, pady=10)
            records: list[TrainingContributionResponse] = []
            offset = 0
            server_url = self.session_server_url
            user_token = self.session_token

            def refresh(page_offset: int | None = None) -> None:
                nonlocal offset
                if page_offset is not None:
                    offset = max(0, page_offset)

                def on_success(
                    contributions: list[TrainingContributionResponse],
                ) -> None:
                    records[:] = contributions
                    listing.delete(0, "end")
                    for contribution in records[:50]:
                        listing.insert(
                            "end",
                            f"{contribution.provider} / {contribution.model} — "
                            f"{contribution.status} — "
                            f"{contribution.contribution_id}",
                        )
                    previous_button.configure(state="normal" if offset else "disabled")
                    next_button.configure(
                        state="normal" if len(records) > 50 else "disabled"
                    )

                self._background(
                    lambda: list_training_candidates(
                        server_url,
                        user_token,
                        limit=51,
                        offset=offset,
                    ),
                    on_success,
                )

            def withdraw_selected() -> None:
                selection = listing.curselection()
                if not selection:
                    messagebox.showerror(
                        "AI@Home",
                        "Selecciona una aportación para retirarla.",
                        parent=window,
                    )
                    return
                contribution_id = records[selection[0]].contribution_id

                def on_success(_result) -> None:
                    self.status_var.set("Aportación retirada y eliminada.")
                    refresh()

                self._background(
                    lambda: withdraw_training_candidate(
                        server_url,
                        user_token,
                        contribution_id,
                    ),
                    on_success,
                )

            def correct_selected():
                selection = listing.curselection()
                if not selection:
                    messagebox.showerror("Corrección", "Selecciona una aportación.", parent=window)
                    return
                contribution_id = records[selection[0]].contribution_id
                self._background(lambda: get_training_candidate(server_url, user_token, contribution_id),
                    lambda result: self._correction_dialog(result, server_url, user_token),
                    is_current=lambda: bool(window.winfo_exists()))

            buttons = ttk.Frame(window)
            buttons.pack(fill="x", padx=10, pady=(0, 10))
            previous_button = ttk.Button(
                buttons,
                text="Anterior",
                command=lambda: refresh(offset - 50),
                state="disabled",
            )
            previous_button.pack(side="left", padx=(0, 6))
            next_button = ttk.Button(
                buttons,
                text="Siguiente",
                command=lambda: refresh(offset + 50),
                state="disabled",
            )
            next_button.pack(side="left", padx=(0, 6))
            ttk.Button(buttons, text="Actualizar", command=refresh).pack(
                side="left",
                padx=(0, 6),
            )
            ttk.Button(
                buttons,
                text="Retirar seleccionada",
                command=withdraw_selected,
            ).pack(side="left")
            ttk.Button(buttons, text="Corregir respuesta", command=correct_selected).pack(side="left", padx=6)
            ttk.Label(
                window,
                text=(
                    "Las aportaciones pendientes no se entrenan automáticamente. "
                    "Una vez congeladas en un dataset o incorporadas a pesos "
                    "publicados, su influencia no puede garantizarse reversible."
                ),
                wraplength=720,
            ).pack(fill="x", padx=10, pady=(0, 10))
            refresh()

        def _start_contribution(self) -> None:
            if self.node_registration_active:
                self.status_var.set("El registro del nodo ya está en curso.")
                return
            if not self.consent_var.get():
                messagebox.showerror(
                    "AI@Home",
                    "Confirma que deseas aportar cómputo voluntariamente.",
                    parent=self,
                )
                return
            if self.session_token is None:
                messagebox.showerror(
                    "AI@Home",
                    "Inicia sesión antes de registrar el nodo.",
                    parent=self,
                )
                return
            user_token = self.session_token
            if self.node_thread is not None and self.node_thread.is_alive():
                self.status_var.set("El nodo ya está ejecutándose.")
                return

            try:
                server_url = normalize_base_url(self.server_var.get())
                if server_url != self.session_server_url:
                    raise ValueError(
                        "Inicia sesión en el servidor seleccionado antes de "
                        "registrar el nodo."
                    )
                existing_id = self.settings.get("node_id", "")
                existing_server = self.settings.get("node_server_url", "")
                if existing_id and existing_server != server_url:
                    raise ValueError(
                        "Revoke the registered node on its original server before "
                        "switching this device to another server."
                    )
                token = (
                    keyring.get_password(
                        KEYRING_SERVICE,
                        f"node:{existing_server}:{existing_id}",
                    )
                    if existing_id and existing_server == server_url
                    else None
                )
            except (ValueError, keyring.errors.KeyringError) as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return

            def register_and_store() -> NodeCredentialResponse:
                credentials = register_node_with_server(
                    server_url=server_url,
                    user_token=user_token,
                    name=os.environ.get("COMPUTERNAME", "AI-Home node")[:120],
                )
                keyring.set_password(
                    KEYRING_SERVICE,
                    f"node:{server_url}:{credentials.node_id}",
                    credentials.node_token,
                )
                return credentials

            def on_registered(credentials: NodeCredentialResponse) -> None:
                self.node_registration_active = False
                self.settings["node_id"] = credentials.node_id
                self.settings["node_server_url"] = server_url
                save_settings(self.settings)
                self._launch_node(server_url, credentials.node_id, credentials.node_token)

            if token:
                self._launch_node(server_url, existing_id, token)
            else:
                self.node_registration_active = True
                self._background(
                    register_and_store,
                    on_registered,
                    lambda: setattr(self, "node_registration_active", False),
                )

        def _launch_node(self, server_url: str, node_id: str, token: str) -> None:
            stop_event = threading.Event()
            self.node_stop = stop_event
            self.node_thread = threading.Thread(
                target=self._node_loop,
                args=(server_url, node_id, token, stop_event),
                daemon=True,
            )
            self.node_thread.start()
            self.status_var.set(f"Nodo {node_id} activo; trabajos voluntarios habilitados.")

        def _node_loop(
            self,
            server_url: str,
            node_id: str,
            token: str,
            stop_event: threading.Event,
        ) -> None:
            agent = NodeAgent(server_url=server_url, node_token=token)
            try:
                while not stop_event.is_set():
                    did_work = agent.run_once()
                    status = (
                        f"Nodo {node_id}: trabajo completado"
                        if did_work
                        else f"Nodo {node_id}: esperando trabajo"
                    )
                    self.events.put(("node_status", status))
                    stop_event.wait(10)
            except Exception as error:
                self.events.put(("error", error))
            finally:
                agent.close()
                self.events.put(("node_status", f"Nodo {node_id} detenido."))

        def _stop_contribution(self) -> None:
            if self.node_stop is None:
                self.status_var.set("No hay un nodo activo.")
                return
            self.node_stop.set()
            self.status_var.set("Deteniendo el nodo…")

        def _revoke_node(self) -> None:
            node_id = self.settings.get("node_id", "")
            node_server = self.settings.get("node_server_url", "")
            if not node_id or not node_server:
                self.status_var.set("No hay un nodo registrado en este equipo.")
                return
            if self.session_token is None:
                messagebox.showerror(
                    "AI@Home",
                    "Inicia sesión en el servidor donde se registró este nodo.",
                    parent=self,
                )
                return
            if self.session_server_url != node_server:
                messagebox.showerror(
                    "AI@Home",
                    "Cambia al servidor del nodo e inicia sesión allí para revocarlo.",
                    parent=self,
                )
                return
            if self.node_stop is not None:
                self.node_stop.set()

            def revoke() -> bool:
                with httpx.Client(base_url=node_server, timeout=20) as client:
                    response = client.delete(
                        f"/api/v1/nodes/{node_id}",
                        headers={"Authorization": f"Bearer {self.session_token}"},
                    )
                    response.raise_for_status()
                try:
                    keyring.delete_password(
                        KEYRING_SERVICE,
                        f"node:{node_server}:{node_id}",
                    )
                except keyring.errors.PasswordDeleteError:
                    return False
                return True

            def on_success(credential_deleted: bool) -> None:
                self.settings.pop("node_id", None)
                self.settings.pop("node_server_url", None)
                save_settings(self.settings)
                self.status_var.set(
                    "Nodo revocado."
                    if credential_deleted
                    else "Nodo revocado; no había una credencial local que borrar."
                )

            self._background(revoke, on_success)

        def _close(self) -> None:
            evolution = getattr(self, "evolution_worker", None)
            if evolution is not None and evolution.is_alive():
                self.evolution_cancel.set()
                self.status_var.set("Esperando a detener el experimento antes de cerrar…")
                self.after(100, self._close)
                return
            worker = getattr(self, "training_worker", None)
            if worker is not None and worker.is_alive():
                self.training_cancel.set()
                self.status_var.set("Esperando a guardar el entrenamiento antes de cerrar…")
                self.after(100, self._close)
                return
            self._stop_generation()
            if self.node_stop is not None:
                self.node_stop.set()
            self.destroy()

    try:
        app = AIHomeDesktop()
    except (OSError, RuntimeError, ValueError) as error:
        messagebox.showerror("AI@Home", str(error))
        return
    app.mainloop()


if __name__ == "__main__":
    main()

