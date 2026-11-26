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
from uuid import uuid4

import httpx
import keyring

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
    provider_completion,
)
from ai_home.privacy import contains_common_secret
from ai_home.schemas import (
    KnowledgeReference,
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


def _is_training_provider(value: str) -> TypeGuard[ProviderId]:
    return value in {"openai", "anthropic", "gemini", "openai-compatible"}


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
    provider: ProviderId,
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


def server_chat_completion(server_url, user_token, messages, **options) -> str:
    return server_chat_result(server_url, user_token, messages, **options).answer


def server_chat_result(
    server_url: str,
    user_token: str,
    messages: list[dict[str, str]],
    *,
    use_knowledge: bool = False,
    timeout_seconds: float = 180,
    transport: httpx.BaseTransport | None = None,
) -> ServerChatResult:
    with httpx.Client(
        base_url=normalize_base_url(server_url),
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={"messages": messages, **({"use_knowledge": True} if use_knowledge else {})},
        )
        response.raise_for_status()
        data = response.json()
        references = tuple(KnowledgeReference.model_validate(item) for item in data.get("references", []))
        return ServerChatResult(extract_chat_content(data), references)


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
    from tkinter import messagebox, ttk

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
            self.conversation_id = str(uuid4())
            self.conversation_source: tuple[str, str, str] | None = None
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
            provider_tab = ttk.Frame(self.page_stack, padding=20)
            account_tab = ttk.Frame(self.page_stack, padding=20)
            node_tab = ttk.Frame(self.page_stack, padding=20)
            self.page_stack.add(chat_tab, text="Chat")
            self.page_stack.add(provider_tab, text="Conectar IA")
            self.page_stack.add(account_tab, text="Cuenta y aportes")
            self.page_stack.add(node_tab, text="Colaborar")
            self._build_chat_tab(ttk, chat_tab)
            self._build_provider_tab(ttk, provider_tab)
            self._build_account_tab(ttk, account_tab)
            self._build_node_tab(ttk, node_tab)
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
            ).pack(fill="x", padx=8, pady=(0, 3))

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
            ttk.Label(
                memory_controls,
                text=(
                    "La memoria local se usa automáticamente; solo los recuerdos "
                    "relevantes se envían al proveedor. No entrena el modelo."
                ),
                foreground="#666666",
            ).pack(side="left", padx=(8, 0))
            ttk.Button(memory_controls, text="Fuentes recuperadas",
                       command=self._show_knowledge_references).pack(side="right")
            conversation = ttk.Frame(parent)
            conversation.pack(fill="both", expand=True, padx=24, pady=(0, 8))
            self.chat_messages_canvas = tk.Canvas(
                conversation,
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
                    self.status_var.set("Error")
                    messagebox.showerror("AI@Home", str(error), parent=self)
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
            selected_memories = retrieve_memories(prompt, self.memory_notes)
            request_messages = messages_with_memory_context(
                messages,
                selected_memories,
            )
            try:
                use_server = self.use_server_ai_var.get()
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
                    model = "AI@Home"
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
                    operation = lambda: server_chat_result(
                        server_url,
                        user_token,
                        request_messages,
                        use_knowledge=use_knowledge,
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
            self.messages.append({"role": "user", "content": prompt})
            self._append_chat("Tú", prompt)
            messages = list(self.messages)

            request_conversation_id = self.conversation_id

            def on_success(result: str | ServerChatResult) -> None:
                if self.conversation_id != request_conversation_id:
                    return
                answer = result.answer if isinstance(result, ServerChatResult) else result
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
                self.status_var.set(f"Respuesta de {model}{suffix}")

            self._background(
                operation,
                on_success,
                lambda: setattr(self, "chat_request_active", False) if self.conversation_id == request_conversation_id else None,
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

        def _show_knowledge_references(self) -> None:
            window = tk.Toplevel(self)
            self.private_windows.append(window)
            window.title("Fuentes recuperadas de esta conversación")
            window.geometry("760x500")
            content = tk.Text(window, wrap="word")
            content.pack(fill="both", expand=True, padx=12, pady=12)
            content.insert("end", format_knowledge_references(self.knowledge_references))
            content.configure(state="disabled")

        def _clear_chat(self) -> None:
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
                    "Solo se pueden aportar conversaciones de proveedores "
                    "externos con procedencia verificable.",
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
            provider: ProviderId = source_provider
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

