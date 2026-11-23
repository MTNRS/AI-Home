"""Desktop client for OpenAI-compatible chat providers and volunteer nodes."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
import queue
import threading
from pathlib import Path
import sys
from typing import TypeVar

import httpx
import keyring

from ai_home.inference import chat_completion, extract_chat_content
from ai_home.node_worker import NodeAgent, register_node_with_server
from ai_home.schemas import NodeCredentialResponse, SessionResponse
from ai_home.urls import normalize_base_url

KEYRING_SERVICE = "ai-home-desktop"
DEFAULT_CHAT_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_CHAT_MODEL = "llama3.2"
T = TypeVar("T")


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
        response.raise_for_status()
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
        response.raise_for_status()


def server_chat_completion(
    server_url: str,
    user_token: str,
    messages: list[dict[str, str]],
    *,
    timeout_seconds: float = 180,
    transport: httpx.BaseTransport | None = None,
) -> str:
    with httpx.Client(
        base_url=normalize_base_url(server_url),
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={"messages": messages},
        )
        response.raise_for_status()
        return extract_chat_content(response.json())


def settings_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "ai-home" / "desktop.json"


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
            self.messages: list[dict[str, str]] = []
            self.chat_request_active = False
            self.node_registration_active = False
            self.node_stop: threading.Event | None = None
            self.node_thread: threading.Thread | None = None
            self._build_ui(ttk)
            self.protocol("WM_DELETE_WINDOW", self._close)
            self.after(100, self._drain_events)

        def _build_ui(self, ttk) -> None:
            account = ttk.LabelFrame(self, text="Cuenta y servidor AI@Home")
            account.pack(fill="x", padx=12, pady=(12, 6))
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
            ttk.Button(account, text="Crear cuenta", command=self._register).grid(
                row=1, column=2, padx=6, pady=3
            )
            ttk.Button(account, text="Salir de la cuenta", command=self._logout).grid(
                row=2, column=2, padx=6, pady=3
            )
            account.columnconfigure(1, weight=1)

            tabs = ttk.Notebook(self)
            tabs.pack(fill="both", expand=True, padx=12, pady=6)
            chat_tab = ttk.Frame(tabs, padding=12)
            node_tab = ttk.Frame(tabs, padding=12)
            tabs.add(chat_tab, text="Usar IA")
            tabs.add(node_tab, text="Colaborar")
            self._build_chat_tab(ttk, chat_tab)
            self._build_node_tab(ttk, node_tab)
            self.status_var = tk.StringVar(value="Desconectado")
            ttk.Label(self, textvariable=self.status_var, anchor="w").pack(
                fill="x",
                padx=14,
                pady=(0, 10),
            )

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
            self.provider_var = tk.StringVar(
                value=self.settings.get("chat_endpoint", DEFAULT_CHAT_ENDPOINT)
            )
            self.model_var = tk.StringVar(
                value=self.settings.get("chat_model", DEFAULT_CHAT_MODEL)
            )
            self.api_key_var = tk.StringVar()
            self._entry_row(ttk, parent, 0, "Proveedor OpenAI-compatible", self.provider_var)
            self._entry_row(ttk, parent, 1, "Modelo", self.model_var)
            self._entry_row(
                ttk,
                parent,
                2,
                "Clave API (opcional)",
                self.api_key_var,
                show="*",
            )
            ttk.Label(
                parent,
                text=(
                    "En modo directo, los mensajes se envían al proveedor indicado "
                    "(Ollama local por defecto). El modo AI@Home usa el proveedor "
                    "configurado por la instancia seleccionada."
                ),
                wraplength=800,
            ).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=8)
            self.use_server_ai_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                parent,
                text="Usar el proveedor configurado en el servidor AI@Home",
                variable=self.use_server_ai_var,
            ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=3)
            self.chat_view = tk.Text(parent, wrap="word", state="disabled", height=18)
            self.chat_view.grid(
                row=5,
                column=0,
                columnspan=2,
                sticky="nsew",
                padx=6,
                pady=6,
            )
            self.prompt = tk.Text(parent, height=4, wrap="word")
            self.prompt.grid(
                row=6,
                column=0,
                sticky="ew",
                padx=6,
                pady=6,
            )
            ttk.Button(parent, text="Enviar", command=self._send_message).grid(
                row=6,
                column=1,
                sticky="n",
                padx=6,
                pady=6,
            )
            ttk.Button(
                parent,
                text="Limpiar conversación",
                command=self._clear_chat,
            ).grid(row=7, column=0, sticky="w", padx=6, pady=6)
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(5, weight=1)

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
        ) -> None:
            self.status_var.set("Procesando…")

            def execute() -> None:
                try:
                    result = operation()
                except Exception as error:
                    self.events.put(("error", (error, on_failure)))
                else:
                    self.events.put(("success", (on_success, result)))

            threading.Thread(target=execute, daemon=True).start()

        def _drain_events(self) -> None:
            while True:
                try:
                    kind, payload = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "error":
                    error, on_failure = payload
                    if on_failure is not None:
                        on_failure()
                    self.status_var.set("Error")
                    messagebox.showerror("AI@Home", str(error), parent=self)
                elif kind == "success":
                    callback, result = payload
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
                updated_settings = {
                    **self.settings,
                    "server_url": server_url,
                    "email": email,
                }
                save_settings(updated_settings)
                self.settings = updated_settings
                self.session_token = session.access_token
                self.session_server_url = server_url
                self.password_var.set("")
                self.status_var.set(f"Sesión iniciada: {email}")

            self._background(
                lambda: login_to_server(server_url, email, password),
                on_success,
            )

        def _register(self) -> None:
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
                self.status_var.set("Cuenta creada; ya puedes iniciar sesión.")
                messagebox.showinfo(
                    "AI@Home",
                    "Cuenta creada correctamente.",
                    parent=self,
                )

            self._background(
                lambda: register_account(server_url, email, password),
                on_success,
            )

        def _logout(self) -> None:
            if self.session_token is None:
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
                self.session_token = None
                self.session_server_url = None
                self.status_var.set("Sesión cerrada.")

            self._background(
                logout_request,
                on_success,
            )

        def _send_message(self) -> None:
            if self.chat_request_active:
                return
            prompt = self.prompt.get("1.0", "end").strip()
            if not prompt:
                return
            messages = [
                *self.messages,
                {"role": "user", "content": prompt},
            ]
            try:
                use_server = self.use_server_ai_var.get()
                if use_server:
                    if self.session_token is None or self.session_server_url is None:
                        raise ValueError(
                            "Inicia sesión antes de usar el proveedor del servidor."
                        )
                    server_url = self.session_server_url
                    user_token = self.session_token
                    model = "AI@Home"
                    operation = lambda: server_chat_completion(
                        server_url,
                        user_token,
                        messages,
                    )
                else:
                    provider_url = normalize_base_url(self.provider_var.get())
                    model = self.model_var.get().strip()
                    api_key = self.api_key_var.get()
                    self.settings["chat_endpoint"] = provider_url
                    self.settings["chat_model"] = model
                    save_settings(self.settings)
                    operation = lambda: chat_completion(
                        provider_url,
                        model,
                        messages,
                        api_key=api_key,
                    )
            except ValueError as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return
            except RuntimeError as error:
                messagebox.showerror("AI@Home", str(error), parent=self)
                return

            self.prompt.delete("1.0", "end")
            self.chat_request_active = True
            self.messages.append({"role": "user", "content": prompt})
            self._append_chat("Tú", prompt)
            messages = list(self.messages)

            def on_success(answer: str) -> None:
                self.chat_request_active = False
                self.messages.append({"role": "assistant", "content": answer})
                self._append_chat("IA", answer)
                self.status_var.set(f"Respuesta de {model}")

            self._background(
                operation,
                on_success,
                lambda: setattr(self, "chat_request_active", False),
            )

        def _append_chat(self, speaker: str, content: str) -> None:
            self.chat_view.configure(state="normal")
            self.chat_view.insert("end", f"{speaker}:\n{content}\n\n")
            self.chat_view.configure(state="disabled")
            self.chat_view.see("end")

        def _clear_chat(self) -> None:
            self.messages.clear()
            self.chat_view.configure(state="normal")
            self.chat_view.delete("1.0", "end")
            self.chat_view.configure(state="disabled")

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
    except (OSError, RuntimeError) as error:
        messagebox.showerror("AI@Home", str(error))
        return
    app.mainloop()


if __name__ == "__main__":
    main()
