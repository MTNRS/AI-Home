"""Local, explicit-consent planning for public distributed evaluation."""
import tkinter as tk
from tkinter import ttk, filedialog


def open_evaluation_plan_dialog(app):
    window = tk.Toplevel(app)
    app.private_windows.append(window)
    window.title("Planificar evaluacion distribuida")
    window.geometry("800x560")
    window.minsize(700, 540)
    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Crea un plan local para datos y pesos publicos. No envia trabajos ni reserva creditos. "
              "El archivo contiene los pesos y los textos del holdout.", wraplength=660).grid(
                  row=0, column=0, columnspan=3, sticky="w", pady=8)
    fields = {}
    specifications = (("checkpoint", "Checkpoint", ""), ("dataset", "Dataset holdout JSON", ""),
        ("output", "Plan nuevo JSON", ""), ("server", "Servidor", app.session_server_url or "http://localhost:8000"),
        ("owner", "ID del propietario", ""), ("shards", "Fragmentos (1-8)", "2"),
        ("reward", "Creditos por fragmento", "0"))
    for row, (key, label, default) in enumerate(specifications, start=1):
        value = fields[key] = tk.StringVar(value=default)
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=value).grid(row=row, column=1, sticky="ew")
        if key in {"checkpoint", "dataset", "output"}:
            def choose(variable=value, kind=key):
                if kind == "output":
                    path = filedialog.asksaveasfilename(parent=window, defaultextension=".json")
                else:
                    path = filedialog.askopenfilename(parent=window)
                if path:
                    variable.set(path)
            ttk.Button(frame, text="Elegir", command=choose).grid(row=row, column=2, padx=5)
    frame.columnconfigure(1, weight=1)
    public_data, public_weights = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="Autorizo procesar estos datos publicos en nodos voluntarios", variable=public_data).grid(
        row=8, column=0, columnspan=3, sticky="w", pady=6)
    ttk.Checkbutton(frame, text="Confirmo que estos pesos son publicos y redistribuibles", variable=public_weights).grid(
        row=9, column=0, columnspan=3, sticky="w", pady=6)
    status = tk.StringVar(value="No activa modelos. Cerrar la ventana no cancela una escritura local ya iniciada.")
    ttk.Label(frame, textvariable=status, wraplength=660).grid(row=11, column=0, columnspan=3, sticky="w", pady=10)

    def execute():
        values = {key: value.get().strip() for key, value in fields.items()}
        try:
            count, reward = int(values["shards"]), int(values["reward"])
            if not all(values.values()) or not 1 <= count <= 8 or not 0 <= reward <= 1_000_000:
                raise ValueError("Completa todos los campos y revisa fragmentos y creditos.")
            if not public_data.get() or not public_weights.get():
                raise ValueError("Se requieren ambos consentimientos explicitos.")
        except ValueError as error:
            status.set(str(error))
            return
        start.configure(state="disabled")
        status.set("Validando dataset y checkpoint; creando plan local...")
        def operation():
            from .foundation.evaluation_coordinator import create_plan, dataset_shards
            return create_plan(values["checkpoint"], dataset_shards(values["dataset"], count), values["output"],
                server=values["server"], owner_user_id=values["owner"], reward_credits=reward,
                allow_public_node_processing=True, weights_public_and_redistributable=True)
        def success(plan):
            status.set(f"Plan guardado: {len(plan['payloads'])} fragmentos, {count * reward} creditos previstos. "
                       "Sin envio ni reservas. Para ejecutar, usa advance con la cuenta propietaria.")
            start.configure(state="normal")
        def failure():
            status.set("No se pudo crear el plan. Revisa el error; no se enviaron trabajos.")
            start.configure(state="normal")
        app._background(operation, success, failure, is_current=lambda: bool(window.winfo_exists()))
    start = ttk.Button(frame, text="Guardar plan local", command=execute)
    start.grid(row=10, column=1, sticky="w", pady=8)
    return window
