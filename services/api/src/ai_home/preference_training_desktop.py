"""Local reviewed preference training; no network submission or model promotion."""
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText


def open_preference_training_dialog(app):
    existing = getattr(app, "preference_training_window", None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return existing
    window = tk.Toplevel(app)
    app.preference_training_window = window
    app.private_windows.append(window)
    window.title("Entrenar preferencias revisadas")
    window.geometry("840x720")
    cancel = threading.Event()
    window.bind("<Destroy>", lambda event: cancel.set() if event.widget == window else None)
    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Requiere exportación consentida y revisión explícita de pares. "
              "Guarda un candidato local restringido. No activa ni distribuye pesos.",
              wraplength=780).pack(anchor="w")
    fields = {}
    for key, label in (("parent", "Checkpoint padre"), ("export", "Exportación revisada JSON"),
                       ("review", "Revisión de pares JSON"), ("output", "Carpeta nueva de salida"),
                       ("resume", "Reanudar checkpoint (opcional)")):
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=3)
        value = fields[key] = tk.StringVar()
        ttk.Label(row, text=label, width=32).pack(side="left")
        ttk.Entry(row, textvariable=value).pack(side="left", fill="x", expand=True)
        if key != "output":
            def choose(var=value):
                path = filedialog.askopenfilename(parent=window)
                if path:
                    var.set(path)
            ttk.Button(row, text="Elegir", command=choose).pack(side="left")
    numbers = {}
    for key, label, default, convert in (
            ("steps", "Pasos totales", "20", int),
            ("learning_rate", "Tasa de aprendizaje", "0.0003", float),
            ("beta", "Beta DPO", "0.1", float), ("seed", "Semilla", "42", int),
            ("validation_fraction", "Fracción reservada", "0.2", float),
            ("checkpoint_every", "Guardar cada N pasos", "10", int)):
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        value = tk.StringVar(value=default)
        numbers[key] = (value, convert)
        ttk.Label(row, text=label, width=32).pack(side="left")
        ttk.Entry(row, textvariable=value, width=16).pack(side="left")
    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=8)
    device = tk.StringVar(value="cpu")
    ttk.Combobox(controls, textvariable=device, values=("cpu", "cuda"),
                 state="readonly", width=8).pack(side="left")
    status = tk.StringVar(value="Reanudar exige el mismo padre, datos, ajustes y dispositivo.")
    ttk.Label(frame, textvariable=status, wraplength=780).pack(anchor="w")
    output = ScrolledText(frame, state="disabled", height=10, wrap="word")
    output.pack(fill="both", expand=True)
    # Retain all Tk variables for the lifetime of the window.
    window.preference_fields = fields
    window.preference_numbers = numbers
    window.preference_device = device
    window.preference_status = status

    def failed():
        start.configure(state="normal")
        template_button.configure(state="normal")
        status.set("No terminó. Consulta el error y los checkpoints que llegaron a guardarse.")

    def success(report):
        start.configure(state="normal")
        template_button.configure(state="normal")
        status.set("Cancelado." if report.get("cancelled") else "Candidato guardado; pendiente de evaluación independiente.")
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("1.0", json.dumps(report, ensure_ascii=False, indent=2))
        output.configure(state="disabled")

    def execute():
        values = {key: var.get().strip() for key, var in fields.items()}
        if not all(values[key] for key in ("parent", "export", "review", "output")):
            status.set("Selecciona padre, exportación, revisión y carpeta nueva.")
            return
        try:
            settings = {key: convert(var.get()) for key, (var, convert) in numbers.items()}
        except ValueError:
            status.set("Revisa los valores numéricos.")
            return
        selected_device = device.get()
        cancel.clear()
        start.configure(state="disabled")
        template_button.configure(state="disabled")
        status.set("Preparando pares y referencia. Cancelar se comprueba entre pasos.")
        metrics = queue.Queue(maxsize=1)
        done = threading.Event()
        session = app.session_token

        def record(metric):
            try:
                metrics.get_nowait()
            except queue.Empty:
                pass
            metrics.put_nowait(metric)

        def poll():
            if done.is_set():
                return
            if not window.winfo_exists() or app.session_token != session:
                cancel.set()
                return
            try:
                metric = metrics.get_nowait()
                status.set(f"Paso {metric['step']}/{settings['steps']} · pérdida DPO {metric['dpo_loss']:.6f}")
            except queue.Empty:
                pass
            window.after(100, poll)

        window.after(100, poll)

        def operation():
            from .foundation.preference_training import train_preferences
            from .foundation.training import TrainingCancelled
            try:
                return train_preferences(values["parent"], values["export"], values["review"],
                    values["output"], device=selected_device, resume=values["resume"] or None,
                    cancel=cancel, on_metric=record, **settings)
            except TrainingCancelled:
                return {"cancelled": True, "scope": "Cancelado antes de guardar un candidato nuevo"}
            finally:
                done.set()

        app._background(operation, success, failed, is_current=lambda: bool(window.winfo_exists()))

    start = ttk.Button(controls, text="Entrenar preferencias", command=execute)
    start.pack(side="left", padx=8)
    ttk.Button(controls, text="Cancelar", command=cancel.set).pack(side="left")

    def prepare_review():
        source = fields["export"].get().strip()
        if not source:
            status.set("Selecciona una exportación consentida y revisada.")
            return
        target = filedialog.asksaveasfilename(parent=window, title="Nueva plantilla pendiente de revisión",
                                            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not target:
            return
        start.configure(state="disabled")
        template_button.configure(state="disabled")
        status.set("Preparando alternativas; no se aprueba ninguna preferencia.")
        def operation():
            from .foundation.preference_data import create_review_template
            return create_review_template(source, target)
        def prepared(report):
            start.configure(state="normal")
            template_button.configure(state="normal")
            status.set(f"Plantilla guardada: {report['pending_pairs']} pares pendientes. "
                       "Revisa respuestas, orden, identidad, fecha y confirmaciones antes de entrenar.")
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("1.0", target + "\n\nEl orden de IDs no expresa preferencia. "
                          "Consulta las respuestas en la exportación original y conserva ambos archivos en privado.")
            output.configure(state="disabled")
        app._background(operation, prepared, failed, is_current=lambda: bool(window.winfo_exists()))

    template_button = ttk.Button(controls, text="Preparar revisión", command=prepare_review)
    template_button.pack(side="left", padx=8)
    return window
