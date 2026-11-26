"""Local visual training form using the same configuration workflow as the CLI."""
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText


def open_visual_training_dialog(app):
    existing = getattr(app, "visual_training_window", None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return existing
    window = tk.Toplevel(app)
    app.visual_training_window = window
    app.private_windows.append(window)
    window.title("Entrenar modelo visual experimental")
    window.geometry("780x580")
    window.minsize(640, 480)
    cancel = threading.Event()
    window.bind("<Destroy>", lambda event: cancel.set() if event.widget == window else None)
    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Solo manifiestos visuales públicos revisados. Produce un candidato local; no lo publica ni activa. La carpeta de salida debe ser nueva.",
              wraplength=580).pack(anchor="w", pady=6)
    fields = {}
    for key, label in (("dataset", "Manifiesto JSON"), ("config", "Configuración JSON"),
                       ("output", "Carpeta nueva"), ("resume", "Reanudar .pt (opcional)")):
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=4)
        value = tk.StringVar()
        fields[key] = value
        ttk.Label(row, text=label, width=24).pack(side="left")
        ttk.Entry(row, textvariable=value).pack(side="left", fill="x", expand=True)
        if key != "output":
            def choose(var=value, kind=key):
                path = filedialog.askopenfilename(parent=window, filetypes=[("Archivo", "*.pt" if kind == "resume" else "*.json")])
                if path:
                    var.set(path)
            ttk.Button(row, text="Elegir", command=choose).pack(side="left", padx=4)
    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=8)
    device = tk.StringVar(value="cpu")
    ttk.Combobox(controls, textvariable=device, values=("cpu", "cuda"), state="readonly", width=8).pack(side="left")
    status = tk.StringVar(value="Las métricas se muestran al finalizar; cancelar guarda al límite del paso.")
    ttk.Label(frame, textvariable=status, wraplength=580).pack(anchor="w", pady=5)
    output = ScrolledText(frame, wrap="word", state="disabled")
    output.pack(fill="both", expand=True)
    def success(report):
        output.configure(state="normal")
        output.delete("1.0", "end")
        lines = [f"Checkpoint: {report['checkpoint'] or 'ninguno'}",
                 f"Precisión: {report['training']['precision']}",
                 "Estado: candidato local, sin aprobación de calidad.", "",
                 "Paso | Pérdida entrenamiento | Pérdida validación | Tokens supervisados/s"]
        lines.extend(f"{m['step']} | {m['train_loss']:.6f} | {m['validation_loss']:.6f} | {m['supervised_tokens_per_second']:.2f}"
                     for m in report["history"])
        lines.extend(["", "Informe completo: training-report.json en la carpeta de salida.",
                      "Una pérdida menor no demuestra mejor comprensión visual."])
        output.insert("1.0", "\n".join(lines))
        output.configure(state="disabled")
        start.configure(state="normal")
        status.set("Cancelado con informe." if report["cancelled"] else "Candidato guardado; consulta su evaluación antes de atribuirle capacidades.")
    def failure():
        start.configure(state="normal")
        status.set("El entrenamiento no terminó correctamente. Revisa el error y los checkpoints guardados.")
    def execute():
        values = {key: value.get().strip() for key, value in fields.items()}
        if not all(values[key] for key in ("dataset", "config", "output")):
            status.set("Selecciona manifiesto, configuración y una carpeta de salida nueva.")
            return
        selected_device = device.get()
        cancel.clear()
        start.configure(state="disabled")
        status.set("Entrenando localmente…")
        metrics = queue.Queue(maxsize=1)
        done = threading.Event()
        session = app.session_token
        def record(metric):
            # Keep only the most recent display update; complete history is saved.
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
                status.set(f"Paso {metric['step']} · pérdida {metric['train_loss']:.4f} · validación {metric['validation_loss']:.4f} · {metric['supervised_tokens_per_second']:.1f} tokens/s")
            except queue.Empty:
                pass
            window.after(100, poll)
        def finished(report):
            done.set()
            success(report)
        def failed():
            done.set()
            failure()
        window.after(100, poll)
        def operation():
            from .foundation.vision_train_cli import train_from_config
            return train_from_config(values["dataset"], values["config"], values["output"],
                device=selected_device, resume=values["resume"] or None, cancel=cancel, on_metric=record)
        app._background(operation, finished, failed, is_current=lambda: bool(window.winfo_exists()))
    start = ttk.Button(controls, text="Entrenar", command=execute)
    start.pack(side="left", padx=8)
    ttk.Button(controls, text="Cancelar", command=cancel.set).pack(side="left")
    return window
