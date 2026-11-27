"""Explicit public source acquisition; never approves downloaded documents."""
import threading
import tkinter as tk
from tkinter import ttk, filedialog


def open_acquisition_dialog(app):
    window = tk.Toplevel(app)
    app.private_windows.append(window)
    cancellation = threading.Event()
    window.bind("<Destroy>", lambda event: cancellation.set() if event.widget == window else None)
    window.title("Adquirir fuentes publicas")
    window.geometry("760x340")
    window.minsize(680, 320)
    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Descarga las fuentes enumeradas en un manifiesto GitHub/Wikipedia. "
              "No concede permisos de entrenamiento ni recuperacion. Revisa derechos y privacidad antes de aprobar.",
              wraplength=640).grid(row=0, column=0, columnspan=3, sticky="w", pady=8)
    values = {}
    for row, (key, label) in enumerate((("manifest", "Manifiesto JSON"), ("output", "Archivo pendiente JSONL nuevo")), start=1):
        value = values[key] = tk.StringVar()
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=8)
        ttk.Entry(frame, textvariable=value).grid(row=row, column=1, sticky="ew")
        def choose(variable=value, kind=key):
            path = (filedialog.askopenfilename(parent=window) if kind == "manifest" else
                    filedialog.asksaveasfilename(parent=window, defaultextension=".jsonl"))
            if path:
                variable.set(path)
        ttk.Button(frame, text="Elegir", command=choose).grid(row=row, column=2, padx=5)
    frame.columnconfigure(1, weight=1)
    status = tk.StringVar(value="Cancelar se comprueba entre bloques de descarga y antes de guardar el lote.")
    ttk.Label(frame, textvariable=status, wraplength=640).grid(row=4, column=0, columnspan=3, sticky="w", pady=10)
    def execute():
        paths = {key: value.get().strip() for key, value in values.items()}
        if not all(paths.values()):
            status.set("Selecciona el manifiesto y un archivo pendiente nuevo.")
            return
        cancellation.clear()
        start.configure(state="disabled")
        status.set("Descargando fuentes publicas para revision...")
        def operation():
            from .knowledge_connectors import acquire_manifest, AcquisitionCancelled
            try:
                return acquire_manifest(paths["manifest"], paths["output"], cancel=cancellation)
            except AcquisitionCancelled:
                return {"state": "cancelled"}
        def success(result):
            if result.get("state") == "cancelled":
                status.set("Adquisicion cancelada antes de publicar el lote.")
                start.configure(state="normal")
                return
            status.set(f"Guardados {result['documents']} documentos ({result['bytes']} bytes). "
                       "Pendientes de revision de licencia y privacidad; permisos desactivados.")
            start.configure(state="normal")
        def failure():
            status.set("No se completo la adquisicion. Revisa el error antes de repetir.")
            start.configure(state="normal")
        app._background(operation, success, failure, is_current=lambda: bool(window.winfo_exists()))
    start = ttk.Button(frame, text="Descargar para revision", command=execute)
    start.grid(row=3, column=1, sticky="w", pady=12)
    ttk.Button(frame, text="Cancelar", command=cancellation.set).grid(row=3, column=2, padx=5)
    return window
