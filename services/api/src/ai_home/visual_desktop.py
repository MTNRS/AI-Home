"""Explicit local visual experiment window; no conversation persistence."""

import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText


def run_visual(checkpoint, image_path, prompt, device, cancel, on_chunk=None):
    if cancel.is_set():
        return "Cancelado."
    from .foundation.vision_checkpoint import load_vision_weights
    from .foundation.vision_images import load_image
    from .foundation.vision_inference import stream_vision_text
    from .foundation.training import select_device
    selected = select_device(device)
    model, tokenizer, _ = load_vision_weights(checkpoint, selected)
    pixels, _ = load_image(image_path, model.vision_config.image_size)
    if cancel.is_set():
        return "Cancelado."
    chunks = []
    for chunk in stream_vision_text(model, tokenizer, pixels.to(selected),
            [{"role": "user", "content": prompt}], temperature=0, max_new_tokens=64, cancel=cancel):
        chunks.append(chunk)
        if on_chunk is not None:
            on_chunk(chunk)
    answer = "".join(chunks)
    return ("Cancelado.\n" if cancel.is_set() else "") + (answer or "El modelo no produjo texto.")


def enqueue_chunk(chunks, text, cancel):
    while not cancel.is_set():
        try:
            chunks.put(text, timeout=.05)
            return
        except queue.Full:
            pass


def open_visual_dialog(app):
    existing = getattr(app, "visual_window", None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return existing
    window = tk.Toplevel(app)
    app.visual_window = window
    app.private_windows.append(window)
    window.title("Imagen local — modelo propio experimental")
    window.geometry("760x560")
    window.minsize(600, 450)
    cancel = threading.Event()
    window.bind("<Destroy>", lambda event: cancel.set() if event.widget == window else None)
    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Inferencia local experimental. La imagen y la respuesta no se guardan en el historial ni se aportan al entrenamiento.",
              wraplength=550).pack(anchor="w", pady=6)
    checkpoint, image_path = tk.StringVar(), tk.StringVar()
    for label, variable, types in (("Checkpoint visual", checkpoint, [("Pesos visuales", "*.pt")]),
                                   ("Imagen PNG/JPEG", image_path, [("Imagen", "*.png *.jpg *.jpeg")])):
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=19).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        def choose(var=variable, filetypes=types):
            selected = filedialog.askopenfilename(parent=window, filetypes=filetypes)
            if selected:
                var.set(selected)
        ttk.Button(row, text="Elegir", command=choose).pack(side="left", padx=4)
    prompt = tk.StringVar(value="Describe the image.")
    ttk.Label(frame, text="Instrucción (usa la del entrenamiento para comparar resultados)").pack(anchor="w", pady=(8, 2))
    ttk.Entry(frame, textvariable=prompt).pack(fill="x")
    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=8)
    device = tk.StringVar(value="cpu")
    ttk.Combobox(controls, textvariable=device, values=("cpu", "cuda"), state="readonly", width=8).pack(side="left")
    notice = tk.StringVar(value="Selecciona un candidato visual; los checkpoints textuales no son compatibles.")
    ttk.Label(frame, textvariable=notice, wraplength=550).pack(anchor="w", pady=5)
    result = ScrolledText(frame, height=12, wrap="word", state="disabled")
    result.pack(fill="both", expand=True)
    def finished(text):
        result.configure(state="normal")
        result.delete("1.0", "end")
        result.insert("1.0", text)
        result.configure(state="disabled")
        generate.configure(state="normal")
        notice.set("Finalizado. La salida puede ser incorrecta; no demuestra comprensión visual general.")
    def failed():
        generate.configure(state="normal")
        notice.set("No se pudo ejecutar la inferencia visual.")
    def start():
        values = checkpoint.get().strip(), image_path.get().strip(), prompt.get().strip(), device.get()
        if not all(values) or len(values[2]) > 8192:
            notice.set("Selecciona checkpoint, imagen e instrucción de hasta 8192 caracteres.")
            return
        cancel.clear()
        generate.configure(state="disabled")
        notice.set("Ejecutando localmente…")
        result.configure(state="normal")
        result.delete("1.0", "end")
        result.configure(state="disabled")
        chunks = queue.Queue(maxsize=16)
        done = threading.Event()
        session = app.session_token
        def success(text):
            done.set()
            finished(text)
        def failure():
            done.set()
            failed()
        def poll():
            if done.is_set():
                return
            if not window.winfo_exists() or app.session_token != session:
                cancel.set()
                return
            fragments = []
            for _ in range(16):
                try:
                    fragments.append(chunks.get_nowait())
                except queue.Empty:
                    break
            if fragments:
                result.configure(state="normal")
                result.insert("end", "".join(fragments))
                result.see("end")
                result.configure(state="disabled")
            window.after(40, poll)
        window.after(40, poll)
        app._background(lambda: run_visual(*values, cancel, on_chunk=lambda text: enqueue_chunk(chunks, text, cancel)), success, failure,
                        is_current=lambda: bool(window.winfo_exists()))
    generate = ttk.Button(controls, text="Generar", command=start)
    generate.pack(side="left", padx=8)
    ttk.Button(controls, text="Cancelar", command=cancel.set).pack(side="left")
    return window
