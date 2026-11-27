"""Local adapter training controls for authorized public datasets."""
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText


def open_adapter_training_dialog(app):
    existing = getattr(app, 'adapter_training_window', None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return existing
    window = tk.Toplevel(app)
    app.adapter_training_window = window
    app.private_windows.append(window)
    window.title('Entrenar adaptadores locales')
    window.geometry('840x720')
    cancel = threading.Event()
    window.bind('<Destroy>', lambda event: cancel.set() if event.widget == window else None)
    frame = ttk.Frame(window, padding=12)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='Requiere datos públicos autorizados. Guarda factores locales restringidos; '
              'no distribuye ni activa pesos. Reanudar exige los mismos datos, padre y ajustes.',
              wraplength=780).pack(anchor='w')
    fields = {}
    for key, label in (('parent', 'Checkpoint padre'), ('dataset', 'Dataset autorizado JSON'),
                       ('output', 'Carpeta nueva de salida'), ('resume', 'Carpeta step-N (opcional)')):
        row = ttk.Frame(frame)
        row.pack(fill='x', pady=4)
        var = fields[key] = tk.StringVar()
        ttk.Label(row, text=label, width=30).pack(side='left')
        ttk.Entry(row, textvariable=var).pack(side='left', fill='x', expand=True)
        if key != 'output':
            def choose(value=var, directory=key == 'resume'):
                path = (filedialog.askdirectory(parent=window) if directory
                        else filedialog.askopenfilename(parent=window))
                if path:
                    value.set(path)
            ttk.Button(row, text='Elegir', command=choose).pack(side='left')
    numbers = {}
    for key, label, default, convert in (
            ('steps', 'Pasos totales', '20', int), ('rank', 'Rango', '4', int),
            ('alpha', 'Escala alpha', '4', float), ('learning_rate', 'Tasa de aprendizaje', '.001', float),
            ('seed', 'Semilla', '42', int), ('batch_size', 'Tamaño de lote', '2', int),
            ('checkpoint_every', 'Guardar cada N pasos', '10', int)):
        row = ttk.Frame(frame)
        row.pack(fill='x', pady=2)
        var = tk.StringVar(value=default)
        numbers[key] = (var, convert)
        ttk.Label(row, text=label, width=30).pack(side='left')
        ttk.Entry(row, textvariable=var, width=16).pack(side='left')
    controls = ttk.Frame(frame)
    controls.pack(fill='x', pady=8)
    device = tk.StringVar(value='cpu')
    ttk.Combobox(controls, textvariable=device, values=('cpu', 'cuda'), state='readonly', width=8).pack(side='left')
    objective = tk.StringVar(value='causal')
    ttk.Label(controls, text='Objetivo:').pack(side='left', padx=(8, 2))
    ttk.Combobox(controls, textvariable=objective, values=('causal', 'sft'), state='readonly', width=8).pack(side='left')
    ttk.Label(frame, text='Causal: todo el texto. SFT: solo respuestas del asistente; requiere mensajes de conversación.',
              wraplength=780).pack(anchor='w')
    status = tk.StringVar(value='Preparado. Cancelar se comprueba entre pasos.')
    ttk.Label(frame, textvariable=status, wraplength=780).pack(anchor='w')
    output = ScrolledText(frame, state='disabled', height=10, wrap='word')
    output.pack(fill='both', expand=True)
    window.adapter_fields, window.adapter_numbers = fields, numbers
    window.adapter_device, window.adapter_status = device, status
    window.adapter_objective = objective

    def failed():
        start.configure(state='normal')
        status.set('No terminó. Consulta el error y los checkpoints guardados.')

    def success(report):
        start.configure(state='normal')
        status.set('Cancelado.' if report.get('cancelled') else 'Candidato guardado; requiere evaluación independiente.')
        output.configure(state='normal')
        output.delete('1.0', 'end')
        output.insert('1.0', json.dumps(report, ensure_ascii=False, indent=2))
        output.configure(state='disabled')

    def execute():
        values = {key: var.get().strip() for key, var in fields.items()}
        if not all(values[key] for key in ('parent', 'dataset', 'output')):
            status.set('Selecciona padre, dataset y carpeta nueva.')
            return
        try:
            settings = {key: convert(var.get()) for key, (var, convert) in numbers.items()}
        except ValueError:
            status.set('Revisa los valores numéricos.')
            return
        selected_device = device.get()
        selected_objective = objective.get()
        cancel.clear()
        start.configure(state='disabled')
        status.set('Entrenando. Cancelar conserva el último estado recuperable.')
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
                status.set(f"Paso {metric['step']}/{settings['steps']} · pérdida {metric['loss']:.6f} · "
                           f"{metric['tokens_per_second']:.1f} tokens/s")
            except queue.Empty:
                pass
            window.after(100, poll)
        window.after(100, poll)
        def operation():
            from .foundation.adapter_training import train_adapter
            from .foundation.training import TrainingCancelled
            try:
                return train_adapter(values['parent'], values['dataset'], values['output'],
                    device=selected_device, resume=values['resume'] or None, cancel=cancel,
                    on_metric=record, objective=selected_objective, **settings)
            except TrainingCancelled:
                return {'cancelled': True, 'scope': 'Cancelado antes de guardar un candidato nuevo'}
            finally:
                done.set()
        app._background(operation, success, failed, is_current=lambda: bool(window.winfo_exists()))

    start = ttk.Button(controls, text='Entrenar adaptadores', command=execute)
    start.pack(side='left', padx=8)
    ttk.Button(controls, text='Cancelar', command=cancel.set).pack(side='left')
    return window
