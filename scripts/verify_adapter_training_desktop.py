"""Real Tk adapter training, cancellation and exact resume with authored fixtures."""
import argparse
import json
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import torch
from ai_home import desktop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--require-installed', action='store_true')
parser.add_argument('--objective', choices=('causal', 'sft'), default='causal')
args = parser.parse_args()
if args.require_installed:
    assert 'site-packages' in desktop.__file__
assert not args.output_dir.exists()
dataset = args.fixture_dir / 'dataset.json'
if args.objective == 'sft':
    args.output_dir.mkdir(parents=True)
    records = []
    for i in range(8):
        prompt = f'Pregunta {i}: responde A.'
        records.append({'source': 'Authored SFT adapter UI fixture', 'url': 'local:adapter-sft-ui',
            'license': 'Apache-2.0', 'date': '2026-11-26', 'language': 'es',
            'visibility': 'public', 'training_allowed': True, 'privacy_reviewed': True,
            'text': prompt + '\nA', 'messages': [{'role': 'user', 'content': prompt},
                                                {'role': 'assistant', 'content': 'A'}]})
    dataset = args.output_dir / 'dataset.json'
    dataset.write_text(json.dumps({'format_version': 1, 'records': records}), encoding='utf-8')
torch.set_num_threads(1)
desktop.load_settings = lambda: {}
desktop.load_memories = lambda *a: []
desktop.save_settings = lambda *a: None

def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)

def run(app, *unused):
    app.withdraw()
    errors = []
    messagebox.showerror = lambda *a, **kw: errors.append(a)
    original_step = torch.optim.AdamW.step
    release = threading.Event()
    try:
        next(w for w in descendants(app) if isinstance(w, ttk.Button)
             and w.cget('text') == 'Entrenar adaptadores').invoke()
        window = app.adapter_training_window
        window.withdraw()
        for key, name in (('parent', 'parent.pt'), ('dataset', 'dataset.json')):
            window.adapter_fields[key].set(str((args.fixture_dir / name).resolve()))
        assert window.adapter_objective.get() == 'causal'
        window.adapter_objective.set(args.objective)
        window.adapter_fields['dataset'].set(str(dataset.resolve()))
        window.adapter_numbers['steps'][0].set('2')
        window.adapter_numbers['checkpoint_every'][0].set('1')
        button = next(w for w in descendants(window) if isinstance(w, ttk.Button)
                      and w.cget('text') == 'Entrenar adaptadores')
        cancel = next(w for w in descendants(window) if isinstance(w, ttk.Button)
                      and w.cget('text') == 'Cancelar')
        def launch(name):
            window.adapter_fields['output'].set(str((args.output_dir / name).resolve()))
            button.invoke()
        def wait_finished():
            deadline = time.monotonic() + 60
            while str(button.cget('state')) == 'disabled' and time.monotonic() < deadline:
                app.update()
                time.sleep(.02)
            assert not errors, errors
            assert str(button.cget('state')) == 'normal'
        second = threading.Event()
        updates = 0
        def pause_second(optimizer, *a, **kw):
            nonlocal updates
            result = original_step(optimizer, *a, **kw)
            updates += 1
            if updates == 2:
                second.set()
                assert release.wait(20)
            return result
        torch.optim.AdamW.step = pause_second
        launch('full')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            app.update()
            if second.is_set() and 'Paso 1/2' in window.adapter_status.get():
                break
            time.sleep(.02)
        assert second.is_set() and 'Paso 1/2' in window.adapter_status.get()
        assert 'tokens/s' in window.adapter_status.get()
        release.set()
        wait_finished()
        release.clear()
        reached = threading.Event()
        def pause(optimizer, *a, **kw):
            result = original_step(optimizer, *a, **kw)
            reached.set()
            assert release.wait(20)
            return result
        torch.optim.AdamW.step = pause
        launch('cancelled')
        deadline = time.monotonic() + 20
        while not reached.is_set() and time.monotonic() < deadline:
            app.update()
            time.sleep(.02)
        assert reached.is_set()
        cancel.invoke()
        release.set()
        wait_finished()
        torch.optim.AdamW.step = original_step
        cancelled = json.loads((args.output_dir / 'cancelled/report.json').read_text())
        assert cancelled['cancelled'] and cancelled['step'] == 1 and cancelled['candidate'] is None
        window.adapter_fields['resume'].set(str((args.output_dir / 'cancelled/step-1').resolve()))
        launch('resumed')
        wait_finished()
        full = torch.load(args.output_dir / 'full/step-2/adapter.pt', weights_only=True)
        resumed = torch.load(args.output_dir / 'resumed/step-2/adapter.pt', weights_only=True)
        assert all(torch.equal(full['factors'][k], resumed['factors'][k]) for k in full['factors'])
        report = json.loads((args.output_dir / 'resumed/report.json').read_text())
        assert report['objective'] == args.objective
        if args.objective == 'sft':
            assert report['run_resources']['supervised_tokens'] == 4
        reached.clear()
        release.clear()
        queued = threading.Event()
        original_put = app.events.put
        def record_event(event, *a, **kw):
            original_put(event, *a, **kw)
            if event[0] in ('success', 'error'):
                queued.set()
        app.events.put = record_event
        torch.optim.AdamW.step = pause
        window.adapter_fields['resume'].set('')
        launch('profile-cancelled')
        deadline = time.monotonic() + 20
        while not reached.is_set() and time.monotonic() < deadline:
            app.update()
            time.sleep(.02)
        assert reached.is_set()
        app._switch_memory_profile()
        assert not window.winfo_exists()
        release.set()
        deadline = time.monotonic() + 20
        while not queued.is_set() and time.monotonic() < deadline:
            app.update()
            time.sleep(.02)
        assert queued.is_set(), 'Training did not terminate after profile switch'
        app._drain_events()
        app.events.put = original_put
        assert not errors, errors
        isolated = json.loads((args.output_dir / 'profile-cancelled/report.json').read_text())
        assert isolated['cancelled'] and isolated['step'] == 1 and isolated['candidate'] is None
        print('Real Tk live metrics, adapter training, cancellation, exact resume and active profile isolation passed.')
    finally:
        release.set()
        torch.optim.AdamW.step = original_step
        app.destroy()

tk.Tk.mainloop = run
desktop.main()
