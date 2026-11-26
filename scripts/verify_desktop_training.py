"""Real asynchronous Tk Foundation training with disk sequence cache enabled."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import torch
from ai_home import desktop
from ai_home.foundation.checkpoint import load_checkpoint
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.training import TrainingConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--require-installed', action='store_true')
args = parser.parse_args()
if args.require_installed:
    assert 'site-packages' in desktop.__file__
output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
records = [{'source': 'AI@Home authored desktop training smoke', 'url': 'local:desktop-training',
    'license': 'Apache-2.0', 'date': '2026-11-26', 'language': 'es',
    'training_allowed': True, 'privacy_reviewed': True, 'visibility': 'public',
    'text': f'Documento {i}: entrenamos un modelo propio con datos autorizados.'} for i in range(8)]
dataset, model_config, training_config = [output / name for name in ('dataset.json', 'model.json', 'training.json')]
dataset.write_text(json.dumps({'format_version': 1, 'records': records}), encoding='utf-8')
model_config.write_text(json.dumps(asdict(FoundationConfig(vocab_size=270, width=16,
    layers=1, heads=2, kv_heads=1, hidden=32, context=16))), encoding='utf-8')
training_config.write_text(json.dumps(asdict(TrainingConfig(steps=4, warmup=1,
    checkpoint_every=4))), encoding='utf-8')
desktop.load_settings = lambda: {}
desktop.load_memories = lambda *a: []
desktop.save_settings = lambda *a: None
original_top = tk.Toplevel


class HiddenTop(original_top):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.withdraw()


tk.Toplevel = HiddenTop


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def run(app, *unused):
    app.withdraw()
    try:
        app._run_training_dialog()
        window = next(w for w in app.winfo_children() if isinstance(w, tk.Toplevel)
                      and w.title() == 'Entrenar Foundation')
        widgets = list(descendants(window))
        entries = [w for w in widgets if isinstance(w, ttk.Entry) and not isinstance(w, ttk.Combobox)]
        for widget, value in zip(entries, (dataset, model_config, training_config, output / 'run')):
            widget.insert(0, str(value))
        cache = next(w for w in widgets if isinstance(w, ttk.Checkbutton) and 'secuencias en disco' in w.cget('text'))
        cache.invoke()
        start = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == 'Entrenar')
        start.invoke()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            app.update()
            if (output / 'run/result.json').exists() and not app.training_worker.is_alive() and str(start.cget('state')) == 'normal':
                break
            time.sleep(.02)
        assert (output / 'run/result.json').is_file(), 'Desktop training did not finish'
        app.update()
        result = json.loads((output / 'run/result.json').read_text())
        configuration = json.loads((output / 'run/configuration.json').read_text())
        assert configuration['disk_sequences'] is True
        assert result['step'] == 4 and result['training_complete']
        assert result['embedding_delta_l2'] > 0
        assert any(result['checkpoint'] in str(w.cget('text')) for w in widgets if isinstance(w, ttk.Label))
        _, _, payload = load_checkpoint(result['checkpoint'])
        assert payload['step'] == 4
        for split in ('train', 'validation'):
            manifest = json.loads((output / f'run/sequence-cache/{split}/manifest.json').read_text())
            assert manifest['rows'] > 0
            assert manifest['bytes'] == (output / f'run/sequence-cache/{split}/sequences.i64').stat().st_size
        print('Real Tk training completed four updates with disk sequences and a loadable checkpoint.')
    finally:
        if getattr(app, 'training_cancel', None) is not None:
            app.training_cancel.set()
        if getattr(app, 'training_worker', None) is not None:
            app.training_worker.join(timeout=10)
            assert not app.training_worker.is_alive()
        app.destroy()


tk.Tk.mainloop = run
desktop.main()
