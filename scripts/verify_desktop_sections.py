"""Hidden real-Tk navigation smoke; requires a Tk-capable desktop session.
Uses empty development settings and does not train or access an account.
"""
import tkinter as tk
import argparse
from tkinter import ttk
from ai_home import desktop
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--require-installed', action='store_true')
args = parser.parse_args()
if args.require_installed:
    assert 'site-packages' in desktop.__file__, 'Expected the installed wheel, not a source checkout'
desktop.load_settings=lambda: {}
desktop.load_memories=lambda *args: []
original_top=tk.Toplevel
class HiddenTop(original_top):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.withdraw()
tk.Toplevel=HiddenTop
def run(self,*args):
    self.withdraw()
    expected={'FOUNDATION MODEL','TRAINING','EVALUATION','EVOLUTION','KNOWLEDGE','DISTRIBUTED COMPUTE','MODELS','CHAT','PRIVACY','CREDITS'}
    assert set(self.control_sections)==expected
    def descendants(widget):
        for child in widget.winfo_children():
            yield child; yield from descendants(child)
    selector=next(w for w in descendants(self) if isinstance(w,ttk.Combobox) and set(w.cget('values'))==expected)
    for name in expected:
        self.control_section_var.set(name); selector.event_generate('<<ComboboxSelected>>'); self.update()
        assert self.page_stack.select()==str(self.control_sections[name])
        assert self.control_section_var.get()==name
    assert self.transfer_probe_var.get() is False
    assert self.node_training_var.get() is False
    assert self.node_device_var.get() == 'cpu'
    assert self.node_evaluation_var.get() is False
    assert self.node_adapters_var.get() is False
    adapters = next(w for w in descendants(self.control_sections['DISTRIBUTED COMPUTE'])
                    if isinstance(w, ttk.Checkbutton) and 'entrenamiento de adaptadores' in w.cget('text'))
    adapters.invoke()
    assert self.node_adapters_var.get() is True
    assert not self.node_training_var.get() and not self.node_evaluation_var.get()
    adapters.invoke()
    assert self.node_adapters_var.get() is False
    evaluation = next(w for w in descendants(self.control_sections['DISTRIBUTED COMPUTE'])
                      if isinstance(w, ttk.Checkbutton) and 'Permitir evaluación' in w.cget('text'))
    evaluation.invoke()
    assert self.node_evaluation_var.get() is True and self.node_training_var.get() is False
    evaluation.invoke()
    assert self.node_evaluation_var.get() is False
    training = next(w for w in descendants(self.control_sections['DISTRIBUTED COMPUTE'])
                    if isinstance(w, ttk.Checkbutton) and 'Permitir entrenamiento' in w.cget('text'))
    training.invoke()
    assert self.node_training_var.get() is True
    training.invoke()
    assert self.node_training_var.get() is False
    probe = next(w for w in descendants(self.control_sections['DISTRIBUTED COMPUTE'])
                 if isinstance(w, ttk.Checkbutton) and 'rendimiento HTTP' in w.cget('text'))
    probe.invoke()
    assert self.transfer_probe_var.get() is True
    probe.invoke()
    assert self.transfer_probe_var.get() is False
    self._show_page(self.control_sections['TRAINING']); self.update()
    buttons={w.cget('text'):w for w in descendants(self.control_sections['TRAINING']) if isinstance(w,ttk.Button)}
    buttons['Entrenar o reanudar'].invoke()
    windows=[w for w in self.winfo_children() if isinstance(w,tk.Toplevel)]
    assert any(w.title()=='Entrenar Foundation' for w in windows)
    training_window = next(w for w in windows if w.title()=='Entrenar Foundation')
    disk = next(w for w in descendants(training_window)
                if isinstance(w, ttk.Checkbutton) and 'secuencias en disco' in w.cget('text'))
    assert not bool(self.getvar(disk.cget('variable')))
    disk.invoke()
    assert bool(self.getvar(disk.cget('variable')))
    self._run_evolution_dialog()
    evolution_window = next(w for w in self.winfo_children()
                            if isinstance(w, tk.Toplevel) and w.title() == 'Experimento de evolución Foundation')
    evolution_disk = next(w for w in descendants(evolution_window)
                         if isinstance(w, ttk.Checkbutton) and 'secuencias en disco' in w.cget('text'))
    assert not bool(self.getvar(evolution_disk.cget('variable')))
    evolution_disk.invoke()
    assert bool(self.getvar(evolution_disk.cget('variable')))
    for entry in descendants(evolution_window):
        if isinstance(entry, ttk.Entry) and int(entry.grid_info().get('row', -1)) < 7:
            entry.insert(0, 'navigation-fixture')
    # Stub only the job boundary: this smoke verifies UI argument transport.
    # Real experiments are separately exercised by test_foundation_evolution.py.
    from ai_home.foundation import evolution_job
    original_run, captured = evolution_job.run_evolution, []
    def capture_job(*paths, **options):
        captured.append(options)
        return {'decision': 'navigation fixture', 'reason': 'no training performed'}
    evolution_job.run_evolution = capture_job
    try:
        next(w for w in descendants(evolution_window)
             if isinstance(w, ttk.Button) and w.cget('text') == 'Ejecutar experimento').invoke()
        self.evolution_worker.join(timeout=5)
        assert not self.evolution_worker.is_alive()
        assert len(captured) == 1 and captured[0]['disk_sequences'] is True
    finally:
        evolution_job.run_evolution = original_run
    self.destroy()
    print('Ten real Tk sections route correctly; Training opens existing form. No visual or training capability claim.')
tk.Tk.mainloop=run
desktop.main()
