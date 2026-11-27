"""Train a real preference candidate through Tk using authored smoke inputs."""
import argparse
import json
from pathlib import Path
import time
import threading
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ai_home import desktop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fixture-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--require-installed", action="store_true")
args = parser.parse_args()
if args.require_installed:
    assert "site-packages" in desktop.__file__
assert not args.output_dir.exists()
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
    try:
        app._preference_training_dialog()
        window = app.preference_training_window
        window.withdraw()
        for key, name in (("parent", "parent.pt"), ("export", "synthetic-export.json"),
                          ("review", "synthetic-review.json")):
            window.preference_fields[key].set(str((args.fixture_dir / name).resolve()))
        window.preference_fields["output"].set(str((args.output_dir / "full").resolve()))
        window.preference_numbers["steps"][0].set("2")
        window.preference_numbers["validation_fraction"][0].set("0.5")
        button = next(w for w in descendants(window) if isinstance(w, ttk.Button)
                      and w.cget("text") == "Entrenar preferencias")
        def wait_finished():
            deadline = time.monotonic() + 60
            while str(button.cget("state")) == "disabled" and time.monotonic() < deadline:
                app.update()
                time.sleep(.02)
            assert not errors, errors
            assert str(button.cget("state")) == "normal"
        template_path = args.output_dir / "pending-review.json"
        filedialog.asksaveasfilename = lambda **kw: str(template_path.resolve())
        next(w for w in descendants(window) if isinstance(w, ttk.Button)
             and w.cget("text") == "Preparar revisión").invoke()
        wait_finished()
        template = json.loads(template_path.read_text(encoding="utf-8"))
        assert len(template["pairs"]) == 12 and not any(p["preference_confirmed"] for p in template["pairs"])
        assert template["reviewer_id"] == "" and template["reviewed_at_epoch"] == 0
        assert "pares pendientes" in window.preference_status.get()
        from ai_home.foundation.preference_data import load_preference_pairs
        try:
            load_preference_pairs(args.fixture_dir / "synthetic-export.json", template_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Unreviewed template was admitted")
        second_update, continue_training = threading.Event(), threading.Event()
        initial_step = torch.optim.AdamW.step
        updates = 0
        def pause_second(optimizer, *a, **kw):
            nonlocal updates
            result = initial_step(optimizer, *a, **kw)
            updates += 1
            if updates == 2:
                second_update.set()
                assert continue_training.wait(20)
            return result
        torch.optim.AdamW.step = pause_second
        try:
            button.invoke()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                app.update()
                if second_update.is_set() and "Paso 1/2" in window.preference_status.get():
                    break
                time.sleep(.02)
            assert second_update.is_set() and "Paso 1/2" in window.preference_status.get()
            assert "pérdida DPO" in window.preference_status.get()
        finally:
            continue_training.set()
            torch.optim.AdamW.step = initial_step
        wait_finished()
        report = json.loads((args.output_dir / "full/report.json").read_text(encoding="utf-8"))
        assert report["step"] == 2 and report["training_complete"]
        assert report["data_visibility"] == "reviewed-training-only" and report["promoted"] is False
        assert (args.output_dir / "full/candidate.pt").is_file()
        assert "Candidato guardado" in window.preference_status.get()
        # Pause after an actual AdamW update so the main Tk thread can press Cancel.
        updated, release = threading.Event(), threading.Event()
        original_step = torch.optim.AdamW.step
        def controlled_step(optimizer, *a, **kw):
            result = original_step(optimizer, *a, **kw)
            updated.set()
            assert release.wait(20)
            return result
        torch.optim.AdamW.step = controlled_step
        window.preference_fields["output"].set(str((args.output_dir / "cancelled").resolve()))
        try:
            button.invoke()
            deadline = time.monotonic() + 20
            while not updated.is_set() and time.monotonic() < deadline:
                app.update()
                time.sleep(.02)
            assert updated.is_set()
            next(w for w in descendants(window) if isinstance(w, ttk.Button)
                 and w.cget("text") == "Cancelar").invoke()
        finally:
            release.set()
            torch.optim.AdamW.step = original_step
        wait_finished()
        cancelled = json.loads((args.output_dir / "cancelled/report.json").read_text(encoding="utf-8"))
        assert cancelled["step"] == 1 and cancelled["cancelled"] and cancelled["candidate"] is None
        window.preference_fields["resume"].set(str((args.output_dir / "cancelled/candidate.pt").resolve()))
        window.preference_fields["output"].set(str((args.output_dir / "resumed").resolve()))
        button.invoke()
        wait_finished()
        from ai_home.foundation.checkpoint import load_checkpoint
        full, _, _ = load_checkpoint(args.output_dir / "full/candidate.pt")
        resumed, _, _ = load_checkpoint(args.output_dir / "resumed/candidate.pt")
        assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in full.state_dict().items())
        # Switch profile while the real optimizer is paused after an update.
        updated.clear()
        release.clear()
        queued = threading.Event()
        original_put = app.events.put
        def record_event(event, *a, **kw):
            original_put(event, *a, **kw)
            if event[0] in ("success", "error"):
                queued.set()
        app.events.put = record_event
        torch.optim.AdamW.step = controlled_step
        window.preference_fields["resume"].set("")
        window.preference_fields["output"].set(str((args.output_dir / "profile-cancelled").resolve()))
        try:
            button.invoke()
            deadline = time.monotonic() + 20
            while not updated.is_set() and time.monotonic() < deadline:
                app.update()
                time.sleep(.02)
            assert updated.is_set()
            app._switch_memory_profile()
            assert not window.winfo_exists()
        finally:
            release.set()
            torch.optim.AdamW.step = original_step
        deadline = time.monotonic() + 20
        while not queued.is_set() and time.monotonic() < deadline:
            app.update()
            time.sleep(.02)
        assert queued.is_set(), "Background training did not terminate"
        app._drain_events()
        app.events.put = original_put
        assert not errors, errors
        isolated = json.loads((args.output_dir / "profile-cancelled/report.json").read_text(encoding="utf-8"))
        assert isolated["cancelled"] and isolated["step"] == 1 and isolated["candidate"] is None
        assert isolated["data_visibility"] == "reviewed-training-only"
        assert not any(isinstance(w, tk.Toplevel) for w in app.winfo_children())
        print("Real Tk verified live loss, training, cancellation, exact resume and active profile-switch cancellation without late callbacks.")
    finally:
        app.destroy()


tk.Tk.mainloop = run
desktop.main()
