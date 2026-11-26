"""Real Tk/session integration with a controlled stream, not a model-quality test."""
import argparse
import threading
import time
import tkinter as tk
from tkinter import ttk

from ai_home import desktop, visual_desktop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--require-installed", action="store_true")
args = parser.parse_args()
if args.require_installed:
    assert "site-packages" in desktop.__file__
desktop.load_settings = lambda: {}
desktop.load_memories = lambda *args: []
desktop.save_settings = lambda *args, **kwargs: None
started, stopped = threading.Event(), threading.Event()


def controlled_stream(checkpoint, image, prompt, device, cancel, on_chunk=None):
    try:
        on_chunk("PRIVATE-OLD-PROFILE")
        started.set()
        if not cancel.wait(5):
            raise AssertionError("Profile switch did not cancel stream")
        return "PRIVATE-LATE-RESULT"
    finally:
        stopped.set()


visual_desktop.run_visual = controlled_stream
original_top = tk.Toplevel


class HiddenTop(original_top):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.withdraw()


tk.Toplevel = HiddenTop


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def run(app, *args):
    app.withdraw()
    try:
        app.session_token = "synthetic-old-session"
        old = visual_desktop.open_visual_dialog(app)
        widgets = list(descendants(old))
        entries = [w for w in widgets if isinstance(w, ttk.Entry) and not isinstance(w, ttk.Combobox)]
        for widget, value in zip(entries, ("candidate.pt", "image.png", "Q")):
            widget.delete(0, "end")
            widget.insert(0, value)
        next(w for w in widgets if isinstance(w, ttk.Button) and w.cget("text") == "Generar").invoke()
        assert started.wait(2), "Worker did not start"
        app.session_token = "synthetic-new-session"
        node_stop = threading.Event()
        app.node_stop = node_stop
        app.consent_var.set(True)
        app.node_training_var.set(True)
        app._switch_memory_profile("https://example.invalid", "00000000-0000-0000-0000-000000000002")
        assert node_stop.is_set() and app.node_stop is None
        assert not app.consent_var.get() and not app.node_training_var.get()
        assert not old.winfo_exists()
        assert stopped.wait(2), "Destroyed profile window did not cancel producer"
        new = visual_desktop.open_visual_dialog(app)
        # Let the actual background success event arrive and run the actual guard.
        deadline = time.monotonic() + .25
        while time.monotonic() < deadline:
            app.update()
            app._drain_events()
            time.sleep(.01)
        text = next(w for w in descendants(new) if isinstance(w, tk.Text))
        assert text.get("1.0", "end").strip() == "", "Old private result leaked into new window"
        assert app.visual_window is new
        print("Real profile switch cancelled visual producer and discarded old queued/final output.")
    finally:
        app.destroy()


tk.Tk.mainloop = run
desktop.main()
