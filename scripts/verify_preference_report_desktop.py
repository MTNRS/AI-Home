"""Exercise the real Tk preference-report button and asynchronous reader."""
import argparse
import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

from ai_home import desktop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--require-installed", action="store_true")
args = parser.parse_args()
if args.require_installed:
    assert "site-packages" in desktop.__file__
expected = json.loads(args.report.read_text(encoding="utf-8"))
assert expected["format"] in {"ai-home-preference-training-v1", "ai-home-adapter-training-v1"}
desktop.load_settings = lambda: {}
desktop.load_memories = lambda *a: []
desktop.save_settings = lambda *a: None
filedialog.askopenfilename = lambda **kw: str(args.report.resolve())
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
    errors = []
    messagebox.showerror = lambda *a, **kw: errors.append(a)
    try:
        label = ("Abrir resultado de preferencias" if expected["format"] == "ai-home-preference-training-v1"
                 else "Abrir informe")
        button = next(w for w in descendants(app) if isinstance(w, ttk.Button)
                      and w.cget("text") == label)
        button.invoke()
        window = next(w for w in app.winfo_children() if isinstance(w, tk.Toplevel))
        content = next(w for w in descendants(window) if isinstance(w, tk.Text))
        deadline = time.monotonic() + 10
        while not content.get("1.0", "end").strip() and time.monotonic() < deadline:
            app.update()
            time.sleep(.02)
        assert not errors, errors
        document = json.loads(content.get("1.0", "end"))
        assert document["report"] == expected
        summaries = [app.getvar(w.cget("textvariable")) for w in descendants(window)
                     if isinstance(w, ttk.Label) and w.cget("textvariable")]
        assert any(f"paso {expected['step']}" in text and "no mide calidad" in text
                   for text in summaries), summaries
        assert str(content.cget("state")) == "disabled"
        app._switch_memory_profile()
        assert not window.winfo_exists()
        print("Real Tk report button loaded exact report, displayed metrics and closed on profile switch.")
    finally:
        app.destroy()


tk.Tk.mainloop = run
desktop.main()
