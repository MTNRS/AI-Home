"""Real Tk form, asynchronous plan creation, no network or account credentials."""
import argparse
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, messagebox
from ai_home import desktop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--require-installed", action="store_true")
args = parser.parse_args()
if args.require_installed:
    assert "site-packages" in desktop.__file__
assert not args.output.exists()
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
    errors = []
    messagebox.showerror = lambda *a, **kw: errors.append(a)
    try:
        app._evaluation_plan_dialog()
        window = next(w for w in app.winfo_children() if isinstance(w, tk.Toplevel)
                      and w.title() == "Planificar evaluacion distribuida")
        entries = sorted((w for w in descendants(window) if isinstance(w, ttk.Entry)),
                         key=lambda w: int(w.grid_info()["row"]))
        for entry, value in zip(entries, [args.checkpoint, args.dataset, str(args.output),
                "http://127.0.0.1:8000", "desktop-plan-owner", "2", "3"]):
            entry.delete(0, "end")
            entry.insert(0, value)
        checks = [w for w in descendants(window) if isinstance(w, ttk.Checkbutton)]
        assert len(checks) == 2 and all(not bool(app.getvar(w.cget("variable"))) for w in checks)
        start = next(w for w in descendants(window) if isinstance(w, ttk.Button)
                     and w.cget("text") == "Guardar plan local")
        start.invoke()
        assert not args.output.exists() and str(start.cget("state")) == "normal"
        checks[0].invoke()
        start.invoke()
        assert not args.output.exists()
        checks[1].invoke()
        start.invoke()
        deadline = time.monotonic() + 45
        while str(start.cget("state")) == "disabled" and time.monotonic() < deadline:
            app.update()
            time.sleep(.02)
        assert not errors, errors
        assert str(start.cget("state")) == "normal" and args.output.exists()
        plan = json.loads(args.output.read_text(encoding="utf-8"))
        assert len(plan["payloads"]) == 2 and plan["reward_credits"] == 3
        assert plan["owner_user_id"] == "desktop-plan-owner"
        assert plan["allow_public_node_processing"] is True
        assert plan["weights_public_and_redistributable"] is True
        assert window in app.private_windows
        print("Real Tk plan saved with two explicit consents, two shards and no network submission.")
    finally:
        app.destroy()

tk.Tk.mainloop = run
desktop.main()
