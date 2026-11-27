"""Real Tk/background visual training, live metric, cancel and checkpoint check."""
import argparse
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

from ai_home import desktop
from ai_home.foundation.vision_checkpoint import load_vision_weights
from ai_home.visual_training_desktop import open_visual_training_dialog

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--require-installed", action="store_true")
args = parser.parse_args()
if args.require_installed:
    assert "site-packages" in desktop.__file__
args.output_dir.mkdir(parents=True, exist_ok=False)
config = args.output_dir / "config.json"
config.write_text(json.dumps({"model": {"width": 32, "layers": 2, "heads": 4, "kv_heads": 2,
    "hidden": 64, "context": 32}, "vision": {"image_size": 8, "patch_size": 4},
    "training": {"steps": 200, "warmup": 1, "batch_size": 2, "checkpoint_every": 200,
                 "objective": "sft"}, "prompt": "Q", "tokenizer_vocab_size": 270}), encoding="utf-8")
desktop.load_settings = lambda: {}
desktop.load_memories = lambda *args: []
desktop.save_settings = lambda *args, **kwargs: None
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


def run(app, *unused):
    app.withdraw()
    try:
        window = open_visual_training_dialog(app)
        widgets = list(descendants(window))
        entries = [w for w in widgets if isinstance(w, ttk.Entry) and not isinstance(w, ttk.Combobox)]
        destination = args.output_dir / "training"
        for widget, value in zip(entries, (str(args.dataset), str(config), str(destination), "")):
            widget.insert(0, value)
        buttons = {w.cget("text"): w for w in widgets if isinstance(w, ttk.Button)}
        buttons["Entrenar"].invoke()
        metric_seen = False
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            app.update()
            app._drain_events()
            if not metric_seen and any(str(w.cget("text")).startswith("Paso ")
                    for w in widgets if isinstance(w, ttk.Label)):
                metric_seen = True
                buttons["Cancelar"].invoke()
            if str(buttons["Entrenar"].cget("state")) == "normal":
                break
            time.sleep(.01)
        assert metric_seen, "No real live metric was displayed"
        report = json.loads((destination / "training-report.json").read_text(encoding="utf-8"))
        assert report["cancelled"] and 0 < len(report["history"]) < 200
        _, _, checkpoint = load_vision_weights(destination / report["checkpoint"])
        assert checkpoint["training_state"]["step"] == report["history"][-1]["step"]
        shown = next(w for w in widgets if isinstance(w, tk.Text)).get("1.0", "end")
        assert report["checkpoint"] in shown
        print("Real desktop worker displayed live metrics, cancelled and saved the matching visual checkpoint.")
    finally:
        app.destroy()


tk.Tk.mainloop = run
desktop.main()
