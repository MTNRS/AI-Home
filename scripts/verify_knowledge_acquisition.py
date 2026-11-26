"""Real Tk acquisition with controlled HTTP, not live source availability."""
import argparse
import json
from pathlib import Path
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import httpx
from ai_home import desktop, knowledge_connectors

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--require-installed", action="store_true")
args = parser.parse_args()
if args.require_installed:
    assert "site-packages" in desktop.__file__
args.output_dir.mkdir(parents=True, exist_ok=False)
manifest, output = args.output_dir / "manifest.json", args.output_dir / "pending.jsonl"
manifest.write_text(json.dumps([dict(connector="wikipedia-revision-v1", language="es",
                                    revision=123, license="fixture declared license")]), encoding="utf-8")
def respond(request):
    assert request.url.host == "es.wikipedia.org" and request.url.params["revids"] == "123"
    assert "authorization" not in request.headers
    return httpx.Response(200, json={"query": {"pages": [{"pageid": 42, "ns": 0, "title": "Fixture",
        "revisions": [{"revid": 123, "timestamp": "2026-11-26T00:00:00Z", "slots": {
            "main": {"contentmodel": "wikitext", "content": "A public [[fixture]] for acquisition."}}}]}]}})
original_acquire = knowledge_connectors.acquire_manifest
knowledge_connectors.acquire_manifest = lambda manifest, output, **options: original_acquire(
    manifest, output, transport=httpx.MockTransport(respond), **options)
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
        app._knowledge_acquisition_dialog()
        window = next(w for w in app.winfo_children() if isinstance(w, tk.Toplevel))
        entries = sorted((w for w in descendants(window) if isinstance(w, ttk.Entry)),
                         key=lambda w: int(w.grid_info()["row"]))
        for entry, path in zip(entries, (manifest, output)):
            entry.insert(0, str(path.resolve()))
        start = next(w for w in descendants(window) if isinstance(w, ttk.Button)
                     and w.cget("text") == "Descargar para revision")
        start.invoke()
        deadline = time.monotonic() + 20
        while str(start.cget("state")) == "disabled" and time.monotonic() < deadline:
            app.update()
            time.sleep(.02)
        assert not errors, errors
        row = json.loads(output.read_text(encoding="utf-8"))
        assert row["revision_id"] == "123" and row["visibility"] == "pending-review"
        assert not row["training_allowed"] and not row["retrieval_allowed"] and not row["privacy_reviewed"]
        started, release = threading.Event(), threading.Event()
        class PausedStream(httpx.SyncByteStream):
            def __iter__(self):
                started.set()
                assert release.wait(10)
                yield b"cancelled before parsing"
        knowledge_connectors.acquire_manifest = lambda manifest, output, **options: original_acquire(
            manifest, output, transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=PausedStream())),
            **options)
        cancelled_output = args.output_dir / "cancelled.jsonl"
        entries[1].delete(0, "end")
        entries[1].insert(0, str(cancelled_output.resolve()))
        try:
            start.invoke()
            assert started.wait(5)
            next(w for w in descendants(window) if isinstance(w, ttk.Button)
                 and w.cget("text") == "Cancelar").invoke()
        finally:
            release.set()
        deadline = time.monotonic() + 10
        while str(start.cget("state")) == "disabled" and time.monotonic() < deadline:
            app.update()
            time.sleep(.02)
        assert str(start.cget("state")) == "normal" and not cancelled_output.exists() and not errors
        assert any("cancelada" in str(app.getvar(w.cget("textvariable"))) for w in descendants(window)
                   if isinstance(w, ttk.Label) and w.cget("textvariable"))
        app._switch_memory_profile()
        assert not window.winfo_exists()
        print("Real Tk acquired pending source and cancelled active HTTP without publishing; profile closes window.")
    finally:
        app.destroy()
tk.Tk.mainloop = run
desktop.main()
