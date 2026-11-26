"""Exercise the nested Tk callbacks without a display or real network."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ai_home.desktop as desktop


def method(name):
    tree = ast.parse(Path(desktop.__file__).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(vars(desktop))
    exec(compile(ast.Module(body=[node], type_ignores=[]), desktop.__file__, "exec"), scope)
    return scope[name]


def test_late_answer_cannot_enter_another_conversation():
    app = SimpleNamespace(
        chat_request_active=False, prompt_is_placeholder=False,
        prompt=Mock(), messages=[], memory_notes=[],
        use_server_ai_var=Mock(), session_token="private-token",
        session_server_url="https://one.example", conversation_source=None,
        api_key_var=Mock(), conversation_id="alice-chat", active_memory_ids=set(),
        _append_chat=Mock(), _background=Mock(), _layout_home=Mock(), status_var=Mock(),
    )
    app.prompt.get.return_value = "Private question"
    app.use_server_ai_var.get.return_value = True
    method("_send_message")(app)
    operation, success, failure = app._background.call_args.args
    assert len(app.messages) == 1
    # Switching account or clearing chat invalidates this pending callback.
    app.conversation_id = "bob-chat"
    app.messages = []
    app.chat_request_active = True  # Bob has his own request running.
    success("Private answer for Alice")
    failure()
    assert app.messages == []
    assert app.chat_request_active is True
    app._layout_home.assert_not_called()


def test_profile_switch_clears_chat_and_closes_old_memory_dialogs(tmp_path, monkeypatch):
    from uuid import uuid4
    import ai_home.memory as memory
    monkeypatch.setattr(memory, "memory_vault_path", lambda: tmp_path / "legacy")
    window = Mock()
    app = SimpleNamespace(
        private_windows=[window], memory_notes=["old secret"],
        prompt=Mock(), api_key_var=Mock(), prompt_is_placeholder=True,
        _clear_chat=Mock(),
    )
    method("_switch_memory_profile")(app, "https://one.example", str(uuid4()))
    assert app.memory_notes == []
    assert app.private_windows == []
    window.destroy.assert_called_once()
    app._clear_chat.assert_called_once()
    app.prompt.delete.assert_called_once_with("1.0", "end")
    assert app.prompt_is_placeholder is False


def test_stale_background_error_is_discarded_before_display():
    import queue
    events = queue.Queue()
    events.put(("error", (ValueError("Alice private data"), Mock(), lambda: False)))
    events.put(("success", (Mock(), "Alice private result", lambda: False)))
    app = SimpleNamespace(events=events, status_var=Mock(), after=Mock(), _drain_events=Mock())
    method("_drain_events")(app)
    app.status_var.set.assert_not_called()
    assert events.empty()
