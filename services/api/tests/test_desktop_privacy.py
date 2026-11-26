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
        stream_enabled_var=Mock(), chat_preview_var=Mock(), generation_settings=None,
        prompt=Mock(), messages=[], memory_notes=[],
        pinned_server_version=None, server_version_var=Mock(),
        use_server_ai_var=Mock(), use_knowledge_var=Mock(return_value=False), session_token="private-token",
        session_server_url="https://one.example", conversation_source=None,
        api_key_var=Mock(), conversation_id="alice-chat", active_memory_ids=set(),
        _append_chat=Mock(), _background=Mock(), _layout_home=Mock(), status_var=Mock(),
    )
    app.server_version_var.get.return_value = desktop.DEFAULT_SERVER_VERSION
    app.prompt.get.return_value = "Private question"
    app.use_server_ai_var.get.return_value = True
    app.use_knowledge_var.get.return_value = False
    app.stream_enabled_var.get.return_value = False
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
        prompt=Mock(), api_key_var=Mock(), use_knowledge_var=Mock(), prompt_is_placeholder=True,
        _clear_chat=Mock(), server_version_combo=Mock(), stream_enabled_var=Mock(),
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


def test_retrieval_conversation_cannot_lose_provenance_in_training_submission():
    app = SimpleNamespace(conversation_used_knowledge=True)
    callback = method("_submit_training_candidate")
    dialog = Mock()
    callback.__globals__["messagebox"] = dialog
    callback(app)
    dialog.showerror.assert_called_once()
    assert "licencias" in dialog.showerror.call_args.args[1]


def test_server_response_pins_next_turn_and_records_version_provenance():
    from uuid import uuid4
    version = str(uuid4())
    app = SimpleNamespace(
        chat_request_active=False, prompt_is_placeholder=False,
        stream_enabled_var=Mock(), chat_preview_var=Mock(), generation_settings=None,
        prompt=Mock(), messages=[], memory_notes=[], use_server_ai_var=Mock(),
        use_knowledge_var=Mock(), pinned_server_version=None, server_version_var=Mock(),
        session_token="token", session_server_url="https://example.org", conversation_source=None,
        api_key_var=Mock(), conversation_id="chat", active_memory_ids=set(),
        _append_chat=Mock(), _background=Mock(), _layout_home=Mock(), status_var=Mock(),
    )
    app.prompt.get.return_value = "Hi"
    app.use_server_ai_var.get.return_value = True
    app.use_knowledge_var.get.return_value = False
    app.stream_enabled_var.get.return_value = False
    app.server_version_var.get.return_value = desktop.DEFAULT_SERVER_VERSION
    send = method("_send_message")
    send(app)
    app._background.call_args.args[1](desktop.ServerChatResult("answer", model_version=version))
    assert app.pinned_server_version == version
    assert app.conversation_source[1] == f"AI@Home:{version}"
    # Even if the selector says active, the next request keeps the established version.
    completion = Mock(return_value=desktop.ServerChatResult("next", model_version=version))
    send.__globals__["server_chat_result"] = completion
    send(app)
    app._background.call_args.args[0]()
    assert completion.call_args.kwargs["model_version"] == version


def test_stop_sets_cancellation_and_clears_preview():
    import threading
    cancel = threading.Event()
    app = SimpleNamespace(chat_cancel=cancel, chat_preview_var=Mock(), status_var=Mock())
    method("_stop_generation")(app)
    assert cancel.is_set()
    app.chat_preview_var.set.assert_called_with("")


def test_late_stream_fragments_are_discarded_after_stop_or_conversation_change():
    import queue
    import threading
    current, stopped = threading.Event(), threading.Event()
    stopped.set()
    events = queue.Queue()
    events.put(("chat_delta", ("old-chat", current, "old private text")))
    events.put(("chat_delta", ("current-chat", stopped, "cancelled text")))
    events.put(("chat_delta", ("current-chat", current, "accepted")))
    preview = Mock()
    preview.get.return_value = ""
    app = SimpleNamespace(events=events, conversation_id="current-chat", chat_cancel=current,
                          chat_preview_var=preview, after=Mock(), _drain_events=Mock())
    method("_drain_events")(app)
    preview.set.assert_called_once_with("accepted")
