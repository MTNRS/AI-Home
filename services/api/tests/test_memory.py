import pytest

from ai_home.memory import (
    delete_memory,
    load_memories,
    memory_context,
    messages_with_memory_context,
    retrieve_memories,
    save_memory,
)


def test_memories_round_trip_as_obsidian_markdown_and_links(tmp_path):
    first = save_memory(
        "Trabajo",
        "Estoy creando una aplicación colaborativa.",
        vault_path=tmp_path,
    )
    second = save_memory(
        "Preferencias de desarrollo",
        "Prefiero Python y pruebas automatizadas.",
        links=[first.id],
        vault_path=tmp_path,
    )

    loaded = load_memories(tmp_path)
    assert {note.id: note for note in loaded} == {
        first.id: first,
        second.id: second,
    }
    markdown = (tmp_path / f"{first.id}.md").read_text(encoding="utf-8")
    assert markdown.startswith("---\nid: ")
    assert "title: \"Trabajo\"" in markdown
    assert "Estoy creando una aplicación colaborativa." in markdown
    linked_markdown = (tmp_path / f"{second.id}.md").read_text(encoding="utf-8")
    assert f"[[{first.id}|Trabajo]]" in linked_markdown


def test_linked_note_preserves_its_own_links_heading_and_following_text(tmp_path):
    linked = save_memory("Referencia", "Contenido enlazado.", vault_path=tmp_path)
    content = "Antes.\n\n## Enlaces\n\nEste encabezado pertenece a mi nota.\nDespués."

    note = save_memory(
        "Nota con encabezado",
        content,
        links=[linked.id],
        vault_path=tmp_path,
    )

    assert {item.id: item for item in load_memories(tmp_path)} == {
        linked.id: linked,
        note.id: note,
    }
    markdown = (tmp_path / f"{note.id}.md").read_text(encoding="utf-8")
    assert "<!-- ai-home:links-start -->" in markdown
    assert f"[[{linked.id}|Referencia]]" in markdown


def test_legacy_generated_links_footer_is_removed_without_truncating_note(tmp_path):
    linked = save_memory("Referencia", "Contenido enlazado.", vault_path=tmp_path)
    note = save_memory(
        "Nota heredada",
        "Contenido de la nota.",
        links=[linked.id],
        vault_path=tmp_path,
    )
    path = tmp_path / f"{note.id}.md"
    markdown = path.read_text(encoding="utf-8")
    markdown = markdown.replace(
        "<!-- ai-home:links-start -->\n",
        "",
    ).replace("\n<!-- ai-home:links-end -->", "")
    path.write_text(markdown, encoding="utf-8")

    assert {item.id: item for item in load_memories(tmp_path)} == {
        linked.id: linked,
        note.id: note,
    }


def test_memory_search_ranks_title_and_returns_only_relevant_notes(tmp_path):
    work = save_memory(
        "Proyecto AI@Home",
        "Aplicación de inteligencia artificial colaborativa.",
        vault_path=tmp_path,
    )
    unrelated = save_memory(
        "Receta de pan",
        "Harina, agua, levadura y sal.",
        vault_path=tmp_path,
    )

    assert retrieve_memories("¿Cómo va mi proyecto AI Home?", [unrelated, work]) == [
        work
    ]
    assert retrieve_memories("el para con", [work, unrelated]) == []
    with pytest.raises(ValueError, match="positivo"):
        retrieve_memories("proyecto", [work], limit=0)


def test_memory_context_contains_local_notes_and_guidance(tmp_path):
    note = save_memory(
        "Proyecto",
        "La interfaz prioriza privacidad.",
        vault_path=tmp_path,
    )
    context = memory_context([note])

    assert "Recuerdos personales locales" in context
    assert "La interfaz prioriza privacidad." in context
    assert memory_context([]) == ""


def test_memory_context_is_added_only_to_request_copy(tmp_path):
    note = save_memory(
        "Proyecto",
        "La interfaz prioriza privacidad.",
        vault_path=tmp_path,
    )
    conversation = [{"role": "user", "content": "¿Qué prioriza mi proyecto?"}]

    with_memory = messages_with_memory_context(conversation, [note])
    without_memory = messages_with_memory_context(conversation, [])

    assert with_memory[0]["role"] == "system"
    assert note.content in with_memory[0]["content"]
    assert with_memory[1:] == conversation
    assert without_memory == conversation
    assert len(conversation) == 1


def test_deleting_memory_removes_links_without_changing_other_note_id(tmp_path):
    first = save_memory("Origen", "Primer recuerdo.", vault_path=tmp_path)
    second = save_memory(
        "Relacionado",
        "Segundo recuerdo.",
        links=[first.id],
        vault_path=tmp_path,
    )

    delete_memory(first.id, tmp_path)

    assert load_memories(tmp_path) == [
        type(second)(second.id, second.title, second.content, ())
    ]

