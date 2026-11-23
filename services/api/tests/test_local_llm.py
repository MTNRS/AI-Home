import bz2
import gzip
import json

import pytest

from ai_home.conversation_corpus import import_oasst_conversations
from ai_home.local_llm import ModelConfig, complete_locally, train_from_corpus
from ai_home.wikipedia_corpus import import_wikipedia_dump


def test_oasst_import_keeps_reviewed_spanish_dialogue_and_lineage(tmp_path):
    source_path = tmp_path / "messages.jsonl.gz"
    records = [
        {
            "message_id": "user-1",
            "parent_id": None,
            "message_tree_id": "tree-1",
            "role": "prompter",
            "lang": "es",
            "text": "Hola",
            "deleted": False,
            "review_result": True,
            "tree_state": "ready_for_export",
        },
        {
            "message_id": "assistant-1",
            "parent_id": "user-1",
            "message_tree_id": "tree-1",
            "role": "assistant",
            "lang": "es",
            "text": "¡Hola! ¿En qué puedo ayudarte?",
            "deleted": False,
            "review_result": True,
            "rank": 0,
            "tree_state": "ready_for_export",
        },
        {
            "message_id": "user-2",
            "parent_id": None,
            "message_tree_id": "tree-2",
            "role": "prompter",
            "lang": "en",
            "text": "Not Spanish",
            "deleted": False,
            "review_result": True,
            "tree_state": "ready_for_export",
        },
    ]
    with gzip.open(source_path, "wt", encoding="utf-8") as source:
        for record in records:
            source.write(json.dumps(record, ensure_ascii=False) + "\n")

    corpus_path = tmp_path / "dialogue.txt"
    examples, attribution_path, manifest_path = import_oasst_conversations(
        source_path,
        corpus_path,
    )

    assert examples == 1
    assert corpus_path.read_text(encoding="utf-8") == (
        "### Usuario\nHola\n"
        "### AI@Home\n¡Hola! ¿En qué puedo ayudarte?\n\n"
    )
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    assert attribution["message_ids"] == ["user-1", "assistant-1"]
    assert "user_id" not in attribution
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["license"] == "Apache-2.0"
    assert manifest["purpose"] == "language_model_training"


def test_wikipedia_import_writes_conversational_examples_and_attribution(
    tmp_path,
):
    pytest.importorskip("mwparserfromhell")
    dump_path = tmp_path / "tiny.xml.bz2"
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Ejemplo</title><ns>0</ns><id>10</id>
    <revision><id>20</id><text>'''Ejemplo''' es una prueba.</text></revision>
  </page>
  <page>
    <title>Redirección</title><ns>0</ns><id>11</id>
    <redirect title="Ejemplo" />
    <revision><id>21</id><text>#REDIRECT [[Ejemplo]]</text></revision>
  </page>
</mediawiki>
"""
    dump_path.write_bytes(bz2.compress(xml.encode("utf-8")))
    corpus_path = tmp_path / "wiki.txt"

    pages, attribution_path, manifest_path = import_wikipedia_dump(
        dump_path,
        corpus_path,
        source_url=(
            "https://dumps.wikimedia.org/eswiki/latest/"
            "eswiki-latest-pages-articles-multistream.xml.bz2"
        ),
    )

    assert pages == 1
    assert "### Usuario\n¿Qué es Ejemplo?" in corpus_path.read_text(
        encoding="utf-8"
    )
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    assert attribution["title"] == "Ejemplo"
    assert attribution["revision_id"] == "20"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["purpose"] == "language_model_training"


def test_training_from_scratch_saves_both_corpus_manifests(tmp_path):
    torch = pytest.importorskip("torch")
    corpus_paths = [tmp_path / "wiki.txt", tmp_path / "dialogue.txt"]
    manifest_paths = [
        tmp_path / "wiki.manifest.json",
        tmp_path / "dialogue.manifest.json",
    ]
    for index, (corpus_path, manifest_path) in enumerate(
        zip(corpus_paths, manifest_paths, strict=True)
    ):
        corpus_path.write_text(
            ("texto wikipedia " if index == 0 else "### Usuario\nHola\n")
            * 20,
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "source": "test corpus",
                    "license": "test license",
                    "purpose": "language_model_training",
                }
            ),
            encoding="utf-8",
        )

    checkpoint_path = tmp_path / "models" / "test-model.pt"
    result = train_from_corpus(
        corpus_paths,
        checkpoint_path,
        manifest_paths=manifest_paths,
        steps=2,
        batch_size=2,
        config=ModelConfig(
            context_length=16,
            embedding_dim=16,
            layers=1,
            heads=4,
        ),
        device="cpu",
    )

    assert result.checkpoint_path == checkpoint_path
    assert result.steps == 2
    assert result.final_loss > 0
    answer = complete_locally(
        [{"role": "user", "content": "Hola"}],
        checkpoint_path=checkpoint_path,
        max_new_bytes=16,
    )
    assert answer
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert len(checkpoint["training"]["corpora"]) == 2


def test_training_refuses_corpus_without_provenance(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("texto de prueba " * 4, encoding="utf-8")

    with pytest.raises(ValueError, match="manifiesto"):
        train_from_corpus(
            [corpus_path],
            tmp_path / "model.pt",
            config=ModelConfig(
                context_length=8,
                embedding_dim=8,
                layers=1,
                heads=1,
            ),
        )
