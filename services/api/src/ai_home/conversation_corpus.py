"""Prepare Spanish dialogue examples from the licensed OpenAssistant OASST1 export."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


OASST1_MESSAGES_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
    "2023-04-12_oasst_ready.messages.jsonl.gz"
)


def _validate_source_url(source_url: str) -> str:
    parsed = urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"huggingface.co", "github.com"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "La procedencia debe ser una URL HTTPS de Hugging Face o GitHub."
        )
    return source_url


def _is_usable_message(record: dict[str, Any], language: str) -> bool:
    if (
        record.get("lang") != language
        or record.get("deleted") is not False
        or record.get("review_result") is not True
        or record.get("tree_state") != "ready_for_export"
    ):
        return False
    role = record.get("role")
    if role not in {"prompter", "assistant"}:
        return False
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        return False
    message_id = record.get("message_id")
    if not isinstance(message_id, str) or not message_id.strip():
        return False
    if role == "assistant" and record.get("rank") not in (0, None):
        return False

    labels = record.get("labels")
    if isinstance(labels, dict):
        excluded_labels = {
            "spam",
            "lang_mismatch",
            "not_appropriate",
            "hate_speech",
            "sexual_content",
            "pii",
        }
        for label in excluded_labels:
            value = labels.get(label)
            if isinstance(value, dict):
                score = value.get("value")
                if isinstance(score, (int, float)) and score > 0:
                    return False
    return True


def import_oasst_conversations(
    source_path: Path,
    corpus_path: Path,
    *,
    source_url: str = OASST1_MESSAGES_URL,
    language: str = "es",
) -> tuple[int, Path, Path]:
    source_path = Path(source_path)
    corpus_path = Path(corpus_path)
    source_url = _validate_source_url(source_url)
    if not source_path.is_file() or source_path.suffix.casefold() != ".gz":
        raise ValueError("Selecciona el archivo de mensajes OASST1 comprimido en .gz.")
    if not language or len(language) > 16:
        raise ValueError("El código de idioma no es válido.")

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    attribution_path = Path(f"{corpus_path}.attribution.jsonl")
    manifest_path = Path(f"{corpus_path}.manifest.json")
    temporary_corpus = Path(f"{corpus_path}.tmp")
    temporary_attribution = Path(f"{attribution_path}.tmp")
    temporary_manifest = Path(f"{manifest_path}.tmp")
    valid_messages: dict[str, dict[str, Any]] = {}

    try:
        with gzip.open(source_path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON inválido en {source_path}, línea {line_number}."
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Registro inválido en {source_path}, línea {line_number}."
                    )
                if not _is_usable_message(record, language):
                    continue
                message_id = record["message_id"]
                if message_id in valid_messages:
                    raise ValueError(
                        f"El dataset repite el mensaje {message_id!r}."
                    )
                parent_id = record.get("parent_id")
                if parent_id is not None and not isinstance(parent_id, str):
                    continue
                valid_messages[message_id] = {
                    "message_id": message_id,
                    "parent_id": parent_id,
                    "message_tree_id": record.get("message_tree_id"),
                    "role": record["role"],
                    "text": record["text"].strip(),
                }

        child_ids = {
            record["parent_id"]
            for record in valid_messages.values()
            if record["parent_id"] is not None
        }
        examples: list[list[dict[str, Any]]] = []
        for record in valid_messages.values():
            if record["role"] != "assistant" or record["message_id"] in child_ids:
                continue
            chain: list[dict[str, Any]] = []
            seen: set[str] = set()
            current: dict[str, Any] | None = record
            while current is not None:
                message_id = current["message_id"]
                if message_id in seen:
                    chain = []
                    break
                seen.add(message_id)
                chain.append(current)
                parent_id = current["parent_id"]
                if parent_id is None:
                    break
                current = valid_messages.get(parent_id)
                if current is None:
                    chain = []
                    break
            chain.reverse()
            if (
                chain
                and chain[0]["role"] == "prompter"
                and chain[-1]["role"] == "assistant"
                and all(
                    left["role"] != right["role"]
                    for left, right in zip(chain, chain[1:])
                )
            ):
                examples.append(chain)
        if not examples:
            raise ValueError(
                "No se encontraron conversaciones completas y revisadas en "
                f"el idioma {language!r}."
            )

        with (
            temporary_corpus.open("w", encoding="utf-8", newline="\n") as corpus,
            temporary_attribution.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as attribution,
        ):
            for chain in examples:
                for message in chain:
                    role = "Usuario" if message["role"] == "prompter" else "AI@Home"
                    corpus.write(f"### {role}\n{message['text']}\n")
                corpus.write("\n")
                attribution.write(
                    json.dumps(
                        {
                            "dataset": "OpenAssistant OASST1",
                            "source_url": source_url,
                            "license": "Apache-2.0",
                            "message_ids": [
                                message["message_id"] for message in chain
                            ],
                            "message_tree_id": chain[-1]["message_tree_id"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        manifest = {
            "source": "OpenAssistant OASST1",
            "source_url": source_url,
            "dump_filename": source_path.name,
            "language": language,
            "license": "Apache-2.0",
            "purpose": "language_model_training",
            "conversation_examples": len(examples),
            "filtered_messages": len(valid_messages),
            "attribution_file": attribution_path.name,
            "accessed_at": datetime.now(timezone.utc).isoformat(),
            "notice": (
                "Only Spanish, reviewed, non-deleted ready-for-export messages "
                "without positive spam, inappropriate-content or PII labels "
                "were considered. This filter is not a safety guarantee; "
                "review content and dataset terms before redistributing weights."
            ),
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_corpus.replace(corpus_path)
        temporary_attribution.replace(attribution_path)
        temporary_manifest.replace(manifest_path)
    except (OSError, EOFError):
        for temporary_path in (
            temporary_corpus,
            temporary_attribution,
            temporary_manifest,
        ):
            temporary_path.unlink(missing_ok=True)
        raise

    return len(examples), attribution_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara diálogos en español del corpus OASST1."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-url", default=OASST1_MESSAGES_URL)
    parser.add_argument("--language", default="es")
    args = parser.parse_args()
    examples, attribution, manifest = import_oasst_conversations(
        args.source,
        args.output,
        source_url=args.source_url,
        language=args.language,
    )
    print(f"Ejemplos de conversación importados: {examples}")
    print(f"Corpus: {args.output}")
    print(f"Atribuciones: {attribution}")
    print(f"Manifiesto: {manifest}")


if __name__ == "__main__":
    main()
