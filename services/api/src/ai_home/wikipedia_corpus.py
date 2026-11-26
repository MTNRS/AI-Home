"""Import a Wikimedia articles dump into a text corpus with attribution records."""

from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from xml.etree import ElementTree


def _require_wikitext_parser():
    try:
        import mwparserfromhell
    except ImportError as error:
        raise RuntimeError(
            "La importación de Wikipedia necesita mwparserfromhell. Instala "
            '`python -m pip install -e ".[llm]"`.'
        ) from error
    return mwparserfromhell


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _child_text(element: ElementTree.Element, name: str) -> str:
    return next(
        (
            child.text or ""
            for child in element
            if _local_name(child.tag) == name
        ),
        "",
    )


def _validate_source_url(source_url: str) -> str:
    parsed = urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname != "dumps.wikimedia.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "La URL del dump debe ser HTTPS y pertenecer a dumps.wikimedia.org."
        )
    return source_url


def import_wikipedia_dump(
    dump_path: Path,
    corpus_path: Path,
    *,
    source_url: str,
    language: str = "es",
) -> tuple[int, Path, Path]:
    dump_path = Path(dump_path)
    corpus_path = Path(corpus_path)
    source_url = _validate_source_url(source_url)
    if not re.fullmatch(r"[a-z0-9-]{2,16}", language):
        raise ValueError("El código de idioma de Wikipedia no es válido.")
    if not dump_path.is_file() or dump_path.suffix.casefold() != ".bz2":
        raise ValueError("Selecciona un dump XML de Wikipedia comprimido en .bz2.")

    mwparserfromhell = _require_wikitext_parser()
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    attribution_path = Path(f"{corpus_path}.attribution.jsonl")
    manifest_path = Path(f"{corpus_path}.manifest.json")
    temporary_corpus = Path(f"{corpus_path}.tmp")
    temporary_attribution = Path(f"{attribution_path}.tmp")
    temporary_manifest = Path(f"{manifest_path}.tmp")
    imported_pages = 0

    try:
        with (
            bz2.open(dump_path, "rb") as source,
            temporary_corpus.open("w", encoding="utf-8", newline="\n") as corpus,
            temporary_attribution.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as attribution,
        ):
            for _event, page in ElementTree.iterparse(source, events=("end",)):
                if _local_name(page.tag) != "page":
                    continue
                title = _child_text(page, "title")
                namespace = _child_text(page, "ns")
                page_id = _child_text(page, "id")
                revision_id = ""
                wikitext = ""
                for element in page.iter():
                    if _local_name(element.tag) != "revision":
                        continue
                    revision_id = _child_text(element, "id")
                    wikitext = next(
                        (
                            child.text or ""
                            for child in element
                            if _local_name(child.tag) == "text"
                        ),
                        "",
                    )

                is_redirect = any(
                    _local_name(child.tag) == "redirect" for child in page
                )
                if namespace == "0" and not is_redirect and title and wikitext:
                    article_text = (
                        mwparserfromhell.parse(wikitext)
                        .strip_code(normalize=True, collapse=True)
                        .strip()
                    )
                    if article_text:
                        introduction = article_text.splitlines()[0][:1200]
                        corpus.write(
                            f"### Usuario\n¿Qué es {title}?\n"
                            f"### AI@Home\n{introduction}\n\n"
                            f"### Artículo: {title}\n{article_text}\n\n"
                        )
                        attribution.write(
                            json.dumps(
                                {
                                    "title": title,
                                    "page_id": page_id,
                                    "revision_id": revision_id,
                                    "article_url": (
                                        f"https://{language}.wikipedia.org/"
                                        f"?curid={page_id}"
                                    ),
                                    "revision_url": (
                                        f"https://{language}.wikipedia.org/"
                                        f"w/index.php?oldid={revision_id}"
                                    ),
                                    "source_dump_url": source_url,
                                    "license": "CC BY-SA 4.0",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        imported_pages += 1
                page.clear()

        if imported_pages == 0:
            raise ValueError(
                "El dump no contiene artículos de texto del espacio principal."
            )

        manifest = {
            "source": f"Wikipedia {language}",
            "source_url": source_url,
            "dump_filename": dump_path.name,
            "language": language,
            "license": "CC BY-SA 4.0",
            "purpose": "language_model_training",
            "imported_pages": imported_pages,
            "attribution_file": attribution_path.name,
            "accessed_at": datetime.now(timezone.utc).isoformat(),
            "notice": (
                "Review page-specific exceptions and attribution/share-alike "
                "obligations before distributing derived model weights."
            ),
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_corpus.replace(corpus_path)
        temporary_attribution.replace(attribution_path)
        temporary_manifest.replace(manifest_path)
    except (OSError, ElementTree.ParseError, ValueError):
        for temporary_path in (
            temporary_corpus,
            temporary_attribution,
            temporary_manifest,
        ):
            temporary_path.unlink(missing_ok=True)
        raise

    return imported_pages, attribution_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importa artículos de un dump oficial de Wikipedia en español."
    )
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--language", default="es")
    args = parser.parse_args()
    pages, attribution, manifest = import_wikipedia_dump(
        args.dump,
        args.output,
        source_url=args.source_url,
        language=args.language,
    )
    print(f"Artículos importados: {pages}")
    print(f"Corpus: {args.output}")
    print(f"Atribuciones: {attribution}")
    print(f"Manifiesto: {manifest}")


if __name__ == "__main__":
    main()

