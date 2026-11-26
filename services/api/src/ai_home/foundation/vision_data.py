"""Reviewed local visual manifests; lazy pixel loading with hash binding."""

import json
from pathlib import Path
import re
import unicodedata

from ai_home.privacy import contains_common_secret
from .data import canonical_hash
from .vision_images import PREPROCESSING, load_image


FORMAT = "ai-home-visual-dataset-v1"


def _image_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Image path must be a relative portable path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Image path escapes dataset root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Image path escapes dataset root")
    return resolved


def load_visual_manifest(path, max_bytes=2_000_000):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Invalid manifest budget")
    path = Path(path).resolve()
    with path.open("rb") as source:
        raw = source.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Visual manifest exceeds file budget")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or data.get("format") != FORMAT or data.get("preprocessing") != PREPROCESSING:
        raise ValueError("Invalid visual dataset format or preprocessing")
    records = data.get("records")
    if not isinstance(records, list) or not 2 <= len(records) <= 4096:
        raise ValueError("Expected 2 to 4096 visual records")
    seen, splits = set(), set()
    required = {"image", "image_sha256", "caption", "source", "url", "license", "date", "language",
                "training_allowed", "privacy_reviewed", "visibility", "split"}
    for record in records:
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError("Incomplete visual provenance or unknown fields")
        for key in ("caption", "source", "url", "license", "date", "language"):
            value = record[key]
            if not isinstance(value, str) or not value.strip() or len(value) > 8192:
                raise ValueError("Invalid visual provenance text")
            if contains_common_secret(value) or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value):
                raise ValueError("Potential personal information or credential")
        if record["caption"] != unicodedata.normalize("NFC", record["caption"]).replace("\r\n", "\n").strip():
            raise ValueError("Caption must be reviewed in normalized form")
        if record["training_allowed"] is not True or record["privacy_reviewed"] is not True or record["visibility"] != "public":
            raise ValueError("Only explicitly public authorized reviewed images accepted")
        if record["split"] not in ("train", "validation"):
            raise ValueError("Invalid explicit dataset split")
        splits.add(record["split"])
        digest = record["image_sha256"]
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid image SHA256")
        if digest in seen:
            raise ValueError("Duplicate image, including potential split contamination")
        seen.add(digest)
        _image_path(path.parent, record["image"])
    if splits != {"train", "validation"}:
        raise ValueError("Separate train and validation records required")
    version = canonical_hash({"format": FORMAT, "preprocessing": PREPROCESSING, "records": records})
    if data.get("dataset_sha256") != version:
        raise ValueError("Visual dataset integrity mismatch")
    return records, version


def load_visual_record(manifest_path, record, image_size):
    """Load a validated record lazily; verify the same bytes that are decoded.

    Caller must first validate the complete manifest. Paths are rechecked on each
    use. This is not a sandbox against concurrent local filesystem attackers.
    """
    root = Path(manifest_path).resolve().parent
    image, metadata = load_image(_image_path(root, record["image"]), image_size,
                                expected_sha256=record["image_sha256"])
    return image, record["caption"], metadata
