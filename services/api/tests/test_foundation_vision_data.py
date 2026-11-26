import hashlib
import json

import pytest

pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
from ai_home.foundation.data import canonical_hash
from ai_home.foundation.vision_data import FORMAT, load_visual_manifest, load_visual_record
from ai_home.foundation.vision_images import PREPROCESSING


def publish(path, data):
    data["dataset_sha256"] = canonical_hash({k: data[k] for k in ("format", "preprocessing", "records")})
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    records = []
    for color, split in (("red", "train"), ("blue", "validation")):
        image = tmp_path / f"{color}.png"
        Image.new("RGB", (8, 8), color).save(image)
        records.append(dict(image=image.name, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            caption=color, split=split, source="authored procedural fixture", url="local:fixture",
            license="Apache-2.0", date="2026-11-26", language="en", training_allowed=True,
            privacy_reviewed=True, visibility="public"))
    data = dict(format=FORMAT, preprocessing=PREPROCESSING, records=records)
    path = tmp_path / "manifest.json"
    publish(path, data)
    return path, data


def test_reviewed_manifest_lazy_load_and_changed_image(dataset):
    path, data = dataset
    records, version = load_visual_manifest(path)
    assert version == data["dataset_sha256"]
    pixels, caption, metadata = load_visual_record(path, records[0], 8)
    assert pixels.shape == (1, 3, 8, 8) and caption == "red"
    assert metadata["sha256"] == records[0]["image_sha256"]
    (path.parent / "red.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        load_visual_record(path, records[0], 8)


@pytest.mark.parametrize("field,value", [("training_allowed", False), ("privacy_reviewed", False),
    ("visibility", "private"), ("image", "../outside.png"), ("image", "C:/private.png"),
    ("caption", "owner@example.com"), ("caption", " padded "), ("license", ""),
    ("image_sha256", "bad"), ("split", "test")])
def test_rejects_unapproved_or_invalid_records(dataset, field, value):
    path, data = dataset
    data["records"][0][field] = value
    publish(path, data)
    with pytest.raises(ValueError):
        load_visual_manifest(path)


def test_duplicate_split_and_manifest_tampering(dataset):
    path, data = dataset
    data["records"][1]["image_sha256"] = data["records"][0]["image_sha256"]
    publish(path, data)
    with pytest.raises(ValueError, match="Duplicate"):
        load_visual_manifest(path)
    data["records"][1]["image_sha256"] = "a" * 64
    publish(path, data)
    data["records"][0]["caption"] = "altered after review"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_visual_manifest(path)
    with pytest.raises(ValueError, match="budget"):
        load_visual_manifest(path, max_bytes=10)
