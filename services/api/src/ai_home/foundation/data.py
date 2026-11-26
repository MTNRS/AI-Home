import hashlib
import json
from pathlib import Path
import re
import unicodedata

from ai_home.privacy import contains_common_secret


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def prepare_records(records, allow_reviewed=False, *, purpose="training"):
    if purpose not in {"training", "retrieval"}:
        raise ValueError("Unknown dataset purpose")
    if purpose == "retrieval" and allow_reviewed:
        raise ValueError("Private reviewed training exports cannot be retrieved")
    result = []
    seen = set()
    for record in records:
        required = ("source", "url", "license", "date", "language", "text")
        if any(not isinstance(record.get(k), str) or not record[k].strip() for k in required):
            raise ValueError("Incomplete source provenance")
        permission = record.get("training_allowed")
        if purpose == "retrieval":
            # Legacy public datasets predate the separate retrieval permission.
            # An explicit retrieval denial always overrides that legacy default.
            permission = record.get("retrieval_allowed", permission)
        if permission is not True or record.get("privacy_reviewed") is not True:
            raise ValueError(f"{purpose.title()} authorization and privacy review required")
        if record.get("visibility") != "public" and not (
            allow_reviewed and record.get("visibility") == "reviewed-training-only"
        ):
            raise ValueError("Only explicitly public reviewed records are accepted")
        text = unicodedata.normalize("NFC", record["text"]).replace("\r\n", "\n").strip()
        if "messages" in record:
            from ai_home.schemas import TrainingConversationMessage
            messages = [TrainingConversationMessage.model_validate(m).model_dump() for m in record["messages"]]
            joined = unicodedata.normalize("NFC", "\n".join(m["content"] for m in messages)).replace("\r\n", "\n").strip()
            if joined != text:
                raise ValueError("Conversation text differs from reviewed content")
        if contains_common_secret(text) or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
            raise ValueError("Potential personal information or credential; review required")
        digest = hashlib.sha256(text.encode()).hexdigest()
        if "sha256" in record and record["sha256"] != digest:
            raise ValueError("Document hash does not match normalized text")
        if digest in seen:
            continue
        seen.add(digest)
        result.append({**record, "text": text, "sha256": digest})
    if len(result) < 2:
        raise ValueError("At least two distinct documents required")
    return sorted(result, key=lambda r: r["sha256"])


def split_records(records, validation_fraction=0.2, allow_reviewed=False):
    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation fraction")
    records = prepare_records(records, allow_reviewed=allow_reviewed)
    count = max(1, min(len(records) - 1, round(len(records) * validation_fraction)))
    return records[count:], records[:count]


def load_records(path, max_bytes=8_000_000, *, purpose="training"):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Invalid dataset read budget")
    with Path(path).open("rb") as source:
        raw = source.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Reference dataset loader budget exceeded")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or data.get("format_version") != 1 or not isinstance(data.get("records"), list):
        raise ValueError("Invalid knowledge dataset")
    declared_version = data.get("dataset_version")
    if "dataset_version" in data and declared_version != canonical_hash(data["records"]):
        raise ValueError("Dataset version integrity check failed")
    records = prepare_records(data["records"], purpose=purpose)
    digest = canonical_hash(records)
    if "dataset_version" in data and declared_version != digest:
        raise ValueError("Versioned dataset is not in canonical prepared form")
    if "review_sha256" in data:
        review_hash = data["review_sha256"]
        if (not isinstance(review_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", review_hash)
                or any(record.get("review_sha256") != review_hash for record in records)):
            raise ValueError("Dataset review binding mismatch")
    return records, digest
