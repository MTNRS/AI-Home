import hashlib
import json
from pathlib import Path
import re
import unicodedata

from ai_home.privacy import contains_common_secret


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def prepare_records(records, allow_reviewed=False):
    result = []
    seen = set()
    for record in records:
        required = ("source", "url", "license", "date", "language", "text")
        if any(not isinstance(record.get(k), str) or not record[k].strip() for k in required):
            raise ValueError("Incomplete source provenance")
        if record.get("training_allowed") is not True or record.get("privacy_reviewed") is not True:
            raise ValueError("Training authorization and privacy review required")
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


def load_records(path, max_bytes=8_000_000):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Reference dataset loader budget exceeded")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != 1 or not isinstance(data.get("records"), list):
        raise ValueError("Invalid knowledge dataset")
    records = prepare_records(data["records"])
    return records, canonical_hash(records)
