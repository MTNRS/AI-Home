"""Explicit local import of an administrator-exported, frozen training dataset.

This is not a public-data release and does not authorize volunteer distribution.
Only use exports obtained from the authenticated administrative API.
"""
import json
from pathlib import Path

from ai_home.schemas import TrainingDatasetExportResponse
from .data import canonical_hash, prepare_records


def load_reviewed_export(path, max_bytes=8_000_000):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Reviewed export exceeds input budget")
    raw = json.loads(path.read_text(encoding="utf-8"))
    export = TrainingDatasetExportResponse.model_validate(raw)
    manifest = {"dataset_version": raw["dataset_version"], "records": raw["records"]}
    if canonical_hash(manifest) != export.manifest_sha256 or export.item_count != len(export.records):
        raise ValueError("Frozen dataset manifest mismatch")
    records, seen = [], set()
    for item in export.records:
        if item.contribution_id in seen:
            raise ValueError("Duplicate contribution")
        seen.add(item.contribution_id)
        if (item.consent_version != "training_consent_v1"
            or item.consent_scope != "conversation_for_aihome_training_candidates"
            or not item.reviewer_id.strip()
            or item.consent_at_epoch <= 0 or item.reviewed_at_epoch < item.consent_at_epoch
            or any(item.review_checklist.get(key) is not True for key in (
                "provider_terms_verified", "sensitive_data_reviewed", "reuse_rights_verified"))):
            raise ValueError("Missing consent or independent review")
        messages = [m.model_dump() for m in item.messages]
        if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
            raise ValueError("Expected a user/assistant training conversation")
        if any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(messages)):
            raise ValueError("Conversation roles must alternate")
        records.append({
            "source": "AI@Home reviewed contribution", "url": "local:reviewed-export",
            "license": "consented-training-only; no redistribution grant",
            "date": export.created_at.isoformat(), "language": "und",
            "training_allowed": True, "privacy_reviewed": True,
            "visibility": "reviewed-training-only", "messages": messages,
            "text": "\n".join(m["content"] for m in messages),
            "dataset_version": export.dataset_version, "manifest_sha256": export.manifest_sha256,
        })
    return prepare_records(records, allow_reviewed=True)
