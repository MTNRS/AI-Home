"""Explicit pair reviews bound to an existing consented frozen export."""
import json
import math
import re
from pathlib import Path
from typing import Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr
from .contributions import validate_reviewed_export
from .data import canonical_hash


class PairReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chosen_contribution_id: StrictStr = Field(min_length=1, max_length=200)
    rejected_contribution_id: StrictStr = Field(min_length=1, max_length=200)
    preference_confirmed: StrictBool


class PreferenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["ai-home-preference-review-v1"]
    export_manifest_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: StrictStr = Field(min_length=1, max_length=200)
    reviewed_at_epoch: StrictInt = Field(gt=0)
    pairs: list[PairReview] = Field(min_length=1, max_length=1000)


def _read(path, limit):
    with Path(path).open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Preference input exceeds read budget")
    return json.loads(raw)


def load_preference_pairs(export_path, review_path):
    export = _read(export_path, 8_000_000)
    validated = validate_reviewed_export(export)
    review = PreferenceReview.model_validate(_read(review_path, 1_000_000))
    if (not review.reviewer_id.strip()
            or review.export_manifest_sha256 != export["manifest_sha256"]):
        raise ValueError("Preference review must bind the validated export")
    by_id = {row["contribution_id"]: row for row in export["records"]}
    families = {canonical_hash(normalized_messages(row["messages"])): row["split_group_sha256"] for row in validated}
    result, seen = [], set()
    for pair in review.pairs:
        chosen, rejected = by_id.get(pair.chosen_contribution_id), by_id.get(pair.rejected_contribution_id)
        if not pair.preference_confirmed or chosen is None or rejected is None:
            raise ValueError("Both frozen contributions and explicit pair confirmation are required")
        if review.reviewed_at_epoch < max(chosen["reviewed_at_epoch"], rejected["reviewed_at_epoch"]):
            raise ValueError("Pair review predates source review")
        c, r = normalized_messages(chosen["messages"]), normalized_messages(rejected["messages"])
        if (c[:-1] != r[:-1] or not c[-1]["content"] or not r[-1]["content"]
                or c[-1]["content"] == r[-1]["content"]):
            raise ValueError("Preference pair requires identical prompt history and distinct final responses")
        identity = canonical_hash(sorted((canonical_hash(c), canonical_hash(r))))
        if identity in seen:
            raise ValueError("Duplicate or contradictory preference pair")
        seen.add(identity)
        result.append({"messages": c[:-1], "chosen": c[-1]["content"], "rejected": r[-1]["content"],
            "chosen_contribution_id": pair.chosen_contribution_id,
            "rejected_contribution_id": pair.rejected_contribution_id,
            "split_group_sha256": canonical_hash({"family": "preference-prompt-v1", "messages": c[:-1]}),
            "source_split_groups": sorted({families[canonical_hash(c)], families[canonical_hash(r)]}),
            "export_manifest_sha256": review.export_manifest_sha256,
            "preference_review_sha256": canonical_hash(review.model_dump()),
            "visibility": "reviewed-training-only", "training_allowed": True})
    return result


def normalized_messages(messages):
    return [{"role": row["role"], "content": unicodedata.normalize("NFC", row["content"]).replace("\r\n", "\n").strip()}
            for row in messages]


def create_review_template(export_path, output_path):
    """Enumerate eligible alternatives without choosing or approving a preference."""
    from ai_home.immutable_artifacts import publish_bytes
    export = _read(export_path, 8_000_000)
    validate_reviewed_export(export)
    groups = {}
    for row in sorted(export["records"], key=lambda item: item["contribution_id"]):
        messages = normalized_messages(row["messages"])
        group = groups.setdefault(canonical_hash(messages[:-1]), {})
        # Identical responses provide no preference contrast, even under new IDs.
        if messages[-1]["content"]:
            group.setdefault(messages[-1]["content"], row["contribution_id"])
    pairs = []
    for key in sorted(groups):
        alternatives = list(groups[key].values())
        for index, first in enumerate(alternatives):
            for second in alternatives[index + 1:]:
                if len(pairs) == 1000:
                    raise ValueError("More than 1000 eligible pairs; use a smaller reviewed export")
                pairs.append({"chosen_contribution_id": first, "rejected_contribution_id": second,
                              "preference_confirmed": False})
    if not pairs:
        raise ValueError("No distinct responses with identical prompt history")
    template = {"format": "ai-home-preference-review-v1",
                "export_manifest_sha256": export["manifest_sha256"],
                "reviewer_id": "", "reviewed_at_epoch": 0, "pairs": pairs}
    publish_bytes(output_path, (json.dumps(template, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {"pending_pairs": len(pairs), "export_manifest_sha256": export["manifest_sha256"],
            "notice": "ID order is not a preference. Review both responses, choose order, remove unwanted pairs, then record reviewer, time and confirmation. Keep this file private."}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create an unapproved local preference-review template")
    parser.add_argument("--export", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(create_review_template(args.export, args.output), indent=2))


def split_preference_pairs(pairs, validation_fraction=0.2):
    """Split validated pairs by connected prompt/source families, never individual rows."""
    if (type(validation_fraction) not in (int, float) or not math.isfinite(validation_fraction)
            or not 0 < validation_fraction < 1):
        raise ValueError("Invalid preference validation fraction")
    if not isinstance(pairs, list) or not 2 <= len(pairs) <= 1000:
        raise ValueError("Preference splitting requires two to 1000 reviewed pairs")
    parent, groups_by_pair = {}, []
    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for pair in pairs:
        sources = pair.get("source_split_groups")
        if not isinstance(sources, list) or not sources:
            raise ValueError("Preference source families are required")
        groups = [pair.get("split_group_sha256"), *sources]
        if any(not isinstance(group, str) or not re.fullmatch(r"[0-9a-f]{64}", group) for group in groups):
            raise ValueError("Invalid preference family fingerprint")
        for group in groups:
            parent.setdefault(group, group)
        for group in groups[1:]:
            parent[root(group)] = root(groups[0])
        groups_by_pair.append(groups)
    components = {}
    for group in parent:
        components.setdefault(root(group), []).append(group)
    fingerprints = {key: canonical_hash({"family": "preference-component-v1", "groups": sorted(groups)})
                    for key, groups in components.items()}
    train, validation = [], []
    threshold = int(validation_fraction * 2**256)
    for pair, groups in zip(pairs, groups_by_pair):
        fingerprint = fingerprints[root(groups[0])]
        target = validation if int(fingerprint, 16) < threshold else train
        target.append(pair)
    if not train or not validation:
        raise ValueError("Preference family split requires nonempty training and validation; supply independent families")
    return sorted(train, key=canonical_hash), sorted(validation, key=canonical_hash)


if __name__ == "__main__":
    main()
