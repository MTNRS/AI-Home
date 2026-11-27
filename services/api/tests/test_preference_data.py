import copy
import json

import pytest
from ai_home.foundation.data import canonical_hash
from ai_home.foundation.preference_data import load_preference_pairs, split_preference_pairs


def test_review_template_stays_unapproved_and_never_overwrites(tmp_path):
    from ai_home.foundation.preference_data import create_review_template
    export, _ = fixture(tmp_path)
    source, target = tmp_path / "export.json", tmp_path / "template.json"
    source.write_text(json.dumps(export), encoding="utf-8")
    result = create_review_template(source, target)
    assert result["pending_pairs"] == 1
    template = json.loads(target.read_text(encoding="utf-8"))
    assert not template["pairs"][0]["preference_confirmed"]
    assert template["reviewer_id"] == "" and template["reviewed_at_epoch"] == 0
    with pytest.raises(ValueError):
        load_preference_pairs(source, target)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        create_review_template(source, target)
    assert target.read_bytes() == original
    template.update(reviewer_id="fixture-reviewer", reviewed_at_epoch=3)
    template["pairs"][0]["preference_confirmed"] = True
    target.write_text(json.dumps(template), encoding="utf-8")
    assert len(load_preference_pairs(source, target)) == 1


def test_review_template_rejects_unrelated_prompts(tmp_path):
    from ai_home.foundation.preference_data import create_review_template
    export, _ = fixture(tmp_path)
    export["records"][1]["messages"][0]["content"] = "Different history"
    export["manifest_sha256"] = canonical_hash({"dataset_version": export["dataset_version"], "records": export["records"]})
    source, target = tmp_path / "export.json", tmp_path / "template.json"
    source.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError, match="No distinct"):
        create_review_template(source, target)
    assert not target.exists()


def fixture(tmp_path):
    records = []
    for index, answer in enumerate(("Preferred fixture answer", "Rejected fixture answer"), start=1):
        records.append({"contribution_id": f"00000000-0000-0000-0000-{index:012d}",
            "provider": "ai-home-foundation", "source_endpoint": "local", "model": "own-fixture",
            "consent_version": "training_consent_v1",
            "consent_scope": "conversation_for_aihome_training_candidates", "consent_at_epoch": 1,
            "reviewer_id": "source-reviewer", "reviewed_at_epoch": 2,
            "review_checklist": {"provider_terms_verified": True, "sensitive_data_reviewed": True,
                                 "reuse_rights_verified": True},
            "messages": [{"role": "user", "content": "Authored fixture prompt"},
                         {"role": "assistant", "content": answer}]})
    export = {"dataset_id": "00000000-0000-0000-0000-000000000010", "dataset_version": "pairs-v1",
        "item_count": 2, "created_at": "2026-11-26T00:00:00Z", "records": records}
    export["manifest_sha256"] = canonical_hash({"dataset_version": export["dataset_version"], "records": records})
    review = {"format": "ai-home-preference-review-v1", "export_manifest_sha256": export["manifest_sha256"],
        "reviewer_id": "pair-reviewer", "reviewed_at_epoch": 3, "pairs": [{
            "chosen_contribution_id": records[0]["contribution_id"],
            "rejected_contribution_id": records[1]["contribution_id"], "preference_confirmed": True}]}
    return export, review


def write_and_load(tmp_path, export, review):
    paths = [tmp_path / "export.json", tmp_path / "review.json"]
    for path, value in zip(paths, (export, review)):
        path.write_text(json.dumps(value), encoding="utf-8")
    return load_preference_pairs(*paths)


def test_explicit_pair_review_keeps_restricted_visibility(tmp_path):
    export, review = fixture(tmp_path)
    pairs = write_and_load(tmp_path, export, review)
    assert len(pairs) == 1 and pairs[0]["visibility"] == "reviewed-training-only"
    assert pairs[0]["chosen"] == export["records"][0]["messages"][-1]["content"]
    assert len(pairs[0]["source_split_groups"]) == 2
    swapped = copy.deepcopy(review)
    pair = swapped["pairs"][0]
    pair["chosen_contribution_id"], pair["rejected_contribution_id"] = pair["rejected_contribution_id"], pair["chosen_contribution_id"]
    assert write_and_load(tmp_path, export, swapped)[0]["split_group_sha256"] == pairs[0]["split_group_sha256"]
    review["pairs"].append(pair)
    with pytest.raises(ValueError, match="contradictory"):
        write_and_load(tmp_path, export, review)


@pytest.mark.parametrize("change", ["consent", "manifest", "confirmation", "unknown", "early", "prompt", "identical"])
def test_invalid_pair_review_rejected(tmp_path, change):
    export, review = fixture(tmp_path)
    if change == "consent":
        export["records"][0]["consent_scope"] = "unapproved"
    elif change == "manifest":
        review["export_manifest_sha256"] = "0" * 64
    elif change == "confirmation":
        review["pairs"][0]["preference_confirmed"] = False
    elif change == "unknown":
        review["pairs"][0]["chosen_contribution_id"] = "missing"
    elif change == "early":
        review["reviewed_at_epoch"] = 1
    elif change == "prompt":
        export["records"][1]["messages"][0]["content"] = "Another prompt"
    else:
        review["pairs"][0]["rejected_contribution_id"] = review["pairs"][0]["chosen_contribution_id"]
    if change in {"consent", "prompt"}:
        export["manifest_sha256"] = canonical_hash({"dataset_version": export["dataset_version"], "records": export["records"]})
        review["export_manifest_sha256"] = export["manifest_sha256"]
    with pytest.raises(ValueError):
        write_and_load(tmp_path, export, review)


def test_preference_partition_preserves_transitive_source_and_prompt_families():
    def group(name):
        return canonical_hash({"fixture-family": name})
    def pair(name, prompt, sources):
        return {"name": name, "split_group_sha256": group(prompt),
                "source_split_groups": [group(source) for source in sources]}
    # A--B share a source; B--C share a prompt. A and C must stay together too.
    connected = [pair("a", "prompt-a", ["source-a", "source-b"]),
                 pair("b", "prompt-b", ["source-b", "source-c"]),
                 pair("c", "prompt-b", ["source-d", "source-e"])]
    independent = [pair(str(i), f"prompt-{i}", [f"source-{i}"]) for i in range(20)]
    rows = connected + independent
    original = copy.deepcopy(rows)
    train, validation = split_preference_pairs(rows, .5)
    assert rows == original
    assert split_preference_pairs(list(reversed(rows)), .5) == (train, validation)
    assert sum(row["name"] in {"a", "b", "c"} for row in train) in (0, 3)
    def all_groups(partition):
        return {group for row in partition for group in [row["split_group_sha256"], *row["source_split_groups"]]}
    assert all_groups(train).isdisjoint(all_groups(validation))
    assert len(train) + len(validation) == len(rows)
    with pytest.raises(ValueError, match="independent families"):
        split_preference_pairs(connected, .5)
    with pytest.raises(ValueError, match="fraction"):
        split_preference_pairs(rows, True)
