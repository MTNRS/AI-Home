import hashlib
import json
import httpx
from uuid import uuid4

from test_training_review import client, register_and_login, create_candidate, approve_candidate, REVIEW_TOKEN
from test_training_contributions import register_and_login as another_user, candidate_payload


def test_real_foundation_reply_can_enter_consented_correction_pipeline(client, tmp_path):
    from ai_home.foundation.training import train, TrainingConfig
    from ai_home.foundation.checkpoint import load_checkpoint
    from ai_home.foundation.inference import stream_text
    from test_foundation_model import config, records
    checkpoint = tmp_path / "own.pt"
    train(records(), checkpoint, config(), TrainingConfig(steps=1, warmup=0), "cpu")
    model, tokenizer, _ = load_checkpoint(checkpoint)
    messages = [{"role": "user", "content": "Hi"}]
    reply = "".join(stream_text(model, tokenizer, messages, max_new_tokens=4, temperature=0))
    assert reply.strip()
    messages.append({"role": "assistant", "content": reply})
    token = register_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}
    data = candidate_payload(provider="ai-home-foundation", model=model.checkpoint_sha256,
                             source_endpoint=f"ai-home-foundation://sha256/{model.checkpoint_sha256}", messages=messages)
    for change in ({"model": "mutable-model.pt"}, {"source_endpoint": "https://example.org"},
                   {"consent": False}, {"provider_terms_allow_reuse": False}):
        assert client.post("/api/v1/training/conversations", headers=headers, json={**data, **change}).status_code == 422
    response = client.post("/api/v1/training/conversations", headers=headers, json=data)
    assert response.status_code == 201 and response.json()["status"] == "pending_review"
    correction = {key: data[key] for key in ("consent", "consent_version", "consent_scope",
                                            "provider_terms_allow_reuse", "content_reviewed_for_sensitive_data")}
    correction.update(conversation_id=str(uuid4()), message_index=1, replacement="A human-authored correction.")
    corrected = client.post("/api/v1/training/conversations/" + response.json()["contribution_id"] + "/corrections",
                            headers=headers, json=correction)
    assert corrected.status_code == 201
    assert corrected.json()["provider"] == "ai-home-foundation"
    assert corrected.json()["model"] == model.checkpoint_sha256


def test_correction_is_separate_consented_reviewed_and_exported(client, tmp_path):
    from ai_home.foundation.contributions import load_reviewed_export
    token = register_and_login(client)
    headers = {"Authorization": f"Bearer {token}"}
    created = client.post("/api/v1/training/conversations", headers=headers, json=candidate_payload(messages=[
        {"role": "user", "content": "Question 1"}, {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Follow-up"}, {"role": "assistant", "content": "Depends on old answer"}]))
    assert created.status_code == 201
    source = created.json()["contribution_id"]
    route = f"/api/v1/training/conversations/{source}"
    original = client.get(route, headers=headers).json()
    correction = dict(conversation_id=str(uuid4()), message_index=1, replacement="Corrected public answer",
                      consent=True, consent_version="training_consent_v1",
                      consent_scope="conversation_for_aihome_training_candidates",
                      provider_terms_allow_reuse=True, content_reviewed_for_sensitive_data=True)
    other = another_user(client, "other@example.com")
    assert client.post(route + "/corrections", headers={"Authorization": f"Bearer {other}"}, json=correction).status_code == 404
    for change in ({"consent": False}, {"provider_terms_allow_reuse": False},
                   {"content_reviewed_for_sensitive_data": False}, {"message_index": 0},
                   {"replacement": "sk-" + "x" * 30}):
        assert client.post(route + "/corrections", headers=headers, json={**correction, **change}).status_code == 422
    approved = {"Authorization": f"Bearer {REVIEW_TOKEN}"}
    approve_candidate(client, source)
    from ai_home.desktop import submit_training_correction
    def transport(request):
        response = client.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
        return httpx.Response(response.status_code, content=response.content)
    candidate = submit_training_correction("http://127.0.0.1", token, source, 1, correction["replacement"],
        consent=True, reviewed=True, conversation_id=correction["conversation_id"],
        transport=httpx.MockTransport(transport)).model_dump(mode="json")
    assert candidate["status"] == "pending_review"
    assert candidate["correction"]["source_contribution_id"] == source
    assert candidate["correction"]["original_content_sha256"] == hashlib.sha256(b"Answer 1").hexdigest()
    assert client.get(route, headers=headers).json()["messages"] == original["messages"]
    assert client.post(route + "/corrections", headers=headers, json=correction).status_code == 409
    candidate_id = candidate["contribution_id"]
    second = create_candidate(client, token, "2")
    ids = [candidate_id, second]
    assert client.post("/api/v1/admin/training/datasets", headers=approved,
                       json={"dataset_version": "correction-v1", "contribution_ids": ids}).status_code == 409
    for item in ids:
        approve_candidate(client, item)
    frozen = client.post("/api/v1/admin/training/datasets", headers=approved,
                         json={"dataset_version": "correction-v1", "contribution_ids": ids})
    assert frozen.status_code == 201
    export = client.get("/api/v1/admin/training/datasets/" + frozen.json()["dataset_id"], headers=approved).json()
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    records = load_reviewed_export(path)
    corrected = next(r for r in records if "correction" in r)
    assert len(corrected["messages"]) == 2
    assert corrected["messages"][-1]["content"] == correction["replacement"]
    assert corrected["visibility"] == "reviewed-training-only"
    assert corrected["correction"] == candidate["correction"]
    from ai_home.foundation.training import train, TrainingConfig
    from ai_home.foundation.model import FoundationConfig
    from ai_home.foundation.checkpoint import load_checkpoint
    output = tmp_path / "corrected-sft.pt"
    result = train(None, output, FoundationConfig(vocab_size=280, width=16, layers=1,
                   heads=2, kv_heads=1, hidden=32, context=16),
                   TrainingConfig(steps=1, warmup=0, objective="sft"), "cpu", reviewed_export=path)
    assert result["embedding_delta_l2"] > 0
    assert load_checkpoint(output)[2]["data_visibility"] == "reviewed-training-only"
