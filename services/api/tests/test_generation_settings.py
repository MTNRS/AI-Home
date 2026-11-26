import pytest
from pydantic import ValidationError

from ai_home.schemas import FoundationGeneration
from test_chat_api import client, create_session


@pytest.mark.parametrize("options", [
    {"temperature": -0.1}, {"temperature": float("nan")}, {"temperature": True},
    {"top_p": 0}, {"top_p": 1.1}, {"top_k": -1}, {"top_k": 1.5},
    {"max_new_tokens": True}, {"max_new_tokens": 4097}, {"max_new_tokens": "10"},
    {"seed": -1}, {"seed": 2**63}, {"repetition_penalty": 0.5}, {"unknown": 1},
])
def test_generation_options_are_strict_and_bounded(options):
    with pytest.raises(ValidationError):
        FoundationGeneration.model_validate(options)


def test_explicit_generation_controls_are_not_ignored_by_external_backend(client):
    token = create_session(client)
    response = client.post("/api/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hi"}], "generation": {"temperature": 0}},
        headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 422
    assert "Foundation" in response.json()["detail"]
