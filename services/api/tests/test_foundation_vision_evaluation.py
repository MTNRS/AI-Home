import pytest
torch = pytest.importorskip("torch")
from test_foundation_vision_data import dataset
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.foundation.vision import VisionConfig, VisionPrefixModel
from ai_home.foundation.vision_checkpoint import save_vision_weights
from ai_home.foundation.vision_evaluation import evaluate_visual


def test_real_heldout_generation_control_and_protocol(dataset, tmp_path):
    manifest, data = dataset
    torch.manual_seed(17)
    model = VisionPrefixModel(FoundationConfig(vocab_size=262, width=16, layers=1,
        heads=2, kv_heads=1, hidden=32, context=16), VisionConfig(8, 4))
    path = tmp_path / "candidate.pt"
    save_vision_weights(model, ByteBPE(), path, provenance={"dataset_sha256": data["dataset_sha256"],
        "visibility": "public", "training_allowed": True, "privacy_reviewed": True})
    report = evaluate_visual(path, manifest, prompt="Q", max_new_tokens=3)
    again = evaluate_visual(path, manifest, prompt="Q", max_new_tokens=3)
    assert report["count"] == 1
    assert report["cases"][0]["expected"] == "blue"
    assert report["training_dataset_matches"] is True
    assert report["protocol_sha256"] == again["protocol_sha256"]
    assert report["cases"][0]["image"]["answer"] == again["cases"][0]["image"]["answer"]
    for branch in ("image", "white_control"):
        case = report["cases"][0][branch]
        assert case["generated_tokens"] <= 3 and case["seconds"] > 0
        assert case["exact_match"] == (case["answer"].strip() == "blue")
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_visual(path, manifest, prompt="Q" * 20)
