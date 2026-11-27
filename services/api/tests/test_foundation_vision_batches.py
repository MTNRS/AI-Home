import pytest

torch = pytest.importorskip("torch")
from test_foundation_vision_data import dataset
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.foundation.vision import VisionConfig, VisionPrefixModel
from ai_home.foundation.vision_batches import VisualCaptionDataset


def config(context=40):
    return FoundationConfig(vocab_size=262, width=16, layers=1, heads=2, kv_heads=1,
                            hidden=32, context=context)


def test_exact_assistant_targets_padding_and_split(dataset):
    path, _ = dataset
    tokenizer = ByteBPE()
    train = VisualCaptionDataset(path, tokenizer, config(), VisionConfig(8, 4), split="train", prompt="Q")
    validation = VisualCaptionDataset(path, tokenizer, config(), VisionConfig(8, 4), split="validation", prompt="Q")
    assert len(train) == len(validation) == 1
    pixels, inputs, targets = train[0]
    assert inputs[:8].tolist() == [257, 260, ord("Q"), 258, 261, *b"red"]
    assert targets[:8].tolist() == [-100] * 4 + [*b"red", 258]
    assert targets[8:].eq(-100).all() and inputs[8:].eq(256).all()
    assert pixels.shape == (3, 8, 8)
    assert train.records[0]["image_sha256"] != validation.records[0]["image_sha256"]
    inputs[0] = 0
    assert train[0][1][0].item() == 257


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_real_loaded_image_caption_training_updates_visual_weights(dataset, device):
    path, _ = dataset
    samples = VisualCaptionDataset(path, ByteBPE(), config(), VisionConfig(8, 4), split="train", prompt="Q")
    model = VisionPrefixModel(config(), VisionConfig(8, 4)).to(device)
    before = model.projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    images, inputs, targets = next(iter(torch.utils.data.DataLoader(samples, batch_size=1)))
    images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
    logits, _ = model(inputs, images)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    assert model.projection.weight.grad.abs().sum() > 0
    optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.projection.weight)


def test_context_overflow_rejected_without_truncation(dataset):
    path, _ = dataset
    with pytest.raises(ValueError, match="exceeds context"):
        VisualCaptionDataset(path, ByteBPE(), config(8), VisionConfig(8, 4), split="train", prompt="Q")
