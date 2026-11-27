from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.vision import VisionConfig, VisionPrefixModel


def model():
    return VisionPrefixModel(FoundationConfig(vocab_size=262, width=16, layers=1,
        heads=2, kv_heads=1, hidden=32, context=12), VisionConfig(8, 4))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_image_conditioning_cache_causality_and_real_update(device):
    torch.manual_seed(42)
    net = model().to(device).eval()
    pixels = torch.rand(2, 3, 8, 8, device=device)
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]], device=device)
    full, _ = net(tokens, pixels)
    changed, _ = net(tokens, torch.zeros_like(pixels))
    assert not torch.allclose(full, changed)
    altered = tokens.clone()
    altered[:, -1] = 9
    causal, _ = net(altered, pixels)
    torch.testing.assert_close(full[:, :-1], causal[:, :-1])
    first, cache = net(tokens[:, :1], pixels, use_cache=True)
    rest, cache = net(tokens[:, 1:], past=cache, use_cache=True)
    torch.testing.assert_close(full, torch.cat((first, rest), 1), atol=1e-6, rtol=1e-5)
    assert cache[0][0].shape[-2] == 7
    net.train()
    net.text.config = replace(net.text.config, gradient_checkpointing=True)
    before = net.projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    logits, _ = net(tokens, pixels)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flip(1).flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert net.projection.weight.grad.abs().sum() > 0
    assert net.text.embedding.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, net.projection.weight)


def test_pixel_validation_context_and_prefix_reuse():
    net = model().eval()
    tokens = torch.ones(1, 2, dtype=torch.long)
    pixels = torch.zeros(1, 3, 8, 8)
    for bad in (pixels.long(), pixels + float("nan"), pixels - 1, pixels + 2, pixels[:, :, :4]):
        with pytest.raises(ValueError):
            net(tokens, bad)
    with pytest.raises(ValueError, match="Context"):
        net(torch.ones(1, 9, dtype=torch.long), pixels)
    with pytest.raises(ValueError, match="requires"):
        net(tokens)
    _, cache = net(tokens, pixels, use_cache=True)
    with pytest.raises(ValueError, match="repeated"):
        net(tokens, pixels, past=cache, use_cache=True)
    with pytest.raises(ValueError, match="Context"):
        net(torch.ones(1, 7, dtype=torch.long), past=cache, use_cache=True)


def test_text_path_and_state_dict_unchanged():
    net = model()
    tokens = torch.ones(1, 2, dtype=torch.long)
    direct, _ = net.text(tokens)
    continuous, _ = net.text.hidden_embeddings(net.text.embedding(tokens))
    torch.testing.assert_close(direct, net.text.output(continuous), rtol=0, atol=0)
    clone = model()
    clone.load_state_dict(net.state_dict(), strict=True)
    pixels = torch.rand(1, 3, 8, 8)
    torch.testing.assert_close(net(tokens, pixels)[0], clone(tokens, pixels)[0], rtol=0, atol=0)
