import threading

import pytest

torch = pytest.importorskip("torch")
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.foundation.vision import VisionConfig, VisionPrefixModel
from ai_home.foundation.vision_inference import stream_vision_tokens, stream_vision_text


def setup():
    torch.manual_seed(17)
    model = VisionPrefixModel(FoundationConfig(vocab_size=262, width=16, layers=1,
        heads=2, kv_heads=1, hidden=32, context=7), VisionConfig(8, 4))
    return model, torch.rand(1, 3, 8, 8)


def test_real_greedy_generation_cache_rollover_matches_reference():
    model, image = setup()
    prompt = [1, 2, 3, 4, 5]
    calls = []
    hook = model.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[1]))
    actual = list(stream_vision_tokens(model, image, prompt, temperature=0, max_new_tokens=8))
    hook.remove()
    expected, ids, cache = [], list(prompt), None
    with torch.inference_mode():
        for _ in range(8):
            if cache is None or cache[0][0].shape[-2] >= 7:
                current, cache, pixels = ids[-3:], None, image
            else:
                current, pixels = ids[-1:], None
            logits, cache = model(torch.tensor([current]), pixels, past=cache, use_cache=True)
            scores = logits[0, -1].clone()
            scores[[256, 257, 259, 260, 261]] = -torch.inf
            token = scores.argmax().item()
            if token == 258:
                break
            ids.append(token)
            expected.append(token)
    assert actual == expected
    assert len(actual) == 8
    assert calls == [3] * 8  # Full prompt rolls immediately, retaining four image patches.


def test_short_prompt_uses_cache_and_sampling_is_repeatable():
    model, image = setup()
    calls = []
    hook = model.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[1]))
    first = list(stream_vision_tokens(model, image, [1], temperature=.8, top_k=4,
                                     top_p=.9, repetition_penalty=1.2, seed=32, max_new_tokens=6))
    hook.remove()
    second = list(stream_vision_tokens(model, image, [1], temperature=.8, top_k=4,
                                      top_p=.9, repetition_penalty=1.2, seed=32, max_new_tokens=6))
    assert first == second and len(first) == 6
    assert calls[:4] == [1, 1, 1, 3]


def test_cancellation_text_and_invalid_image():
    model, image = setup()
    cancel = threading.Event()
    cancel.set()
    assert list(stream_vision_tokens(model, image, [1], cancel=cancel)) == []
    cancel.clear()
    stream = stream_vision_tokens(model, image, [1], temperature=0, cancel=cancel)
    next(stream)
    cancel.set()
    assert list(stream) == []
    text = "".join(stream_vision_text(model, ByteBPE(), image,
        [{"role": "user", "content": "Describe"}], temperature=0, max_new_tokens=2))
    assert isinstance(text, str)
    with pytest.raises(ValueError, match="exactly one"):
        list(stream_vision_tokens(model, image.repeat(2, 1, 1, 1), [1]))
