import codecs
import math
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .training import select_device


def stream_tokens(model, prompt, max_new_tokens=128, temperature=0.8, top_k=40,
                  top_p=0.95, repetition_penalty=1.0, cancel=None, seed=42):
    if not prompt or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Invalid prompt or generation length")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling settings")
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1 or top_k < 0:
        raise ValueError("Invalid repetition penalty or top-k")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = list(prompt)
    cache = None
    model.eval()
    for _ in range(max_new_tokens):
        if cancel is not None and cancel.is_set():
            return
        if cache is None or cache[0][0].shape[-2] >= model.config.context:
            current = ids[-model.config.context:]
            cache = None
        else:
            current = ids[-1:]
        with torch.inference_mode():
            logits, cache = model(torch.tensor([current], device=device), past=cache, use_cache=True)
            scores = logits[0, -1].float().clone()
            previous = torch.tensor(sorted(set(ids)), device=device)
            scores[previous] = torch.where(scores[previous] < 0,
                                           scores[previous] * repetition_penalty,
                                           scores[previous] / repetition_penalty)
            scores[[256, 257, 259, 260, 261]] = -torch.inf
            if temperature == 0:
                token = scores.argmax().item()
            else:
                scores /= temperature
                if top_k:
                    threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                    scores[scores < threshold] = -torch.inf
                ordered, indices = scores.sort(descending=True)
                cumulative = ordered.softmax(-1).cumsum(-1)
                remove = cumulative - ordered.softmax(-1) >= top_p
                scores[indices[remove]] = -torch.inf
                token = torch.multinomial(scores.softmax(-1), 1, generator=generator).item()
        if token == 258:
            return
        ids.append(token)
        yield token


def stream_text(model, tokenizer, messages, **options):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    prompt = tokenizer.conversation(messages)
    for token in stream_tokens(model, prompt, **options):
        text = decoder.decode(tokenizer.pieces[token])
        if text:
            yield text
    final = decoder.decode(b"", final=True)
    if final:
        yield final


def complete(messages, checkpoint_path, device="auto", max_bytes=2_000_000_000, **options):
    if not Path(checkpoint_path).is_file():
        raise RuntimeError("No existe el checkpoint del modelo fundacional seleccionado.")
    model, tokenizer, _ = load_checkpoint(checkpoint_path, select_device(device), max_bytes=max_bytes)
    answer = "".join(stream_text(model, tokenizer, messages, **options))
    if not answer.strip():
        raise RuntimeError("El modelo no ha producido texto; necesita entrenamiento y evaluación.")
    return answer
