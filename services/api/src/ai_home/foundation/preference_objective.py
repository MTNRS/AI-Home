"""Experimental DPO math for own-model experiments; no dataset admission or promotion."""
import math

import torch
from torch.nn import functional as F


def response_log_probability(model, prompt, response):
    """Sum response-token log probabilities; callers include EOS explicitly.

    Prompt targets never contribute to the sum. No truncation, padding or length
    normalization is performed. This low-level function does not authorize data.
    """
    for values in (prompt, response):
        if (not isinstance(values, (list, tuple)) or not values
                or any(type(token) is not int or not 0 <= token < model.config.vocab_size for token in values)):
            raise ValueError("Expected nonempty valid token sequences")
    if len(prompt) + len(response) - 1 > model.config.context:
        raise ValueError("Preference response exceeds model context; truncation is not permitted")
    device = next(model.parameters()).device
    tokens = torch.tensor([*prompt, *response], dtype=torch.long, device=device)
    logits, _ = model(tokens[:-1].unsqueeze(0))
    # Position len(prompt)-1 predicts the first response token.
    selected = logits[0, len(prompt)-1:].float().log_softmax(-1)
    result = selected.gather(1, tokens[len(prompt):].unsqueeze(1)).sum()
    if not torch.isfinite(result):
        raise ValueError("Non-finite response log probability")
    return result


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, *, beta=0.1):
    """Mean negative log-sigmoid of reference-relative preference log odds."""
    if type(beta) not in (float, int) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    values = (chosen, rejected, reference_chosen, reference_rejected)
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point() for value in values):
        raise ValueError("DPO requires floating-point log probability tensors")
    if (chosen.numel() == 0 or chosen.ndim > 1
            or any(value.shape != chosen.shape or value.device != chosen.device for value in values)
            or any(not torch.isfinite(value).all() for value in values)):
        raise ValueError("DPO log probabilities must have matching finite scalar/vector shapes")
    margin = ((chosen.float() - rejected.float())
              - (reference_chosen.detach().float() - reference_rejected.detach().float()))
    result = -F.logsigmoid(beta * margin).mean()
    if not torch.isfinite(result):
        raise ValueError("Non-finite DPO loss")
    return result
