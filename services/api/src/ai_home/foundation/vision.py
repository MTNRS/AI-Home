"""Experimental own-weight image prefix. No pretrained encoder or image decoder."""

from dataclasses import dataclass

import torch
from torch import nn

from .model import FoundationModel


@dataclass(frozen=True)
class VisionConfig:
    image_size: int = 32
    patch_size: int = 8

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.image_size, self.patch_size)):
            raise ValueError("Image and patch sizes must be positive integers")
        if self.image_size > 1024 or self.image_size % self.patch_size:
            raise ValueError("Image must be bounded and divisible into patches")

    @property
    def patches(self):
        return (self.image_size // self.patch_size) ** 2


class VisionPrefixModel(nn.Module):
    """One fixed-size RGB image followed by text, using the existing causal decoder.

    Caller supplies finite float pixels in [0, 1]. No resizing, file ingestion,
    consent inference, checkpoint promotion or training-data collection occurs here.
    Returned logits align with text input positions, for next-token targets.
    """

    def __init__(self, text_config, vision_config=VisionConfig()):
        super().__init__()
        if vision_config.patches >= text_config.context:
            raise ValueError("Image prefix leaves no text context")
        self.vision_config = vision_config
        self.text = FoundationModel(text_config)
        self.projection = nn.Linear(3 * vision_config.patch_size ** 2, text_config.width, bias=False)
        self.projection.apply(FoundationModel._initialize)

    def forward(self, tokens, images=None, past=None, use_cache=False):
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("Expected nonempty text batch")
        if past is not None and images is not None:
            raise ValueError("Image prefix must not be repeated with a KV cache")
        if past is None and images is None:
            raise ValueError("Initial call requires an image")
        x = self.text.embedding(tokens)
        prefix_length = 0
        if images is not None:
            c = self.vision_config
            if images.shape != (tokens.shape[0], 3, c.image_size, c.image_size):
                raise ValueError("Expected fixed-size RGB batch matching text")
            if not images.is_floating_point() or images.device != x.device:
                raise ValueError("Pixels must be floating point on the model device")
            if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
                raise ValueError("Pixels must be finite in [0, 1]")
            if c.patches + tokens.shape[1] > self.text.config.context:
                raise ValueError("Context budget exceeded")
            patches = torch.nn.functional.unfold(images, c.patch_size, stride=c.patch_size).transpose(1, 2)
            prefix = self.projection(patches.to(self.projection.weight.dtype)).to(x.dtype)
            prefix_length = c.patches
            x = torch.cat((prefix, x), dim=1)
        hidden, cache = self.text.hidden_embeddings(x, past, use_cache)
        return self.text.output(hidden[:, prefix_length:]), cache
