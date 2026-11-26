"""Local image-conditioned generation using Foundation's shared sampler."""

from torch import nn

from .inference import stream_tokens, stream_text


class _ImageSession(nn.Module):
    def __init__(self, model, image):
        super().__init__()
        self.model = model
        self.config = model.text.config
        self.input_context = self.config.context - model.vision_config.patches
        if image.shape != (1, 3, model.vision_config.image_size, model.vision_config.image_size):
            raise ValueError("Generation requires exactly one fixed-size RGB image")
        # Capture caller input once: a rolling-context rebuild must use the same image.
        self.image = image.detach().clone()

    def forward(self, tokens, past=None, use_cache=False):
        return self.model(tokens, self.image if past is None else None,
                          past=past, use_cache=use_cache)


def stream_vision_tokens(model, image, prompt, **options):
    """Stream real tokens; prepend the same image whenever the context rolls over.

    Image must already be normalized and on the model device. There is no file
    ingestion, network transmission or automatic training from this call.
    """
    yield from stream_tokens(_ImageSession(model, image), prompt, **options)


def stream_vision_text(model, tokenizer, image, messages, **options):
    """Conversation formatting and incremental UTF-8 decoding via the text engine."""
    yield from stream_text(_ImageSession(model, image), tokenizer, messages, **options)
