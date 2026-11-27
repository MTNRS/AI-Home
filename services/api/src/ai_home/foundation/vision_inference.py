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


def main():
    import argparse
    from .training import select_device
    from .vision_checkpoint import load_vision_weights
    from .vision_images import load_image

    parser = argparse.ArgumentParser(description="Experimental local visual candidate inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    model, tokenizer, _ = load_vision_weights(args.checkpoint, select_device(args.device))
    image, _ = load_image(args.image, model.vision_config.image_size)
    image = image.to(next(model.parameters()).device)
    for text in stream_vision_text(model, tokenizer, image,
            [{"role": "user", "content": args.prompt}],
            max_new_tokens=args.max_new_tokens, temperature=args.temperature):
        print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
