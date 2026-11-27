"""Experimental own-model embeddings; no claim of semantic retrieval quality."""
import torch

from .checkpoint import load_checkpoint


class FoundationEmbedder:
    """Token-weighted mean of final causal states across non-overlapping windows.

    Windows reset positions and cannot attend to earlier windows. No padding,
    special tokens, or silent truncation is used. CPU is the explicit default.
    """
    def __init__(self, checkpoint, *, device="cpu", max_bytes=2_000_000_000):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Embedding device must be cpu or cuda")
        self.model, self.tokenizer, _ = load_checkpoint(checkpoint, device=device, max_bytes=max_bytes)
        self.model.eval()

    @property
    def identity(self):
        return {"method": "foundation-windowed-mean-v1",
                "checkpoint_sha256": self.model.checkpoint_sha256,
                "tokenizer_sha256": self.tokenizer.fingerprint,
                "dimensions": self.model.config.width,
                "window_tokens": self.model.config.context}

    @classmethod
    def from_loaded(cls, model, tokenizer):
        if model.training or not getattr(model, "checkpoint_sha256", None):
            raise ValueError("Embeddings require an evaluated, checkpoint-backed model")
        instance = cls.__new__(cls)
        instance.model, instance.tokenizer = model, tokenizer
        return instance

    @torch.inference_mode()
    def embed(self, text):
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 32000:
            raise ValueError("Embedding text must be nonempty and at most 32000 UTF-8 bytes")
        tokens = self.tokenizer.encode(text)
        device = next(self.model.parameters()).device
        total = torch.zeros(self.model.config.width, device=device, dtype=torch.float32)
        window = self.model.config.context
        for start in range(0, len(tokens), window):
            batch = torch.tensor([tokens[start:start + window]], device=device, dtype=torch.long)
            hidden, _ = self.model.hidden_states(batch)
            total += hidden[0].float().sum(dim=0)
        total /= len(tokens)
        norm = torch.linalg.vector_norm(total)
        if not torch.isfinite(total).all() or not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Cannot normalize non-finite or zero embedding")
        return (total / norm).cpu().tolist()
