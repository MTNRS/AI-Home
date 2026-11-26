from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class FoundationConfig:
    vocab_size: int = 512
    width: int = 128
    layers: int = 4
    heads: int = 4
    kv_heads: int = 2
    hidden: int = 352
    context: int = 256
    rope_theta: float = 10000.0
    tied_embeddings: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "width", "layers", "heads", "kv_heads", "hidden", "context"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Invalid {name}")
        if self.vocab_size < 262 or self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("Incompatible vocabulary or attention dimensions")
        if (self.width // self.heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
            raise ValueError("Invalid RoPE base")

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def rotary(x, offset=0, theta=10000.0):
    d = x.shape[-1]
    frequency = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
    angles = positions[:, None] * frequency[None, :]
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width // config.heads
        self.q = nn.Linear(config.width, config.width, bias=False)
        self.k = nn.Linear(config.width, config.kv_heads * d, bias=False)
        self.v = nn.Linear(config.width, config.kv_heads * d, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, past=None, use_cache=False):
        c = self.config
        b, length, _ = x.shape
        d = c.width // c.heads
        offset = 0 if past is None else past[0].shape[-2]
        q = self.q(x).view(b, length, c.heads, d).transpose(1, 2)
        k = self.k(x).view(b, length, c.kv_heads, d).transpose(1, 2)
        v = self.v(x).view(b, length, c.kv_heads, d).transpose(1, 2)
        q, k = rotary(q, offset, c.rope_theta), rotary(k, offset, c.rope_theta)
        if past is not None:
            k, v = torch.cat((past[0], k), -2), torch.cat((past[1], v), -2)
        cache = (k, v) if use_cache else None
        repeat = c.heads // c.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        if offset:
            mask = torch.arange(k.shape[-2], device=x.device)[None, :] <= (
                torch.arange(length, device=x.device)[:, None] + offset
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(y.transpose(1, 2).contiguous().view(b, length, c.width)), cache


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.width)
        self.gate = nn.Linear(config.width, config.hidden, bias=False)
        self.up = nn.Linear(config.width, config.hidden, bias=False)
        self.down = nn.Linear(config.hidden, config.width, bias=False)

    def forward(self, x, past=None, use_cache=False):
        y, cache = self.attention(self.attention_norm(x), past, use_cache)
        x = x + y
        y = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y)), cache


class FoundationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tied_embeddings:
            self.output.weight = self.embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, past=None, use_cache=False):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("Expected nonempty [batch, tokens]")
        if past is not None and (len(past) != len(self.blocks) or not use_cache):
            raise ValueError("Invalid KV cache")
        offset = 0 if past is None else past[0][0].shape[-2]
        if offset + tokens.shape[1] > self.config.context:
            raise ValueError("Context budget exceeded")
        if self.training and self.config.gradient_checkpointing and use_cache:
            raise ValueError("KV cache is incompatible with gradient checkpointing")
        x = self.embedding(tokens)
        caches = []
        for i, block in enumerate(self.blocks):
            if self.training and self.config.gradient_checkpointing:
                x = checkpoint(lambda h, layer=block: layer(h)[0], x, use_reentrant=False)
            else:
                x, cache = block(x, None if past is None else past[i], use_cache)
                caches.append(cache)
        return self.output(self.norm(x)), tuple(caches) if use_cache else None

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
