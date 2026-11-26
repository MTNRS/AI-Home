"""Small byte-level decoder-only language model trained from scratch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import ExitStack
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import random
from typing import Any, Callable, Sequence


MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    context_length: int = 1024
    embedding_dim: int = 128
    layers: int = 4
    heads: int = 4
    vocab_size: int = 256

    def validate(self) -> None:
        if self.context_length < 8:
            raise ValueError("El contexto del modelo debe ser de al menos 8 bytes.")
        if self.heads < 1:
            raise ValueError("El modelo necesita al menos una cabeza de atención.")
        if self.embedding_dim < 8 or self.embedding_dim % self.heads:
            raise ValueError(
                "La dimensión del modelo debe ser positiva y divisible por "
                "el número de cabezas."
            )
        if self.layers < 1:
            raise ValueError("El modelo necesita al menos una capa de atención.")
        if self.vocab_size != 256:
            raise ValueError("Esta versión usa un vocabulario fijo de bytes UTF-8.")


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    steps: int
    final_loss: float
    device: str
    parameter_count: int


def default_checkpoint_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "ai-home" / "models" / "aihome-small.pt"


def _torch_runtime():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "El motor local necesita PyTorch. Instálalo con "
            '`python -m pip install -e ".[llm]"`.'
        ) from error
    return torch, nn


def _build_model(config: ModelConfig):
    torch, nn = _torch_runtime()
    config.validate()

    class ByteGPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(
                config.vocab_size,
                config.embedding_dim,
            )
            self.position_embedding = nn.Embedding(
                config.context_length,
                config.embedding_dim,
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.heads,
                dim_feedforward=config.embedding_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(
                layer,
                num_layers=config.layers,
                enable_nested_tensor=False,
            )
            self.final_norm = nn.LayerNorm(config.embedding_dim)
            self.output = nn.Linear(
                config.embedding_dim,
                config.vocab_size,
                bias=False,
            )
            self.output.weight = self.token_embedding.weight

        def forward(self, token_ids):
            _, sequence_length = token_ids.shape
            if sequence_length > config.context_length:
                raise ValueError("La entrada supera el contexto del modelo.")
            positions = torch.arange(sequence_length, device=token_ids.device)
            hidden = self.token_embedding(token_ids) + self.position_embedding(
                positions
            )
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=token_ids.device,
                ),
                diagonal=1,
            )
            hidden = self.blocks(hidden, mask=causal_mask)
            return self.output(self.final_norm(hidden))

    return ByteGPT()


def _read_training_manifest(corpus_path: Path, manifest_path: Path | None) -> dict[str, Any]:
    path = manifest_path or Path(f"{corpus_path}.manifest.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"Falta el manifiesto de procedencia del corpus: {path}."
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"No se pudo leer el manifiesto del corpus: {path}.") from error
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(field), str) or not manifest[field].strip()
        for field in ("source", "license", "purpose")
    ):
        raise ValueError(
            "El manifiesto debe incluir source, license y purpose no vacíos."
        )
    if manifest["purpose"] != "language_model_training":
        raise ValueError(
            "El manifiesto no autoriza el corpus para entrenamiento de lenguaje."
        )
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_from_corpus(
    corpus_paths: Sequence[Path],
    checkpoint_path: Path,
    *,
    manifest_paths: Sequence[Path] | None = None,
    steps: int = 10000,
    batch_size: int = 8,
    config: ModelConfig = ModelConfig(),
    learning_rate: float = 3e-4,
    device: str = "auto",
    seed: int = 42,
    on_step: Callable[[int, float], None] | None = None,
) -> TrainingResult:
    corpus_paths = [Path(path) for path in corpus_paths]
    checkpoint_path = Path(checkpoint_path)
    config.validate()
    if steps < 1 or batch_size < 1:
        raise ValueError("Los pasos y el tamaño del lote deben ser positivos.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("La tasa de aprendizaje debe ser positiva.")
    if not corpus_paths:
        raise ValueError("Selecciona al menos un corpus de entrenamiento.")
    if manifest_paths is not None and len(manifest_paths) != len(corpus_paths):
        raise ValueError("Debe proporcionarse un manifiesto por cada corpus.")
    normalized_manifest_paths = (
        [Path(path) for path in manifest_paths]
        if manifest_paths is not None
        else [None] * len(corpus_paths)
    )
    corpus_sizes: list[int] = []
    corpus_manifests: list[dict[str, Any]] = []
    for corpus_path, manifest_path in zip(
        corpus_paths,
        normalized_manifest_paths,
        strict=True,
    ):
        if not corpus_path.is_file():
            raise ValueError(f"No se encontró el corpus: {corpus_path}.")
        corpus_size = corpus_path.stat().st_size
        if corpus_size <= config.context_length + 1:
            raise ValueError(f"El corpus es demasiado pequeño: {corpus_path}.")
        corpus_sizes.append(corpus_size)
        corpus_manifests.append(
            _read_training_manifest(corpus_path, manifest_path)
        )

    torch, _nn = _torch_runtime()
    torch.manual_seed(seed)
    random.seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    final_loss = 0.0
    model.train()

    with ExitStack() as stack:
        corpora = []
        for corpus_path in corpus_paths:
            corpus_file = stack.enter_context(corpus_path.open("rb"))
            corpora.append(
                stack.enter_context(
                    mmap.mmap(
                        corpus_file.fileno(),
                        0,
                        access=mmap.ACCESS_READ,
                    )
                )
            )
        for step in range(1, steps + 1):
            batch_samples = []
            for _ in range(batch_size):
                corpus_index = random.randrange(len(corpora))
                maximum_start = (
                    corpus_sizes[corpus_index] - config.context_length - 1
                )
                start = random.randrange(maximum_start + 1)
                batch_samples.append(
                    (
                        list(
                            corpora[corpus_index][
                                start : start + config.context_length
                            ]
                        ),
                        list(
                            corpora[corpus_index][
                                start + 1 : start + config.context_length + 1
                            ]
                        ),
                    )
                )
            inputs = torch.tensor(
                [sample[0] for sample in batch_samples],
                dtype=torch.long,
                device=device,
            )
            targets = torch.tensor(
                [sample[1] for sample in batch_samples],
                dtype=torch.long,
                device=device,
            )
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach().item())
            if on_step is not None:
                on_step(step, final_loss)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )
    try:
        torch.save(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "config": asdict(config),
                "model_state": model.state_dict(),
                "training": {
                    "steps": steps,
                    "final_loss": final_loss,
                    "corpora": [
                        {
                            "sha256": _sha256_file(corpus_path),
                            "manifest": manifest,
                        }
                        for corpus_path, manifest in zip(
                            corpus_paths,
                            corpus_manifests,
                            strict=True,
                        )
                    ],
                },
            },
            temporary_path,
        )
        temporary_path.replace(checkpoint_path)
    except (OSError, RuntimeError):
        temporary_path.unlink(missing_ok=True)
        raise

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        steps=steps,
        final_loss=final_loss,
        device=str(device),
        parameter_count=parameter_count,
    )


def _format_messages(messages: list[dict[str, str]]) -> str:
    role_names = {
        "system": "Instrucciones",
        "user": "Usuario",
        "assistant": "AI@Home",
    }
    parts = [
        f"### {role_names[message['role']]}\n{message['content']}\n"
        for message in messages
        if message.get("role") in role_names
    ]
    parts.append("### AI@Home\n")
    return "".join(parts)


def complete_locally(
    messages: list[dict[str, str]],
    *,
    checkpoint_path: Path | None = None,
    max_new_bytes: int = 512,
    temperature: float = 0.8,
) -> str:
    if not messages or not any(message.get("role") == "user" for message in messages):
        raise ValueError("Escribe un mensaje antes de iniciar la conversación.")
    if not 1 <= max_new_bytes <= 2048:
        raise ValueError("La respuesta local debe solicitar entre 1 y 2048 bytes.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("La temperatura debe ser positiva.")

    path = checkpoint_path or default_checkpoint_path()
    if not path.is_file():
        raise RuntimeError(
            "El LLM de AI@Home todavía no está entrenado. Importa un corpus "
            "autorizado y ejecuta `ai-home-train` para crear el modelo local."
        )

    torch, _nn = _torch_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"No se pudo cargar el modelo local: {path}.") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != MODEL_FORMAT_VERSION
        or not isinstance(checkpoint.get("config"), dict)
        or not isinstance(checkpoint.get("model_state"), dict)
    ):
        raise RuntimeError("El archivo no es un checkpoint de AI@Home compatible.")
    try:
        config = ModelConfig(**checkpoint["config"])
        config.validate()
    except (TypeError, ValueError) as error:
        raise RuntimeError("La configuración del checkpoint no es válida.") from error
    model = _build_model(config).to(device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise RuntimeError("El checkpoint del modelo está incompleto o dañado.") from error
    model.eval()

    prompt = _format_messages(messages).encode("utf-8")
    token_ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    initial_length = token_ids.shape[1]
    with torch.inference_mode():
        for _ in range(max_new_bytes):
            context = token_ids[:, -config.context_length :]
            logits = model(context)[:, -1, :] / temperature
            cutoff = torch.topk(logits, k=min(40, config.vocab_size)).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

    generated_bytes = bytes(
        token_ids[0, initial_length:].to("cpu").tolist()
    )
    answer = generated_bytes.decode("utf-8", errors="ignore")
    for stop_marker in ("\n### Usuario\n", "\n### Artículo"):
        answer = answer.split(stop_marker, maxsplit=1)[0]
    answer = answer.strip()
    if not answer:
        raise RuntimeError(
            "El modelo local no generó una respuesta legible. Necesita más "
            "entrenamiento con texto y ejemplos conversacionales autorizados."
        )
    return answer

