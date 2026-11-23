"""Command-line entry point for training the local AI@Home language model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_home.local_llm import (
    ModelConfig,
    default_checkpoint_path,
    train_from_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena desde cero el modelo byte-level experimental de AI@Home."
    )
    parser.add_argument("--corpus", required=True, nargs="+", type=Path)
    parser.add_argument("--manifest", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=default_checkpoint_path())
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.manifest is not None and len(args.manifest) != len(args.corpus):
        parser.error("Indica exactamente un --manifest por cada --corpus.")
    config = ModelConfig(
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        layers=args.layers,
        heads=args.heads,
    )

    def report_progress(step: int, loss: float) -> None:
        if step == 1 or step % 100 == 0:
            print(f"paso={step} pérdida={loss:.4f}")

    result = train_from_corpus(
        args.corpus,
        args.output,
        manifest_paths=args.manifest,
        steps=args.steps,
        batch_size=args.batch_size,
        config=config,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        on_step=report_progress,
    )
    print(
        f"Checkpoint creado: {result.checkpoint_path}; "
        f"parámetros={result.parameter_count}; dispositivo={result.device}; "
        f"pérdida_final={result.final_loss:.4f}"
    )


if __name__ == "__main__":
    main()
