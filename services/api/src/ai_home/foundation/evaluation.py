"""Versioned, deterministic generation probes; no execution of generated code."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import unicodedata

import psutil
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator, model_serializer
import torch

from .checkpoint import load_checkpoint
from .data import canonical_hash
from .inference import stream_tokens
from .training import select_device


CATEGORIES = {"spanish", "english", "instructions", "programming", "mathematics",
              "knowledge", "reasoning", "consistency", "robustness"}


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StrictStr = Field(min_length=1, max_length=80)
    category: StrictStr
    prompt: StrictStr = Field(min_length=1, max_length=8192)
    answers: list[StrictStr] = Field(min_length=1, max_length=20)
    split_group_sha256: StrictStr | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_serializer(mode='wrap')
    def serialize_probe(self, handler):
        result = handler(self)
        if result.get('split_group_sha256') is None:
            result.pop('split_group_sha256', None)
        return result

    @model_validator(mode="after")
    def validate_probe(self):
        if self.category not in CATEGORIES or any(not a.strip() or len(a) > 8192 for a in self.answers):
            raise ValueError("Invalid category or expected answer")
        return self


class ProbeSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: StrictStr = Field(min_length=1, max_length=80)
    source: StrictStr = Field(min_length=1, max_length=500)
    license: StrictStr = Field(min_length=1, max_length=200)
    cases: list[Probe] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({p.id for p in self.cases}) != len(self.cases):
            raise ValueError("Duplicate probe identifiers")
        return self


def normalize(text):
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()


class EvaluationCancelled(RuntimeError):
    """Incomplete evaluations must not be published as complete reports."""


def evaluate_checkpoint(path, suite, *, device="auto", max_new_tokens=64, max_bytes=2_000_000_000,
                        cancel=None, on_probe=None):
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise EvaluationCancelled("Evaluation cancelled; no complete report produced")
    check_cancel()
    suite = ProbeSuite.model_validate(suite)
    device = select_device(device)
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Invalid generation budget")
    model, tokenizer, payload = load_checkpoint(path, device, max_bytes)
    training_hashes = set(payload.get("training_document_hashes", []))
    training_groups = set(payload.get('training_split_groups', []))
    if any(probe.split_group_sha256 in training_groups for probe in suite.cases
           if probe.split_group_sha256 is not None):
        raise ValueError('Probe family overlaps known training split groups')
    results = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for probe in suite.cases:
        check_cancel()
        texts = [probe.prompt, *(probe.prompt + "\n" + a for a in probe.answers)]
        if any(hashlib.sha256(normalize(t).encode()).hexdigest() in training_hashes for t in texts):
            raise ValueError("Probe overlaps known training documents")
        prompt = tokenizer.conversation([{"role": "user", "content": probe.prompt}])
        truncated = len(prompt) > model.config.context
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        tokens = list(stream_tokens(model, prompt, max_new_tokens=max_new_tokens, temperature=0, cancel=cancel))
        check_cancel()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        answer = tokenizer.decode(tokens, errors="replace")
        passed = not truncated and normalize(answer) in {normalize(a) for a in probe.answers}
        results.append({"id": probe.id, "category": probe.category, "generated": answer,
                        "passed": passed, "prompt_truncated": truncated,
                        "prompt_tokens": len(prompt), "generated_tokens": len(tokens),
                        "generation_seconds": elapsed, "tokens_per_second": len(tokens) / elapsed})
        if on_probe is not None:
            on_probe({"completed": len(results), "total": len(suite.cases), "id": probe.id, "passed": passed})
    check_cancel()
    categories = {}
    for category in sorted({p.category for p in suite.cases}):
        rows = [r for r in results if r["category"] == category]
        passed = sum(r["passed"] for r in rows)
        categories[category] = {"passed": passed, "total": len(rows), "accuracy": passed / len(rows)}
    return {"format": "ai-home-evaluation-v1", "checkpoint_sha256": model.checkpoint_sha256,
            'group_lineage_available': 'training_split_groups' in payload,
            'grouped_probe_count': sum(p.split_group_sha256 is not None for p in suite.cases),
            "suite_sha256": canonical_hash(suite.model_dump()), "suite_version": suite.version,
            "tokenizer_sha256": tokenizer.fingerprint, "device": device,
            "torch_version": str(torch.__version__), "generation": {"temperature": 0, "max_new_tokens": max_new_tokens},
            "categories": categories, "results": results,
            "all_prompts_fit": not any(r["prompt_truncated"] for r in results),
            "training_lineage_available": "training_document_hashes" in payload,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "process_rss_bytes_after": psutil.Process().memory_info().rss,
            "claim": "Authored exact-answer probes only; not a general capability certification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.suite.stat().st_size > 8_000_000:
        parser.error("Suite exceeds input budget")
    report = evaluate_checkpoint(args.checkpoint, json.loads(args.suite.read_text(encoding="utf-8")),
                                 device=args.device, max_new_tokens=args.max_new_tokens)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({"categories": report["categories"], "all_prompts_fit": report["all_prompts_fit"]}))


if __name__ == "__main__":
    main()
