"""Fixed small arithmetic generalization experiment; no automatic promotion."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .data import canonical_hash
from .evaluation import evaluate_checkpoint
from .model import FoundationConfig
from .training import TrainingConfig, train


def protocol():
    partitions = {'train': [], 'validation': [], 'test': []}
    for a in range(10):
        for b in range(10):
            group = (10 * min(a, b) + max(a, b)) % 7
            split = 'test' if group == 0 else 'validation' if group == 1 else 'train'
            prompt, answer = f'{a}+{b}=', str(a + b)
            partitions[split].append({'source': 'AI@Home authored arithmetic generator v1',
                'url': 'local:arithmetic-experiment-v1', 'license': 'Apache-2.0',
                'date': '2026-11-26', 'language': 'zxx', 'training_allowed': True,
                'privacy_reviewed': True, 'visibility': 'public', 'text': prompt + '\n' + answer,
                'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]})
    suite = {'version': 'single-digit-addition-unordered-holdout-v1',
        'source': 'AI@Home procedural fixture; pairs and reversed pairs held out together',
        'license': 'Apache-2.0', 'cases': [
            {'id': f'addition-{i}', 'category': 'mathematics',
             'prompt': r['messages'][0]['content'], 'answers': [r['messages'][1]['content']]}
            for i, r in enumerate(partitions['test'])]}
    return partitions, suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error('Seed must be an integer between 0 and 2**32 - 1')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    partitions, suite = protocol()
    config = FoundationConfig(vocab_size=262, width=64, layers=2, heads=4,
                              kv_heads=2, hidden=128, context=16)
    training = TrainingConfig(steps=1000, batch_size=16, learning_rate=.002,
        warmup=30, checkpoint_every=1000, objective='sft', seed=args.seed)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    plan = {'format': 'ai-home-arithmetic-protocol-v1', 'model': asdict(config),
        'training': asdict(training), 'device': args.device, 'max_new_tokens': 4,
        'partition_sha256': canonical_hash(partitions), 'suite_sha256': canonical_hash(suite),
        'comparison': 'one-update checkpoint versus fixed 1000-update checkpoint',
        'selection': 'No tuning or early stopping using test results; no promotion',
        'claim_limit': 'Single seed, procedural single-digit addition; not broad mathematical reasoning'}
    save('protocol.json', plan)
    save('partitions.json', partitions)
    save('suite.json', suite)
    early = output / 'one-update.pt'
    train(partitions['train'], early, config=config, training=training,
          device=args.device, stop_after=1, validation_records=partitions['validation'])
    before = evaluate_checkpoint(early, suite, device=args.device, max_new_tokens=4)
    save('before.json', before)
    final = output / 'candidate.pt'
    checkpoint = train(partitions['train'], final, config=config, training=training,
        device=args.device, resume=early, validation_records=partitions['validation'])
    after = evaluate_checkpoint(final, suite, device=args.device, max_new_tokens=4)
    save('after.json', after)
    report = {'format': 'ai-home-arithmetic-experiment-v1', 'protocol': plan,
        'partition_counts': {k: len(v) for k, v in partitions.items()},
        'before': before, 'after': after, 'parameter_count': checkpoint['parameter_count'],
        'embedding_delta_l2_after_resume': checkpoint['embedding_delta_l2'],
        'final_training_metric': checkpoint['metrics'][-1], 'promoted': False}
    save('report.json', report)
    print(json.dumps({'before': before['categories'], 'after': after['categories'],
                      'report': str(output / 'report.json')}, indent=2))


if __name__ == '__main__':
    main()
