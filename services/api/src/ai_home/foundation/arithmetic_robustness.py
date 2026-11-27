"""Evaluate fixed arithmetic checkpoints under prompt changes; no training."""
import argparse
import json
from pathlib import Path

from .arithmetic_experiment import protocol
from .data import canonical_hash
from .evaluation import evaluate_checkpoint


def robustness_suite():
    _, original = protocol()
    cases = []
    for case in original['cases']:
        a, b = case['prompt'].removesuffix('=').split('+')
        for name, category, prompt in (
            ('original', 'mathematics', case['prompt']),
            ('spaces', 'robustness', f'{a} + {b} ='),
            ('spanish', 'spanish', f'Suma {a}+{b}'),
            ('english', 'english', f'Add {a}+{b}'),
        ):
            cases.append({'id': case['id'] + '-' + name, 'category': category,
                          'prompt': prompt, 'answers': case['answers']})
    return {'version': 'arithmetic-format-robustness-v1',
        'source': 'AI@Home authored transformations of reserved single-digit pairs; category names label formats only',
        'license': 'Apache-2.0', 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    suite = robustness_suite()
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    save('suite.json', suite)
    save('protocol.json', {'format': 'ai-home-arithmetic-robustness-v1',
        'suite_sha256': canonical_hash(suite), 'device': args.device, 'max_new_tokens': 4,
        'training': False, 'selection': 'Report every supplied checkpoint; no tuning or promotion',
        'scope': '14 existing held-out pairs in four formats, not 56 independent problems'})
    reports = []
    for index, checkpoint in enumerate(args.checkpoints):
        result = evaluate_checkpoint(checkpoint, suite, device=args.device, max_new_tokens=4)
        save(f'evaluation-{index}.json', result)
        reports.append(result)
    save('report.json', {'format': 'ai-home-arithmetic-robustness-result-v1',
                        'evaluations': reports, 'promoted': False})
    print(json.dumps([r['categories'] for r in reports], indent=2))


if __name__ == '__main__':
    main()
