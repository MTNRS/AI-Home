"""Reproduce paired arithmetic augmentation comparisons from retained artifacts."""
import argparse
import json
from pathlib import Path

from .arithmetic_experiment import augment_formats, protocol
from .arithmetic_robustness import robustness_suite
from .data import canonical_hash
from .evolution import probe_regressions
from .evaluation import normalize


def read_json(path):
    with Path(path).open('rb') as source:
        raw = source.read(16_000_001)
    if len(raw) > 16_000_000:
        raise ValueError('Comparison artifact exceeds read budget')
    return json.loads(raw)


def compare_pair(control, candidate, control_evaluation, candidate_evaluation):
    """Check protocol comparability and exact planned partitions before scoring."""
    original, suite = protocol()
    augmented = {**original, 'train': augment_formats(original['train'])}
    for artifact, expected in ((control, original), (candidate, augmented)):
        plan, partitions = artifact['protocol'], artifact['partitions']
        if artifact['report']['protocol'] != plan:
            raise ValueError('Experiment report and protocol disagree')
        if partitions != expected or plan['partition_sha256'] != canonical_hash(partitions):
            raise ValueError('Partitions differ from the fixed arithmetic protocol')
        if plan['suite_sha256'] != canonical_hash(suite):
            raise ValueError('Original probe protocol mismatch')
    left, right = control['protocol'], candidate['protocol']
    for field in ('model', 'training', 'device', 'max_new_tokens'):
        if left[field] != right[field]:
            raise ValueError(f'Paired control differs in {field}')
    if left.get('format_augmentation', False) or not right.get('format_augmentation', False):
        raise ValueError('Expected original control and augmented candidate')
    expected_suite = canonical_hash(robustness_suite())
    expected_cases = {case['id']: case for case in robustness_suite()['cases']}
    for artifact, evaluation in ((control, control_evaluation), (candidate, candidate_evaluation)):
        if evaluation['checkpoint_sha256'] != artifact['report']['after']['checkpoint_sha256']:
            raise ValueError('Evaluation does not identify the retained experiment checkpoint')
        if evaluation['suite_sha256'] != expected_suite:
            raise ValueError('Unexpected robustness suite')
        if (evaluation['device'] != artifact['protocol']['device'] or
            evaluation['generation'] != {'temperature': 0, 'max_new_tokens': artifact['protocol']['max_new_tokens']} or
            evaluation['tokenizer_sha256'] != artifact['report']['after']['tokenizer_sha256']):
            raise ValueError('Evaluation runtime or tokenizer mismatch')
        if len(evaluation['results']) != len(expected_cases) or {r['id'] for r in evaluation['results']} != set(expected_cases):
            raise ValueError('Incomplete probe results')
        for row in evaluation['results']:
            expected = expected_cases[row['id']]
            passed = not row['prompt_truncated'] and normalize(row['generated']) in {normalize(a) for a in expected['answers']}
            if row['category'] != expected['category'] or row['passed'] is not passed:
                raise ValueError('Probe score differs from generated answer')
        if not evaluation['all_prompts_fit'] or not evaluation['training_lineage_available']:
            raise ValueError('Incomplete context or training lineage evidence')
    regressions = probe_regressions(control_evaluation, candidate_evaluation)
    return {'seed': left['training']['seed'],
        'control_sha256': control_evaluation['checkpoint_sha256'],
        'candidate_sha256': candidate_evaluation['checkpoint_sha256'],
        'control': control_evaluation['categories'], 'candidate': candidate_evaluation['categories'],
        'regressions': regressions}


def load_experiment(directory):
    directory = Path(directory)
    return {key: read_json(directory / name) for key, name in (
        ('protocol', 'protocol.json'), ('partitions', 'partitions.json'), ('report', 'report.json'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', nargs='+', required=True)
    parser.add_argument('--candidates', nargs='+', required=True)
    parser.add_argument('--control-evaluations', required=True)
    parser.add_argument('--candidate-evaluations', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    controls = read_json(args.control_evaluations)['evaluations']
    candidates = read_json(args.candidate_evaluations)['evaluations']
    if len({len(args.controls), len(args.candidates), len(controls), len(candidates)}) != 1:
        parser.error('Every control requires one candidate and two evaluations')
    rows = [compare_pair(load_experiment(a), load_experiment(b), c, d)
            for a, b, c, d in zip(args.controls, args.candidates, controls, candidates)]
    if len({row['seed'] for row in rows}) != len(rows):
        raise ValueError('Duplicate seed in paired comparison')
    report = {'format': 'ai-home-arithmetic-augmentation-comparison-v2', 'rows': rows,
        'evaluation_role': 'Previously inspected development suite; not independent confirmation',
        'artifact_scope': 'Checks retained metadata and hashes, not an attestation or rerun of weights',
        'decision': 'reject_unconditional_promotion' if any(r['regressions'] for r in rows)
                    else 'requires_independent_review', 'promoted': False}
    with args.output.open('x', encoding='utf-8') as target:
        json.dump(report, target, indent=2)
    print(report['decision'])


if __name__ == '__main__':
    main()
