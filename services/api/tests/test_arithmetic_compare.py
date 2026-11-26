import copy
import pytest

pytest.importorskip('torch')
from ai_home.foundation.arithmetic_compare import compare_pair
from ai_home.foundation.arithmetic_experiment import protocol, augment_formats
from ai_home.foundation.arithmetic_robustness import robustness_suite
from ai_home.foundation.data import canonical_hash


def evidence():
    partitions, original_suite = protocol()
    suite = robustness_suite()
    evaluation = {'checkpoint_sha256': 'a' * 64, 'suite_sha256': canonical_hash(suite),
        'device': 'cpu', 'generation': {'temperature': 0, 'max_new_tokens': 4},
        'torch_version': 'test', 'tokenizer_sha256': 'b' * 64,
        'all_prompts_fit': True, 'training_lineage_available': True,
        'results': [{'id': c['id'], 'category': c['category'], 'generated': c['answers'][0],
                     'passed': True, 'prompt_truncated': False} for c in suite['cases']],
        'categories': {c: {'passed': 14, 'total': 14, 'accuracy': 1.0}
                       for c in ('mathematics', 'robustness', 'spanish', 'english')}}
    artifacts = []
    for augmented in (False, True):
        data = {**partitions, 'train': augment_formats(partitions['train'])} if augmented else partitions
        plan = {'model': {}, 'training': {'seed': 42}, 'device': 'cpu', 'max_new_tokens': 4,
                'partition_sha256': canonical_hash(data), 'suite_sha256': canonical_hash(original_suite),
                'format_augmentation': augmented}
        artifacts.append({'protocol': plan, 'partitions': data,
                          'report': {'protocol': plan, 'after': evaluation}})
    return *artifacts, copy.deepcopy(evaluation), copy.deepcopy(evaluation)


def test_paired_comparison_checks_retained_evidence():
    assert compare_pair(*evidence())['regressions'] == []


@pytest.mark.parametrize('change', ['seed', 'partition', 'missing', 'score', 'checkpoint'])
def test_paired_comparison_rejects_incomparable_or_incomplete_reports(change):
    control, candidate, before, after = evidence()
    if change == 'seed':
        candidate['protocol']['training']['seed'] = 7
    elif change == 'partition':
        candidate['partitions']['train'].append(candidate['partitions']['test'][0])
    elif change == 'missing':
        after['results'].pop()
    elif change == 'score':
        after['results'][0]['generated'] = 'incorrect'
    else:
        after['checkpoint_sha256'] = 'c' * 64
    with pytest.raises(ValueError):
        compare_pair(control, candidate, before, after)
