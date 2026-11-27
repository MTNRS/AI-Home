import pytest

pytest.importorskip("torch")
from ai_home.foundation.arithmetic_experiment import protocol
from ai_home.foundation.arithmetic_experiment import augment_formats


def test_arithmetic_protocol_reserves_commutative_pairs_and_correct_targets():
    partitions, suite = protocol()
    seen = {}
    for split, records in partitions.items():
        for record in records:
            prompt, answer = [m['content'] for m in record['messages']]
            a, b = map(int, prompt.removesuffix('=').split('+'))
            assert answer == str(a + b)
            key = tuple(sorted((a, b)))
            assert seen.setdefault(key, split) == split
    assert len(seen) == 55
    assert {k: len(v) for k, v in partitions.items()} == {'train': 70, 'validation': 16, 'test': 14}
    assert {p['prompt'] for p in suite['cases']} == {
        r['messages'][0]['content'] for r in partitions['test']}


def test_format_augmentation_keeps_reserved_pairs_out_of_training():
    import re
    partitions, _ = protocol()
    expanded = augment_formats(partitions['train'])
    assert len(expanded) == 280
    def pairs(records):
        return {tuple(sorted(map(int, re.findall(r'\d+', r['messages'][0]['content'])))) for r in records}
    assert pairs(expanded) == pairs(partitions['train'])
    assert not pairs(expanded) & pairs(partitions['validation'] + partitions['test'])
