import hashlib
import pytest

from ai_home.foundation.data import prepare_records, split_records


def records(group, prefix):
    return [{'source': 'authored group fixture', 'url': 'local:groups', 'license': 'Apache-2.0',
        'date': '2026-11-26', 'language': 'es', 'visibility': 'public', 'training_allowed': True,
        'privacy_reviewed': True, 'split_group_sha256': hashlib.sha256(group.encode()).hexdigest(),
        'text': f'{prefix} Documento {i} de una fuente autorizada.'} for i in range(2)]


def test_groups_stay_together_and_existing_assignments_are_stable():
    source = [r for i in range(30) for r in records(str(i), str(i))]
    def assignment(values):
        result = {}
        for split, rows in zip(('train', 'validation'), split_records(values)):
            for row in rows:
                group = row['split_group_sha256']
                assert result.setdefault(group, split) == split
        return result
    before = assignment(source)
    after = assignment(list(reversed(source)) + records('new', 'new'))
    assert all(after[k] == v for k, v in before.items())


def test_group_metadata_rejects_conflicting_duplicates_and_invalid_hashes():
    source = records('a', 'same')
    with pytest.raises(ValueError, match='conflicting'):
        prepare_records(source + [{**source[0], 'split_group_sha256': 'f' * 64}])
    with pytest.raises(ValueError, match='fingerprint'):
        prepare_records([{**source[0], 'split_group_sha256': 'arbitrary-name'}, source[1]])
    with pytest.raises(ValueError, match='nonempty'):
        split_records(source)


def test_training_rejects_family_leakage_and_retains_lineage(tmp_path):
    torch = pytest.importorskip('torch')
    from ai_home.foundation.model import FoundationConfig
    from ai_home.foundation.training import TrainingConfig, train
    torch.set_num_threads(1)
    kwargs = dict(config=FoundationConfig(vocab_size=262, width=16, layers=1, heads=2,
                  kv_heads=1, hidden=32, context=16),
                  training=TrainingConfig(steps=1, warmup=0), device='cpu')
    with pytest.raises(ValueError, match='share a split group'):
        train(records('a', 'train'), tmp_path / 'invalid.pt',
              validation_records=records('a', 'distinct validation text'), **kwargs)
    parent = train(records('a', 'train'), tmp_path / 'parent.pt',
                   validation_records=records('b', 'validation'), **kwargs)
    assert parent['training_split_groups'] == [hashlib.sha256(b'a').hexdigest()]
    from ai_home.foundation.vocabulary import expand_checkpoint
    from ai_home.foundation.checkpoint import load_checkpoint
    with pytest.raises(ValueError, match='holdout split group'):
        expand_checkpoint(tmp_path / 'parent.pt', tmp_path / 'invalid-expansion.pt',
                          records('b', 'unseen revision'), 270)
    expand_checkpoint(tmp_path / 'parent.pt', tmp_path / 'expanded.pt', records('c', 'new corpus'), 270)
    expanded = load_checkpoint(tmp_path / 'expanded.pt')[2]
    assert expanded['training_split_groups'] == sorted(hashlib.sha256(g).hexdigest() for g in (b'a', b'c'))
    assert expanded['validation_split_groups'] == parent['validation_split_groups']
    with pytest.raises(ValueError, match='checkpoint lineage'):
        train(records('b', 'new unseen text'), tmp_path / 'candidate.pt',
              validation_records=records('c', 'new validation'), initial_checkpoint=tmp_path / 'parent.pt', **kwargs)
    assert not (tmp_path / 'candidate.pt').exists()
