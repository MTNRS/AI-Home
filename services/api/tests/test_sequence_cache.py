import json
import pytest

torch = pytest.importorskip('torch')
from ai_home.foundation.arithmetic_experiment import protocol
from ai_home.foundation.model import FoundationConfig
from ai_home.foundation.tokenizer import ByteBPE
from ai_home.foundation.training import TrainingConfig, _sequences, train


@pytest.mark.parametrize('objective', ['causal', 'sft'])
def test_disk_sequences_match_memory_and_refuse_existing_directory(tmp_path, objective):
    records = protocol()[0]['train'][:4]
    tokenizer = ByteBPE.train([r['text'] for r in records], vocab_size=262)
    expected = _sequences(records, tokenizer, 7, objective)
    actual = _sequences(records, tokenizer, 7, objective, tmp_path / 'cache')
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    indices = torch.tensor([2, 0, 2])
    assert torch.equal(actual[0][indices], expected[0][indices])
    assert torch.equal(actual[1][indices], expected[1][indices])
    manifest = json.loads((tmp_path / 'cache/manifest.json').read_text())
    assert manifest['bytes'] == (tmp_path / 'cache/sequences.i64').stat().st_size
    with pytest.raises(FileExistsError):
        _sequences(records, tokenizer, 7, objective, tmp_path / 'cache')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_disk_training_and_resume_match_in_memory_exactly(tmp_path, device):
    torch.set_num_threads(1)
    records = protocol()[0]
    config = FoundationConfig(vocab_size=262, width=16, layers=1, heads=2,
                              kv_heads=1, hidden=32, context=16)
    settings = TrainingConfig(steps=4, warmup=1, checkpoint_every=4, objective='sft')
    kwargs = dict(config=config, training=settings, device=device, validation_records=records['validation'])
    reference = train(records['train'], tmp_path / 'memory.pt', **kwargs)
    train(records['train'], tmp_path / 'partial.pt', stop_after=2,
          sequence_cache_dir=tmp_path / 'first-cache', **kwargs)
    resumed = train(records['train'], tmp_path / 'resumed.pt', resume=tmp_path / 'partial.pt',
                    sequence_cache_dir=tmp_path / 'second-cache', **kwargs)
    assert reference['validation'] == resumed['validation']
    assert reference['dataset_sha256'] == resumed['dataset_sha256']
    for name in reference['model_state']:
        assert torch.equal(reference['model_state'][name], resumed['model_state'][name])
