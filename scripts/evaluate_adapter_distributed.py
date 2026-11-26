"""Compare saved loopback adapter checkpoints on the experiment's held-out docs."""
import argparse
import json
from pathlib import Path
import torch
from ai_home.foundation.checkpoint import load_checkpoint
from ai_home.foundation.data import prepare_records, canonical_hash
from ai_home.foundation.training import _sequences, evaluate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
experiment = json.loads((args.experiment / 'report.json').read_text(encoding='utf-8'))
if experiment['format'] != 'ai-home-adapter-distributed-smoke-v1':
    raise ValueError('Expected the authored distributed adapter experiment')
records = json.loads((args.experiment / 'dataset.json').read_text(encoding='utf-8'))['records']
if len(records) != 6:
    raise ValueError('Expected four training and two held-out fixture documents')
holdout = prepare_records(records[4:])
metrics = {}
for name in ('parent', 'candidate'):
    model, tokenizer, payload = load_checkpoint(args.experiment / (name + '.pt'))
    hashes = {row['sha256'] for row in holdout}
    if hashes & set(payload['training_document_hashes']):
        raise ValueError('Holdout overlaps training lineage')
    if not hashes <= set(payload['validation_document_hashes']):
        raise ValueError('Holdout is absent from saved validation lineage')
    metrics[name] = {'checkpoint_sha256': model.checkpoint_sha256,
        'tokenizer_sha256': tokenizer.fingerprint,
        **evaluate(model, _sequences(holdout, tokenizer, model.config.context), 'cpu')}
report = {'format': 'ai-home-adapter-distributed-holdout-v1',
    'torch_version': str(torch.__version__), 'device': 'cpu',
    'holdout_sha256': canonical_hash(holdout), 'documents': len(holdout),
    'metrics': metrics, 'promoted': False,
    'claim': 'Two authored fixture documents; loss comparison only, not general language quality'}
with args.output.open('x', encoding='utf-8') as target:
    json.dump(report, target, indent=2, allow_nan=False)
print(json.dumps(report, indent=2))
