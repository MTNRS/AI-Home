"""Bounded, immutable distribution of the authored mechanical smoke experiment."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

from ai_home.immutable_artifacts import publish_bytes
from .checkpoint import load_checkpoint
from .data import canonical_hash, load_records, prepare_records, split_records
from .smoke import smoke_records

MAX_BYTES = 64 * 1024 * 1024
FILES = {'foundation-nano.pt', 'dataset.json', 'report.json', 'tokenizer.json',
         'config.json', 'training.json', 'LICENSE', 'README.txt'}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def bounded_read(path):
    with Path(path).open('rb') as source:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Artifact exceeds smoke package budget')
    return raw


def validate(files):
    # Snapshot bytes into a private temporary directory: never extract archive paths.
    with tempfile.TemporaryDirectory(prefix='aihome-smoke-package-') as temporary:
        root = Path(temporary)
        for name in ('foundation-nano.pt', 'dataset.json'):
            (root / name).write_bytes(files[name])
        records, _ = load_records(root / 'dataset.json')
        if records != prepare_records(smoke_records()):
            raise ValueError('Only the authored public smoke corpus may be packaged')
        model, tokenizer, payload = load_checkpoint(root / 'foundation-nano.pt', max_bytes=MAX_BYTES)
    train, validation = split_records(records)
    dataset_hash = canonical_hash({'train': train, 'validation': validation,
                                   'replay': [], 'replay_fraction': 0.25})
    report = json.loads(files['report.json'])
    if (payload.get('data_visibility') != 'public' or payload.get('parent_checkpoint_sha256') is not None
            or not payload.get('training_complete') or payload.get('status') != 'candidate'):
        raise ValueError('Expected a completed public from-scratch candidate')
    if payload.get('dataset_sha256') != dataset_hash or report.get('dataset_sha256') != dataset_hash:
        raise ValueError('Dataset binding mismatch')
    if report.get('checkpoint_sha256') != model.checkpoint_sha256:
        raise ValueError('Report checkpoint binding mismatch')
    if report.get('tokenizer_sha256') != tokenizer.fingerprint:
        raise ValueError('Report tokenizer binding mismatch')
    for name, value in [('tokenizer.json', tokenizer.to_dict()), ('config.json', payload['config']),
                        ('training.json', payload['training_config'])]:
        if json.loads(files[name]) != value:
            raise ValueError(f'Checkpoint metadata mismatch: {name}')
    for key in ('step', 'parameter_count', 'validation', 'metrics'):
        if report.get(key) != payload.get(key):
            raise ValueError(f'Report metadata mismatch: {key}')
    if not payload.get('embedding_delta_l2', 0) > 0:
        raise ValueError('No measured parameter change')
    if report.get('embedding_delta_l2_since_resume') != payload['embedding_delta_l2']:
        raise ValueError('Reported parameter change mismatch')
    return {'checkpoint_sha256': model.checkpoint_sha256, 'dataset_sha256': dataset_hash,
            'tokenizer_sha256': tokenizer.fingerprint, 'parameter_count': model.parameter_count}


def build(directory, output, *, revision, license_path):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('A full source commit is required')
    root = Path(directory)
    files = {name: bounded_read(root / name) for name in ('foundation-nano.pt', 'dataset.json', 'report.json')}
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / 'checkpoint.pt'
        snapshot.write_bytes(files['foundation-nano.pt'])
        _, tokenizer, payload = load_checkpoint(snapshot, max_bytes=MAX_BYTES)
    files.update({'tokenizer.json': encoded(tokenizer.to_dict()), 'config.json': encoded(payload['config']),
                  'training.json': encoded(payload['training_config']), 'LICENSE': bounded_read(license_path)})
    if b'Apache License' not in files['LICENSE'] or b'Version 2.0' not in files['LICENSE']:
        raise ValueError('Expected repository Apache-2.0 license')
    files['README.txt'] = (f'AI@Home experimental Nano smoke package\nSource revision: {revision}\n'
        'Authored synthetic corpus and random-initialized weights. Apache-2.0.\n'
        'Candidate only: no conversational quality or registry approval established.\n'
        'The report contains a real generated sample and training/validation metrics, not a capability benchmark.\n'
        'From the source checkout at the stated revision, install .[llm] and run:\n'
        'python -m ai_home.foundation.smoke --output-dir reproduced --device cpu\n'
        'Use --device cuda for CUDA. Versions are recorded in report.json.\n'
        'Timing and sampled text may differ; reproduction does not promise cross-device byte identity.\n'
        'Verify this ZIP before use with python -m ai_home.foundation.smoke_package verify PACKAGE.zip\n'
        'SHA256 detects corruption, not trusted authorship; this package is unsigned.\n').encode('utf-8')
    identity = validate(files)
    manifest = {'format': 'ai-home-smoke-package-v1', 'source_revision': revision, **identity,
                'files': {name: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                          for name, raw in sorted(files.items())}}
    files['manifest.json'] = encoded(manifest)
    if sum(map(len, files.values())) > MAX_BYTES:
        raise ValueError('Package exceeds budget')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    raw = buffer.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError('Compressed package exceeds budget')
    publish_bytes(output, raw)
    return manifest


def verify(path):
    with zipfile.ZipFile(io.BytesIO(bounded_read(path))) as archive:
        members = archive.infolist()
        if len(members) != len(FILES) + 1 or {m.filename for m in members} != FILES | {'manifest.json'}:
            raise ValueError('Unexpected or duplicate archive entries')
        if sum(m.file_size for m in members) > MAX_BYTES:
            raise ValueError('Expanded package exceeds budget')
        files = {m.filename: archive.read(m) for m in members}
    manifest = json.loads(files.pop('manifest.json'))
    if manifest.get('format') != 'ai-home-smoke-package-v1' or set(manifest.get('files', {})) != FILES:
        raise ValueError('Invalid package manifest')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('source_revision', '')):
        raise ValueError('Invalid source revision')
    for name, raw in files.items():
        if manifest['files'][name] != {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}:
            raise ValueError(f'Artifact integrity mismatch: {name}')
    identity = validate(files)
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError('Manifest identity mismatch')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('build')
    create.add_argument('directory', type=Path)
    create.add_argument('output', type=Path)
    create.add_argument('--revision', required=True)
    create.add_argument('--license', type=Path, default=Path('LICENSE'))
    check = commands.add_parser('verify')
    check.add_argument('archive', type=Path)
    args = parser.parse_args()
    result = (verify(args.archive) if args.command == 'verify' else
              build(args.directory, args.output, revision=args.revision, license_path=args.license))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
