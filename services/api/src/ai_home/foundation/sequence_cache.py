"""Local write-once token sequence cache; never loads pre-existing cache files."""
from array import array
import hashlib
import json
from pathlib import Path
import sys

import torch


def write_sequences(rows, directory, context):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    count, digest = 0, hashlib.sha256()
    path = directory / 'sequences.i64'
    with path.open('xb') as target:
        for inputs, labels in rows:
            if len(inputs) != context or len(labels) != context:
                raise ValueError('Invalid cached sequence dimensions')
            packed = array('q', inputs + labels)
            if packed.itemsize != 8:
                raise RuntimeError('Sequence cache requires 64-bit signed integers')
            raw = packed.tobytes()
            target.write(raw)
            digest.update(raw)
            count += 1
    if not count:
        raise ValueError('Split has no full sequences; reduce context or supply longer documents')
    metadata = {'format': 'ai-home-local-sequence-cache-v1', 'rows': count,
                'context': context, 'dtype': 'int64', 'byteorder': sys.byteorder,
                'sha256': digest.hexdigest(), 'bytes': count * context * 16,
                'scope': 'Local derived data; may contain private training content; not a public artifact'}
    with (directory / 'manifest.json').open('x', encoding='utf-8') as target:
        json.dump(metadata, target, indent=2)
    mapped = torch.from_file(str(path), shared=False, size=count * 2 * context,
                             dtype=torch.int64).reshape(count, 2, context)
    return mapped[:, 0, :], mapped[:, 1, :]
