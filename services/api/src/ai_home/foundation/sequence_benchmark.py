"""Separate-process measurement of sequence preparation, not model training."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def peak_rss():
    info = psutil.Process().memory_info()
    if hasattr(info, 'peak_wset'):
        return info.peak_wset
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def worker(mode, output):
    import torch
    from .tokenizer import ByteBPE
    from .training import _sequences
    torch.set_num_threads(1)
    records = json.loads((output / 'corpus.json').read_text(encoding='utf-8'))['records']
    before_rss, before_peak = psutil.Process().memory_info().rss, peak_rss()
    began = time.perf_counter()
    sequences = _sequences(records, ByteBPE(), 256,
        cache_path=output / 'cache' if mode == 'disk' else None)
    seconds = time.perf_counter() - began
    after_rss, after_peak = psutil.Process().memory_info().rss, peak_rss()
    digest = hashlib.sha256()
    # Hash bounded blocks after the memory observation, forcing a full read.
    for index in range(0, len(sequences[0]), 128):
        block = torch.stack([x[index:index+128] for x in sequences], dim=1)
        digest.update(array('q', block.flatten().tolist()).tobytes())
    return {'mode': mode, 'rows': len(sequences[0]), 'context': 256,
        'sequence_sha256': digest.hexdigest(), 'preparation_seconds': seconds,
        'rss_before': before_rss, 'rss_after_preparation': after_rss,
        'lifetime_peak_before': before_peak, 'lifetime_peak_after_preparation': after_peak,
        'peak_growth_over_prior_peak': max(0, after_peak - before_peak),
        'rss_after_full_read': psutil.Process().memory_info().rss,
        'lifetime_peak_after_full_read': peak_rss(),
        'note': 'RSS and lifetime peak include Python/PyTorch; OS paging and allocator dependent; hashing occurs afterward'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker', choices=('memory', 'disk'))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        print(json.dumps(worker(args.worker, output)))
        return
    if psutil.virtual_memory().available < 1024**3:
        raise RuntimeError('At least 1 GiB available RAM required for this bounded benchmark')
    output.mkdir(parents=True, exist_ok=False)
    records = [{'text': (f'AI@Home public authored fixture {i}. ' * 32000),
        'source': 'AI@Home authored sequence benchmark v1', 'url': 'local:sequence-benchmark',
        'license': 'Apache-2.0', 'date': '2026-11-26', 'language': 'en',
        'training_allowed': True, 'privacy_reviewed': True, 'visibility': 'public'} for i in range(2)]
    (output / 'corpus.json').write_text(json.dumps({'records': records}), encoding='utf-8')
    results = []
    for mode in ('memory', 'disk'):
        process = subprocess.run([sys.executable, *(['-I'] if sys.flags.isolated else []),
            '-m', 'ai_home.foundation.sequence_benchmark', '--output-dir', str(output), '--worker', mode],
            capture_output=True, text=True, check=True, timeout=120)
        results.append(json.loads(process.stdout))
    if results[0]['sequence_sha256'] != results[1]['sequence_sha256']:
        raise RuntimeError('Sequence preparation changed tensor contents')
    report = {'format': 'ai-home-sequence-memory-benchmark-v1', 'runs': results,
        'text_bytes': sum(len(r['text'].encode()) for r in records),
        'scope': 'Two synthetic documents, one run per mode on this host; preparation only, not end-to-end training memory'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
