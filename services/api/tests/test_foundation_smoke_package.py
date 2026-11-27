import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

from ai_home.foundation import smoke, smoke_package


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    root = tmp_path_factory.mktemp('smoke-package')
    old = sys.argv
    try:
        sys.argv = ['smoke', '--output-dir', str(root), '--device', 'cpu']
        smoke.main()
    finally:
        sys.argv = old
    archive = root / 'release.zip'
    smoke_package.build(root, archive, revision='a' * 40, license_path=Path('LICENSE'))
    return root, archive


def test_real_smoke_package_round_trip_and_exclusive_publish(package):
    root, archive = package
    manifest = smoke_package.verify(archive)
    assert manifest['parameter_count'] > 0
    with zipfile.ZipFile(archive) as source:
        assert json.loads(source.read('report.json'))['embedding_delta_l2_since_resume'] > 0
    with pytest.raises(FileExistsError):
        smoke_package.build(root, archive, revision='a' * 40, license_path=Path('LICENSE'))


def rewrite(source, destination, mutate):
    with zipfile.ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    mutate(files)
    with zipfile.ZipFile(destination, 'w') as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)


def test_corruption_and_metadata_forgery_rejected(package, tmp_path):
    _, source = package
    corrupt = tmp_path / 'corrupt.zip'
    rewrite(source, corrupt, lambda files: files.update({'config.json': b'{}'}))
    with pytest.raises(ValueError, match='integrity'):
        smoke_package.verify(corrupt)
    def forge(files):
        files['config.json'] = b'{}'
        manifest = json.loads(files['manifest.json'])
        manifest['files']['config.json'] = {'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}
        files['manifest.json'] = json.dumps(manifest).encode()
    rewrite(source, corrupt, forge)
    with pytest.raises(ValueError, match='metadata mismatch'):
        smoke_package.verify(corrupt)


def test_extra_paths_and_private_dataset_rejected(package, tmp_path):
    _, source = package
    corrupt = tmp_path / 'extra.zip'
    rewrite(source, corrupt, lambda files: files.update({'../outside': b'x'}))
    with pytest.raises(ValueError, match='entries'):
        smoke_package.verify(corrupt)
    with zipfile.ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    data = json.loads(files['dataset.json'])
    data['records'][0]['visibility'] = 'reviewed-training-only'
    files['dataset.json'] = json.dumps(data).encode()
    with pytest.raises(ValueError, match='public'):
        smoke_package.validate(files)
