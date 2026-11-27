import json
import sys

import pytest
pytest.importorskip("torch")
from test_foundation_vision_data import dataset
from ai_home.foundation.vision_train_cli import main
from ai_home.foundation.tokenizer import ByteBPE


def test_cli_real_training_and_exclusive_output(dataset, tmp_path, monkeypatch, capsys):
    manifest, data = dataset
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": {"width": 16, "layers": 1, "heads": 2, "kv_heads": 1,
        "hidden": 32, "context": 16}, "vision": {"image_size": 8, "patch_size": 4},
        "training": {"steps": 1, "warmup": 0, "objective": "sft"}, "prompt": "Q",
        "tokenizer_vocab_size": 270}), encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["vision_train_cli", "--dataset", str(manifest),
        "--config", str(config), "--output-dir", str(output)])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["tokenizer_sha256"] == ByteBPE.train(["red"], vocab_size=270).fingerprint
    assert report["history"][0]["step"] == 1
    checkpoint = output / report["checkpoint"]
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert checkpoint.read_bytes() == before
