import hashlib

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
from ai_home.foundation.vision_images import load_image


def test_rgb_padding_hash_and_alpha(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 2), (255, 0, 0)).save(path)
    pixels, metadata = load_image(path, 4)
    assert pixels.shape == (1, 3, 4, 4)
    assert pixels.dtype == torch.float32
    assert torch.equal(pixels[0, :, 0, 0], torch.ones(3))
    assert torch.equal(pixels[0, :, 1, 0], torch.tensor([1., 0., 0.]))
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(metadata) == {"sha256", "format", "preprocessing", "image_size"}
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(path)
    assert load_image(path, 4)[0].eq(1).all()


def test_orientation_and_jpeg(tmp_path):
    path = tmp_path / "oriented.png"
    image = Image.new("RGB", (2, 1), "red")
    image.putpixel((1, 0), (0, 0, 255))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    pixels, _ = load_image(path, 2)
    assert torch.equal(pixels[0, :, 0, 0], torch.tensor([1., 0., 0.]))
    assert torch.equal(pixels[0, :, 1, 0], torch.tensor([0., 0., 1.]))
    image.save(tmp_path / "image.jpg")
    assert load_image(tmp_path / "image.jpg", 2)[1]["format"] == "JPEG"


def test_budgets_corruption_and_animation(tmp_path):
    path = tmp_path / "image.png"
    image = Image.new("RGB", (4, 4))
    image.save(path)
    with pytest.raises(ValueError, match="file budget"):
        load_image(path, 4, max_bytes=8)
    with pytest.raises(ValueError, match="pixel budget"):
        load_image(path, 4, max_pixels=15)
    image.save(path, save_all=True, append_images=[Image.new("RGB", (4, 4), "red")], duration=100)
    with pytest.raises(ValueError, match="Animated"):
        load_image(path, 4)
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        load_image(path, 4)


def test_cli_loads_actual_weights_and_image(tmp_path, monkeypatch, capsys):
    import sys
    from ai_home.foundation.model import FoundationConfig
    from ai_home.foundation.tokenizer import ByteBPE
    from ai_home.foundation.vision import VisionConfig, VisionPrefixModel
    from ai_home.foundation.vision_checkpoint import save_vision_weights
    from ai_home.foundation.vision_inference import main
    torch.manual_seed(17)
    model = VisionPrefixModel(FoundationConfig(vocab_size=262, width=16, layers=1,
        heads=2, kv_heads=1, hidden=32, context=12), VisionConfig(8, 4))
    image = tmp_path / "own.png"
    Image.new("RGB", (8, 8), "red").save(image)
    checkpoint = tmp_path / "candidate.pt"
    save_vision_weights(model, ByteBPE(), checkpoint, provenance={
        "dataset_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "visibility": "public", "training_allowed": True, "privacy_reviewed": True})
    monkeypatch.setattr(sys, "argv", ["vision_inference", "--checkpoint", str(checkpoint),
        "--image", str(image), "--prompt", "Describe", "--max-new-tokens", "2"])
    main()
    assert capsys.readouterr().out.endswith("\n")
