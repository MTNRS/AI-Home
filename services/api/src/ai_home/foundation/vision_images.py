"""Bounded local PNG/JPEG ingestion; pixels only, no metadata propagation."""

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
import torch


PREPROCESSING = "rgb-exif-white-letterbox-bilinear-v1"


def load_image(path, image_size, *, max_bytes=8_000_000, max_pixels=4_000_000, expected_sha256=None):
    """Return [1,3,H,W] CPU float pixels plus non-personal format/hash metadata.

    EXIF orientation is applied. Alpha is composited onto white. Aspect ratio is
    preserved with white padding; ICC color conversion is not performed.
    """
    if type(image_size) is not int or not 1 <= image_size <= 1024:
        raise ValueError("Invalid target image size")
    if any(type(v) is not int or v < 1 for v in (max_bytes, max_pixels)):
        raise ValueError("Invalid image budget")
    with Path(path).open("rb") as source:
        data = source.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Image exceeds file budget")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Image hash differs from reviewed dataset")
    with Image.open(BytesIO(data), formats=["PNG", "JPEG"]) as source:
        if source.width * source.height > max_pixels:
            raise ValueError("Image exceeds pixel budget")
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Animated images are not supported")
        image_format = source.format
        source.load()
        oriented = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
        background.alpha_composite(oriented)
        rgb = ImageOps.pad(background.convert("RGB"), (image_size, image_size),
                           method=Image.Resampling.BILINEAR, color=(255, 255, 255))
        pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    return pixels, {"sha256": digest, "format": image_format,
                    "preprocessing": PREPROCESSING, "image_size": image_size}
