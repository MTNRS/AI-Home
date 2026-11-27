"""Publish complete local artifact bytes without replacing an existing version."""
import os
from pathlib import Path
import tempfile


def publish_bytes(path, raw, *, before_publish=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".artifact.tmp")
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        if before_publish is not None:
            before_publish()
        # Same-directory hard link is exclusive and exposes only complete bytes.
        # Unsupported filesystems fail explicitly rather than falling back to overwrite.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
