"""Explicit public-source connectors producing unapproved, review-bound records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import quote

import httpx
from ai_home.immutable_artifacts import publish_bytes
from ai_home.foundation.data import canonical_hash

MAX_DOCUMENT_BYTES = 500_000
MAX_BATCH_BYTES = 8_000_000


def github_text(spec, client):
    """Fetch one public UTF-8 text file at a full commit SHA, without credentials."""
    required = {"connector", "repository", "commit", "path", "license", "language"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Invalid GitHub source specification")
    repo, commit, path = spec["repository"], spec["commit"], spec["path"]
    if (not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo)
            or repo.split("/")[-1] in {".", ".."}):
        raise ValueError("Invalid public repository")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full immutable commit SHA is required")
    if (not isinstance(path, str) or not path or len(path) > 1024
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(ord(char) < 32 or char in "\\?#" for char in path)
            or Path(path).suffix.lower() not in {".md", ".txt", ".rst", ".py", ".json", ".csv"}):
        raise ValueError("Unsupported repository text path")
    if any(not isinstance(spec[key], str) or not spec[key].strip() or len(spec[key]) > 200
           for key in ("license", "language")):
        raise ValueError("Declared license and language required")
    encoded = quote(path, safe="/")
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{encoded}"
    with client.stream("GET", url, follow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Expected public text response without redirection")
        raw = bytearray()
        for block in response.iter_bytes():
            raw.extend(block)
            if len(raw) > MAX_DOCUMENT_BYTES:
                raise ValueError("Source document exceeds download budget")
    text = unicodedata.normalize("NFC", bytes(raw).decode("utf-8")).replace("\r\n", "\n").strip()
    if not text.strip() or "\x00" in text:
        raise ValueError("Source is empty or not supported text")
    return {"source": f"GitHub public text: {repo}",
            "url": f"https://github.com/{repo}/blob/{commit}/{encoded}",
            "license": spec["license"], "license_status": "declared-unverified",
            "date": datetime.now(timezone.utc).date().isoformat(), "language": spec["language"],
            "format": "utf-8-text", "text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "connector": "github-text-v1", "source_commit": commit, "source_path": path,
            "split_group_sha256": canonical_hash({'family': 'github-file-v1', 'repository': repo.lower(), 'path': path}),
            "training_allowed": False, "retrieval_allowed": False,
            "privacy_reviewed": False, "visibility": "public"}


# Extension points are installed Python code, never module names from manifests.
CONNECTORS = {"github-text-v1": github_text}


def acquire(specifications, output, *, transport=None):
    if not isinstance(specifications, list) or not 1 <= len(specifications) <= 20:
        raise ValueError("Expected 1..20 explicit source specifications")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Pending output already exists")
    rows, seen, size = [], set(), 0
    with httpx.Client(timeout=20, trust_env=False, transport=transport,
                      headers={"User-Agent": "AI-Home-Knowledge/1.0", "Accept": "text/plain"}) as client:
        for spec in specifications:
            name = spec.get("connector") if isinstance(spec, dict) else None
            if not isinstance(name, str) or name not in CONNECTORS:
                raise ValueError("Unknown installed knowledge connector")
            record = CONNECTORS[name](spec, client)
            if record["sha256"] in seen:
                raise ValueError("Duplicate downloaded document; ambiguous review provenance")
            seen.add(record["sha256"])
            row = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
            size += len(row)
            if size > MAX_BATCH_BYTES:
                raise ValueError("Pending batch exceeds review budget")
            rows.append(row)
    publish_bytes(output, b"".join(rows))
    return {"documents": len(rows), "bytes": size, "output": str(output.resolve()),
            "state": "pending-license-and-privacy-review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.manifest.open("rb") as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise ValueError("Connector manifest exceeds budget")
    print(json.dumps(acquire(json.loads(raw), args.output)))


if __name__ == "__main__":
    main()
