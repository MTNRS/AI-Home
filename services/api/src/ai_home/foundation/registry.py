"""Local operator-managed model candidates, evaluated approval and audited rollback."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from uuid import uuid4

from .evaluation import evaluate_checkpoint


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_evaluations(baseline, candidate):
    for key in ("suite_sha256", "generation", "device", "torch_version"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Evaluation comparison requires matching {key}")
    if not baseline["all_prompts_fit"] or not candidate["all_prompts_fit"]:
        raise ValueError("Truncated evaluation cannot establish regression safety")
    old = {row["id"]: row for row in baseline["results"]}
    new = {row["id"]: row for row in candidate["results"]}
    if set(old) != set(new) or any(old[key]["category"] != new[key]["category"] for key in old):
        raise ValueError("Evaluation probe sets differ")
    regressions = sorted(key for key in old if old[key]["passed"] and not new[key]["passed"])
    gains = sorted(key for key in old if not old[key]["passed"] and new[key]["passed"])
    categories = {}
    for category in sorted({row["category"] for row in old.values()}):
        ids = [key for key in old if old[key]["category"] == category]
        before = sum(old[key]["passed"] for key in ids)
        after = sum(new[key]["passed"] for key in ids)
        categories[category] = {"total": len(ids), "baseline_passed": before,
                                "candidate_passed": after, "delta": after - before}
    return {"baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
            "suite_sha256": candidate["suite_sha256"], "regressions": regressions,
            "gains": gains, "categories": categories, "passed": not regressions}


class ModelRegistry:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "weights").mkdir(exist_ok=True)
        with self.database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY, checkpoint_hash TEXT NOT NULL,
                    evaluation TEXT NOT NULL, created REAL NOT NULL,
                    approved_by TEXT, approved_at REAL, minimum_accuracy REAL);
                CREATE TABLE IF NOT EXISTS approval_comparisons (
                    candidate_id TEXT PRIMARY KEY REFERENCES candidates(id),
                    baseline_id TEXT REFERENCES candidates(id), comparison TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS activation (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL REFERENCES candidates(id),
                    operator TEXT NOT NULL, reason TEXT NOT NULL, created REAL NOT NULL);
            """)

    @contextmanager
    def database(self):
        db = sqlite3.connect(self.directory / "registry.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def checkpoint(self, digest):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid checkpoint digest")
        path = self.directory / "weights" / (digest + ".pt")
        if file_hash(path) != digest:
            raise ValueError("Registered checkpoint integrity failure")
        return path

    def register(self, checkpoint, suite, *, device="cpu", max_new_tokens=64, max_bytes=2_000_000_000):
        if Path(checkpoint).stat().st_size > max_bytes:
            raise ValueError("Checkpoint exceeds registry budget")
        digest = file_hash(checkpoint)
        destination = self.directory / "weights" / (digest + ".pt")
        temporary = self.directory / "weights" / (str(uuid4()) + ".tmp")
        try:
            with temporary.open("xb") as output, Path(checkpoint).open("rb") as source:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            if file_hash(temporary) != digest:
                raise ValueError("Source checkpoint changed while copying")
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        destination = self.checkpoint(digest)
        # Evaluate the retained bytes, never a report supplied by the candidate producer.
        report = evaluate_checkpoint(destination, suite, device=device, max_new_tokens=max_new_tokens, max_bytes=max_bytes)
        if file_hash(destination) != report["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during evaluation")
        report["suite"] = suite
        version = str(uuid4())
        with self.database() as db:
            db.execute("INSERT INTO candidates VALUES (?, ?, ?, ?, NULL, NULL, NULL)",
                       (version, digest, json.dumps(report, sort_keys=True), time.time()))
        return version

    def list(self):
        with self.database() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, checkpoint_hash, created, approved_by, approved_at, minimum_accuracy FROM candidates ORDER BY created, id")]

    def inspect(self, version):
        with self.database() as db:
            row = db.execute("SELECT * FROM candidates WHERE id=?", (version,)).fetchone()
            if row is None:
                raise ValueError("Unknown model candidate")
            result = dict(row)
            result["evaluation"] = json.loads(result["evaluation"])
            comparison = db.execute("SELECT baseline_id, comparison FROM approval_comparisons WHERE candidate_id=?", (version,)).fetchone()
            result["approval_comparison"] = None if comparison is None else {
                "baseline_id": comparison["baseline_id"], "result": json.loads(comparison["comparison"])}
            return result

    def approve(self, version, *, operator, minimum_accuracy=1.0, baseline_version=None):
        if not isinstance(operator, str) or not operator.strip() or len(operator) > 200:
            raise ValueError("Approval operator required")
        if type(minimum_accuracy) not in (int, float) or not 0 <= minimum_accuracy <= 1:
            raise ValueError("Invalid approval threshold")
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM candidates WHERE id=?", (version,)).fetchone()
            if row is None:
                raise ValueError("Unknown model candidate")
            self.checkpoint(row["checkpoint_hash"])
            report = json.loads(row["evaluation"])
            results = report["results"]
            accuracy = sum(r["passed"] for r in results) / len(results)
            if not report["all_prompts_fit"] or not report["training_lineage_available"] or accuracy < minimum_accuracy:
                raise ValueError("Candidate does not pass approval gates")
            if row["approved_by"] is not None:
                raise ValueError("Candidate approval already recorded")
            baseline_id = baseline_version
            if baseline_id is None:
                active = db.execute("SELECT candidate_id FROM activation ORDER BY sequence DESC LIMIT 1").fetchone()
                if active is not None:
                    baseline_id = active["candidate_id"]
                else:
                    latest = db.execute("SELECT id FROM candidates WHERE approved_by IS NOT NULL ORDER BY approved_at DESC, id LIMIT 1").fetchone()
                    baseline_id = latest["id"] if latest is not None else None
            comparison = {"bootstrap": True, "passed": True}
            if baseline_id is not None:
                baseline = db.execute("SELECT * FROM candidates WHERE id=?", (baseline_id,)).fetchone()
                if baseline is None or baseline["approved_by"] is None or baseline_id == version:
                    raise ValueError("Regression baseline must be another approved candidate")
                self.checkpoint(baseline["checkpoint_hash"])
                comparison = compare_evaluations(json.loads(baseline["evaluation"]), report)
                if not comparison["passed"]:
                    raise ValueError("Candidate regresses on previously passing probes")
            db.execute("INSERT INTO approval_comparisons VALUES (?, ?, ?)",
                       (version, baseline_id, json.dumps(comparison, sort_keys=True)))
            db.execute("UPDATE candidates SET approved_by=?, approved_at=?, minimum_accuracy=? WHERE id=?",
                       (operator, time.time(), minimum_accuracy, version))

    def activate(self, version, *, operator, reason):
        if not all(isinstance(value, str) and value.strip() and len(value) <= 500 for value in (operator, reason)):
            raise ValueError("Activation operator and reason required")
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM candidates WHERE id=?", (version,)).fetchone()
            if row is None or row["approved_by"] is None:
                raise ValueError("Only an approved candidate may be activated")
            self.checkpoint(row["checkpoint_hash"])
            db.execute("INSERT INTO activation(candidate_id, operator, reason, created) VALUES (?, ?, ?, ?)",
                       (version, operator, reason, time.time()))

    def active_checkpoint(self):
        with self.database() as db:
            row = db.execute("""SELECT c.checkpoint_hash FROM activation a
                JOIN candidates c ON c.id=a.candidate_id
                WHERE c.approved_by IS NOT NULL ORDER BY a.sequence DESC LIMIT 1""").fetchone()
            if row is None:
                raise ValueError("No approved model is active")
            return self.checkpoint(row["checkpoint_hash"])

    def history(self):
        with self.database() as db:
            return [dict(row) for row in db.execute("SELECT * FROM activation ORDER BY sequence")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("checkpoint", type=Path)
    register.add_argument("suite", type=Path)
    register.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    approve = commands.add_parser("approve")
    approve.add_argument("version")
    approve.add_argument("--operator", required=True)
    approve.add_argument("--baseline", default=None)
    approve.add_argument("--minimum-accuracy", type=float, default=1.0)
    activate = commands.add_parser("activate")
    activate.add_argument("version")
    activate.add_argument("--operator", required=True)
    activate.add_argument("--reason", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("version")
    commands.add_parser("list")
    commands.add_parser("history")
    args = parser.parse_args()
    registry = ModelRegistry(args.registry)
    if args.command == "register":
        print(registry.register(args.checkpoint, json.loads(args.suite.read_text(encoding="utf-8")), device=args.device))
    elif args.command == "inspect":
        print(json.dumps(registry.inspect(args.version), indent=2))
    elif args.command == "approve":
        registry.approve(args.version, operator=args.operator, minimum_accuracy=args.minimum_accuracy, baseline_version=args.baseline)
    elif args.command == "activate":
        registry.activate(args.version, operator=args.operator, reason=args.reason)
    else:
        print(json.dumps(getattr(registry, args.command)(), indent=2))


if __name__ == "__main__":
    main()
