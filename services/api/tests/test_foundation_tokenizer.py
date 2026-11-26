import tempfile
from pathlib import Path
import unittest

from ai_home.foundation.tokenizer import ByteBPE
from ai_home.foundation.data import split_records


class TokenizerTests(unittest.TestCase):
    def test_unicode_roundtrip_and_serialization(self):
        texts = ["Español, inglés 日本語 🧠\n", "def f(x): return x + 1\t", "<assistant> literal"]
        tokenizer = ByteBPE.train(texts * 4, vocab_size=310)
        self.assertGreater(tokenizer.vocab_size, 262)
        for text in texts + ["", "árbol\x00🌍"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertFalse(set(tokenizer.encode(text)) & set(range(256, 262)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            self.assertEqual(ByteBPE.load(path).fingerprint, tokenizer.fingerprint)
        self.assertEqual(ByteBPE.train(texts * 4, 310).fingerprint, tokenizer.fingerprint)

    def test_specials_are_structural(self):
        tokenizer = ByteBPE()
        ids = tokenizer.conversation([{"role": "user", "content": "<assistant>"}])
        self.assertEqual(ids[:2], [257, 260])
        self.assertEqual(ids[-2:], [258, 261])
        self.assertEqual(ids.count(261), 1)

    def test_reject_invalid_serialization(self):
        for merges in [[(0, 262)], [(256, 0)], [(0, 1), (0, 1)]]:
            with self.assertRaises(ValueError):
                ByteBPE(merges)
        with self.assertRaises(ValueError):
            ByteBPE.train(["x"], 10)

    def test_data_review_and_deduplication(self):
        common = dict(source="test", url="local:test", license="CC0-1.0", date="2026-11-26",
                      language="es", training_allowed=True, privacy_reviewed=True, visibility="public")
        a, b = split_records([{**common, "text": t} for t in ("uno", "dos", "uno")])
        self.assertEqual(len(a) + len(b), 2)
        self.assertFalse({r["sha256"] for r in a} & {r["sha256"] for r in b})
        with self.assertRaises(ValueError):
            split_records([{**common, "text": "email: foo@example.com"}, {**common, "text": "dos"}])
        with self.assertRaises(ValueError):
            split_records([{**common, "text": "uno", "privacy_reviewed": False}])


if __name__ == "__main__":
    unittest.main()
