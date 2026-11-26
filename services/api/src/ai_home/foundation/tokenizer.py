from collections import Counter
import hashlib
import json
from pathlib import Path


class ByteBPE:
    specials = ("<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>")
    version = 1

    def __init__(self, merges=()):
        self.merges = tuple(tuple(pair) for pair in merges)
        self.pieces = [bytes([i]) for i in range(256)] + [b""] * len(self.specials)
        seen = set()
        for pair in self.merges:
            if len(pair) != 2 or pair in seen or any(
                type(i) is not int or i < 0 or i >= len(self.pieces)
                or 256 <= i < 262 for i in pair
            ):
                raise ValueError("Invalid BPE merge table")
            seen.add(pair)
            self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])

    @property
    def vocab_size(self):
        return len(self.pieces)

    @staticmethod
    def _merge(tokens, pair, new_id):
        result = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    @classmethod
    def train(cls, texts, vocab_size=512, max_bytes=8_000_000):
        if not 262 <= vocab_size <= 65536:
            raise ValueError("Vocabulary must be between 262 and 65536")
        sequences = []
        total = 0
        for text in texts:
            data = text.encode("utf-8")
            total += len(data)
            if total > max_bytes:
                raise ValueError("Reference BPE trainer memory budget exceeded")
            sequences.append(list(data))
        if total == 0:
            raise ValueError("Empty tokenizer training corpus")
        merges = []
        for token_id in range(262, vocab_size):
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:
                break
            merges.append(pair)
            sequences = [cls._merge(seq, pair, token_id) for seq in sequences]
        return cls(merges)

    def extend(self, texts, vocab_size, max_bytes=8_000_000):
        """Append merges without changing any existing token ID or piece."""
        if type(vocab_size) is not int or not self.vocab_size < vocab_size <= 65536:
            raise ValueError("Expansion requires a larger vocabulary, at most 65536")
        sequences, total = [], 0
        for text in texts:
            total += len(text.encode("utf-8"))
            if total > max_bytes:
                raise ValueError("Reference BPE trainer memory budget exceeded")
            sequences.append(self.encode(text))
        if not total:
            raise ValueError("Empty tokenizer training corpus")
        merges = list(self.merges)
        for token_id in range(self.vocab_size, vocab_size):
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:
                break
            merges.append(pair)
            sequences = [self._merge(seq, pair, token_id) for seq in sequences]
        return type(self)(merges)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for token_id, pair in enumerate(self.merges, 262):
            tokens = self._merge(tokens, pair, token_id)
        return tokens

    def decode(self, tokens, errors="strict"):
        if any(type(i) is not int or not 0 <= i < self.vocab_size for i in tokens):
            raise ValueError("Token outside vocabulary")
        return b"".join(self.pieces[i] for i in tokens).decode("utf-8", errors=errors)

    def conversation(self, messages, generation_prompt=True):
        result = [257]
        for message in messages:
            role = message["role"]
            if role not in ("system", "user", "assistant"):
                raise ValueError("Invalid conversation role")
            result.append(259 + ("system", "user", "assistant").index(role))
            result.extend(self.encode(message["content"]))
            result.append(258)
        if generation_prompt:
            result.append(261)
        return result

    def to_dict(self):
        return {"version": self.version, "specials": list(self.specials),
                "merges": [list(pair) for pair in self.merges]}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != cls.version or data.get("specials") != list(cls.specials):
            raise ValueError("Incompatible tokenizer format")
        return cls(data["merges"])

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
