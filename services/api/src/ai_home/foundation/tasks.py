"""Bounded, declarative Foundation tasks safe to dispatch to public nodes."""
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator, model_serializer

from ai_home.privacy import contains_common_secret
from .data import canonical_hash, prepare_records
from .tokenizer import ByteBPE

TOKENIZE_JOB = "foundation.tokenize.v1"


class PublicRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: StrictStr = Field(min_length=1, max_length=500)
    url: StrictStr = Field(min_length=1, max_length=1000)
    license: StrictStr = Field(min_length=1, max_length=200)
    date: StrictStr = Field(min_length=1, max_length=80)
    language: StrictStr = Field(min_length=1, max_length=40)
    text: StrictStr = Field(min_length=1, max_length=16_384)
    training_allowed: StrictBool
    privacy_reviewed: StrictBool
    visibility: Literal["public"]
    split_group_sha256: StrictStr | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_serializer(mode='wrap')
    def serialize_record(self, handler):
        result = handler(self)
        if result.get('split_group_sha256') is None:
            result.pop('split_group_sha256', None)
        return result


class TokenizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    specials: list[StrictStr] = Field(min_length=6, max_length=6)
    merges: list[list[StrictInt]] = Field(max_length=250)

    @model_validator(mode="after")
    def bounded_merges(self):
        lengths = [1] * 256 + [0] * 6
        for pair in self.merges:
            if len(pair) != 2 or any(i < 0 or i >= len(lengths) or 256 <= i < 262 for i in pair):
                raise ValueError("Invalid merge references")
            length = sum(lengths[i] for i in pair)
            if length > 8192:
                raise ValueError("Token expansion exceeds node budget")
            lengths.append(length)
        ByteBPE.from_dict(self.model_dump())
        return self


class TokenizePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PublicRecord] = Field(min_length=2, max_length=32)
    tokenizer: TokenizerSpec

    @model_validator(mode="after")
    def reviewed_public_only(self):
        raw = self.model_dump()
        body = json.dumps(raw, ensure_ascii=False)
        if len(body.encode()) > 65_536:
            raise ValueError("Tokenization input exceeds 64 KiB budget")
        if contains_common_secret(body) or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", body):
            raise ValueError("Potential personal information or credential")
        prepare_records(raw["records"])
        return self


def execute_tokenize(payload):
    validated = TokenizePayload.model_validate(payload)
    tokenizer = ByteBPE.from_dict(validated.tokenizer.model_dump())
    records = prepare_records([r.model_dump() for r in validated.records])
    artifact = {"format": "ai-home-tokenized-shard-v1", "tokenizer_sha256": tokenizer.fingerprint,
                "dataset_sha256": canonical_hash(records),
                "documents": [{"sha256": r["sha256"], "tokens": [257, *tokenizer.encode(r["text"]), 258],
                               **({'split_group_sha256': r['split_group_sha256']} if 'split_group_sha256' in r else {})}
                              for r in records]}
    return {"token_count": sum(len(r["tokens"]) for r in artifact["documents"]),
            "artifact_sha256": canonical_hash(artifact),
            "artifact_json": json.dumps(artifact, sort_keys=True, separators=(",", ":"))}
