"""Explicit transfer-only HTTP measurements, never a physical-link benchmark."""
import hashlib
import time
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from typing import Annotated, Literal


class TransferMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["ai-home-http-transfer-probe-v1"]
    payload_bytes: Annotated[StrictInt, Field(ge=1, le=1048576)]
    empty_round_trip_seconds: list[Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)]] = Field(min_length=3, max_length=3)
    download_seconds: float = Field(gt=0, le=60, allow_inf_nan=False)
    upload_seconds: float = Field(gt=0, le=60, allow_inf_nan=False)

    def summary(self, received_epoch):
        return {**self.model_dump(), "received_epoch": received_epoch,
            "expires_epoch": received_epoch + 120,
            "download_bytes_per_second": self.payload_bytes / max(self.download_seconds, 1e-9),
            "upload_bytes_per_second": self.payload_bytes / max(self.upload_seconds, 1e-9),
            "minimum_http_round_trip_seconds": min(self.empty_round_trip_seconds),
            "evidence": "client-reported HTTP service timings; not independently attested"}


def measure_transfer(client, *, size=65536):
    """Use an already authenticated node client; caller owns timeouts/transport.

    Three empty responses estimate HTTP service round trips, followed by one
    bounded download and upload. Does not store telemetry or award credits.
    """
    if type(size) is not int or not 1 <= size <= 1048576:
        raise ValueError("Probe size must be 1 to 1048576 bytes")
    samples = []
    for _ in range(3):
        began = time.perf_counter()
        response = client.get("/api/v1/node/transfer-probe", params={"size": 0})
        response.raise_for_status()
        if response.content:
            raise ValueError("Unexpected empty-probe response")
        samples.append(time.perf_counter() - began)
    expected = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    began = time.perf_counter()
    total, digest = 0, hashlib.sha256()
    with client.stream("GET", "/api/v1/node/transfer-probe", params={"size": size}) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(65536):
            total += len(chunk)
            if total > size:
                raise ValueError("Probe response exceeds requested size")
            digest.update(chunk)
    download_seconds = time.perf_counter() - began
    expected_hash = hashlib.sha256(expected).hexdigest()
    if total != size or digest.hexdigest() != expected_hash:
        raise ValueError("Probe download integrity mismatch")
    began = time.perf_counter()
    response = client.post("/api/v1/node/transfer-probe", content=expected,
                           headers={"Content-Type": "application/octet-stream"})
    response.raise_for_status()
    upload_seconds = time.perf_counter() - began
    if response.json() != {"bytes": size, "sha256": expected_hash}:
        raise ValueError("Probe upload integrity mismatch")
    return {"format": "ai-home-http-transfer-probe-v1", "payload_bytes": size,
            "empty_round_trip_seconds": samples, "download_seconds": download_seconds,
            "upload_seconds": upload_seconds,
            "download_bytes_per_second": size / max(download_seconds, 1e-9),
            "upload_bytes_per_second": size / max(upload_seconds, 1e-9),
            "claim": "HTTP transfer service including authentication and server overhead; not physical bandwidth or network-only RTT"}
