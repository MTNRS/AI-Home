"""Safe node-side hardware reporting and allowlisted job execution."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from csv import Error as CsvError
from csv import reader
from pathlib import Path

import httpx
import psutil

from ai_home.schemas import (
    ClaimedJobResponse,
    HardwareReport,
    MathSumJobPayload,
    NodeCredentialResponse,
)
from ai_home.urls import normalize_base_url


class HardwareDetectionError(RuntimeError):
    """Raised when installed hardware detection fails unexpectedly."""


def detect_hardware() -> HardwareReport:
    cpu_model = platform.processor() or platform.machine() or "Unknown CPU"
    nvidia_smi = _find_nvidia_smi()
    if nvidia_smi is not None:
        try:
            process = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            gpus = _parse_nvidia_smi(process.stdout)
        except (OSError, subprocess.SubprocessError, CsvError, ValueError) as error:
            raise HardwareDetectionError("NVIDIA GPU detection failed.") from error
    elif importlib.util.find_spec("torch") is not None:
        import torch

        gpus = []
        try:
            if torch.cuda.is_available():
                for device_index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(device_index)
                    gpus.append(
                        {
                            "device_index": device_index,
                            "name": properties.name,
                            "memory_total_bytes": properties.total_memory,
                        }
                    )
        except (RuntimeError, OSError) as error:
            raise HardwareDetectionError("CUDA device detection failed.") from error
    else:
        gpus = []

    return HardwareReport(
        platform=f"{platform.system()} {platform.release()}",
        cpu_model=cpu_model[:128],
        logical_cores=os.cpu_count() or 1,
        ram_total_bytes=psutil.virtual_memory().total,
        gpus=gpus,
    )


def _find_nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        return executable
    if sys.platform != "win32":
        return None

    program_roots = {
        os.environ.get("ProgramW6432"),
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    }
    for root in filter(None, program_roots):
        candidate = Path(root) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _parse_nvidia_smi(output: str) -> list[dict[str, int | str]]:
    devices = []
    for row in reader(output.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("Unexpected NVIDIA-SMI output.")
        device_index, name, memory_mib = row
        devices.append(
            {
                "device_index": int(device_index),
                "name": name.strip(),
                "memory_total_bytes": int(memory_mib.strip()) * 1024**2,
            }
        )
    return devices


def execute_job(job_type: str, payload: dict, *, allow_foundation_training=False, foundation_device="cpu") -> dict[str, int | str]:
    if job_type == "foundation.gradient.v1":
        if not allow_foundation_training:
            raise ValueError("Neural work requires explicit worker opt-in")
        if psutil.virtual_memory().available < 1024**3:
            raise ValueError("Insufficient available RAM for neural task")
        from ai_home.foundation.distributed import execute_gradient
        return execute_gradient(payload, device=foundation_device)
    if job_type == "foundation.tokenize.v1":
        from ai_home.foundation.tasks import execute_tokenize
        if psutil.virtual_memory().available < 256 * 1024**2:
            raise ValueError("Insufficient available RAM for tokenization")
        return execute_tokenize(payload)
    if job_type != "math.sum.v1":
        raise ValueError(f"Unsupported job type: {job_type}")
    validated = MathSumJobPayload.model_validate(payload)
    return {"sum": sum(validated.values)}


def register_node_with_server(
    *,
    server_url: str,
    user_token: str,
    name: str,
    timeout_seconds: float = 30,
    transport: httpx.BaseTransport | None = None,
) -> NodeCredentialResponse:
    server_url = normalize_base_url(server_url)
    hardware = detect_hardware()
    with httpx.Client(
        base_url=server_url,
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=timeout_seconds,
        transport=transport,
    ) as client:
        response = client.post(
            "/api/v1/nodes",
            json={
                "name": name,
                "hardware": hardware.model_dump(mode="json"),
                "supported_job_types": ["math.sum.v1", "foundation.tokenize.v1"],
            },
        )
        response.raise_for_status()
        return NodeCredentialResponse.model_validate(response.json())


class NodeAgent:
    """Runs only typed, versioned job handlers defined by this package."""

    def __init__(
        self,
        *,
        server_url: str,
        node_token: str,
        timeout_seconds: float = 30,
        transport: httpx.BaseTransport | None = None,
        allow_foundation_training: bool = False,
        foundation_device: str = "cpu",
        lease_renew_interval: int = 30,
    ) -> None:
        if foundation_device not in {"cpu", "cuda"}:
            raise ValueError("Foundation worker device must be cpu or cuda")
        if type(lease_renew_interval) is not int or not 1 <= lease_renew_interval <= 60:
            raise ValueError("Lease renewal interval must be between 1 and 60 seconds")
        self.lease_renew_interval = lease_renew_interval
        self.foundation_device = foundation_device
        self.last_run_metrics = None
        self.last_measurement_uploaded = False
        self.allow_foundation_training = allow_foundation_training
        server_url = normalize_base_url(server_url)
        self._client = httpx.Client(
            base_url=server_url,
            headers={"Authorization": f"Bearer {node_token}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def run_once(self) -> bool:
        # Only publish a sample after the server accepts a result; never reuse it
        # after an idle/failed attempt. Timings include service processing.
        self.last_run_metrics = None
        self.last_measurement_uploaded = False
        resources = {"ram_available_bytes": psutil.virtual_memory().available,
                     "foundation_device": self.foundation_device}
        if self.foundation_device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise ValueError("CUDA worker requested but CUDA is unavailable")
            resources["cuda_free_bytes"] = torch.cuda.mem_get_info()[0]
        started = time.perf_counter()
        heartbeat = self._client.post("/api/v1/node/heartbeat", json=resources)
        heartbeat.raise_for_status()
        heartbeat_seconds = time.perf_counter() - started

        started = time.perf_counter()
        response = self._client.post("/api/v1/node/jobs/claim")
        claim_seconds = time.perf_counter() - started
        if response.status_code == 204:
            return False
        response.raise_for_status()

        job = ClaimedJobResponse.model_validate(response.json())
        stop_renewal, renewal_failed = threading.Event(), threading.Event()
        def renew():
            while not stop_renewal.wait(self.lease_renew_interval):
                try:
                    response = self._client.post(f"/api/v1/node/jobs/{job.job_id}/renew",
                                                 json={"lease_token": job.lease_token})
                    response.raise_for_status()
                except Exception:
                    renewal_failed.set()
                    return
        renewer = threading.Thread(target=renew, daemon=True)
        renewer.start()
        started = time.perf_counter()
        try:
            result = execute_job(job.job_type, job.payload, allow_foundation_training=self.allow_foundation_training,
                                 foundation_device=self.foundation_device)
            execution_seconds = time.perf_counter() - started
        finally:
            stop_renewal.set()
            renewer.join()
        if renewal_failed.is_set():
            raise RuntimeError("Job lease renewal failed; result was not submitted")
        started = time.perf_counter()
        submitted = self._client.post(
            f"/api/v1/node/jobs/{job.job_id}/result",
            json={
                "lease_token": job.lease_token,
                "result": result,
            },
        )
        submitted.raise_for_status()
        submit_seconds = time.perf_counter() - started
        self.last_run_metrics = {
            "format": "ai-home-worker-service-measurement-v1",
            "job_type": job.job_type,
            "device": self.foundation_device,
            "heartbeat_service_seconds": heartbeat_seconds,
            "claim_service_seconds": claim_seconds,
            "execution_seconds": execution_seconds,
            "submit_service_seconds": submit_seconds,
            "claim_response_bytes": len(response.content),
            "submit_request_bytes": len(submitted.request.content),
            "claim_service_bytes_per_second": len(response.content) / max(claim_seconds, 1e-9),
            "submit_service_bytes_per_second": len(submitted.request.content) / max(submit_seconds, 1e-9),
        }
        if job.job_type == "foundation.gradient.v1":
            self.last_run_metrics["gradient_tokens_per_second"] = result["token_count"] / max(execution_seconds, 1e-9)
        from ai_home.schemas import WorkerServiceMeasurement
        sample = {key: self.last_run_metrics[key] for key in WorkerServiceMeasurement.model_fields}
        try:
            telemetry = self._client.post(f"/api/v1/node/jobs/{job.job_id}/measurement", json=sample)
            self.last_measurement_uploaded = telemetry.is_success
        except httpx.TransportError:
            # The result and credits are already committed. Telemetry loss must
            # not report useful work as failed or cause a second submission.
            pass
        return True

    def close(self) -> None:
        self._client.close()

