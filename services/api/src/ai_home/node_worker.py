"""Safe node-side hardware reporting and allowlisted job execution."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
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


def execute_job(job_type: str, payload: dict) -> dict[str, int]:
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
    ) -> None:
        server_url = normalize_base_url(server_url)
        self._client = httpx.Client(
            base_url=server_url,
            headers={"Authorization": f"Bearer {node_token}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def run_once(self) -> bool:
        heartbeat = self._client.post("/api/v1/node/heartbeat")
        heartbeat.raise_for_status()

        response = self._client.post("/api/v1/node/jobs/claim")
        if response.status_code == 204:
            return False
        response.raise_for_status()

        job = ClaimedJobResponse.model_validate(response.json())
        result = execute_job(job.job_type, job.payload)
        submitted = self._client.post(
            f"/api/v1/node/jobs/{job.job_id}/result",
            json={
                "lease_token": job.lease_token,
                "result": result,
            },
        )
        submitted.raise_for_status()
        return True

    def close(self) -> None:
        self._client.close()

