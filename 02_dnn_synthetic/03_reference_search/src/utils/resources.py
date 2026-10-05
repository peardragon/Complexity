"""Portable worker settings and concurrency limits for synthetic commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping


REQUIRED_WORKER_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class ResourcePolicy:
    schema_version: int
    max_cpu_threads: int
    max_concurrent_gpus: int
    worker_environment: Mapping[str, str]
    # Compatibility for old protocol metadata; these fractions do not detect
    # hardware or constrain portable resource settings.
    cpu_fraction_limit: float = 0.75
    gpu_fraction_limit: float = 0.5

    @classmethod
    def from_json(cls, path: str | Path) -> "ResourcePolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        environment = payload.get(
            "worker_environment",
            payload.get("cpu_worker_policy", {}).get("environment", {}),
        )
        settings = dict(REQUIRED_WORKER_ENVIRONMENT)
        settings.update({str(key): str(value) for key, value in environment.items()})
        policy = cls(
            schema_version=int(payload.get("schema_version", 1)),
            max_cpu_threads=int(payload["max_cpu_threads"]),
            max_concurrent_gpus=int(payload["max_concurrent_gpus"]),
            worker_environment=settings,
            cpu_fraction_limit=float(payload.get("cpu_fraction_limit", 0.75)),
            gpu_fraction_limit=float(payload.get("gpu_fraction_limit", 0.5)),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("resource schema_version must equal 1")
        if self.max_cpu_threads < 1 or self.max_concurrent_gpus < 1:
            raise ValueError("CPU and GPU concurrency limits must be positive")
        if self.worker_environment.get("CUBLAS_WORKSPACE_CONFIG") not in {
            ":4096:8", ":16:8"
        }:
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must select deterministic cuBLAS")
        for key in REQUIRED_WORKER_ENVIRONMENT:
            if key == "CUBLAS_WORKSPACE_CONFIG":
                continue
            value = int(self.worker_environment[key])
            if not 1 <= value <= self.max_cpu_threads:
                raise ValueError(f"{key} must be within the CPU thread limit")

    def worker_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        environment = dict(os.environ if base is None else base)
        self.install_worker_environment(environment)
        return environment

    def install_worker_environment(
        self,
        environment: MutableMapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        target = os.environ if environment is None else environment
        for key, value in self.worker_environment.items():
            if key == "CUDA_VISIBLE_DEVICES":
                # An optional configuration mask is only a default. The caller's
                # explicit mask (including an empty CPU-only mask) takes priority.
                target.setdefault(key, value)
            else:
                target[key] = value
        return self.assert_worker_environment(target)

    def assert_worker_environment(
        self,
        environment: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        active = os.environ if environment is None else environment
        mismatches = {
            key: (value, active.get(key))
            for key, value in self.worker_environment.items()
            if key != "CUDA_VISIBLE_DEVICES" and active.get(key) != value
        }
        if "CUDA_VISIBLE_DEVICES" in self.worker_environment and "CUDA_VISIBLE_DEVICES" not in active:
            mismatches["CUDA_VISIBLE_DEVICES"] = (
                self.worker_environment["CUDA_VISIBLE_DEVICES"], None
            )
        if mismatches:
            raise RuntimeError(f"worker environment does not match configured settings: {mismatches}")
        return {key: str(active[key]) for key in self.worker_environment}

    def assert_request(self, *, cpu_threads: int, gpu_count: int) -> None:
        if not 0 <= int(cpu_threads) <= self.max_cpu_threads:
            raise ValueError(f"requested {cpu_threads} CPU threads; limit is {self.max_cpu_threads}")
        if not 0 <= int(gpu_count) <= self.max_concurrent_gpus:
            raise ValueError(f"requested {gpu_count} GPUs; limit is {self.max_concurrent_gpus}")

    def assert_gpu_devices(self, devices: Iterable[int]) -> tuple[int, ...]:
        resolved = tuple(int(device) for device in devices)
        if len(set(resolved)) != len(resolved) or any(device < 0 for device in resolved):
            raise ValueError("logical GPU indices must be distinct and nonnegative")
        self.assert_request(cpu_threads=0, gpu_count=len(resolved))
        return resolved

    def assert_masked_cuda_runtime(self) -> None:
        self.assert_worker_environment()
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA execution requires at least one visible GPU")

    def assert_worker_plan(
        self,
        *,
        worker_count: int,
        cpu_threads_per_worker: int,
        gpu_devices: Iterable[int] = (),
    ) -> None:
        workers, threads = int(worker_count), int(cpu_threads_per_worker)
        if workers < 0 or threads < 0:
            raise ValueError("worker and per-worker thread counts must be nonnegative")
        devices = self.assert_gpu_devices(gpu_devices)
        self.assert_request(cpu_threads=workers * threads, gpu_count=len(devices))
