from __future__ import annotations

from functools import lru_cache
import os
import platform
import sys
from typing import Any

import numpy as np
import scipy
import sklearn
import torch

from .config import canonical_json_bytes, sha256_bytes
from utils.resources import REQUIRED_WORKER_ENVIRONMENT


@lru_cache(maxsize=1)
def runtime_signature() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    return {
        "signature_schema": "synthetic_runtime_v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_basename": __import__("pathlib").Path(sys.executable).name,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_build_version": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "torch_visible_gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "torch_visible_gpu_names": gpu_names,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "worker_environment": {
            key: os.environ.get(key)
            for key in sorted(REQUIRED_WORKER_ENVIRONMENT)
        },
    }


def runtime_fingerprint() -> str:
    return sha256_bytes(canonical_json_bytes(runtime_signature()))
