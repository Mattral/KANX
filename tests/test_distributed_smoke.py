"""Short distributed training smokes. CPU stands in when no GPU is present."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str) -> dict:
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TF_NUM_INTRAOP_THREADS"] = "1"
    env["TF_NUM_INTEROP_THREADS"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["MASTER_ADDR"] = "127.0.0.1"
    env.pop("MASTER_PORT", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    payload = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            payload = json.loads(stripped)
    assert result.returncode == 0, result.stderr[-2000:]
    assert payload is not None, result.stdout + result.stderr
    assert payload["ok"] is True
    assert math.isfinite(payload["loss"])
    return payload


def test_tensorflow_mirrored_strategy_smoke() -> None:
    payload = _run("distributed_smoke_tf.py")
    assert payload["backend"] == "tensorflow"
    assert payload["strategy"] in {"MirroredStrategy", "OneDeviceStrategy"}
    assert payload["mode"] in {"gpu", "cpu-logical", "single-device"}
    if payload["mode"] == "single-device":
        assert payload["replicas"] == 1
    else:
        assert payload["replicas"] >= 2


def test_pytorch_distributed_smoke() -> None:
    payload = _run("distributed_smoke_torch.py")
    assert payload["backend"] == "torch"
    assert payload["distributed"] in {"gloo", "nccl"}
    assert payload["world_size"] == 2
