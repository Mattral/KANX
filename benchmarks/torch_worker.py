"""PyTorch half of the real-world benchmark, in its own process.

TensorFlow and PyTorch must not initialize CUDA in the same process. The
parent script (``benchmarks/real_world.py``) stays on TensorFlow and launches
this file when a GPU run is requested. This process loads ``kanx.torch``
without executing ``kanx/__init__.py``, which would import TensorFlow.

Stdout is one JSON object. Diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np


def _load_torch_stack():
    """Import PyTorch KAN without importing TensorFlow.

    A fresh interpreter would otherwise execute ``kanx/__init__.py`` on the
    first ``import kanx.torch``, and that module imports TensorFlow.
    """
    import torch
    from torch import nn

    src = Path(__file__).resolve().parents[1] / "src"
    if "kanx" not in sys.modules:
        package = types.ModuleType("kanx")
        package.__path__ = [str(src / "kanx")]
        package.__package__ = "kanx"
        sys.modules["kanx"] = package
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from kanx.torch import KAN, Trainer

    return torch, nn, KAN, Trainer


def split_folds(n: int, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Same fold split as ``benchmarks/real_world.py``."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if n < n_folds:
        raise ValueError(f"need at least {n_folds} rows to build {n_folds} folds, got {n}")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return list(np.array_split(idx, n_folds))


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return rmse, r2


def _latency(torch, model, batch: np.ndarray, device: str, runs: int) -> float:
    model.to(device)
    model.eval()
    values = torch.as_tensor(batch, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(values)
        if device == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(runs):
            start = time.perf_counter()
            model(values)
            if device == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def _fit_one(torch, Trainer, model, X, y, epochs: int, device: str) -> None:
    torch.manual_seed(0)
    Trainer(model, device=device).fit(
        X,
        y,
        epochs=epochs,
        batch_size=min(128, int(X.shape[0])),
        lr=1e-2,
        verbose=0,
    )


def _predict(torch, model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(X, device=next(model.parameters()).device))
    return pred.detach().cpu().numpy()


def _evaluate_model(
    torch,
    Trainer,
    build_model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    dataset: str,
    name: str,
    epochs: int,
    device: str,
    n_folds: int,
    latency_runs: int,
) -> dict:
    folds = split_folds(len(X), n_folds=n_folds, seed=0)
    rmse_scores = []
    r2_scores = []
    train_times = []
    params = int(sum(p.numel() for p in build_model().parameters() if p.requires_grad))
    for fold, test_idx in enumerate(folds):
        train_parts = [part for i, part in enumerate(folds) if i != fold]
        train_idx = np.concatenate(train_parts) if train_parts else test_idx
        model = build_model()
        started = time.perf_counter()
        _fit_one(torch, Trainer, model, X[train_idx], y[train_idx], epochs, device)
        train_times.append(time.perf_counter() - started)
        rmse, r2 = _scores(y[test_idx], _predict(torch, model, X[test_idx]))
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    model = build_model()
    _fit_one(torch, Trainer, model, X, y, epochs, device)
    probe = np.repeat(X[:1], min(256, max(len(X), 1)), axis=0)
    cpu_ms = _latency(torch, model, probe, "cpu", latency_runs)
    gpu_ms = _latency(torch, model, probe, "cuda", latency_runs) if device == "cuda" else None
    return {
        "dataset": dataset,
        "model": name,
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores, ddof=1)) if len(rmse_scores) > 1 else 0.0,
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores, ddof=1)) if len(r2_scores) > 1 else 0.0,
        "train_time_s": float(np.mean(train_times)),
        "params": params,
        "inference_latency_cpu_ms": cpu_ms,
        "inference_latency_gpu_ms": gpu_ms,
    }


def run_job(config: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """Train the PyTorch KAN and MLP. ``config['device']`` is ``cpu`` or ``cuda``."""
    torch, nn, KAN, Trainer = _load_torch_stack()
    device = str(config["device"])
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available in the benchmark subprocess")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if X.ndim != 2 or y.shape[0] != X.shape[0]:
        raise ValueError(f"expected X (n, features) and y (n, 1), got {X.shape} and {y.shape}")

    in_features = int(X.shape[1])
    hidden = int(config["hidden"])
    shared = {
        "dataset": str(config["dataset"]),
        "epochs": int(config["epochs"]),
        "device": device,
        "n_folds": int(config["n_folds"]),
        "latency_runs": int(config["latency_runs"]),
    }
    def build_kan():
        return KAN([in_features, hidden, 1])

    def build_mlp():
        return nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    results = [
        _evaluate_model(torch, Trainer, build_kan, X, y, name="KAN_TORCH", **shared),
        _evaluate_model(torch, Trainer, build_mlp, X, y, name="MLP_TORCH", **shared),
    ]
    return {
        "tensorflow_imported": "tensorflow" in sys.modules,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PyTorch real-world benchmark worker")
    parser.add_argument("--config", required=True, help="JSON job: dataset, epochs, folds, hidden, device")
    parser.add_argument("--data", required=True, help="npz file with arrays X and y")
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        with np.load(args.data) as store:
            X = np.asarray(store["X"])
            y = np.asarray(store["y"])
        payload = run_job(config, X, y)
    except Exception as exc:
        print(f"PyTorch benchmark worker failed: {exc}", file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
