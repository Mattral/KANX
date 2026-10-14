"""Real-world tabular benchmark suite for KANX.

TensorFlow trains in this process. When the run is on a GPU, the PyTorch KAN
and a PyTorch MLP train in ``benchmarks/torch_worker.py`` so the two
libraries do not initialize CUDA together. CPU runs and ``--smoke`` stay
TensorFlow-only.

Run with:
    python benchmarks/real_world.py --datasets california --epochs 3
    python benchmarks/real_world.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import tensorflow as tf

from kanx import KAN as TF_KAN
from kanx import set_global_seed
from kanx.datasets import (
    load_california_housing,
    load_concrete_strength,
    load_energy_efficiency,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "real_world_results.json"


@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    r2_std: float
    train_time_s: float
    params: int
    inference_latency_cpu_ms: float
    inference_latency_gpu_ms: float | None


def cpu_info() -> str:
    info = platform.processor() or platform.machine()
    return f"{info} / {platform.system()}"


def gpu_info() -> str | None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return gpus[0].name
    return None


def param_count_tf(model: tf.keras.Model) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_variables))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def split_folds(n: int, n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if n < n_folds:
        raise ValueError(f"need at least {n_folds} rows to build {n_folds} folds, got {n}")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return list(np.array_split(idx, n_folds))


def train_tf_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    device: str,
    hidden: int = 32,
) -> tf.keras.Model:
    set_global_seed(0)
    if device == "cuda":
        tf_device = "/GPU:0"
    else:
        tf_device = "/CPU:0"
    with tf.device(tf_device):
        model = TF_KAN([X.shape[1], hidden, 1])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="mse")
        model.fit(X, y, epochs=epochs, batch_size=min(128, X.shape[0]), verbose=0)
    return model


def evaluate_tf_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y_pred = model(X, training=False).numpy()
    return rmse_score(y, y_pred), r2_score(y, y_pred)


def inference_latency_tf(model: tf.keras.Model, batch: np.ndarray, device: str, runs: int = 100) -> float:
    if device == "cuda" and not tf.config.list_physical_devices("GPU"):
        return 0.0  # Skip GPU latency if no GPU
    tf_device = "/GPU:0" if device == "cuda" else "/CPU:0"
    batch = batch.astype(np.float32)
    with tf.device(tf_device):
        for _ in range(5):
            _ = model(batch, training=False)
        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(batch, training=False)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(latencies))


def build_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if name == "california":
        return load_california_housing()
    if name == "concrete":
        return load_concrete_strength()
    if name == "energy":
        return load_energy_efficiency()
    raise ValueError(f"Unknown dataset: {name}")


def summary_markdown(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Dataset | Model | Params | Train (s) | RMSE mean ± std | R² mean ± std | CPU latency (ms) | GPU latency (ms) |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    for r in results:
        if r.inference_latency_gpu_ms is None:
            gpu = "N/A"
        else:
            gpu = f"{r.inference_latency_gpu_ms:.2f}"
        lines.append(
            f"| {r.dataset} | {r.model} | {r.params} | {r.train_time_s:.2f} | "
            f"{r.rmse_mean:.4f} ± {r.rmse_std:.4f} | {r.r2_mean:.4f} ± {r.r2_std:.4f} | "
            f"{r.inference_latency_cpu_ms:.2f} | {gpu} |"
        )
    return "\n".join(lines)


def evaluate_dataset(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    device: str,
    n_folds: int = 5,
    hidden: int = 32,
    latency_runs: int = 100,
) -> BenchmarkResult:
    folds = split_folds(len(X), n_folds=n_folds, seed=0)
    rmse_scores = []
    r2_scores = []
    train_times = []
    params = None
    for fold in range(len(folds)):
        test_idx = folds[fold]
        train_parts = [part for i, part in enumerate(folds) if i != fold]
        if train_parts:
            train_idx = np.concatenate(train_parts)
        else:
            train_idx = test_idx
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        start = time.perf_counter()
        model = train_tf_model(X_train, y_train, epochs=epochs, device=device, hidden=hidden)
        train_times.append(time.perf_counter() - start)
        fold_rmse, fold_r2 = evaluate_tf_model(model, X_test, y_test)
        if params is None:
            params = param_count_tf(model)

        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)

    model = train_tf_model(X, y, epochs=epochs, device=device, hidden=hidden)
    probe = np.repeat(X[:1], min(256, max(len(X), 1)), axis=0)
    inference_cpu = inference_latency_tf(model, probe, "cpu", runs=latency_runs)

    return BenchmarkResult(
        dataset=dataset_name,
        model="KAN_TF",
        rmse_mean=mean(rmse_scores),
        rmse_std=stdev(rmse_scores) if len(rmse_scores) > 1 else 0.0,
        r2_mean=mean(r2_scores),
        r2_std=stdev(r2_scores) if len(r2_scores) > 1 else 0.0,
        train_time_s=mean(train_times),
        params=params or 0,
        inference_latency_cpu_ms=inference_cpu,
        inference_latency_gpu_ms=None,
    )


def ensure_real_world_gate(results: list[BenchmarkResult]) -> None:
    """Fail the process when a real-world benchmark result is unusable."""
    problems: list[str] = []
    if not results:
        problems.append("no dataset results")
    for row in results:
        if row.params <= 0:
            problems.append(f"{row.dataset}: no parameters")
        for label, value in (
            ("rmse_mean", row.rmse_mean),
            ("rmse_std", row.rmse_std),
            ("r2_mean", row.r2_mean),
            ("r2_std", row.r2_std),
            ("train_time_s", row.train_time_s),
            ("inference_latency_cpu_ms", row.inference_latency_cpu_ms),
        ):
            if not math.isfinite(value):
                problems.append(f"{row.dataset}: {label} is not finite ({value})")
        if row.inference_latency_gpu_ms is not None and not math.isfinite(
            row.inference_latency_gpu_ms
        ):
            problems.append(
                f"{row.dataset}: inference_latency_gpu_ms is not finite "
                f"({row.inference_latency_gpu_ms})"
            )
    if problems:
        raise SystemExit("benchmark gate failed:\n- " + "\n- ".join(problems))


def launch_torch_worker(config: dict, data_path: Path) -> list[BenchmarkResult]:
    """Run ``torch_worker.py`` and return its rows.

    The child must not import TensorFlow. ``config['device']`` is ``cuda``
    for a GPU benchmark and ``cpu`` for the protocol test.
    """
    worker = Path(__file__).resolve().parent / "torch_worker.py"
    config_path = data_path.with_name("torch_job.json")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(worker), "--config", str(config_path), "--data", str(data_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit("PyTorch benchmark subprocess failed:\n" + detail)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"PyTorch benchmark subprocess returned invalid JSON: {exc}") from exc
    if payload.get("tensorflow_imported"):
        raise SystemExit("PyTorch benchmark subprocess imported TensorFlow")
    rows = [BenchmarkResult(**item) for item in payload["results"]]
    if config.get("device") == "cuda":
        missing = [row.model for row in rows if row.inference_latency_gpu_ms is None]
        if missing:
            raise SystemExit("PyTorch GPU latency missing for " + ", ".join(missing))
    return rows


def run_torch_subprocess(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    n_folds: int,
    hidden: int,
    latency_runs: int,
) -> list[BenchmarkResult]:
    """Train the PyTorch models in a child process on CUDA."""
    config = {
        "dataset": dataset,
        "epochs": epochs,
        "n_folds": n_folds,
        "hidden": hidden,
        "latency_runs": latency_runs,
        "device": "cuda",
    }
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "arrays.npz"
        np.savez(data_path, X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.float32))
        return launch_torch_worker(config, data_path)


def write_results_json(path: Path, results: list[BenchmarkResult], *, pytorch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(),
            "cpu": cpu_info(),
            "gpu": gpu_info(),
            "tf_version": tf.__version__,
        },
        "note": pytorch,
        "results": [asdict(row) for row in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Real-world tabular benchmark suite for KANX. "
            "PyTorch runs in a subprocess when --device cuda is selected."
        )
    )
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--datasets",
        choices=["all", "california", "concrete", "energy"],
        default="all",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "CPU smoke gate: California housing, 256 rows, 2 folds, 1 epoch. "
            "Does not overwrite the committed results JSON."
        ),
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Write JSON results here. Defaults to benchmarks/results/real_world_results.json except for --smoke.",
    )
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--latency-runs", type=int, default=None)
    args = ap.parse_args(argv)

    if args.smoke and args.device == "cuda":
        ap.error("--smoke always runs on CPU")
    if args.smoke:
        device = "cpu"
    elif args.device == "auto":
        device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"
    else:
        device = args.device

    epochs = args.epochs if args.epochs is not None else (1 if args.smoke else 200)
    n_folds = args.folds if args.folds is not None else (2 if args.smoke else 5)
    hidden = args.hidden if args.hidden is not None else (4 if args.smoke else 32)
    latency_runs = args.latency_runs if args.latency_runs is not None else (2 if args.smoke else 100)
    max_samples = args.max_samples if args.max_samples is not None else (256 if args.smoke else None)
    if args.smoke and args.datasets == "all":
        selected = ["california"]
    elif args.datasets == "all":
        selected = ["california", "concrete", "energy"]
    else:
        selected = [args.datasets]

    results: list[BenchmarkResult] = []
    for dataset_name in selected:
        print(f"Benchmarking {dataset_name} on device={device}...", flush=True)
        X, y, _feature_names = build_dataset(dataset_name)
        if max_samples is not None:
            X = X[:max_samples]
            y = y[:max_samples]
        results.append(
            evaluate_dataset(
                dataset_name,
                X,
                y,
                epochs=epochs,
                device=device,
                n_folds=n_folds,
                hidden=hidden,
                latency_runs=latency_runs,
            )
        )
        if device == "cuda":
            results.extend(
                run_torch_subprocess(
                    dataset_name,
                    X,
                    y,
                    epochs=epochs,
                    n_folds=n_folds,
                    hidden=hidden,
                    latency_runs=latency_runs,
                )
            )
    ensure_real_world_gate(results)

    if args.output is not None:
        out = Path(args.output)
    elif args.smoke:
        out = None
    else:
        out = RESULTS_PATH
    print("\n" + summary_markdown(results))
    if out is not None:
        if device == "cuda":
            pytorch = (
                "PyTorch KAN and MLP ran in a subprocess so TensorFlow and "
                "PyTorch did not initialize CUDA in one process."
            )
        else:
            pytorch = (
                "CPU run is TensorFlow-only. Pass --device cuda to run the "
                "PyTorch KAN and MLP in a subprocess."
            )
        write_results_json(out, results, pytorch=pytorch)
        print(f"\nResults saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
