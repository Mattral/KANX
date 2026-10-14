"""CPU smoke coverage for the synthetic and real-world benchmark gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import compare_mlp, real_world

ROOT = Path(__file__).resolve().parents[1]
COMPARE_BASELINE = ROOT / "benchmarks" / "results.md"
REAL_BASELINE = ROOT / "benchmarks" / "results" / "real_world_results.json"


def _tiny_dataset(name):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 3)).astype(np.float32)
    y = rng.normal(size=(24, 1)).astype(np.float32)
    return X, y, ["a", "b", "c"]


def _fake_tf_result(dataset_name, X, y, **kwargs):
    return real_world.BenchmarkResult(
        dataset=dataset_name,
        model="KAN_TF",
        rmse_mean=0.5,
        rmse_std=0.1,
        r2_mean=0.2,
        r2_std=0.0,
        train_time_s=0.01,
        params=12,
        inference_latency_cpu_ms=1.0,
        inference_latency_gpu_ms=None,
    )


def test_compare_gate_rejects_non_finite():
    bad = compare_mlp.Result(
        model="KAN",
        params=3,
        train_s=0.1,
        infer_ms_4k_cpu=1.0,
        infer_ms_4k_gpu=None,
        train_mse=float("nan"),
        test_mse=0.2,
    )
    with pytest.raises(SystemExit, match="not finite"):
        compare_mlp.ensure_compare_gate([bad])


def test_compare_smoke_keeps_published_table(tmp_path):
    before = COMPARE_BASELINE.read_bytes()
    out = tmp_path / "compare.md"
    assert compare_mlp.main(["--smoke", "--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "CPU smoke gate" in text
    assert "KAN[2,4,1]" in text
    assert "MLP[2,8,1]" in text
    assert COMPARE_BASELINE.read_bytes() == before


def test_compare_smoke_rejects_long_flag():
    with pytest.raises(SystemExit):
        compare_mlp.main(["--smoke", "--long"])


def test_real_world_gate_rejects_empty():
    with pytest.raises(SystemExit, match="no dataset results"):
        real_world.ensure_real_world_gate([])


def test_real_world_evaluate_dataset_is_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 3)).astype(np.float32)
    y = (X[:, :1] + 0.1).astype(np.float32)
    result = real_world.evaluate_dataset(
        "synthetic",
        X,
        y,
        epochs=1,
        device="cpu",
        n_folds=2,
        hidden=4,
        latency_runs=1,
    )
    real_world.ensure_real_world_gate([result])
    assert result.params > 0
    assert result.model == "KAN_TF"


def test_real_world_smoke_cli_does_not_clobber_baseline(tmp_path, monkeypatch):
    before = REAL_BASELINE.read_bytes()
    seen = {}

    def fake_dataset(name):
        seen["dataset"] = name
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 3)).astype(np.float32)
        y = rng.normal(size=(80, 1)).astype(np.float32)
        return X, y, ["a", "b", "c"]

    def fake_eval(dataset_name, X, y, **kwargs):
        seen["kwargs"] = kwargs
        seen["rows"] = len(X)
        return real_world.BenchmarkResult(
            dataset=dataset_name,
            model="KAN_TF",
            rmse_mean=0.5,
            rmse_std=0.1,
            r2_mean=0.2,
            r2_std=0.0,
            train_time_s=0.01,
            params=12,
            inference_latency_cpu_ms=1.0,
            inference_latency_gpu_ms=None,
        )

    monkeypatch.setattr(real_world, "build_dataset", fake_dataset)
    monkeypatch.setattr(real_world, "evaluate_dataset", fake_eval)
    out = tmp_path / "real_world.json"
    assert real_world.main(["--smoke", "--output", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["results"][0]["dataset"] == "california"
    assert payload["results"][0]["params"] == 12
    assert seen["dataset"] == "california"
    assert seen["rows"] == 80
    assert seen["kwargs"]["epochs"] == 1
    assert seen["kwargs"]["device"] == "cpu"
    assert seen["kwargs"]["n_folds"] == 2
    assert seen["kwargs"]["hidden"] == 4
    assert seen["kwargs"]["latency_runs"] == 2
    assert len(payload["results"]) == 1
    assert "TensorFlow-only" in payload["note"]
    assert REAL_BASELINE.read_bytes() == before


def test_cpu_benchmark_does_not_spawn_pytorch(tmp_path, monkeypatch):
    calls = []

    def explode(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("CPU benchmark must not start the PyTorch subprocess")

    monkeypatch.setattr(real_world, "launch_torch_worker", explode)
    monkeypatch.setattr(real_world, "build_dataset", _tiny_dataset)
    monkeypatch.setattr(real_world, "evaluate_dataset", _fake_tf_result)
    out = tmp_path / "cpu.json"
    assert real_world.main(
        ["--device", "cpu", "--datasets", "california", "--epochs", "1", "--output", str(out)]
    ) == 0
    assert calls == []
    models = [row["model"] for row in json.loads(out.read_text(encoding="utf-8"))["results"]]
    assert models == ["KAN_TF"]


def test_cuda_benchmark_spawns_pytorch_subprocess(tmp_path, monkeypatch):
    seen = {}

    def fake_launch(config, data_path):
        seen["config"] = config
        seen["exists"] = Path(data_path).is_file()
        return [real_world.BenchmarkResult(
            dataset=config["dataset"],
            model="KAN_TORCH",
            rmse_mean=0.4,
            rmse_std=0.0,
            r2_mean=0.1,
            r2_std=0.0,
            train_time_s=0.2,
            params=8,
            inference_latency_cpu_ms=1.0,
            inference_latency_gpu_ms=0.5,
        )]

    monkeypatch.setattr(real_world, "launch_torch_worker", fake_launch)
    monkeypatch.setattr(real_world, "build_dataset", _tiny_dataset)
    monkeypatch.setattr(real_world, "evaluate_dataset", _fake_tf_result)
    out = tmp_path / "cuda.json"
    assert real_world.main(
        [
            "--device",
            "cuda",
            "--datasets",
            "california",
            "--epochs",
            "1",
            "--folds",
            "2",
            "--hidden",
            "4",
            "--output",
            str(out),
        ]
    ) == 0
    assert seen["exists"] is True
    assert seen["config"]["device"] == "cuda"
    assert seen["config"]["hidden"] == 4
    models = [row["model"] for row in json.loads(out.read_text(encoding="utf-8"))["results"]]
    assert models == ["KAN_TF", "KAN_TORCH"]


def test_torch_worker_subprocess_does_not_import_tensorflow(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(36, 3)).astype(np.float32)
    y = (X[:, :1] * 0.5).astype(np.float32)
    data = tmp_path / "arrays.npz"
    np.savez(data, X=X, y=y)
    rows = real_world.launch_torch_worker(
        {
            "dataset": "synthetic",
            "epochs": 1,
            "n_folds": 2,
            "hidden": 4,
            "latency_runs": 1,
            "device": "cpu",
        },
        data,
    )
    assert [row.model for row in rows] == ["KAN_TORCH", "MLP_TORCH"]
    real_world.ensure_real_world_gate(rows)
    assert all(row.inference_latency_gpu_ms is None for row in rows)
