from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "kanx" / "datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def _normalize_target(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    y = y.reshape(-1, 1)
    mean = y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, keepdims=True)
    if std.item() == 0.0:
        std = np.array([[1.0]], dtype=np.float32)
    return (y - mean) / std


def _load_or_cache(name: str, loader):
    path = CACHE_DIR / f"{name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return data["X"], data["y"], list(data["feature_names"])
    X, y, feature_names = loader()
    np.savez_compressed(path, X=X, y=y, feature_names=np.array(feature_names, dtype=object))
    return X, y, feature_names


def _download_excel(url: str, engine: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to load UCI Excel datasets. Install kanx[dev] or pandas+openpyxl/xlrd") from exc

    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required to download UCI datasets. Install kanx[dev] or requests") from exc

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_excel(BytesIO(resp.content), engine=engine)


def load_california_housing():
    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError as exc:
        raise ImportError("scikit-learn is required to load California Housing. Install kanx[dev] or scikit-learn") from exc

    # as_frame=False keeps this loader on NumPy. as_frame=True pulls in pandas,
    # which the benchmark smoke gate does not install.
    data = fetch_california_housing(as_frame=False)
    X = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.float32).reshape(-1, 1)
    X = _normalize(X)
    y = _normalize_target(y)
    feature_names = list(data.feature_names)
    return X, y, feature_names


def load_concrete_strength():
    def loader():
        df = _download_excel(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            engine="xlrd",
        )
        values = df.to_numpy(dtype=np.float32)
        X = values[:, :-1]
        y = values[:, -1]
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y, feature_names

    X, y, feature_names = _load_or_cache("concrete_strength", loader)
    X = _normalize(X)
    y = _normalize_target(y)
    return X, y, feature_names


def load_energy_efficiency():
    def loader():
        df = _download_excel(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            engine="openpyxl",
        )
        values = df.to_numpy(dtype=np.float32)
        X = values[:, :-2]
        y = values[:, -2]
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y, feature_names

    X, y, feature_names = _load_or_cache("energy_efficiency", loader)
    X = _normalize(X)
    y = _normalize_target(y)
    return X, y, feature_names
