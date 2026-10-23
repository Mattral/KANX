"""Utility helpers: logging, RNG seeding, path management."""
from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np


def get_logger(name: str = "kanx", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger. Idempotent across calls."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Grid-range helpers — critical for correct B-spline behaviour
# ---------------------------------------------------------------------------
def fit_grid_to_data(model, X, pad: float = 0.05) -> tuple[float, float]:
    """Reset every `KANLinear`'s grid so it covers the observed data range.

    **Why this matters.** The stored grid is an interval (default ``[-1, 1]``).
    ``extend_grid`` adds ``spline_order`` knots past each end, and the basis
    is exactly zero only at and beyond that outermost knot. Outside the stored
    interval the basis is no longer a partition of unity, so a model left on
    the default grid is not the function you meant to fit. Nothing raises.

    Call this BEFORE training when you know your data range::

        from kanx import KAN
        from kanx.utils import fit_grid_to_data
        model = KAN([2, 64, 1])
        fit_grid_to_data(model, X_train)
        model.fit(X_train, y_train, epochs=30)

    Args:
        model:  a `kanx.KAN` (TF) or `kanx.torch.KAN`. Must already be built.
        X:      training inputs as np.ndarray / tf.Tensor / torch.Tensor /
                list. Shape ``(N, in_features)``.
        pad:    fractional padding added to each side of the observed range.

    Returns:
        Tuple ``(low, high)`` of the global low/high actually applied.
    """
    import numpy as np
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"X must be rank-2; got shape {arr.shape}")
    low = float(arr.min())
    high = float(arr.max())
    if high <= low:
        raise ValueError(f"Degenerate data range [{low}, {high}]")
    span = high - low
    low -= pad * span
    high += pad * span

    n_updated = 0
    try:
        from kanx.layers import KANLinear as TFKANLinear
        for layer in getattr(model, "layers", []):
            if isinstance(layer, TFKANLinear):
                in_f = layer.in_features
                gs = layer.grid_size
                new_grid = np.linspace(low, high, gs + 1, dtype=np.float32)
                new_grid = np.tile(new_grid[None, :], (in_f, 1))
                layer.grid.assign(new_grid)
                layer.grid_range = (low, high)
                n_updated += 1
    except Exception:
        pass

    try:
        import torch

        from kanx.torch.layers import KANLinear as TorchKANLinear
        for sub in model.modules() if hasattr(model, "modules") else []:
            if isinstance(sub, TorchKANLinear):
                in_f = sub.in_features
                gs = sub.grid_size
                new_grid = torch.linspace(low, high, gs + 1).unsqueeze(0).expand(
                    in_f, -1
                ).contiguous()
                if isinstance(sub.grid, torch.nn.Parameter):
                    with torch.no_grad():
                        sub.grid.copy_(new_grid)
                else:
                    sub.grid = new_grid
                sub.grid_range = (low, high)
                n_updated += 1
    except Exception:
        pass

    if n_updated == 0:
        raise ValueError(
            "No KANLinear layers found on the model — was it built? "
            "Pass an input through it first or call `model.build(...)`."
        )
    get_logger("kanx.utils").info(
        "fit_grid_to_data: updated %d KANLinear grid(s) to [%.4f, %.4f]",
        n_updated, low, high,
    )
    return (low, high)


def check_input_range(model, X, *, name: str = "inputs") -> None:
    """Emit a logged warning if `X` falls outside any KANLinear's grid range.

    This is the single most common silent failure mode of KANs in production.
    Call it before serving inference if you don't fully trust the input
    distribution.
    """
    import numpy as np
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        return  # caller's problem
    obs_low, obs_high = float(arr.min()), float(arr.max())

    grid_lows: list[float] = []
    grid_highs: list[float] = []
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "grid_range"):
            lo, hi = layer.grid_range
            grid_lows.append(float(lo))
            grid_highs.append(float(hi))
    if hasattr(model, "modules"):
        for sub in model.modules():
            if hasattr(sub, "grid_range") and sub is not model:
                lo, hi = sub.grid_range
                grid_lows.append(float(lo))
                grid_highs.append(float(hi))

    if not grid_lows:
        return
    grid_low = min(grid_lows)
    grid_high = max(grid_highs)
    if obs_low < grid_low or obs_high > grid_high:
        get_logger("kanx.utils").warning(
            "%s range [%.4f, %.4f] exceeds model grid range [%.4f, %.4f]. "
            "The basis is exactly zero only past the outermost extended knot; "
            "outside the stored grid it is no longer the interval you fit. "
            "Fix with `kanx.utils.fit_grid_to_data(model, X)` before training.",
            name, obs_low, obs_high, grid_low, grid_high,
        )


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow RNGs for deterministic-ish runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # local import to avoid TF import on package init
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_checkpoint_path(
    ckpt_dir: str, filename: str, root: str | None = None
) -> str:
    root = root or os.getcwd()
    return os.path.join(root, ckpt_dir, filename) if not os.path.isabs(ckpt_dir) \
        else os.path.join(ckpt_dir, filename)
