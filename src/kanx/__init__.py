"""kanx — Production-grade Kolmogorov-Arnold Networks in TensorFlow.

Public API:
    from kanx import KAN, KANLinear, load_config, train, predict

This package provides a clean, vectorized, fully-tested implementation of
Kolmogorov-Arnold Networks (Liu et al., 2024) on top of TensorFlow / Keras 3,
with first-class support for training, inference, checkpointing, configuration
files, REST serving (FastAPI), benchmarking and CI/CD.

Modules:
    layers      — `KANLinear` Keras layer and B-spline primitives.
    model       — `KAN` sequential model and high-level builders.
    train       — Training loop, callbacks, metrics, checkpointing.
    inference   — Inference helpers, batched predict, checkpoint loading.
    config      — YAML config schema and validators.
    utils       — Logging, seeding, IO helpers.
"""
from __future__ import annotations

from . import datasets
from .config import KanxConfig, load_config, validate_config
from .inference import load_model, predict, save_model
from .layers import KANLinear, b_spline_basis, extend_grid
from .model import KAN, build_kan
from .onnx_export import export_onnx_tf
from .quickstart import quickstart
from .train import build_loss, build_optimizer, train
from .utils import check_input_range, fit_grid_to_data, get_logger, set_global_seed

__all__ = [
    "KAN",
    "KANLinear",
    "build_kan",
    "b_spline_basis",
    "extend_grid",
    "load_config",
    "validate_config",
    "KanxConfig",
    "train",
    "build_optimizer",
    "build_loss",
    "predict",
    "load_model",
    "save_model",
    "set_global_seed",
    "get_logger",
    "fit_grid_to_data",
    "check_input_range",
    "export_onnx_tf",
    "quickstart",
    "datasets",
]

__version__ = "0.2.1"
