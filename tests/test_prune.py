"""Edge pruning for the TensorFlow and PyTorch KAN layers."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
import torch

from kanx import KAN, KANLinear
from kanx.prune import keep_mask
from kanx.torch import KAN as TorchKAN
from kanx.torch import MatrixKAN, MatrixKANLinear
from kanx.torch.layers import KANLinear as TorchLinear


def _plant_tf(layer, large=0.5, tiny=1e-8):
    """One strong edge at (0, 0) and one negligible edge at (1, 0)."""
    spline = np.zeros(layer.spline_weight.shape, dtype=np.float32)
    base = np.zeros(layer.base_weight.shape, dtype=np.float32)
    spline[0, 0, :] = large
    base[0, 0] = large
    if spline.shape[0] > 1:
        spline[1, 0, :] = tiny
        base[1, 0] = tiny
    layer.spline_weight.assign(spline)
    layer.base_weight.assign(base)


def _plant_torch(layer, large=0.5, tiny=1e-8):
    with torch.no_grad():
        layer.spline_weight.zero_()
        layer.base_weight.zero_()
        layer.spline_weight[0, 0, :] = large
        layer.base_weight[0, 0] = large
        if layer.spline_weight.shape[0] > 1:
            layer.spline_weight[1, 0, :] = tiny
            layer.base_weight[1, 0] = tiny


def _nonzero_tf(layer) -> int:
    spline = int(np.count_nonzero(layer.spline_weight.numpy()))
    base = int(np.count_nonzero(layer.base_weight.numpy()))
    return spline + base


def _nonzero_torch(layer) -> int:
    return int(torch.count_nonzero(layer.spline_weight)) + int(torch.count_nonzero(layer.base_weight))


def test_keep_mask_uses_the_higher_cutoff():
    scores = np.array([[1.0, 0.2], [0.01, 0.0]])
    keep = keep_mask(scores, threshold=0.05, relative=0.5)
    assert keep.tolist() == [[True, False], [False, False]]


def test_keep_mask_rejects_bad_inputs():
    with pytest.raises(ValueError, match="threshold"):
        keep_mask(np.ones((1, 1)), threshold=-0.1)
    with pytest.raises(ValueError, match="relative"):
        keep_mask(np.ones((1, 1)), relative=1.5)
    with pytest.raises(ValueError, match="needs"):
        keep_mask(np.ones((1, 1)))
    with pytest.raises(ValueError, match="finite"):
        keep_mask(np.array([[np.nan]]), threshold=0.1)


def test_keep_mask_all_zero_scores_stay_finite():
    keep = keep_mask(np.zeros((2, 2)), relative=0.5)
    assert keep.shape == (2, 2)
    assert not keep.any()


def test_tf_prune_reduces_parameters_and_preserves_output():
    layer = KANLinear(2, 1, grid_size=2, spline_order=1)
    x = tf.constant([[0.2, -0.3], [0.7, 0.1], [-0.4, 0.5]], dtype=tf.float32)
    layer(x)
    _plant_tf(layer)
    before = _nonzero_tf(layer)
    y0 = layer(x).numpy()

    stats = layer.prune(threshold=1e-4)
    y1 = layer(x).numpy()

    assert stats == {"pruned": 1, "total": 2, "active": 1}
    assert _nonzero_tf(layer) < before
    assert np.allclose(layer.spline_weight.numpy()[1, 0], 0.0)
    assert np.allclose(layer.spline_weight.numpy()[0, 0], 0.5)
    assert np.allclose(y0, y1, rtol=1e-4, atol=1e-5)
    assert np.isfinite(y1).all()


def test_tf_prune_relative_on_samples():
    layer = KANLinear(2, 1, grid_size=2, spline_order=1)
    x = tf.constant([[0.2, -0.3], [0.7, 0.1]], dtype=tf.float32)
    layer(x)
    _plant_tf(layer, large=1.0, tiny=1e-6)
    y0 = layer(x).numpy()
    stats = layer.prune(relative=0.2, samples=x)
    assert stats["active"] == 1
    assert np.allclose(layer.base_weight.numpy()[1, 0], 0.0)
    assert np.allclose(layer.spline_weight.numpy()[0, 0], 1.0)
    assert np.allclose(layer(x).numpy(), y0, rtol=1e-4, atol=1e-5)


def test_tf_model_prune_visits_every_layer():
    model = KAN([2, 2, 1], grid_size=2, spline_order=1)
    x = tf.constant([[0.2, -0.4], [0.5, 0.3]], dtype=tf.float32)
    model(x)
    for layer in model.layers:
        _plant_tf(layer)
    before = sum(_nonzero_tf(layer) for layer in model.layers)
    y0 = model(x).numpy()
    stats = model.prune(threshold=1e-4, samples=x)
    assert stats["active"] == 2
    assert sum(_nonzero_tf(layer) for layer in model.layers) < before
    assert np.allclose(model(x).numpy(), y0, rtol=1e-4, atol=1e-5)
    assert np.isfinite(model(x).numpy()).all()


def test_tf_prune_requires_a_built_layer():
    with pytest.raises(ValueError, match="built"):
        KANLinear(2, 1).prune(threshold=0.1)


def test_torch_prune_reduces_parameters_and_preserves_output():
    layer = TorchLinear(2, 1, grid_size=2, spline_order=1)
    x = torch.tensor([[0.2, -0.3], [0.7, 0.1], [-0.4, 0.5]])
    _plant_torch(layer)
    before = _nonzero_torch(layer)
    with torch.no_grad():
        y0 = layer(x).clone()
        stats = layer.prune(threshold=1e-4)
        y1 = layer(x)
    assert stats == {"pruned": 1, "total": 2, "active": 1}
    assert _nonzero_torch(layer) < before
    assert torch.allclose(layer.spline_weight[0, 0], torch.full_like(layer.spline_weight[0, 0], 0.5))
    assert torch.count_nonzero(layer.spline_weight[1, 0]) == 0
    assert torch.allclose(y0, y1, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(y1).all()


def test_torch_model_prune_relative():
    model = TorchKAN([2, 2, 1], grid_size=2, spline_order=1)
    x = torch.tensor([[0.2, -0.4], [0.5, 0.3]])
    for layer in model:
        _plant_torch(layer, large=1.0, tiny=1e-6)
    before = sum(_nonzero_torch(layer) for layer in model)
    with torch.no_grad():
        y0 = model(x).clone()
        stats = model.prune(relative=0.2, samples=x)
        y1 = model(x)
    assert stats["active"] == 2
    assert sum(_nonzero_torch(layer) for layer in model) < before
    assert torch.allclose(y0, y1, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(y1).all()


def test_matrix_kan_prune_matches_the_other_backends():
    layer = MatrixKANLinear(2, 1, grid_size=2, spline_order=1)
    x = torch.tensor([[0.2, -0.3], [0.7, 0.1]])
    _plant_torch(layer)
    before = _nonzero_torch(layer)
    with torch.no_grad():
        y0 = layer(x).clone()
        stats = layer.prune(threshold=1e-4, samples=x)
        y1 = layer(x)
    assert stats == {"pruned": 1, "total": 2, "active": 1}
    assert _nonzero_torch(layer) < before
    assert torch.allclose(y0, y1, rtol=1e-4, atol=1e-5)

    model = MatrixKAN([2, 2, 1], grid_size=2, spline_order=1)
    for child in model:
        _plant_torch(child)
    summary = model.prune(threshold=1e-4)
    assert summary["active"] == 2
    assert summary["pruned"] == summary["total"] - summary["active"]
