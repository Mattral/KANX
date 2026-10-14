"""Learned edge curves match the layer, and the viewers write files."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf
import torch

from kanx import KAN, KANLinear
from kanx.torch import KAN as TorchKAN
from kanx.torch import MatrixKAN, MatrixKANLinear
from kanx.torch.layers import KANLinear as TorchLinear
from kanx.viz import edge_contributions, edge_curves, kan_layers, plot_edges, write_edge_viewer


def _assert_reconstructs(layer, forward, inputs):
    contrib = edge_contributions(layer, inputs)
    assert contrib.shape == (inputs.shape[0], layer.in_features, layer.out_features)
    total = contrib.sum(axis=1)
    expected = np.asarray(forward(inputs), dtype=np.float64)
    assert np.isfinite(total).all()
    np.testing.assert_allclose(total, expected, rtol=1e-4, atol=1e-4)


def test_tf_edges_sum_to_the_layer_output():
    layer = KANLinear(2, 1, grid_size=3, spline_order=2)
    layer(tf.zeros((1, 2)))
    spline = np.zeros(layer.spline_weight.shape, dtype=np.float32)
    base = np.zeros(layer.base_weight.shape, dtype=np.float32)
    spline[0, 0, :] = 0.3
    base[0, 0] = 0.7
    spline[1, 0, :] = -0.2
    base[1, 0] = 0.1
    layer.spline_weight.assign(spline)
    layer.base_weight.assign(base)

    inputs = np.array([[0.2, -0.4], [0.0, 0.5], [-0.7, 0.7]], dtype=np.float32)
    _assert_reconstructs(layer, lambda x: layer(tf.constant(x)).numpy(), inputs)

    positions, curves = edge_curves(layer, n=32)
    assert positions.shape == (32,)
    assert curves.shape == (2, 1, 32)
    assert np.isfinite(curves).all()


def test_torch_and_matrix_edges_sum_to_the_layer_output():
    inputs = np.array([[0.25, -0.5], [-0.2, 0.4]], dtype=np.float32)

    torch_layer = TorchLinear(2, 1, grid_size=3, spline_order=2)
    with torch.no_grad():
        torch_layer.spline_weight.normal_(std=0.2)
        torch_layer.base_weight.normal_(std=0.4)
    _assert_reconstructs(
        torch_layer,
        lambda x: torch_layer(torch.as_tensor(x)).detach().numpy(),
        inputs,
    )

    matrix = MatrixKANLinear(
        2, 1, grid_size=3, spline_order=2, scale_base=0.5, scale_spline=2.0
    )
    with torch.no_grad():
        matrix.spline_weight.normal_(std=0.2)
        matrix.base_weight.normal_(std=0.4)
    _assert_reconstructs(
        matrix,
        lambda x: matrix(torch.as_tensor(x)).detach().numpy(),
        inputs,
    )


def test_zero_edge_is_flat_and_model_walker_keeps_order():
    model = KAN([2, 2, 1], grid_size=2, spline_order=1)
    model(tf.zeros((1, 2)))
    layers = kan_layers(model)
    assert len(layers) == 2
    assert layers[0].in_features == 2 and layers[1].out_features == 1

    layer = layers[0]
    layer.spline_weight.assign(tf.zeros_like(layer.spline_weight))
    layer.base_weight.assign(tf.zeros_like(layer.base_weight))
    _, curves = edge_curves(layer, n=16)
    np.testing.assert_allclose(curves, 0.0, atol=1e-6)


def test_viewers_write_a_figure_and_an_html_page(tmp_path):
    model = TorchKAN([2, 1], grid_size=2, spline_order=1)
    with torch.no_grad():
        layer = kan_layers(model)[0]
        layer.spline_weight.zero_()
        layer.base_weight.zero_()
        layer.base_weight[0, 0] = 1.0

    import matplotlib.pyplot as plt

    figure_path = tmp_path / "edges.png"
    fig = plot_edges(model, path=figure_path, n=24)
    assert figure_path.stat().st_size > 0
    plt.close(fig)

    page = write_edge_viewer(model, tmp_path / "edges.html", n=24)
    text = page.read_text(encoding="utf-8")
    assert "kanx-edge-data" in text
    payload = json.loads(text.split('type="application/json">', 1)[1].split("</script>", 1)[0])
    assert payload[0]["in_features"] == 2
    assert payload[0]["out_features"] == 1
    assert len(payload[0]["curves"][0][0]) == 24


def test_matrix_model_viewer_lists_every_layer(tmp_path):
    model = MatrixKAN([2, 2, 1], grid_size=2, spline_order=1)
    page = write_edge_viewer(model, tmp_path / "edges.html", n=8)
    text = page.read_text(encoding="utf-8")
    assert "Layer 0 (2 → 1)" not in text
    assert "Layer 0 (2 → 2)" in text
    assert "Layer 1 (2 → 1)" in text


def test_rejects_bad_inputs_and_unbuilt_layers():
    layer = KANLinear(2, 1, grid_size=2, spline_order=1)
    with pytest.raises(ValueError, match="built"):
        edge_curves(layer)

    layer(tf.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        edge_contributions(layer, np.array([[0.0, np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        edge_contributions(layer, np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="KANLinear"):
        edge_curves(object())
    with pytest.raises(IndexError):
        plot_edges(KAN([2, 1]), layer=3)
