"""Mixed precision and XLA on the TensorFlow spline path."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from kanx import KAN, KANLinear
from kanx.layers import normalize_precision


def _reference_layer():
    layer = KANLinear(3, 2, grid_size=4, spline_order=2)
    x = tf.constant(np.linspace(-0.8, 0.8, 24, dtype=np.float32).reshape(8, 3))
    layer(x)
    return layer, x


def _copy(layer, **kwargs):
    other = KANLinear(
        layer.in_features,
        layer.out_features,
        grid_size=layer.grid_size,
        spline_order=layer.spline_order,
        **kwargs,
    )
    probe = tf.zeros((1, layer.in_features), dtype=tf.float32)
    other(probe)
    other.set_weights(layer.get_weights())
    return other


def test_normalize_precision_aliases():
    assert normalize_precision("BF16") == "bfloat16"
    assert normalize_precision("mixed_float16") == "float16"
    with pytest.raises(ValueError, match="precision"):
        normalize_precision("float64")


def test_reduced_precision_matches_float32_forward():
    ref, x = _reference_layer()
    expected = ref(x).numpy()
    tolerances = {
        "float16": (1e-2, 2e-3),
        "bfloat16": (5e-2, 5e-3),
    }
    for precision, (rtol, atol) in tolerances.items():
        other = _copy(ref, precision=precision)
        got = other(x).numpy()
        assert got.dtype == np.float32
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


def test_xla_float32_matches_eager_and_accepts_another_batch():
    ref, x = _reference_layer()
    compiled = _copy(ref, jit_compile=True)
    np.testing.assert_allclose(compiled(x).numpy(), ref(x).numpy(), rtol=1e-5, atol=1e-5)
    other = tf.random.uniform((3, 3), -0.5, 0.5, seed=1)
    eager = ref(other).numpy()
    np.testing.assert_allclose(compiled(other).numpy(), eager, rtol=1e-5, atol=1e-5)


def test_xla_bfloat16_is_finite_and_close():
    ref, x = _reference_layer()
    compiled = _copy(ref, precision="bfloat16", jit_compile=True)
    got = compiled(x).numpy()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, ref(x).numpy(), rtol=5e-2, atol=1e-2)


def test_reduced_precision_gradients_are_finite():
    layer = KANLinear(2, 1, grid_size=3, spline_order=2, precision="float16", jit_compile=True)
    x = tf.random.uniform((16, 2), -0.7, 0.7, seed=2)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(layer(x)))
    grads = tape.gradient(loss, layer.trainable_weights)
    assert grads
    for grad in grads:
        assert grad is not None
        assert tf.reduce_all(tf.math.is_finite(grad))


def test_kan_forwards_precision_and_jit():
    model = KAN([2, 4, 1], grid_size=3, spline_order=1, precision="bfloat16", jit_compile=True)
    assert model.get_config()["precision"] == "bfloat16"
    assert model.get_config()["jit_compile"] is True
    y = model(tf.random.uniform((5, 2), seed=3))
    assert y.shape == (5, 1)
    assert y.dtype == tf.float32
    assert tf.reduce_all(tf.math.is_finite(y))
    rebuilt = KANLinear.from_config(model.layers[0].get_config())
    assert rebuilt.precision == "bfloat16"
    assert rebuilt.jit_compile is True
