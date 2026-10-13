"""Spline + custom layers for kanx.

This module implements:

* `extend_grid`     — pads a uniform grid by `spline_order` knots on each side
                      following the standard pykan convention so that the
                      number of B-spline basis functions of degree `k`
                      equals `grid_size + k`.
* `b_spline_basis`  — vectorized Cox-de Boor recursion that evaluates the
                      full basis matrix for a batch of inputs (no per-feature
                      Python loop, no per-sample loop).
* `KANLinear`       — drop-in `tf.keras.layers.Layer` implementing one KAN
                      edge bank: a SiLU-residual base + B-spline edge
                      activations (Liu et al., 2024, eq. 2.10).

Design notes
------------
* The grid is **per-input-feature** (shape `(in_features, grid_size + 1)`).
  Per-feature grids are required for adaptive grid updates and match the
  reference pykan implementation.
* `spline_order` is the polynomial *degree* (k = 3 → cubic B-splines).
* The base path applies `silu` to the input *before* multiplying by
  `base_weight`, matching pykan's `base_activation` design.
* The layer does **not** apply a non-linearity to its output: the B-spline
  edge function is itself a learned non-linearity.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Layer

from .prune import keep_mask, prune_stats


# ----------------------------------------------------------------------------
# B-spline primitives
# ----------------------------------------------------------------------------
def extend_grid(grid: tf.Tensor, k: int) -> tf.Tensor:
    """Symmetrically extend a uniform per-feature grid by `k` knots on each side.

    Args:
        grid: float tensor of shape ``(in_features, grid_size + 1)``.
        k:    spline polynomial degree (number of knots to add on each side).

    Returns:
        Tensor of shape ``(in_features, grid_size + 1 + 2k)``.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return grid

    dtype = grid.dtype
    n = tf.cast(tf.shape(grid)[1] - 1, dtype)
    # Uniform step size per feature (works for non-uniform grids too as long
    # as the boundary spacing is representative).
    h = (grid[:, -1:] - grid[:, :1]) / n

    left_offsets = tf.cast(tf.range(-k, 0), dtype)[None, :]
    right_offsets = tf.cast(tf.range(1, k + 1), dtype)[None, :]

    left = grid[:, :1] + left_offsets * h
    right = grid[:, -1:] + right_offsets * h
    return tf.concat([left, grid, right], axis=1)


def b_spline_basis(x: tf.Tensor, grid_ext: tf.Tensor, k: int) -> tf.Tensor:
    """Evaluate the B-spline basis of degree ``k`` for a batch of inputs.

    Uses the Cox-de Boor recursion in a fully vectorized fashion (no Python
    loop over features or batch).

    Args:
        x:        ``(batch, in_features)`` input tensor.
        grid_ext: ``(in_features, G_ext)`` already-extended grid.
        k:        spline degree (k=3 → cubic).

    Returns:
        ``(batch, in_features, G_ext - k - 1)`` basis tensor. For the standard
        case ``G_ext = grid_size + 1 + 2k`` this equals
        ``(batch, in_features, grid_size + k)`` — i.e. ``grid_size + k`` basis
        functions per feature, matching the pykan convention.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    x_e = tf.expand_dims(x, -1)            # (B, F, 1)
    g_e = tf.expand_dims(grid_ext, 0)      # (1, F, G_ext)

    # Order-0 basis (piecewise constant).
    basis = tf.cast(
        (x_e >= g_e[..., :-1]) & (x_e < g_e[..., 1:]),
        x.dtype,
    )                                       # (B, F, G_ext - 1)

    for p in range(1, k + 1):
        # After this iteration, basis length L_new = current_L - 1.
        L = basis.shape[-1] - 1
        g_l = g_e[..., :L]                  # g[i]
        g_lp = g_e[..., p : p + L]          # g[i+p]
        g_r = g_e[..., 1 : L + 1]           # g[i+1]
        g_rp = g_e[..., p + 1 : p + 1 + L]  # g[i+p+1]

        denom_l = g_lp - g_l
        denom_r = g_rp - g_r
        # Stable division: replace 0 denominators with 1 (numerator is also 0).
        denom_l = tf.where(tf.equal(denom_l, 0), tf.ones_like(denom_l), denom_l)
        denom_r = tf.where(tf.equal(denom_r, 0), tf.ones_like(denom_r), denom_r)

        left_w = (x_e - g_l) / denom_l
        right_w = (g_rp - x_e) / denom_r

        basis = left_w * basis[..., :L] + right_w * basis[..., 1 : L + 1]

    return basis


def _as_prune_samples(samples, in_features: int, dtype) -> tf.Tensor | None:
    """Validate an optional ``(batch, in_features)`` scoring batch."""
    if samples is None:
        return None
    batch = tf.convert_to_tensor(samples, dtype=dtype)
    if batch.shape.rank != 2 or int(batch.shape[-1]) != in_features:
        raise ValueError(
            f"samples must have shape (batch, {in_features}), got {tuple(batch.shape)}"
        )
    if int(batch.shape[0]) == 0:
        raise ValueError("samples must contain at least one row")
    return batch


# ----------------------------------------------------------------------------
# KAN edge bank
# ----------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="kanx")
class KANLinear(Layer):
    """A single KAN layer (edge bank) with SiLU residual + B-spline edges.

    Mathematically::

        out(x) = silu(x) @ W_base  +  Σ_{i,j} spline_w[i, o, j] · B_j(x_i)

    Args:
        in_features:           input dimension.
        out_features:          output dimension.
        grid_size:             number of grid intervals (default 5).
        spline_order:          spline polynomial degree (default 3).
        base_activation:       activation applied to the base path (default
                               ``"silu"``).
        grid_range:            ``(low, high)`` of the initial uniform grid.
        scale_noise:           std of noise used to initialize spline weights.
        scale_base:            scale of the base-path Glorot init.
        regularization_factor: L2 regularization strength (0 disables).
        trainable_grid:        whether the grid points are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        grid_range: tuple[float, float] = (-1.0, 1.0),
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        regularization_factor: float = 0.0,
        trainable_grid: bool = False,
        grid_eps: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")
        if grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        if spline_order < 1:
            raise ValueError("spline_order must be >= 1")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("grid_range must be (low, high) with low < high")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.base_activation_name = base_activation
        self.base_activation = tf.keras.activations.get(base_activation)
        self.grid_range = tuple(grid_range)
        self.scale_noise = float(scale_noise)
        self.scale_base = float(scale_base)
        self.regularization_factor = float(regularization_factor)
        self.trainable_grid = bool(trainable_grid)
        self.grid_eps = float(grid_eps)

        # Number of B-spline basis functions of degree `spline_order`.
        self.num_basis = self.grid_size + self.spline_order

    # ---- Keras lifecycle ----------------------------------------------------
    def build(self, input_shape):
        if input_shape[-1] is None or int(input_shape[-1]) != self.in_features:
            raise ValueError(
                f"Last dim of input ({input_shape[-1]}) must equal "
                f"in_features ({self.in_features})"
            )
        reg = (
            regularizers.l2(self.regularization_factor)
            if self.regularization_factor > 0
            else None
        )

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=initializers.GlorotUniform(),
            regularizer=reg,
            trainable=True,
        )
        # Small-noise init for spline weights so the layer starts close to
        # the base path (this is critical for stable training of KANs).
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.in_features, self.out_features, self.num_basis),
            initializer=initializers.RandomNormal(stddev=self.scale_noise),
            regularizer=reg,
            trainable=True,
        )

        # Per-feature uniform grid of `grid_size + 1` knots.
        low, high = self.grid_range
        grid_init = np.linspace(low, high, self.grid_size + 1, dtype=np.float32)
        grid_init = np.tile(grid_init[None, :], (self.in_features, 1))
        self.grid = self.add_weight(
            name="grid",
            shape=(self.in_features, self.grid_size + 1),
            initializer=initializers.Constant(grid_init),
            trainable=self.trainable_grid,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Base residual path: silu(x) @ W_base   →   (B, out_features)
        base_out = tf.matmul(self.base_activation(inputs), self.base_weight)

        # Spline path
        grid_ext = extend_grid(self.grid, self.spline_order)
        basis = b_spline_basis(inputs, grid_ext, self.spline_order)
        #   basis:        (B, F_in, num_basis)
        #   spline_weight:(F_in, F_out, num_basis)
        #   spline_out:   (B, F_out)
        spline_out = tf.einsum("bik,iok->bo", basis, self.spline_weight)
        return base_out + spline_out

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.out_features,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "out_features": self.out_features,
                "grid_size": self.grid_size,
                "spline_order": self.spline_order,
                "base_activation": self.base_activation_name,
                "grid_range": list(self.grid_range),
                "scale_noise": self.scale_noise,
                "scale_base": self.scale_base,
                "regularization_factor": self.regularization_factor,
                "trainable_grid": self.trainable_grid,
                "grid_eps": self.grid_eps,
            }
        )
        return config

    def prune(
        self,
        *,
        threshold: float | None = None,
        relative: float | None = None,
        samples: tf.Tensor | np.ndarray | None = None,
    ) -> dict[str, int]:
        """Zero edges that are small in magnitude or in contribution.

        An edge is one ``(input feature, output feature)`` pair: its spline
        coefficients and its base weight. ``threshold`` is an absolute cutoff
        on the edge score. ``relative`` drops edges at or below that fraction
        of the strongest edge in this layer. Pass either argument, or both
        (the higher cutoff wins).

        The score is the RMS of the edge function on ``samples`` when a batch
        of shape ``(batch, in_features)`` is given. Otherwise it is the L2
        norm of the edge parameters. The layer must already be built.
        """
        if not self.built:
            raise ValueError("KANLinear.prune() requires a built layer")
        batch = _as_prune_samples(samples, self.in_features, self.spline_weight.dtype)
        scores = self._edge_scores(batch)
        keep = keep_mask(scores, threshold=threshold, relative=relative)
        mask = tf.constant(keep)
        dtype = self.spline_weight.dtype
        self.spline_weight.assign(self.spline_weight * tf.cast(mask[..., None], dtype))
        self.base_weight.assign(self.base_weight * tf.cast(mask, self.base_weight.dtype))
        return prune_stats(keep)

    def _edge_scores(self, samples: tf.Tensor | None) -> np.ndarray:
        spline = tf.cast(self.spline_weight, tf.float64)
        base = tf.cast(self.base_weight, tf.float64)
        if samples is None:
            squared = tf.reduce_sum(tf.square(spline), axis=-1) + tf.square(base)
            return tf.sqrt(tf.maximum(squared, 0.0)).numpy()

        acted = tf.cast(self.base_activation(samples), tf.float64)
        grid_ext = extend_grid(tf.cast(self.grid, tf.float64), self.spline_order)
        basis = b_spline_basis(tf.cast(samples, tf.float64), grid_ext, self.spline_order)
        spline_edge = tf.einsum("bik,iok->bio", basis, spline)
        base_edge = acted[..., None] * base[None, :, :]
        mean_sq = tf.reduce_mean(tf.square(spline_edge + base_edge), axis=0)
        return tf.sqrt(tf.maximum(mean_sq, 0.0)).numpy()

    def update_grid_from_samples(self, x: tf.Tensor, margin: float = 0.01) -> None:
        """Adaptive grid update from data samples (pykan parity).

        Recomputes the per-feature grid using quantiles of the input data,
        then interpolates between the uniform grid and the sample-based grid
        using grid_eps as the interpolation parameter.

        Args:
            x: (batch, in_features) input tensor to fit grid to.
            margin: margin applied to grid boundaries (not used in current implementation).
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Last dimension of x ({x.shape[-1]}) must match "
                f"in_features ({self.in_features})"
            )

        # Compute quantile-based grid for each feature
        new_grids = []
        for i in range(self.in_features):
            feat = x[:, i]
            # Sort and compute quantile indices
            sorted_feat = tf.sort(feat)
            n = tf.cast(tf.shape(sorted_feat)[0], tf.float32)
            indices = tf.linspace(0.0, n - 1, self.grid_size + 1)
            indices = tf.cast(tf.round(indices), tf.int32)
            indices = tf.minimum(indices, tf.cast(tf.shape(sorted_feat)[0], tf.int32) - 1)

            feat_grid = tf.gather(sorted_feat, indices)
            new_grids.append(feat_grid)

        new_grid_tensor = tf.stack(new_grids, axis=0)

        # Interpolate between uniform and sample-based grid
        # grid_eps controls how much we favor the sample-based grid
        uniform_grid = self.grid
        interpolated_grid = (1.0 - self.grid_eps) * uniform_grid + self.grid_eps * new_grid_tensor

        # Update grid in-place
        self.grid.assign(interpolated_grid)
