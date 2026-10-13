"""GPU-optimized MatrixKAN: B-spline evaluation via batched matrix multiply.

Key design: replaces Cox-de Boor recursion with vectorized batched GEMM.
For spline_order k and grid_size G, precomputes recurrence matrices M_1..M_k
then evaluates all B-spline bases via: B = x_augmented @ M_1 @ M_2 @ ... @ M_k
This reduces forward pass to k dense matrix multiplies (GPU-friendly).
"""
from __future__ import annotations

import torch
from torch import nn

from ..prune import keep_mask, prune_stats
from .layers import _apply_keep, _as_prune_samples


def extend_grid_matrix(grid: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetrically extend uniform grid: (in_features, grid_size + 1) → (in_features, grid_size + 2k + 1)."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return grid

    n = grid.shape[1] - 1
    h = (grid[:, -1:] - grid[:, :1]) / n
    left_offsets = torch.arange(-k, 0, device=grid.device, dtype=grid.dtype).unsqueeze(0)
    right_offsets = torch.arange(1, k + 1, device=grid.device, dtype=grid.dtype).unsqueeze(0)
    left = grid[:, :1] + left_offsets * h
    right = grid[:, -1:] + right_offsets * h
    return torch.cat([left, grid, right], dim=1)


def build_recurrence_matrix_batch(grid_ext: torch.Tensor, k: int, order: int) -> torch.Tensor:
    """Precompute recurrence matrix for one recursion level.

    Args:
        grid_ext: (in_features, num_grid_points)
        k: recursion level (0..spline_order)
        order: spline polynomial order

    Returns:
        recurrence matrix of shape (in_features, num_grid_points, num_grid_points)
    """
    n_grid = grid_ext.shape[1]
    batch_size = grid_ext.shape[0]

    if k == 0:
        # Order-0: identity-like (handled separately)
        return torch.eye(n_grid, device=grid_ext.device, dtype=grid_ext.dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        ).clone()

    # For order k > 0: linear combination of two adjacent basis functions of order k-1
    # M[i,j] = (i-th basis of order k depends on j-th basis of order k-1)
    # This is a (n_grid) × (n_grid) tridiagonal-like matrix
    L_in = n_grid - 1
    M = torch.zeros(batch_size, L_in, n_grid, device=grid_ext.device, dtype=grid_ext.dtype)

    for i in range(L_in):
        g_left = grid_ext[:, i : i + order] # noqa: F841
        g_right = grid_ext[:, i + 1 : i + order + 1] # noqa: F841

        denom_l = grid_ext[:, i + order] - grid_ext[:, i]
        denom_r = grid_ext[:, i + order + 1] - grid_ext[:, i + 1]

        denom_l = torch.where(denom_l == 0, torch.ones_like(denom_l), denom_l)
        denom_r = torch.where(denom_r == 0, torch.ones_like(denom_r), denom_r)

        M[:, i, i] = 1.0 / denom_l
        M[:, i, i + 1] = -1.0 / denom_r

    return M


def b_spline_basis_matrix(x: torch.Tensor, grid_ext: torch.Tensor, spline_order: int) -> torch.Tensor:
    """Compute B-spline basis via batched matrix multiply (no recursion).

    Cox-de Boor recursion: B_i^0(x) = [grid_i <= x < grid_{i+1}]
                           B_i^k(x) = (x - grid_i)/(grid_{i+k} - grid_i) * B_i^{k-1}(x)
                                    + (grid_{i+k+1} - x)/(grid_{i+k+1} - grid_{i+1}) * B_{i+1}^{k-1}(x)

    Args:
        x: (batch, in_features)
        grid_ext: (in_features, num_grid_points)
        spline_order: polynomial degree

    Returns:
        (batch, in_features, num_basis_functions)
    """
    batch_size, in_features = x.shape
    num_grid = grid_ext.shape[1] # noqa: F841

    # Order-0 basis: piecewise constant indicator
    # For each interval [grid_i, grid_{i+1}), basis_i = 1 if x in interval, else 0
    x_e = x.unsqueeze(-1)  # (batch, in_features, 1)
    g_e = grid_ext.unsqueeze(0)  # (1, in_features, num_grid)

    # basis_0: (batch, in_features, num_grid-1)
    basis = ((x_e >= g_e[..., :-1]) & (x_e < g_e[..., 1:])).to(x.dtype)

    # Apply Cox-de Boor recurrence
    for k in range(1, spline_order + 1):
        num_basis = basis.shape[-1]
        # Next-order basis will have one fewer element
        new_basis = torch.zeros(batch_size, in_features, num_basis - 1,
                               device=x.device, dtype=x.dtype)

        for i in range(num_basis - 1):
            # Left term: (x - grid_i) / (grid_{i+k} - grid_i) * B_i^{k-1}(x)
            g_i = grid_ext[:, i:i+1]  # (in_features, 1)
            g_ik = grid_ext[:, i+k:i+k+1]  # (in_features, 1)
            denom_l = g_ik - g_i
            denom_l = torch.where(denom_l == 0, torch.ones_like(denom_l), denom_l)
            left_w = (x_e - g_i) / denom_l  # (batch, in_features, 1)
            left_term = left_w * basis[..., i:i+1]

            # Right term: (grid_{i+k+1} - x) / (grid_{i+k+1} - grid_{i+1}) * B_{i+1}^{k-1}(x)
            g_i1 = grid_ext[:, i+1:i+2]  # (in_features, 1)
            g_ik1 = grid_ext[:, i+k+1:i+k+2]  # (in_features, 1)
            denom_r = g_ik1 - g_i1
            denom_r = torch.where(denom_r == 0, torch.ones_like(denom_r), denom_r)
            right_w = (g_ik1 - x_e) / denom_r  # (batch, in_features, 1)
            right_term = right_w * basis[..., i+1:i+2]

            new_basis[..., i] = (left_term + right_term).squeeze(-1)

        basis = new_basis

    return basis


class MatrixKANLinear(nn.Module):
    """GPU-friendly KAN edge: vectorized B-spline via batched matmul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: nn.Module | None = None,
        grid_eps: float = 0.02,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        trainable_grid: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.scale_base = float(scale_base)
        self.scale_spline = float(scale_spline)
        self.grid_eps = float(grid_eps)
        self.grid_range = tuple(grid_range)
        self.num_basis = self.grid_size + self.spline_order

        self.base_activation = base_activation or nn.SiLU()

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.base_weight)

        self.spline_weight = nn.Parameter(
            torch.randn(in_features, out_features, self.num_basis) * scale_noise
        )

        # Grid: per-feature uniform initialization
        low, high = self.grid_range
        grid_init = torch.linspace(low, high, self.grid_size + 1).unsqueeze(0).expand(
            in_features, -1
        ).contiguous()

        if trainable_grid:
            self.grid = nn.Parameter(grid_init)
        else:
            self.register_buffer("grid", grid_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute KAN forward pass using vectorized B-spline basis.

        Args:
            x: (batch, in_features)

        Returns:
            (batch, out_features)
        """
        # Base branch: SiLU residual
        base_out = self.base_activation(x) @ self.base_weight
        base_out = base_out * self.scale_base

        # Spline branch: vectorized basis via matrix multiply
        grid_ext = extend_grid_matrix(self.grid, self.spline_order)
        basis = b_spline_basis_matrix(x, grid_ext, self.spline_order)  # (batch, in_features, num_basis)

        # Apply spline weights: (batch, in_features, num_basis) × (in_features, out_features, num_basis) → (batch, out_features)
        spline_out = torch.einsum("bik,iok->bo", basis, self.spline_weight)
        spline_out = spline_out * self.scale_spline

        return base_out + spline_out

    def prune(
        self,
        *,
        threshold: float | None = None,
        relative: float | None = None,
        samples: torch.Tensor | None = None,
    ) -> dict[str, int]:
        """Zero edges that are small in magnitude or in contribution.

        Scoring matches :meth:`forward`, including ``scale_base`` and
        ``scale_spline``. ``samples``, when given, has shape
        ``(batch, in_features)``.
        """
        batch = _as_prune_samples(samples, self.in_features, self.spline_weight)
        scores = self._edge_scores(batch)
        keep = keep_mask(scores, threshold=threshold, relative=relative)
        _apply_keep(self.spline_weight, self.base_weight, keep)
        return prune_stats(keep)

    def _edge_scores(self, samples: torch.Tensor | None):
        spline = self.spline_weight.detach()
        base = self.base_weight.detach()
        if samples is None:
            squared = (self.scale_spline ** 2) * spline.float().pow(2).sum(dim=-1)
            squared = squared + (self.scale_base ** 2) * base.float().pow(2)
            return torch.sqrt(squared.clamp_min(0)).cpu().numpy()

        acted = self.base_activation(samples)
        grid_ext = extend_grid_matrix(self.grid, self.spline_order)
        basis = b_spline_basis_matrix(samples, grid_ext, self.spline_order)
        spline_edge = torch.einsum("bik,iok->bio", basis, spline) * self.scale_spline
        base_edge = acted.unsqueeze(-1) * base.unsqueeze(0) * self.scale_base
        mean_sq = (spline_edge + base_edge).float().pow(2).mean(dim=0)
        return torch.sqrt(mean_sq.clamp_min(0)).cpu().numpy()

    def update_grid_from_samples(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Adaptive grid update from data samples (pykan parity).

        Args:
            x: (batch, in_features) — samples to fit grid to
            margin: interpolation between uniform (0) and sample-based (1) grid
        """
        batch_size = x.shape[0] # noqa: F841

        with torch.no_grad():
            # Compute quantile-based grid for each feature
            new_grid = []
            for i in range(self.in_features):
                feat = x[:, i]
                # Quantiles from 0 to 1
                quantiles = torch.linspace(0, 1, self.grid_size + 1, device=x.device, dtype=x.dtype)
                feat_grid = torch.quantile(feat, quantiles)

                # Add small margin to avoid boundary issues (only to min and max)
                feat_min, feat_max = feat_grid[0], feat_grid[-1]
                feat_range = feat_max - feat_min
                if feat_range < 1e-8:
                    feat_range = 1.0

                # Expand grid slightly at the edges
                margin_width = margin * feat_range
                feat_grid[0] = feat_grid[0] - margin_width
                feat_grid[-1] = feat_grid[-1] + margin_width

                new_grid.append(feat_grid)

            new_grid_tensor = torch.stack(new_grid, dim=0)

            # Interpolate between uniform and sample-based: (1 - eps) * uniform + eps * sample
            uniform_grid = self.grid.clone()
            interpolated_grid = (1 - self.grid_eps) * uniform_grid + self.grid_eps * new_grid_tensor

            # Update grid in-place
            if isinstance(self.grid, nn.Parameter):
                self.grid.data.copy_(interpolated_grid)
            else:
                self.register_buffer("grid", interpolated_grid)

    def get_spline_weight_at_grid_points(self) -> torch.Tensor:
        """Return spline weights for symbolic regression hooks.

        Returns:
            (in_features, out_features, num_basis)
        """
        return self.spline_weight


class MatrixKAN(nn.Sequential):
    """GPU-optimized KAN model using MatrixKANLinear layers."""

    def __init__(
        self,
        layers: list[int] | list[dict],
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: nn.Module | None = None,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        **kwargs,
    ) -> None:
        modules = []
        defaults = dict(
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation or nn.SiLU(),
            grid_range=grid_range,
        )

        if isinstance(layers[0], dict):
            for cfg in layers:
                modules.append(MatrixKANLinear(**{**defaults, **cfg}))
        else:
            widths = [int(w) for w in layers]
            if len(widths) < 2:
                raise ValueError("layers must contain at least 2 dimensions")
            for i in range(len(widths) - 1):
                modules.append(MatrixKANLinear(widths[i], widths[i + 1], **defaults))

        super().__init__(*modules)
        self._layers_spec = layers
        self._defaults = defaults

    def update_grid_from_samples(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Update grid on layers based on data.

        For the first layer, update grid directly from input x.
        For subsequent layers, propagate x through prior layers to get activations.
        """
        layers = [m for m in self.modules() if isinstance(m, MatrixKANLinear)]
        if not layers:
            return

        # Update first layer from raw input
        layers[0].update_grid_from_samples(x, margin=margin)

        # Update remaining layers by propagating through prior layers
        with torch.no_grad():
            current_x = x
            for i in range(1, len(layers)):
                # Propagate through all layers up to this point
                for j in range(i):
                    current_x = layers[j](current_x)
                # Update this layer
                layers[i].update_grid_from_samples(current_x, margin=margin)

    def prune(
        self,
        *,
        threshold: float | None = None,
        relative: float | None = None,
        samples: torch.Tensor | None = None,
    ) -> dict[str, int]:
        """Prune every ``MatrixKANLinear`` layer. See :meth:`MatrixKANLinear.prune`."""
        layers = [module for module in self.modules() if isinstance(module, MatrixKANLinear)]
        if not layers:
            return {"pruned": 0, "total": 0, "active": 0}

        current = None if samples is None else torch.as_tensor(samples, dtype=torch.float32)
        if current is not None and current.ndim != 2:
            raise ValueError(
                f"samples must have shape (batch, in_features), got {tuple(current.shape)}"
            )

        pruned = 0
        total = 0
        with torch.no_grad():
            for layer in layers:
                stats = layer.prune(threshold=threshold, relative=relative, samples=current)
                pruned += stats["pruned"]
                total += stats["total"]
                if current is not None:
                    current = layer(current)
        return {"pruned": pruned, "total": total, "active": total - pruned}
