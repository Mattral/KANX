"""Plot the learned 1-D functions on KAN edges.

Each edge is one ``(input feature, output feature)`` pair. Its function is
the same univariate map the layer adds into that output:

    phi(x) = scale_base * act(x) * base_weight
           + scale_spline * B(x) · spline_weight

TensorFlow ``KANLinear`` and PyTorch ``KANLinear`` do not scale the two
paths, so both scales are 1. ``MatrixKANLinear`` uses ``scale_base`` and
``scale_spline``, matching its forward pass.

``edge_curves`` and ``write_edge_viewer`` need only NumPy and the backend
that built the layer. ``plot_edges`` needs matplotlib (``pip install
"kanx[viz]"``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def kan_layers(model: Any) -> list[Any]:
    """Return the KAN edge banks in ``model``, or ``[model]`` if it is one."""
    try:
        _kind(model)
    except TypeError:
        pass
    else:
        return [model]

    if hasattr(model, "layers"):
        candidates = list(model.layers)
    elif hasattr(model, "children"):
        candidates = list(model.children())
    else:
        candidates = []

    found = []
    for layer in candidates:
        try:
            _kind(layer)
        except TypeError:
            continue
        found.append(layer)
    if not found:
        raise TypeError("model has no KANLinear or MatrixKANLinear layers")
    return found


def edge_contributions(layer: Any, inputs: np.ndarray) -> np.ndarray:
    """Per-edge values of ``layer`` on a batch.

    Args:
        layer: a built TensorFlow ``KANLinear``, PyTorch ``KANLinear``, or
            ``MatrixKANLinear``.
        inputs: array of shape ``(batch, in_features)``.

    Returns:
        Array of shape ``(batch, in_features, out_features)``. Summing over
        the input axis reproduces the layer output.
    """
    _kind(layer)
    _require_built(layer)
    batch = _as_batch(inputs, layer.in_features)
    basis, acted = _basis_and_activation(layer, batch)
    spline = _to_numpy(layer.spline_weight).astype(np.float64, copy=False)
    base = _to_numpy(layer.base_weight).astype(np.float64, copy=False)
    scale_base, scale_spline = _scales(layer)
    spline_edge = np.einsum("bfk,fok->bfo", basis, spline) * scale_spline
    base_edge = acted[:, :, None] * base[None, :, :] * scale_base
    return spline_edge + base_edge


def edge_curves(
    layer: Any,
    x: np.ndarray | None = None,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample every edge of ``layer`` on one shared 1-D grid.

    Args:
        layer: a built KAN edge bank. See :func:`edge_contributions`.
        x: optional 1-D sample positions. The default is a linspace across
            the layer's grid, padded by 2% on each side.
        n: number of samples when ``x`` is omitted.

    Returns:
        ``(x, curves)`` where ``x`` has shape ``(n,)`` and ``curves`` has
        shape ``(in_features, out_features, n)``. ``curves[i, j]`` is the
        learned function on the edge from input ``i`` to output ``j``.
    """
    _kind(layer)
    _require_built(layer)
    positions = _sample_positions(layer, x, n)
    sweep = np.repeat(positions[:, None], layer.in_features, axis=1)
    contrib = edge_contributions(layer, sweep)
    curves = np.transpose(contrib, (1, 2, 0))
    return positions, curves


def plot_edges(
    model: Any,
    *,
    layer: int = 0,
    x: np.ndarray | None = None,
    n: int = 200,
    path: str | Path | None = None,
):
    """Draw one panel per edge of a single layer.

    ``model`` may be a KAN model or one edge-bank layer. ``layer`` selects
    which bank to draw when ``model`` has several. Returns the matplotlib
    figure. When ``path`` is set, the figure is also written there.

    The layer index is checked before matplotlib is imported, so a bad index
    is reported even when the ``viz`` extra is not installed.
    """
    layers = kan_layers(model)
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"layer {layer} is outside 0..{len(layers) - 1}")
    target = layers[layer]
    positions, curves = edge_curves(target, x=x, n=n)
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            'plot_edges needs matplotlib. Install it with: pip install "kanx[viz]"'
        ) from err
    rows = int(target.out_features)
    cols = int(target.in_features)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 2.6 * rows),
        squeeze=False,
    )
    for i in range(cols):
        for j in range(rows):
            ax = axes[j][i]
            ax.plot(positions, curves[i, j], color="#7C3AED", linewidth=2)
            ax.axhline(0, color="#9CA3AF", linewidth=0.6)
            ax.set_title(f"x_{i} → y_{j}", fontsize=10)
            ax.grid(alpha=0.3)
    fig.suptitle(f"Layer {layer} edge functions", fontsize=12)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def write_edge_viewer(
    model: Any,
    path: str | Path,
    *,
    n: int = 200,
    x: np.ndarray | None = None,
) -> Path:
    """Write a self-contained HTML page that plots every layer's edges.

    The page has a layer picker and a hover readout. It does not load
    anything from the network.
    """
    layers = kan_layers(model)
    payload = []
    for index, layer in enumerate(layers):
        positions, curves = edge_curves(layer, x=x, n=n)
        payload.append(
            {
                "name": f"Layer {index} ({layer.in_features} → {layer.out_features})",
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "x": positions.tolist(),
                "curves": curves.tolist(),
            }
        )
    destination = Path(path)
    destination.write_text(_viewer_html(payload), encoding="utf-8")
    return destination


def _kind(layer: Any) -> str:
    name = type(layer).__name__
    module = type(layer).__module__
    if name == "MatrixKANLinear":
        return "matrix"
    if name == "KANLinear" and module.startswith("kanx.torch"):
        return "torch"
    if name == "KANLinear":
        return "tf"
    raise TypeError(f"expected a kanx KANLinear or MatrixKANLinear, got {type(layer).__name__}")


def _require_built(layer: Any) -> None:
    try:
        _ = layer.spline_weight
        _ = layer.base_weight
        _ = layer.grid
    except (AttributeError, ValueError) as err:
        raise ValueError("the layer must be built before its edges can be read") from err


def _scales(layer: Any) -> tuple[float, float]:
    if _kind(layer) == "matrix":
        return float(layer.scale_base), float(layer.scale_spline)
    return 1.0, 1.0


def _as_batch(inputs: np.ndarray, in_features: int) -> np.ndarray:
    batch = np.asarray(inputs, dtype=np.float64)
    if batch.ndim != 2 or batch.shape[1] != in_features:
        raise ValueError(
            f"inputs must have shape (batch, {in_features}), got {tuple(batch.shape)}"
        )
    if batch.shape[0] == 0:
        raise ValueError("inputs must contain at least one row")
    if not np.isfinite(batch).all():
        raise ValueError("inputs must be finite")
    return batch


def _sample_positions(layer: Any, x: np.ndarray | None, n: int) -> np.ndarray:
    if x is None:
        if n < 2:
            raise ValueError("n must be at least 2")
        grid = _to_numpy(layer.grid)
        low = float(np.min(grid[:, 0]))
        high = float(np.max(grid[:, -1]))
        span = high - low
        if span <= 0:
            low, high = low - 1.0, high + 1.0
            span = high - low
        pad = 0.02 * span
        return np.linspace(low - pad, high + pad, int(n), dtype=np.float64)
    positions = np.asarray(x, dtype=np.float64).reshape(-1)
    if positions.size < 2:
        raise ValueError("x must contain at least two positions")
    if not np.isfinite(positions).all():
        raise ValueError("x must be finite")
    return positions


def _basis_and_activation(layer: Any, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kind = _kind(layer)
    if kind == "tf":
        import tensorflow as tf

        from kanx.layers import b_spline_basis, extend_grid

        dtype = layer.spline_weight.dtype
        values = tf.constant(inputs, dtype=dtype)
        grid_ext = extend_grid(tf.cast(layer.grid, dtype), layer.spline_order)
        basis = b_spline_basis(values, grid_ext, layer.spline_order)
        acted = layer.base_activation(values)
        return _to_numpy(basis), _to_numpy(acted)

    import torch

    if kind == "matrix":
        from kanx.torch.matrix_kan import b_spline_basis_matrix, extend_grid_matrix

        basis_fn = b_spline_basis_matrix
        extend = extend_grid_matrix
    else:
        from kanx.torch.layers import b_spline_basis, extend_grid

        basis_fn = b_spline_basis
        extend = extend_grid

    weight = layer.spline_weight
    values = torch.as_tensor(inputs, dtype=weight.dtype, device=weight.device)
    with torch.no_grad():
        grid_ext = extend(layer.grid, layer.spline_order)
        basis = basis_fn(values, grid_ext, layer.spline_order)
        acted = layer.base_activation(values)
    return _to_numpy(basis), _to_numpy(acted)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _viewer_html(payload: list[dict[str, Any]]) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KANX edge functions</title>
<style>
  body {{ font-family: Inter, Segoe UI, sans-serif; margin: 24px; color: #111827; background: #f8f7fb; }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }}
  p {{ margin: 0 0 16px; color: #4b5563; }}
  select {{ font: inherit; padding: 6px 8px; margin-bottom: 16px; }}
  #grid {{ display: grid; gap: 12px; }}
  .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; }}
  .card h2 {{ font-size: 13px; margin: 0 0 4px; font-weight: 600; }}
  canvas {{ width: 220px; height: 140px; display: block; }}
  .readout {{ font-family: ui-monospace, monospace; font-size: 12px; color: #6b7280; min-height: 1.2em; }}
</style>
</head>
<body>
<h1>KANX edge functions</h1>
<p>Each panel is one learned 1-D edge. Hover a curve to read its value.</p>
<label for="layer">Layer</label>
<select id="layer"></select>
<div id="grid"></div>
<script id="kanx-edge-data" type="application/json">{data}</script>
<script>
const payload = JSON.parse(document.getElementById("kanx-edge-data").textContent);
const select = document.getElementById("layer");
const grid = document.getElementById("grid");
payload.forEach((layer, index) => {{
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = layer.name;
  select.appendChild(option);
}});

function draw(index) {{
  const layer = payload[index];
  grid.innerHTML = "";
  grid.style.gridTemplateColumns = "repeat(" + layer.in_features + ", 240px)";
  const xs = layer.x;
  for (let j = 0; j < layer.out_features; j++) {{
    for (let i = 0; i < layer.in_features; i++) {{
      const ys = layer.curves[i][j];
      const card = document.createElement("div");
      card.className = "card";
      const title = document.createElement("h2");
      title.textContent = "x_" + i + " → y_" + j;
      const canvas = document.createElement("canvas");
      canvas.width = 440;
      canvas.height = 280;
      const readout = document.createElement("div");
      readout.className = "readout";
      readout.textContent = "hover the curve";
      card.append(title, canvas, readout);
      grid.appendChild(card);
      const ctx = canvas.getContext("2d");
      let lo = Math.min.apply(null, ys);
      let hi = Math.max.apply(null, ys);
      if (hi - lo < 1e-9) {{ lo -= 1; hi += 1; }}
      const pad = 0.08 * (hi - lo);
      lo -= pad; hi += pad;
      const x0 = 28, y0 = 12, w = canvas.width - 40, h = canvas.height - 28;
      function px(t) {{ return x0 + (t / (xs.length - 1)) * w; }}
      function py(v) {{ return y0 + (hi - v) / (hi - lo) * h; }}
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#e5e7eb";
      ctx.beginPath();
      const zero = py(0);
      ctx.moveTo(x0, zero); ctx.lineTo(x0 + w, zero); ctx.stroke();
      ctx.strokeStyle = "#7C3AED";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ys.forEach((v, t) => {{
        const X = px(t), Y = py(v);
        if (t === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }});
      ctx.stroke();
      canvas.addEventListener("mousemove", (ev) => {{
        const rect = canvas.getBoundingClientRect();
        const t = Math.max(0, Math.min(xs.length - 1,
          Math.round((ev.clientX - rect.left) / rect.width * (xs.length - 1))));
        readout.textContent = "x=" + xs[t].toFixed(4) + "  phi=" + ys[t].toFixed(4);
      }});
    }}
  }}
}}
select.addEventListener("change", () => draw(Number(select.value)));
draw(0);
</script>
</body>
</html>
"""
