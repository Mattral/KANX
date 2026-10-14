"""Train a small KAN and plot its learned 1-D edge functions.

Run:
    python examples/visualize_edges.py

Writes ``edges.png`` and ``edges.html`` next to this file. The HTML page
is the interactive viewer (layer picker and hover readout). The PNG needs
matplotlib.
"""

from __future__ import annotations

import os

import numpy as np

from kanx import KAN, fit_grid_to_data, set_global_seed
from kanx.viz import plot_edges, write_edge_viewer


def main():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise SystemExit('matplotlib is required: pip install "kanx[viz]"') from err

    set_global_seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(2048, 2)).astype("float32")
    y = (np.sin(np.pi * X[:, :1]) + 0.6 * X[:, 1:2] ** 2).astype("float32")

    model = KAN([2, 1])
    model(np.zeros((1, 2), dtype="float32"))
    fit_grid_to_data(model, X)
    model.fit(X, y, epochs=80, batch_size=64, verbose=0)

    here = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(here, "edges.png")
    html = os.path.join(here, "edges.html")
    fig = plot_edges(model, path=png)
    plt.close(fig)
    write_edge_viewer(model, html)
    print(f"wrote {png}")
    print(f"wrote {html}")


if __name__ == "__main__":
    main()
