"""Parameter counts quoted in README.md, docs/ and paper.md.

Counts are trainable parameters, as reported by `benchmarks/compare_mlp.py`.
The KAN grid knots are non-trainable by default and are not counted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from benchmarks.compare_mlp import make_mlp, n_params
from kanx import KAN


def _kan_params(widths, **kwargs) -> int:
    model = KAN(widths, **kwargs)
    model.build((None, widths[0]))
    return n_params(model)


@pytest.mark.parametrize(
    ("widths", "expected"),
    [
        ([2, 16, 1], 432),  # README / benchmarks table
        ([2, 32, 1], 864),  # README headline and benchmarks table
        ([2, 64, 1], 1728),  # paper.md Table 3
        ([10, 32, 1], 3168),  # paper.md diabetes paragraph
    ],
)
def test_kan_tf_param_counts(widths, expected):
    assert _kan_params(widths, grid_size=5, spline_order=3) == expected


def test_kan_default_grid_matches_quoted_counts():
    # The README tables use the default grid_size=5, spline_order=3.
    assert _kan_params([2, 32, 1]) == 864


@pytest.mark.parametrize(
    ("grid_size", "spline_order", "expected"),
    [
        (5, 1, 672),  # paper.md Table 2
        (5, 2, 768),
        (5, 3, 864),
        (5, 5, 1056),
        (5, 7, 1248),
        (8, 3, 1152),  # paper.md grid-size sentence (33% above G=5)
    ],
)
def test_kan_ablation_param_counts(grid_size, spline_order, expected):
    widths = [2, 32, 1]
    assert _kan_params(widths, grid_size=grid_size, spline_order=spline_order) == expected


def test_count_params_includes_grid_knots():
    # The quickstart notebook prints Keras count_params(), which also counts
    # the non-trainable grid: (2 + 32) inputs x (grid_size + 1) knots = 204.
    model = KAN([2, 32, 1], grid_size=5, spline_order=3)
    model.build((None, 2))
    assert model.count_params() == 1068
    assert model.count_params() - n_params(model) == 204


@pytest.mark.parametrize(
    ("in_dim", "hidden", "expected"),
    [
        (2, 32, 129),  # README / benchmarks table
        (2, (16, 16), 337),  # README / benchmarks table
        (2, (64, 64), 4417),  # README headline and paper.md Table 3
        (10, (64, 64), 4929),  # paper.md diabetes paragraph
    ],
)
def test_mlp_param_counts(in_dim, hidden, expected):
    assert n_params(make_mlp(in_dim, hidden)) == expected


def test_sklearn_mlp_diabetes_param_count():
    # The diabetes comparison uses scikit-learn's MLPRegressor, not Keras.
    neural_network = pytest.importorskip("sklearn.neural_network")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 10))
    y = rng.normal(size=32)
    mlp = neural_network.MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # max_iter=1 does not converge
        mlp.fit(X, y)
    count = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    assert count == 4929
