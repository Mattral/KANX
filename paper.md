---
title: 'KANX: A Production-Grade Open-Source Library for Kolmogorov–Arnold Networks'
tags:
  - Python
  - machine learning
  - Kolmogorov-Arnold Networks
  - B-spline neural networks
  - TensorFlow
  - PyTorch
  - ONNX
  - function approximation
authors:
  - name: Min Htet Myet
    orcid: 0009-0005-0853-8301
    affiliation: '1'
affiliations:
  - index: 1
    name: Independent Researcher
date: 7 June 2026
bibliography: paper.bib
---

# Summary

KANX (`kanx`) is an open-source Python library that provides a production-ready
implementation of Kolmogorov–Arnold Networks (KANs) [@liu2025kan], a neural
architecture in which every learnable parameter is a univariate B-spline function
rather than a fixed scalar weight. Unlike standard multilayer perceptrons (MLPs),
which apply fixed nonlinearities at nodes, KANs place learnable activation functions
on edges, grounded in the Kolmogorov–Arnold representation theorem
[@kolmogorov1957; @arnold1957]. The result is a model family with strong inductive
bias for smooth, separable regression targets and directly interpretable edge
functions.

KANX exposes KANs through four access surfaces — a TensorFlow primary backend, a
PyTorch secondary backend, a FastAPI REST service, and a command-line interface —
all sharing a common one-liner API (`model.fit(X, y, epochs=30)`). The library
includes real ONNX export for both backends with verified numerical parity, Docker
and Kubernetes manifests for cloud-native deployment, a 113-test suite at 94%
library coverage, and automated PyPI releases via CI/CD. KANX is the first KAN
library purpose-built for production deployment rather than research demonstration.

# Statement of Need

Kolmogorov–Arnold Networks were proposed by @liu2025kan and demonstrated strong
parameter efficiency on smooth synthetic regression tasks, achieving lower
mean-squared error than MLPs of comparable or larger size. The paper's acceptance at
ICLR 2025 prompted rapid community interest and a growing ecosystem of variant
architectures [@liu2025kan2; @temporal_kan; @ka_gnn; @kan_ode].

However, every existing open-source KAN implementation — `pykan` [@pykan],
`efficient-kan` [@efficient_kan], and `mlx-kan` [@mlx_kan] — is a research
artifact. None provides ONNX export, a deployment-ready serving layer, property-based
tests, or a CI/CD release pipeline. This creates a gap between research
reproducibility and practical adoption: practitioners cannot confidently deploy KANs
in production environments, and researchers cannot run fair, standardised benchmarks
against a common baseline.

KANX addresses both gaps. Its target audience is (1) ML engineers evaluating KANs
for deployment in scientific computing, edge ML, or regression services; (2)
researchers who need a reproducible, multi-backend baseline for benchmarking KAN
variants; and (3) practitioners in physics, biology, and engineering who need
interpretable function approximators with a standard deployment path.

# State of the Field

The four most widely used open-source KAN libraries are compared in Table 1.

| Feature                  | pykan | efficient-kan | mlx-kan | **kanx** |
|--------------------------|:-----:|:-------------:|:-------:|:--------:|
| Framework                | PyTorch | PyTorch | MLX | TF + PyTorch |
| Vectorised B-spline      | partial | ✓ | ✓ | ✓ |
| ONNX export              | ✗ | ✗ | ✗ | **✓ both backends** |
| REST API service         | ✗ | ✗ | ✗ | **✓ FastAPI** |
| Docker + Kubernetes      | ✗ | ✗ | ✗ | **✓** |
| Property-based tests     | ✗ | ✗ | ✗ | **✓ Hypothesis** |
| Test coverage            | research | research | research | **94%** |
| CI/CD release pipeline   | ✗ | ✗ | ✗ | **✓ PyPI + GHCR** |

: Comparison of open-source KAN libraries (June 2026). {#tbl:comparison}

KANX does not duplicate the theoretical contributions of `pykan` — the reference
implementation of @liu2025kan — but complements it with the engineering infrastructure
required for deployment. The design choice to implement both TensorFlow and PyTorch
backends, rather than contributing to `pykan` alone, reflects the need to serve teams
with heterogeneous framework stacks without framework lock-in.

# Software Design

## Core Architecture

KANX is structured as a layered system. The core library (`src/kanx/`) depends only
on TensorFlow, NumPy, and PyYAML; the REST service (`api/`) imports the library but
the library never imports FastAPI; the PyTorch backend (`src/kanx/torch/`) is a
parallel surface with identical semantics. This separation keeps the library
lightweight for research use while enabling the full serving stack for deployment.

The central computational primitive is `KANLinear`, which implements a generalization
of a dense layer. For input $\mathbf{x} \in \mathbb{R}^n$ and output
$\mathbf{y} \in \mathbb{R}^m$, the layer computes:

$$y_j = \sum_{i=1}^{n} f_{ij}(x_i) + \text{SiLU}(x) \cdot W_{\text{base}}, \quad j = 1, \ldots, m$$

where each $f_{ij} : \mathbb{R} \to \mathbb{R}$ is a degree-$k$ B-spline over a
uniform grid of $G$ intervals, and $W_{\text{base}}$ is a learnable residual weight
matrix. The B-spline basis is computed via the Cox–de Boor recursion
[@deboor1978], vectorised as a single `einsum` operation for JIT compatibility.
Each spline satisfies two mathematical invariants asserted by the test suite:
*partition of unity* ($\sum_i B_i^k(x) = 1$ within the grid) and
*non-negativity* ($B_i^k(x) \geq 0$ everywhere), verified by Hypothesis
property-based tests against randomly generated grids.

A critical design decision is the `fit_grid_to_data` function, which aligns
per-feature B-spline knot vectors to the observed input distribution before training.
Without this step, inputs past the outermost extended knot produce zero
spline contribution. The stored grid defaults to `[-1, 1]`, and the default
cubic basis is still nonzero between that endpoint and about ±2.2. Leaving
the stored interval is still the wrong function, and nothing raises. This
is the most common failure mode for new users and is prominently documented.

## Key Engineering Trade-offs

The choice of B-spline parameterisation over alternatives (e.g., Fourier bases or
radial functions) follows @liu2025kan for comparability. The default spline order
$k = 3$ (cubic) is empirically validated by ablation on a T4 GPU: cubic splines
achieve 46× lower test MSE than linear splines ($k = 1$), with marginal additional
gain at $k = 5$ or $k = 7$ (Table 2). The default grid size $G = 5$ is near-optimal;
$G = 8$ yields a modest improvement at 37% higher parameter cost.

| Spline order $k$ | Basis functions | Parameters | Test MSE |
|-----------------:|----------------:|-----------:|---------:|
| 1 (linear)       | 6               | 876        | 7.1 × 10⁻⁴ |
| 2                | 7               | 972        | 2.6 × 10⁻⁴ |
| **3 (cubic) ★**  | **8**           | **1,068**  | **1.5 × 10⁻⁵** |
| 5                | 10              | 1,260      | 4.0 × 10⁻⁵ |
| 7                | 12              | 1,452      | 2.8 × 10⁻⁵ |

: Spline order ablation. KAN [2,32,1], 30 epochs, Adam(lr=1e-2), batch=128, T4 GPU.
Target: $y = \sin(\pi x_1) + \cos(2\pi x_2)$. {#tbl:ablation}

ONNX export is implemented via `tf2onnx` for TensorFlow and `torch.onnx` for
PyTorch. Both produce models with dynamic batch axes, verified to agree with eager
inference within $10^{-5}$ absolute error across batch sizes 1 to 1,024. This is the
first ONNX export path for any KAN library.

The `MatrixKAN` variant in the PyTorch backend replaces the Cox–de Boor recursion
with precomputed recurrence matrices, enabling batched GEMM operations. On CPU and
small models (hidden dimension ≤ 32), this is slower than the standard implementation;
the advantage materialises for larger hidden dimensions on GPU hardware.

## Quality Infrastructure

The test suite comprises 113 tests across six files: unit tests for layer shape
contracts and mathematical invariants, integration tests for the training loop and
checkpoint roundtrip (verified to zero absolute error), end-to-end tests for all
five API endpoints with boundary validation, and Hypothesis property-based tests
for partition-of-unity and non-negativity across randomly generated grids. Library
coverage is 94% overall (100% `model.py`, 99% `layers.py`). The CI matrix runs on
Python 3.10, 3.11, and 3.12 with lint, type-checking, Docker smoke test, and MkDocs
build on every push.

# Research Impact Statement

## Reproducible Benchmarks

KANX includes a benchmark harness (`benchmarks/compare_mlp.py`) that compares KAN
and MLP models under honest, parameter-matched conditions. Results on a canonical
smooth 2-D regression target ($y = \sin(\pi x_1) + \cos(2\pi x_2)$, 30 epochs,
T4 GPU) are shown in Table 3. KAN [2,64,1] achieves a test $R^2$ of 0.9999 with
2,124 parameters, outperforming an MLP with twice as many parameters. An ablation
over architecture depth confirms that a single wide hidden layer outperforms deeper
narrow alternatives for this target class — consistent with the two-layer
representability guarantee of the Kolmogorov–Arnold theorem.

| Model             | Parameters | Test MSE       | Test $R^2$ |
|-------------------|-----------:|---------------:|-----------:|
| KAN-TF [2,16,1]   | 564        | ~3 × 10⁻⁴      | 0.9997     |
| **KAN-TF [2,64,1]** | **2,124** | **1.4 × 10⁻⁴** | **0.9999** |
| MLP [2,64,64,1]   | 4,417      | 5.5 × 10⁻⁴     | 0.9995     |

: Canonical benchmark. KAN-TF vs. MLP, 30 epochs, Adam(lr=1e-2), batch=128, T4 GPU.
{#tbl:benchmark}

On a real-world task — the UCI Diabetes dataset [@diabetes_dataset], 442 samples, 10
features, 5-fold cross-validation — KAN-TF [10,32,1] achieves mean $R^2 = 0.449
\pm 0.130$ (RMSE = 56.3) with 1,068 parameters, comparable to Ridge regression
($R^2 = 0.490$) and substantially ahead of an MLP with 4× more parameters
($R^2 = 0.089$). These results are fully reproducible via the public Colab
notebook linked in the repository.

## Community Signals

The KANX repository has accumulated 25 GitHub stars and 8 forks within three months
of initial release, with active issue tracking and pull request engagement. The
library is pip-installable as `pip install kanx`, is available on PyPI at version
0.1.8 (June 2026), and includes a Zenodo DOI [@kanx_zenodo] for citation by
downstream work. The accompanying research preprint [@kanx_preprint] is available
for citation. A comprehensive user guide notebook is publicly hosted on Google Colab,
enabling zero-install reproduction of all benchmark results within 15 minutes on a
free T4 GPU.

# AI Usage Disclosure

Claude (Anthropic) was used as an AI assistant during the development of KANX. Its
role was as a coding and writing aid: it helped draft documentation sections, review
code structure, and suggest test cases. All generated content was reviewed,
verified, and substantially revised by the author. The core algorithm implementation
(B-spline basis, Cox–de Boor recursion, KANLinear forward pass, ONNX export logic)
was written and validated by the author; correctness is verified by the 113-test
suite and numerical invariant checks. The paper text was drafted with AI assistance
and edited by the author for accuracy and completeness.

# Acknowledgements

The author thanks the developers of `pykan` [@pykan] and `efficient-kan`
[@efficient_kan] for their open-source implementations, which informed the design of
KANX's B-spline layer. No financial support was received for this work.

# References
