<h1 align="center">🚀 kanx</h1>

<p align="center">
  <strong>Production-grade Kolmogorov-Arnold Networks</strong><br>
  <em>TensorFlow + PyTorch + ONNX — one library, four surfaces. </em><br>
  <em> > `kanx` is the only Kolmogorov-Arnold Network (KAN) library purpose-built for production deployment.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/kanx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/kanx?style=for-the-badge&logo=pypi&logoColor=white&color=7C3AED"></a>
  <a href="https://pypi.org/project/kanx/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/kanx?style=for-the-badge&color=A78BFA"></a>
  <a href="https://pepy.tech/project/kanx">  <img alt="Total Downloads" src="https://img.shields.io/pepy/dt/kanx?style=for-the-badge&color=F87171"></a>
  <a href="./CITATION.cff"><img alt="Cite" src="https://img.shields.io/badge/cite-CITATION.cff-brightgreen?style=for-the-badge"></a>

  <img alt="Python" src="https://img.shields.io/pypi/pyversions/kanx?style=for-the-badge&logo=python&logoColor=white">
  <a href="https://mattral.github.io/KANX/"><img alt="Docs" src="https://img.shields.io/badge/docs-mkdocs--material-22C55E?style=for-the-badge&logo=readthedocs&logoColor=white"></a>
  <a href="https://colab.research.google.com/github/Mattral/KANX/blob/main/notebooks/quickstart.ipynb"><img alt="Colab" src="https://img.shields.io/badge/Colab-train_in_2_to_5_min-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-A78BFA?style=for-the-badge"></a>
  <a href="https://doi.org/10.5281/zenodo.20615396">
    <img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20430883-7C3AED?style=for-the-badge&logo=zenodo&logoColor=white">
  </a>

</p>

> **`pip install kanx`** &nbsp;·&nbsp; A small KAN beats a 10× larger MLP on smooth, separable targets — **honest, param-matched benchmark below.**
> One library. Two backends. Real ONNX export. Docker + Kubernetes ready. Prometheus metrics, TensorBoard logging, Hub and symbolic extras are now implemented.

---


## ⭐ Why kanx?
<div align="center">

Every other KAN library stops at research. `kanx` goes the full distance:

| | [pykan](https://github.com/KindXiaoming/pykan) | [efficient-kan](https://github.com/Blealtan/efficient-kan) | [mlx-kan](https://github.com/Goekdeniz-Guelmez/mlx-kan) | **kanx** |
|---|:---:|:---:|:---:|:---:|
| Framework         | PyTorch | PyTorch | MLX (Apple Silicon) | **TF + PyTorch** |
| Vectorized B-spline | partial | ✅ | ✅ | ✅ |
| ONNX export       | ❌ | ❌ | ❌ | ✅ **both backends** |
| REST API service  | ❌ | ❌ | ❌ | ✅ FastAPI |
| Docker + K8s      | ❌ | ❌ | ❌ | ✅ |
| Property-based tests | ❌ | ❌ | ❌ | ✅ Hypothesis |
| Test coverage     | research | research | research | **CI + Hypothesis** |
| PyPI              | ✅ | ✅ | ✅ | ✅ |
| CI/CD release pipeline | ❌ | ❌ | ❌ | ✅ PyPI + GHCR + Pages |

</div>

Most of the other KAN implementations are strictly academic. `KANX` bridges the gap between theory and reality by providing:
1. **Multi-Backend Support:** Native TensorFlow and PyTorch (`MatrixKAN`) implementations.
2. **Real Deployment:** True ONNX export, FastAPI (`/api/predict`), Docker, and Kubernetes configs out-of-the-box.
3. **High-Throughput Options:** Vectorized `MatrixKAN` to replace recursive B-splines with batched GEMMs for GPU acceleration.
4. **Grid Calibration:** Native adaptive and static grid fitting to prevent out-of-bounds input collapse.


---

## 📊 Benchmarks (reproducible, fair, multi-baseline)

<div align="center">
  
Synthetic 2-D regression target `y = sin(π·x₁) + cos(2π·x₂)`,
100 epochs, Adam(lr=1e-2), batch=128, CPU.


| Model              | Params | Train (s) | Infer 4k (ms) | **Test MSE** |
|--------------------|------:|---------:|-------------:|-------------:|
| **KAN[2,16,1]**    |   432 |    12.50 |        68.64 | **2.14 × 10⁻⁵** |
| KAN[2,32,1]        |   864 |    16.62 |        25.52 | 4.44 × 10⁻⁴ |
| MLP[2,32,1]        |   129 |     5.07 |         6.17 | 4.61 × 10⁻¹ (undersized) |
| MLP[2,16,16,1]     |   337 |     5.46 |         4.08 | 1.60 × 10⁻³ |
| MLP[2,64,64,1]     | 4 417 |     6.00 |         5.74 | 5.51 × 10⁻⁴ |

</div>

<p align="center">
  <img src="docs/assets/benchmark.png" alt="Test MSE for the published KAN and MLP runs" width="720"/>
</p>

**Honest read.** The smallest KAN (432 params) wins on this smooth separable
target. The same KAN is ~10–15× *slower at inference* than a same-MSE MLP
because each edge does a B-spline evaluation. On non-smooth or
high-dimensional targets, this picture often reverses. We do not claim KANs
are universally better than MLPs.

Reproduce with `python benchmarks/compare_mlp.py` (quick, 100 epochs) or
`python benchmarks/compare_mlp.py --long` (1000 epochs + early-stopping).

CI runs a CPU smoke gate that does not rewrite the published tables:

```bash
python benchmarks/compare_mlp.py --smoke
python benchmarks/real_world.py --smoke   # needs scikit-learn for California housing
```

`python benchmarks/real_world.py --device cuda` trains the PyTorch KAN and MLP in a child process, so TensorFlow and PyTorch do not initialize CUDA in one process. CPU runs and `--smoke` stay TensorFlow-only.

---

## ⚡ The 30-second magic moment

```python
import kanx

# Build, train, predict — in one call. No config files. No compile dance.
model = kanx.quickstart()                       # trains on synthetic 2-D data
model.predict([[0.5, 0.2]])                     # → array([[1.04…]])
```

> **⚠️ Grid calibration — two methods**
> 
> KANs use B-splines on a fixed input range (default `[-1, 1]`). If your inputs fall outside that range, the spline path **silently returns zero** and you only get the SiLU residual. Fix it one of two ways:
> 
> **Static approach** (pre-training):
> ```python
> from kanx import KAN, fit_grid_to_data
> model = KAN([n_features, 64, 1])
> fit_grid_to_data(model, X_train)              # one-time grid fit
> model.fit(X_train, y_train, epochs=30)
> ```
> 
> **Adaptive approach** (during training — recommended):
> ```python
> model = KAN([n_features, 64, 1])
> model.fit(X_train, y_train, epochs=15)
> model.update_grid_from_samples(X_train)       # ← refine grid based on data
> model.fit(X_train, y_train, epochs=15)        # continue training
> ```
> 
> `kanx.check_input_range(model, X)` will log a warning at inference if input exceeds the grid.

Drop edges that barely contribute. `threshold` is an absolute cutoff on the edge score. `relative` drops edges at or below that fraction of the strongest edge in the same layer. Pass `samples` to score each edge by its RMS on real inputs. The same method is on the PyTorch `KAN` and on `MatrixKAN`.

```python
model.prune(relative=0.01)                  # keep edges above 1% of the peak
model.prune(threshold=1e-3, samples=X_train)
```

Plot what is left. Each edge is a learned 1-D function (SiLU residual plus B-spline). `plot_edges` needs matplotlib (`pip install "kanx[viz]"`). `write_edge_viewer` writes a single HTML file with a layer picker and a hover readout.

```python
from kanx.viz import plot_edges, write_edge_viewer

plot_edges(model, path="edges.png")
write_edge_viewer(model, "edges.html")
```

<p align="center">
  <img src="docs/assets/edge_functions.png" alt="Learned edge functions on a two-input KAN" width="640"/>
</p>

Layer 0 of a `KAN([2, 1])` fit to `sin(π x₁) + 0.6 x₂²`. The first edge carries the sine. The second carries the quadratic.

The TensorFlow spline contraction can run in reduced precision, and the forward pass can be XLA-compiled. Weights and the returned tensor stay float32. The Cox-de Boor basis stays float32 as well: knot differences underflow in float16.

```python
model = KAN([8, 16, 1], precision="bfloat16", jit_compile=True)
```

`precision` is `float32` (default), `float16`, or `bfloat16`. `jit_compile=True` compiles on the first call. Batch size may change; the feature width may not. The same arguments exist on `KANLinear`. PyTorch is unchanged. Do not export a `jit_compile=True` layer to ONNX; trace the default float32 layer instead. Reduced precision is close to float32, not bit-exact. bfloat16 is the practical training dtype.

---

Want more control? Same simplicity, your data:

```python
from kanx import KAN
import numpy as np

X = np.random.uniform(-1, 1, (1024, 2)).astype("float32")
y = np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2

model = KAN([2, 64, 1])
model.fit(X, y, epochs=30, verbose=0)           # auto-compiles with Adam+MSE
model.predict(X[:3])
```

---

### 🔥 PyTorch? Same API.

```python
from kanx.torch import KAN
import torch

model = KAN([2, 64, 1])
X = torch.randn(1024, 2); y = torch.sin(torch.pi * X[:, :1])
model.fit(X, y, epochs=30, lr=1e-2)             # one-liner, same semantics
model.predict([[0.5, 0.2]])
```

---


### ⚡ GPU-optimized MatrixKAN

For higher throughput on accelerators, use the vectorized `MatrixKAN` (replaces recursion with batched GEMM):

```python
from kanx.torch import MatrixKAN

model = MatrixKAN([4, 32, 1])  # same interface as KAN
model.fit(X, y, epochs=30)      # ~1.5–2× faster on GPU vs standard KAN
```

TensorBoard logging and symbolic edge fitting are included:

```bash
python -m kanx train --config configs/default.yaml --tensorboard
tensorboard --logdir logs/kanx
```

```python
import torch
from kanx.torch import KAN, SymbolicFitter

model = KAN([1, 1])
x_1d = torch.linspace(-1, 1, 64)
name, r2 = SymbolicFitter(model).fit_edge(0, 0, 0, x_1d)
```

---

## 📦 Installation

```bash
pip install kanx                # core (TensorFlow)
pip install "kanx[torch]"       # +PyTorch backend
pip install "kanx[onnx]"        # +tf2onnx + onnxruntime
pip install "kanx[api]"         # +FastAPI service
pip install "kanx[hub]"         # +HuggingFace Hub integration
pip install "kanx[symbolic]"    # +Symbolic regression hooks
pip install "kanx[viz]"         # +matplotlib for plot_edges
pip install "kanx[all]"         # everything (api + torch + onnx + hub + symbolic + viz + dev + docs)
```

Optional extras:
* `kanx[api]` adds FastAPI serving with `/metrics` Prometheus scraping.
* `kanx[torch]` adds the PyTorch backend, `MatrixKAN`, and symbolic helpers.
* `kanx[hub]` adds `push_to_hub()` / `from_pretrained()` for HuggingFace integration.
* `kanx[symbolic]` adds `SymbolicFitter` for post-hoc edge function extraction.
* `kanx[viz]` adds matplotlib so `kanx.viz.plot_edges` can draw the learned curves. The HTML viewer does not need it.

→ Open in Colab: **[Train a KAN in 2-to-5 minutes](https://colab.research.google.com/github/Mattral/KANX/blob/main/notebooks/quickstart.ipynb)**

---



## 🏗️ Production Serving

Install `kanx[api]` and start the FastAPI app with uvicorn. Prometheus scrapes `/metrics`.

```bash
uvicorn api.app:app --port 8000
```

Kubernetes manifests (Deployment, Service, Ingress, HPA, PVC) live in [`k8s/`](k8s/). The same objects are parameterized in the Helm chart [`charts/kanx`](charts/kanx/).

**API Contract:**

* `GET /api/health` - Liveness & model load source
* `GET /api/info` - TF/Torch backend version and summary
* `POST /api/predict` - Batched inference

---

## 🌐 REST API

```bash
docker run --rm -p 8000:8000 ghcr.io/mattral/kanx:latest
# or
uvicorn api.app:app --port 8000
```

<div align="center">
  
| Method | Path           | Purpose |
|-------:|:--------------|:--------|
| `GET`  | `/api/health`  | Liveness + model load source |
| `GET`  | `/api/info`    | Version + TF/Torch + model summary |
| `GET`  | `/metrics`     | Prometheus scrape endpoint |
| `POST` | `/api/predict` | Inference (single or batch) |
| `POST` | `/api/load`    | Hot-swap checkpoint |
| `POST` | `/api/reset`   | Re-init from `KANX_CONFIG` |

</div>

```bash
curl -X POST http://localhost:8000/api/predict \
     -H 'content-type: application/json' \
     -d '{"x": [[0.1, -0.2], [0.5, 0.7]]}'
```

The startup contract loads `KANX_CHECKPOINT` if it exists, otherwise falls
back to a fresh model built from `KANX_CONFIG`. Boundaries are validated:
wrong feature count → `400`, oversized batch → `413`, missing checkpoint → `404`.

---

## 🔄 ONNX export

```python
# From PyTorch
from kanx.torch import KAN, export_onnx
model = KAN([2, 64, 1])
export_onnx(model, "kan.onnx")
```

```python
# From TensorFlow
from kanx import KAN, export_onnx_tf
import tensorflow as tf
model = KAN([2, 64, 1]); model(tf.zeros((1, 2)))
export_onnx_tf(model, "kan.onnx")
```

✔ Dynamic batch
✔ Verified numerical consistency (1e-5)
✔ Works with ONNX Runtime / TensorRT / OpenVINO

---

## 🐳 Docker / ☸️ Kubernetes

```bash
docker run --rm -p 8000:8000 ghcr.io/mattral/kanx:latest
kubectl apply -f k8s/                 # Deployment + Service + Ingress + HPA + PVC
helm install kanx-api charts/kanx     # same objects, parameterized
```

K8s manifests ship with rolling updates, readiness/liveness probes on
`/api/health`, an HPA (2 ↔ 10 replicas, CPU-target 70%) and a PVC for the
model registry. Use the raw manifests or the chart, not both, when the
release name is `kanx-api`. Install notes: [`charts/kanx/README.md`](charts/kanx/README.md).

---

## 🛠️ CLI

```bash
python -m kanx info                                          # versions
python -m kanx train --config configs/default.yaml           # train
python -m kanx predict --checkpoint model.keras --input X.json
```

---

## ⭐ Quality

- Unit, integration, live API, property-based, and performance tests in [`tests/`](tests/)
- **Hypothesis property tests**: partition of unity, shape invariants, gradient finiteness
- **Numerical contracts**: ONNX parity within 1e-5, save/load roundtrip identity
- **Performance regression alarms**: latency budgets on forward pass and predict
- CI: Python 3.10 / 3.11 / 3.12, lint, Docker smoke, MkDocs, and a CPU benchmark smoke gate
- CPU run of `pytest tests/ --ignore=tests/test_e2e_live.py`: **162 passed, 4 skipped** (Helm render checks when `helm` is not installed, and MatrixKAN CUDA checks). Branch coverage of `src/kanx` on that run is **77%** (`layers.py` 95%, `model.py` 91%). The older 94% / 113-test figures described a smaller suite. The distributed smoke in that count runs `MirroredStrategy` on two logical CPUs and `torch.distributed` with two gloo processes when no GPU is present.

```bash
pytest tests/ -v --cov=src/kanx
```

---

## 📚 Documentation

<div align="center">

→ **<https://mattral.github.io/KANX/>** (MkDocs Material)

| Page | What's inside |
|------|---------------|
| [Quickstart](https://mattral.github.io/KANX/quickstart/) | Train your first KAN in 60 seconds |
| [Architecture](https://mattral.github.io/KANX/architecture/) | Package layout, module contracts |
| [System Design](https://mattral.github.io/KANX/system_design/) | Serving topology, scaling, failure modes |
| [REST API](https://mattral.github.io/KANX/api/) | Endpoint reference + curl examples |
| [Testing](https://mattral.github.io/KANX/testing/) | Test pyramid, numerical invariants |
| [Deployment](https://mattral.github.io/KANX/deployment/) | CI/CD, rollout, observability |
| [Benchmarks](https://mattral.github.io/KANX/benchmarks/) | KAN vs MLP — methodology + numbers |

</div>

---

## 📄 Research Paper

If you use kanx in academic work, please cite both the original paper and
the library. 

Our work is formally documented and available as a preprint:

- 📘 Title: *KANX: A Production-Grade Open-Source Library for Kolmogorov-Arnold Networks*
- 📍 DOI: https://doi.org/10.5281/zenodo.20615396
- 📂 Zenodo: https://zenodo.org/records/20430883
- 📄 [Read Paper (preprint)](docs/preprint.pdf)
- 📄 [Read Paper (ArXiv)](docs/KANX_ArXiv_Paper.pdf)



### Citation

```bibtex
@article{mattral2026kanx,
  title={KANX: A Production-Grade Open-Source Library for Kolmogorov-Arnold Networks},
  author={Myet, Min Htet},
  year={2026},
  doi={10.5281/zenodo.20615396},
  publisher={Zenodo}
}

@article{liu2024kan,
  title   = {KAN: Kolmogorov-Arnold Networks},
  author  = {Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle,
             Fabian and Halverson, James and Soljačić, Marin and
             Hou, Thomas Y. and Tegmark, Max},
  journal = {arXiv preprint arXiv:2404.19756},
  year    = {2024}
}
```

### References

- Liu et al., *KAN: Kolmogorov-Arnold Networks* — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- The Kolmogorov-Arnold representation theorem ([Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem))
- B-splines & de Boor algorithm — [Carl de Boor (1972)](https://doi.org/10.1016/0021-9045(72)90080-9)

---

## 🤝 Contributing

PRs welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md). Good places to start:

- 🔖 [Good first issues](https://github.com/Mattral/KANX/labels/good%20first%20issue)
- 🗺️ [`roadmap.md`](roadmap.md) — P0 / P1 / P2 backlog
- 💬 [Discussions](https://github.com/Mattral/KANX/discussions)

---

## 📜 License

[Apache 2.0](LICENSE). Use it. Ship it. Tell us when you do — we'd love to
hear how kanx is being used in the wild.

<p align="center">⭐ <strong>Star the <a href="https://github.com/Mattral/KANX">repo</a></strong> if kanx saved you time!</p>
