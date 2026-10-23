# kanx Roadmap

Single source of truth for what's shipped and what's next.

## ✅ Shipped — v0.1.0 (May 2026)

### Library — TensorFlow backend (primary)
- ✅ `src/kanx/` (`layers`, `model`, `train`, `inference`, `config`, `utils`, `__main__`, `onnx_export`)
- ✅ Vectorized Cox-de Boor B-spline (no Python loops)
- ✅ `KANLinear` with SiLU residual + per-feature grids (Liu et al. 2024)
- ✅ `@register_keras_serializable` → safe `save_model`/`load_model`
- ✅ CLI: `python -m kanx {info,train,predict}`

### Library — PyTorch backend (parallel surface)
- ✅ `src/kanx/torch/` (`layers`, `model`, `trainer`, `onnx_export`)
- ✅ Native `torch.nn.Module` integration (autograd, DataLoader, DDP-ready)
- ✅ `Trainer` mirrors `kanx.train` semantics for one-liner training
- ✅ `KAN.save()`/`KAN.load()` checkpoint format

### ONNX export
- ✅ TF → ONNX via tf2onnx (`kanx.export_onnx_tf`)
- ✅ PyTorch → ONNX native (`kanx.torch.export_onnx`)
- ✅ Dynamic batch axis
- ✅ Numerical-parity tests (within 1e-5 of eager)

### Serving
- ✅ FastAPI service (`api/app.py`) with thread-safe `ModelRegistry`
- ✅ Endpoints: `/api/health`, `/api/info`, `/api/predict`, `/api/load`, `/api/reset`
- ✅ Lifespan-based startup with checkpoint-with-fallback contract
- ✅ Boundary validation (400 / 413 / 404 at right places)
- ✅ Supervisor-managed deployment via `backend/server.py`

### Quality
- ✅ **162 tests passed** on a CPU run that skips `tests/test_e2e_live.py`, with 4 skips when Helm and CUDA are absent
- ✅ **Distributed smoke** (`tests/test_distributed_smoke.py`): one `tf.distribute.MirroredStrategy` epoch and one two-process `torch.distributed` step. CPU CI uses two logical CPUs and gloo. Two GPUs use those devices. Run `pytest tests/test_distributed_smoke.py -q`.
- ✅ **77% branch coverage** of `src/kanx` (`layers.py` 95%, `model.py` 91%). The older 94% figure was a smaller core.
- ✅ Hypothesis property tests (partition of unity, shape invariants, gradient finiteness)
- ✅ Performance regression alarms (latency budgets on forward + predict)
- ✅ Numerical-contract tests: partition of unity, non-negativity, save/load roundtrip, ONNX parity
- ✅ TF + Torch coexistence (single-threaded BLAS via `conftest.py`)

### Infra & docs
- ✅ Dockerfile + docker-compose
- ✅ Kubernetes manifests (Deployment + Service + Ingress + HPA + PVC)
- ✅ GitHub Actions CI: matrix py3.10/3.11/3.12 + lint + Docker smoke + MkDocs build
- ✅ Release pipeline (`release.yml`): PyPI (OIDC) + GHCR Docker push + GitHub Release + MkDocs gh-deploy
- ✅ **MkDocs Material site** (`mkdocs.yml` + `docs/`) — 12 pages with code-tabs, dark/light, search
- ✅ `docs/` — MkDocs source for the website and long-form reference docs
- ✅ `CHANGELOG.md` (Keep-a-Changelog format)
- ✅ `notebooks/quickstart.ipynb` — Colab-ready "Train KAN in 2 minutes"
- ✅ `notebooks/LAUNCH_POST.md` — community launch copy
- ✅ Benchmark chart (`docs/assets/benchmark.png`) plots the warmed 100-epoch CPU table (three init seeds, Intel Core i9-9900, TensorFlow 2.21.0). KAN[2,32,1] mean test MSE is 1.09×10⁻⁴. MLP[2,64,64,1] is 6.43×10⁻⁴. The old 265× headline compared a shorter, unfair run and stays withdrawn.

## ✅ Shipped in 0.2.0

Package 0.2.0 collects the work after 0.1.9: edge pruning, `kanx.viz`, the PyTorch GPU benchmark subprocess, mixed precision and XLA, the Helm chart, and the distributed smoke tests. Those items are checked off below.

## ✅ Shipped — 0.1.7 through 0.1.9 (June 2026)

These features shipped in package versions 0.1.7, 0.1.8, and 0.1.9.
0.1.7 shipped MatrixKAN, adaptive grids, the dataset loaders, and the
real-world benchmark. 0.1.8 added Hub, symbolic regression, TensorBoard, and
Prometheus. 0.1.9 fixed multi-layer grid updates and the Docker image.

### New features
- ✅ **MatrixKAN** (PyTorch, experimental): same Cox–de Boor basis as `KAN`, looped per basis index. The 0.1.7 notes called this a batched GEMM and ~1.5–2× faster on GPU. The forward pass does not use `build_recurrence_matrix_batch`. A Tesla T4 sweep of the loop (torch 2.11.0+cu130, kanx 0.2.0, shape `[2, width, 1]`, width 32–512, batch 1,024–16,384) is slower. The call-time ratio falls from about 6.4× to about 3.3×.
  - `src/kanx/torch/matrix_kan.py` (MatrixKANLinear, MatrixKAN classes)
  - Numerical parity with standard KAN (within 1e-4)
  - Full test coverage: shape, numerical agreement, GPU throughput, grid updates, ONNX export
  - **CPU+GPU compatible**: runs on either device. The Tesla T4 sweep above times the Python-loop basis, and that loop is slower than standard KAN
  
- ✅ **Adaptive grid update** (both TensorFlow + PyTorch): `model.update_grid_from_samples(x, margin=0.01)`
  - Per-feature quantile-based grid recalibration from data statistics
  - Interpolation between uniform and sample-based grids via `grid_eps` parameter
  - Propagates through multi-layer models correctly
  - Differentiable in-place updates
  - **CPU+GPU compatible**: Works on both backends with same semantics
  
- ✅ **`kanx.datasets` mini-module**: Unified dataset loading
  - `load_california_housing()`, `load_concrete_strength()`, `load_energy_efficiency()` (UCI tabular)
  - Per-feature normalization to zero-mean unit-variance
  - Caching to `~/.cache/kanx/datasets/`
  - Used in `benchmarks/real_world.py` for credible baselines
  
- ✅ **GPU timing in benchmarks**: `benchmarks/compare_mlp.py` now measures inference latency on GPU
  - TensorFlow: `tf.config.list_physical_devices('GPU')` + median over 100 passes
  - PyTorch: `torch.cuda.synchronize()` + wall-clock measurement
  - Graceful CPU fallback: GPU field is N/A on CPU-only systems
  - **CPU+GPU compatible**: Works on both, adapts to available hardware
  
- ✅ **Real-world benchmark suite** with reproducible artifact:
  - `benchmarks/real_world.py`: 5-fold cross-validation on 3 UCI datasets
  - TensorFlow-only on CPU (PyTorch separately via subprocess on GPU)
  - Outputs `benchmarks/results/real_world_results.json` (committed baseline)
  - Includes train time, RMSE, R², inference latency (CPU + GPU)
  - **CPU+GPU compatible**: Runs on CPU; GPU fields populated when available
- ✅ **HuggingFace Hub** (`kanx[hub]`): `KAN.from_pretrained()` and `push_to_hub()` on TensorFlow and PyTorch
- ✅ **Symbolic regression** (`kanx[symbolic]`): `kanx.torch.SymbolicFitter` fits closed forms to learned edges
- ✅ **TensorBoard** on both training paths (`python -m kanx train --tensorboard`)
- ✅ **Prometheus** scrape endpoint at `GET /metrics` in the FastAPI service
- ✅ **CPU benchmark smoke gate**: `benchmarks/compare_mlp.py --smoke` and `benchmarks/real_world.py --smoke`, run from the `benchmark-smoke` GitHub Actions job. Smoke runs do not overwrite the committed baseline artifacts.
- ✅ **Edge pruning** on TensorFlow `KAN`, PyTorch `KAN`, and `MatrixKAN`: `prune(threshold=..., relative=..., samples=...)` zeros low-magnitude or low-contribution edges
- ✅ **Edge visualisation** (`kanx.viz`): `edge_curves`, `plot_edges`, and `write_edge_viewer` plot the learned 1-D edge (base residual + spline) for TensorFlow `KAN`, PyTorch `KAN`, and `MatrixKAN`. Example: `examples/visualize_edges.py`. Notebook: `notebooks/visualize_edges.ipynb`.
- ✅ **PyTorch subprocess benchmark wrapper**: `benchmarks/real_world.py` runs the PyTorch KAN and MLP in a child process when a GPU is requested, so TensorFlow and PyTorch do not share a CUDA context. CPU runs and `--smoke` stay in-process and TensorFlow-only.
- ✅ **Mixed precision + XLA** on the TensorFlow spline path: `precision="float16"` or `"bfloat16"` casts the residual matmul and spline einsum, and `jit_compile=True` compiles the forward pass. The Cox-de Boor basis stays float32.

### Infrastructure & quality
- ✅ **CITATION.cff** (CFF v1.2.0 format) — machine-readable citation for academic attribution
- ✅ **SECURITY.md** — vulnerability disclosure policy, version support lifecycle, 48-hour SLA
- ✅ **Docs consolidation**: `documentations/` merged into `docs/`; single source of truth in MkDocs Material site
- ✅ **README updates**: Grid calibration best practices, MatrixKAN introduction, adaptive approach recommended
- ✅ **Quickstart.md** additions: Grid update example (TF + PyTorch), MatrixKAN section
- ✅ **System Design docs**: MatrixKAN architecture, adaptive grid implementation details, GPU considerations
- ✅ **Roadmap clarity**: shipped 0.1.7–0.1.9 work is listed above; open items stay in Next and the backlogs
- ✅ **Helm chart** (`charts/kanx`): values for the image, replica count, resources, probes, CPU HPA, model PVC, and Ingress. `kubectl apply -f k8s/` still works.

### Test coverage
- ✅ **7 MatrixKAN tests** (`tests/test_matrix_kan.py`): output shape, numerical agreement, GPU throughput, grid update, ONNX export
- ✅ **8 grid-update tests** (`tests/test_grid_update.py`): TensorFlow + PyTorch shape/stability/improvement
- Cumulative CPU suite: **162 passed**, 4 skipped without Helm or CUDA; **77%** branch coverage of `src/kanx` (`layers.py` 95%, `model.py` 91%)

## 🔵 Backlog (P1)

- [ ] Profiling: replace per-edge einsum with fused custom op
- [ ] Bayesian / dropout KAN variant for uncertainty estimation
- [ ] gRPC serving alongside REST
- [ ] JAX backend (`kanx.jax`) as a third parallel surface

## 🟣 Backlog (P2)

- [ ] Reproduce subset of Liu et al. (2024) benchmarks (Feynman dataset)
- [ ] Sphinx auto-API reference
- [ ] Triton inference server adapter
- [ ] Quantization-aware training pass
- [ ] Streaming inference endpoint (`/api/predict-stream` via SSE/WS)

## 📜 Decision log

- **TensorFlow primary + PyTorch secondary** — TF matches the upstream repo;
  Torch added because it's where most ML research happens (~80% of new papers).
- **ONNX as the deployment lingua-franca** — instead of integrating with each
  serving runtime, export to ONNX once and let the user pick (TensorRT,
  OpenVINO, ONNX Runtime, CoreML, …).
- **Per-feature grids** — required for adaptive grid updates (pykan parity).
- **Hand-rolled config validator** — no pydantic dep in the core lib.
- **`save_best_only=True` + final-model fallback** — inference always works.
- **Single-threaded BLAS in tests** — only way to keep TF+Torch hermetic in
  one process. Production code does not impose this.

