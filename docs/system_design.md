# System Design

## KAN Architecture Overview

Kolmogorov-Arnold Networks (Liu et al., 2024) replace the weight matrix in MLPs with a **learnable nonlinear basis function** per edge. Each `KANLinear` layer computes:

$$\text{out}(x) = \text{SiLU}(x) \mathbf{W}_{\text{base}} + \sum_{i,j} w^{\text{spline}}_{i,o,j} \cdot B_j(x_i)$$

where:
- $\text{SiLU}(x)$ is the residual path (fixed activation)
- $B_j$ is a B-spline basis function (learnable via grid + weights)
- Each input feature $x_i$ has its own B-spline grid (per-feature adaptation)

### B-spline Basis Functions

B-splines are evaluated via **Cox-de Boor recursion**:
$$B_i^0(x) = [x_i \in [t_i, t_{i+1}))$$
$$B_i^k(x) = \frac{x - t_i}{t_{i+k} - t_i} B_i^{k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1}^{k-1}(x)$$

The implementation is **fully vectorized** (no per-feature or per-sample Python loops) via einsum operations on GPU/TPU.

### Grid System

Each layer maintains a **per-feature uniform grid** with `grid_size + 1` knots over `[grid_range_min, grid_range_max]` (default `[-1, 1]`). The number of basis functions is:
$$\text{num\_basis} = \text{grid\_size} + \text{spline\_order}$$

Example: `grid_size=5`, `spline_order=3` → 8 basis functions per feature.

---

## B-spline Grid Management

### Static Grids (v0.1.0)

By default, KAN models use uniform grids fixed at initialization. For production, users must explicitly calibrate grids to their data using `fit_grid_to_data(model, X_train)` before training.

**Limitation:** The basis is exactly zero only past the outermost extended knot. Outside the stored grid interval it is no longer a partition of unity. Either way, accuracy drops and nothing raises.

### Adaptive Grid Updates (v0.2.0)

v0.2.0 introduces `model.update_grid_from_samples(x, margin=0.01)` for both TensorFlow and PyTorch:

```python
model = KAN([4, 16, 8, 1])
model.fit(X_train, y_train, epochs=15)
model.update_grid_from_samples(X_train)  # recompute grids from data
model.fit(X_train, y_train, epochs=15)   # resume training
```

**Algorithm:**
1. For each input feature, compute quantiles `[0, 1/(grid_size), ..., 1]` from observed data
2. Interpolate between uniform grid and quantile-based grid: $(1 - \epsilon) \cdot \text{uniform} + \epsilon \cdot \text{sample\_based}$
3. Use `grid_eps=0.02` (default) to blend conservatively toward sample-based grid
4. Update layer grids in-place (differentiable)

**Multi-layer handling:**
- First layer updates from raw input $x$
- Subsequent layers: propagate $x$ through prior layers, then update grid from activations
- Ensures coherent grid adaptation across depth

**Benefits:**
- Automatically aligns B-spline bases to observed data distribution
- No manual range calibration needed
- Can be called multiple times during training
- Particularly useful for tabular data with non-uniform feature ranges

**CPU+GPU compatibility:** Works identically on both backends using backend-native operations (TensorFlow `sort` + `gather`, PyTorch `quantile`).

---

## MatrixKAN: experimental (v0.2.1)

`MatrixKAN` is a PyTorch class whose basis is the same Cox–de Boor recursion as `KANLinear`, with a Python loop over each basis index. `build_recurrence_matrix_batch` is not called. Its entries do not depend on `x`, so it is not the recursion. The 0.2.0 docs described a product of precomputed matrices. The code does not do that.

On a Tesla T4 (torch 2.11.0+cu130, kanx 0.2.0) that loop is slower than vectorized `KAN` for `[2, width, 1]`, width 32–512, batch 1,024–16,384. The call-time ratio falls from about 6.4× to about 3.3×. Copied weights match the vectorized basis to a max absolute error of 0 on a CPU check of one 32→1 layer. A profiler on that layer, batch 1,024, torch 2.14.1+cpu, counted 1,394 ATen ops against 205.

### Recommendations

- Use standard `KAN` as the default.
- Treat `MatrixKAN` as experimental until a uniform-grid basis, or a fallback for the non-uniform grids `update_grid_from_samples` writes, is implemented and timed.

---

## REST API & Serving

## Goals

1. Serve KAN inference behind a stable REST contract.
2. Hot-swap checkpoints without a restart.
3. Fall back to a fresh model when no checkpoint exists (zero-downtime cold start).
4. Scale horizontally on Kubernetes with CPU-based HPA.
5. Surface a meaningful `/api/health` signal for orchestrators.

## High-level topology

```
       ┌─────────────────────────────────────────────────┐
       │           Ingress (nginx)                       │
       │   kanx.example.com  →  Service kanx-api:80      │
       └────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
      ┌──────────┐                   ┌──────────┐
      │ Pod #1   │                   │ Pod #N   │
      │ uvicorn  │   ... HPA 2–10    │ uvicorn  │
      │ + kanx   │                   │ + kanx   │
      └────┬─────┘                   └────┬─────┘
           │                              │
           └──────── PVC (RO) ────────────┘
                    /mnt/models/kanx_model.keras
```

* Each pod is **stateless**: it owns a single in-process `tf.keras.Model`
  inside a `ModelRegistry` and serves `/api/*` over uvicorn.
* The checkpoint lives on a **read-only PersistentVolume** mounted into every
  pod at `/mnt/models`. Updates are rolled by re-creating the PV contents and
  POSTing `/api/load` (or restarting the rollout).

## Request lifecycle: `POST /api/predict`

```
client → ingress → svc → pod
              │
              ▼
       Pydantic parse  ──fail──▶ 422 (FastAPI default)
              │
              ▼
       Boundary checks
        • rank ∈ {1, 2}        ──fail──▶ 400
        • last dim == in_feat  ──fail──▶ 400
        • batch <= MAX_BATCH   ──fail──▶ 413
              │
              ▼
       ModelRegistry.model  (R-lock)
              │
              ▼
       kanx.inference.predict(model, x, batch_size)
              │
              ▼
       PredictResponse(output, shape, inference_ms)
```

All boundary checks happen **before** any TF graph is invoked — invalid
requests are cheap.

## Startup contract (checkpoint + fallback)

The user explicitly requested "(b) Serve a trained checkpoint loaded from
disk" **and** "(c) Both: load checkpoint if available, fallback to fresh
model". `api/app.py:_initialise` implements:

```
if exists(KANX_CHECKPOINT):
    model = load_model(KANX_CHECKPOINT)
    source = "checkpoint:<path>"
else:
    model = build_from_config(KANX_CONFIG)
    source = "fresh:<config>"
```

`source` is surfaced through `/api/info` and `/api/health` for observability.

## Scaling model

* **Stateless pods + sticky GET cache** — replicas can be killed at will.
* **HPA target = 70% CPU.** TF inference is CPU-bound for the supported
  model sizes (no GPU).
* **Per-pod throughput.** With `KAN[2,64,64,1]` we measured ~4 ms / 4 k samples
  on a 2-vCPU pod → ~10⁶ predictions/sec single replica. Real workloads will
  be dominated by model architecture, not the API layer.
* **MAX_BATCH guardrail** prevents a single request from monopolising a pod.

## Failure modes & responses

| Failure | Detection | User-visible | Action |
|---|---|---|---|
| Checkpoint missing at startup | `os.path.exists` | `/api/health.source` shows `fresh:…` | Fall back to fresh model |
| Corrupt checkpoint at runtime | `tf.keras.models.load_model` raises | `400` from `/api/load` | Caller retries with valid path |
| Bad input shape | Boundary check | `400` | Caller fixes payload |
| Batch too large | Boundary check | `413` | Caller chunks |
| OOM during inference | TF raises `ResourceExhaustedError` | `500` (default) | Pod gets restarted by k8s liveness |

## Concurrency

* `ModelRegistry` uses an `RLock` around `set / get / status`.
* Reads (`/predict`, `/info`, `/health`) acquire the lock for the duration
  of the model handle access (microseconds) — the heavy TF call happens
  outside the lock.
* `POST /api/load` and `/api/reset` are **write** operations and block
  concurrent reads only momentarily during the swap.

## Observability hooks

* Structured stdout logs (`kanx.train`, `kanx.inference`, `kanx.cli`,
  `[kanx-api] initialised from …`).
* `inference_ms` returned on every `/api/predict` response for client-side
  histogram metrics.
* Prometheus `/metrics` endpoint is implemented by the FastAPI service and documented in `docs/deployment.md`.

