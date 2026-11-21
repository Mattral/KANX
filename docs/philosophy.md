# Philosophy

## What kanx is

`kanx` is a Python library for Kolmogorov-Arnold Networks (KANs, Liu et al.
2024). TensorFlow is the primary backend and PyTorch is the second. It adds
ONNX export, a FastAPI service, a CLI, and Docker and Kubernetes configs.

The library that ships in this repo is the same library that runs in the
Docker image, that is served by FastAPI, that is tested by CI, that is
benchmarked against MLP baselines. **One artifact, many surfaces.**

## Design principles

1. **One canonical surface per concept.** Building a model: `KAN([...])` or
   from a dict-config — that's it. Loading a checkpoint:
   `kanx.load_model(path)`. Training: `kanx.train(cfg, X, y)`. Avoid four
   helpers that do almost the same thing.
2. **Fail loud at the boundary.** All public functions validate their inputs
   and raise `ValueError` / `FileNotFoundError` / `HTTPException` with a
   precise message. No silent fallbacks, no "best effort" guesses.
3. **Tiny dependency surface in the core.** The library depends only on
   `tensorflow`, `numpy`, `pyyaml`. FastAPI / Pydantic are *optional extras*
   for the API surface. This makes `pip install kanx` a 30-second proposition.
4. **Vectorize everything.** The B-spline basis is computed with a single
   broadcasted Cox-de Boor recursion (no Python loop over features or
   batch). Anything that touches the hot path is benchmarked.
5. **Numerical contracts have tests.** Partition-of-unity for B-splines,
   gradient flow for `KANLinear`, save/load roundtrip for `KAN`. Maths gets
   a test, not just a docstring.
6. **Determinism is a feature.** `kanx.set_global_seed(s)` seeds Python,
   NumPy and TensorFlow RNGs in one call; `train()` calls it automatically.
7. **The API is contractual.** `/api/predict` checks `in_features`, max batch
   size, and tensor rank *at the boundary* — not deep inside Keras.

## Non-goals

- **Pykan parity.** kanx implements the same maths but is its own library,
  optimised for the TF ecosystem. We will not chase 100% behavioral identity.
- **CUDA custom kernels.** Out of scope for v0.1 — the einsum path is fast
  enough for low-D smooth tasks. Tracked in `roadmap.md`.
- **Auto-magic everything.** No autotuners, no auto-architecture search.
  Users own their config.

## What we borrow from FAANG / OpenAI / Anthropic style

- A `roadmap.md` that's the source of truth and lives with the code.
- A test pyramid (unit → integration → end-to-end via TestClient) with
  precise coverage targets per layer.
- A clear "release artifact" (Docker image) that mirrors local dev.
- Observability hooks documented in `deployment.md`.
- Explicit configuration via 12-factor env vars (`KANX_CONFIG`,
  `KANX_CHECKPOINT`, `KANX_MAX_BATCH`).

