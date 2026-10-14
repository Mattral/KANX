# Architecture

## Package layout

```
kanx/
├── src/kanx/             # the installable library (`pip install kanx`)
│   ├── __init__.py       # public API re-exports
│   ├── __main__.py       # CLI: `python -m kanx {info,train,predict}`
│   ├── layers.py         # KANLinear + B-spline primitives
│   ├── model.py          # KAN sequential model + builder
│   ├── train.py          # train() pipeline + optimiser/loss factories
│   ├── inference.py      # predict() + save_model / load_model
│   ├── config.py         # KanxConfig dataclasses + YAML validator
│   ├── utils.py          # logging, seeding, path helpers
│   └── torch/            # PyTorch parallel surface
│       ├── __init__.py
│       ├── layers.py
│       ├── model.py
│       ├── trainer.py
│       ├── matrix_kan.py
│       └── symbolic.py
│
├── api/                  # serving layer (depends on kanx, not vice-versa)
│   ├── __init__.py
│   └── app.py            # FastAPI app + ModelRegistry
│
├── backend/server.py     # supervisor entrypoint — re-exports api.app
│
├── tests/                # pytest suite
├── examples/             # runnable demos
├── benchmarks/           # KAN vs MLP harness
├── configs/              # YAML training configs
├── scripts/              # train.sh / benchmark.sh / test.sh
├── docs/                 # MkDocs source and reference docs
├── charts/kanx/          # Helm chart for the API
├── k8s/                  # Kubernetes manifests
├── Dockerfile            # release image
└── pyproject.toml        # build / install / pytest / ruff / black config
```

## Module contracts

### `kanx.layers`

```python
extend_grid(grid: tf.Tensor[F, G+1], k: int) -> tf.Tensor[F, G+1+2k]
b_spline_basis(x: tf.Tensor[B, F], grid_ext: tf.Tensor[F, G_ext], k: int)
    -> tf.Tensor[B, F, G_ext - k - 1]   # = (B, F, grid_size + k)
KANLinear(Layer)   # call: (B, in_features) -> (B, out_features)
```

Pre-conditions: `k >= 0`, `grid` is a uniformly spaced rank-2 tensor,
`x.dtype == grid.dtype`. Post-conditions: basis values are non-negative
and sum to 1 inside the inner knot range (asserted by tests).

### `kanx.model`

```python
KAN(layers: list[int] | list[dict], **default_layer_kwargs)
build_kan(layers, *, grid_size, spline_order, base_activation, ...) -> KAN
KAN.predict_tensor(x) -> tf.Tensor   # bypasses Keras progress UI
```

### `kanx.config`

```python
load_config(path: str) -> KanxConfig
validate_config(raw: dict) -> KanxConfig
KanxConfig = ModelConfig + TrainingConfig + CheckpointConfig
```

`validate_config` raises `ValueError` with a precise message on any schema
violation. Always preferred over silent defaults.

### `kanx.train`

```python
train(cfg: KanxConfig | dict, X, y, *, verbose=1, extra_callbacks=None)
    -> (KAN, tf.keras.callbacks.History)
build_optimizer(name, lr) -> tf.keras.optimizers.Optimizer
build_loss(name) -> tf.keras.losses.Loss
```

Side effects: seeds RNGs deterministically; writes a checkpoint to
`cfg.checkpoint.dir/cfg.checkpoint.filename`.

### `kanx.inference`

```python
load_model(path) -> tf.keras.Model         # auto-discovers KAN/KANLinear
save_model(model, path) -> str
predict(model, x, *, batch_size=None) -> np.ndarray
```

## Dependency graph

```
       ┌────────────┐
       │  api.app   │  (FastAPI surface)
       └─────┬──────┘
             │ imports
             ▼
       ┌──────────────┐
       │  kanx (lib)  │
       └─────┬────────┘
             │
       ┌─────┴────┬──────────┬──────────┬────────────┐
       ▼          ▼          ▼          ▼            ▼
    layers     model      train     inference  config / utils
       │          │          │          │
       └─── tensorflow / numpy / pyyaml (only core deps) ───┘
```

The library does **not** import FastAPI / Pydantic. The API package does
**not** mutate library state at import time (the model is loaded inside
`@app.on_event("startup")`).

## Key design choices

| Choice | Rationale |
|---|---|
| `tf.keras` Sequential KAN | Out-of-the-box Keras callbacks, save/load, distribute. |
| Vectorized B-spline via einsum | 1 line, no Python loop, JIT-friendly. |
| Per-feature grids | Required for future adaptive grid updates (pykan parity). |
| `@register_keras_serializable` | Custom layer survives `model.save() / load_model()`. |
| `ModelRegistry` mutex | Hot-swap checkpoints without restarting the API. |
| Hand-rolled YAML validator | Zero pydantic dep in the core lib. |
| 12-factor env vars on the API | `KANX_CONFIG`, `KANX_CHECKPOINT`, `KANX_MAX_BATCH`. |

