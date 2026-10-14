# Build & Run

## Local install

Requires **Python 3.10+** and `pip >= 23`.

```bash
git clone <repo>
cd kanx
pip install -e .[dev,api]    # editable install + dev + api extras
```

Smoke-test:

```bash
python -c "from kanx import KAN; import tensorflow as tf; print(KAN([2,8,1])(tf.zeros((1,2))))"
```

## Run the test suite

```bash
bash scripts/test.sh
# or, equivalently:
pytest tests/ -v --cov=src/kanx
```

Expected: **113 passed, ~94% coverage** on the library.

## Train from the CLI

```bash
bash scripts/train.sh                    # uses configs/default.yaml
bash scripts/train.sh configs/mnist.yaml # custom config
```

The checkpoint is written to `checkpoints/kanx_model.keras` by default
(configurable in YAML).

## Benchmark

```bash
bash scripts/benchmark.sh                # regenerates benchmarks/results.md
```

## Serve the REST API locally

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
curl http://localhost:8000/api/health
```

## Docker

```bash
docker build -t kanx:latest .
docker run --rm -p 8000:8000 \
    -e KANX_CHECKPOINT=/app/checkpoints/kanx_model.keras \
    -v $(pwd)/checkpoints:/app/checkpoints \
    kanx:latest
```

Or via compose:

```bash
docker compose up --build
```

## Kubernetes

```bash
docker build -t kanx:latest .
kind load docker-image kanx:latest    # if you use kind
kubectl apply -f k8s/
# or: helm install kanx-api charts/kanx
kubectl port-forward svc/kanx-api 8000:80
curl http://localhost:8000/api/health
```

See `k8s/` — Deployment with rolling updates, Service, Ingress, HPA, PVC for
model storage. `charts/kanx` is the parameterized form of those objects. Use
one install path in a namespace.

## Environment variables

| Variable             | Default                              | Used by             |
|----------------------|--------------------------------------|---------------------|
| `KANX_CONFIG`        | `configs/default.yaml`               | `api/app.py` startup |
| `KANX_CHECKPOINT`    | `checkpoints/kanx_model.keras`       | `api/app.py` startup |
| `KANX_MAX_BATCH`     | `4096`                               | `/api/predict` guard |
| `TF_CPP_MIN_LOG_LEVEL` | `2` (recommended in prod)          | TensorFlow stderr noise |

## Reproducibility

```python
from kanx import set_global_seed
set_global_seed(42)
```

Seeds Python `random`, `numpy`, `tf.random` and `tf.keras.utils.set_random_seed`
in one call. Called automatically by `kanx.train`.

