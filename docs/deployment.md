# Deployment

## CI/CD (`.github/workflows/ci.yml`)

Two jobs run on every push / PR:

1. **lint-and-test** (matrix: Python 3.10 / 3.11 / 3.12)
   * `pip install -e .[dev,api]`
   * `ruff check src tests api benchmarks examples`
   * `black --check` (warning-only)
   * `pytest tests/ -v --cov=src/kanx --cov-report=xml`
   * Uploads `coverage.xml` artifact on Python 3.11.
2. **docker** (after tests pass)
   * `docker build -t kanx:ci .`
   * Boots the container, polls `/api/health` until ready, hits `/api/info`.

This gives the same `green check` semantics as the largest open-source ML
projects: every PR has a compiled, served, predict-able artifact.

## Release artifact

The Docker image is the canonical release artifact. It contains:

* The library installed from source (`pip install -e .`).
* The FastAPI app exposed on `:8000`.
* `configs/default.yaml` as the fallback architecture.
* A `HEALTHCHECK` that polls `/api/health`.

Tagging convention (recommended once the repo goes public):
`kanx:0.1.0`, `kanx:0.1.0-cpu`, `kanx:latest` → push to GHCR via a
`release.yml` workflow that triggers on `v*` tags (P1 in `roadmap.md`).

## Kubernetes rollout

`k8s/deployment.yaml` ships with:

* `replicas: 2` for HA.
* `RollingUpdate` with `maxSurge: 1`, `maxUnavailable: 0` (zero downtime).
* `readinessProbe` on `/api/health` — pods don't receive traffic until the
  model is loaded.
* `livenessProbe` on `/api/health` with a longer initial delay — kills pods
  whose TF state has gone bad.
* CPU requests/limits sized for the default `KAN[2,64,64,1]`. Adjust for
  larger models.

`k8s/service.yaml` includes the `HorizontalPodAutoscaler` (CPU-target 70%,
2 ↔ 10 replicas) and the PVC that mounts the model.

`k8s/ingress.yaml` exposes the service via the `nginx` IngressClass — wire
into your existing TLS termination.

## Helm

```bash
helm install kanx-api charts/kanx
helm upgrade --install kanx-api charts/kanx --set image.tag=0.2.1
```

`charts/kanx` parameterizes the image, replica count, resources, probes, CPU
HPA, model PVC, and Ingress. Defaults match `k8s/`. The raw manifests still
apply with `kubectl apply -f k8s/`. Do not install release `kanx-api` into a
namespace that already has those objects: the Deployment, Service, Ingress,
and HPA names match. The chart claim is `kanx-api-models`. The raw claim is
`kanx-models-pvc`.

A Helm upgrade writes `replicaCount` back onto the Deployment. Keep it equal
to `autoscaling.minReplicas` (both default to 2). CPU HPA is still a starter
signal for latency-sensitive inference. Value reference:
`charts/kanx/README.md`.

## Rolling out a new model

Two supported flows:

### A. Rebuild + redeploy (immutable)
1. Bake the new `kanx_model.keras` into the image (`COPY checkpoints/`).
2. Tag, push, `kubectl set image deployment/kanx-api kanx-api=kanx:0.1.1`.
3. Pods drain via `RollingUpdate`.

### B. Hot-swap via PVC + `/api/load` (zero-restart)
1. Write the new checkpoint to the PVC (`kubectl cp` or a sidecar job).
2. Iterate over pods (`kubectl get pods -l app=kanx -o name | xargs ...`),
   POSTing `/api/load` to each.
3. `ModelRegistry`'s RLock guarantees in-flight requests on the old model
   finish before the swap completes.

Flow B is recommended when checkpoints are large (GBs) and image rebuilds
would be slow.

## Observability hooks

| Signal | Where |
|---|---|
| Liveness | `/api/health.status` |
| Model source | `/api/health.source` (`fresh:…` or `checkpoint:…`) |
| Per-request latency | `PredictResponse.inference_ms` |
| TF logs | container stdout (set `TF_CPP_MIN_LOG_LEVEL=2/3` in prod) |
| Library logs | `kanx.train`, `kanx.inference`, `kanx.cli` via stdlib `logging` |

Production observability is available today:

* Prometheus `/metrics` endpoint via `prometheus-fastapi-instrumentator`.
* Kubernetes service annotations and scrape config are already included.
* Structured JSON logs and OpenTelemetry traces remain future work.

### Prometheus scrape config

```yaml
scrape_configs:
  - job_name: kanx-api
    metrics_path: /metrics
    static_configs:
      - targets: ['kanx-api.default.svc.cluster.local:8000']
```

## Disaster recovery

* The image's `KANX_CONFIG` is a known-good architecture — if the PVC is
  unreadable, pods still come up serving a fresh-random model and `/health`
  reports `source: "fresh:…"`. Operators can re-mount and `/api/load`.
* `pytest` is part of the image's test layer (when built with the dev
  extras) so a `docker run --entrypoint pytest` is enough to validate a
  release candidate in a foreign environment.

