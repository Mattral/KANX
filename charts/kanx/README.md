# kanx Helm chart

Parameterized form of the raw manifests in [`k8s/`](../../k8s/). Those files still apply with `kubectl apply -f k8s/`. Use either the chart or the raw manifests in a namespace. A release named `kanx-api` creates the same Deployment, Service, Ingress, and HPA names as the raw files, so applying both will conflict. The chart claim is `<release>-models` (`kanx-api-models`). The raw claim is `kanx-models-pvc`.

## Install

```bash
helm install kanx-api charts/kanx
helm upgrade --install kanx-api charts/kanx --set image.tag=0.1.9
helm template kanx-api charts/kanx
```

The model checkpoint is mounted from a PersistentVolumeClaim at `/mnt/models`. The container runs as UID 1000 with a read-only root filesystem. `/tmp` and `/home/kanx/.cache` are emptyDirs.

## Values

| Value | Default | Purpose |
|---|---|---|
| `image.repository` / `image.tag` / `image.pullPolicy` | `ghcr.io/mattral/kanx` / `latest` / `IfNotPresent` | Serving image |
| `replicaCount` | `2` | Deployment replicas. A Helm upgrade writes this count back while the HPA is also applied, so keep it equal to `autoscaling.minReplicas` |
| `resources` | 500m/1Gi request, 2 CPU/4Gi limit | Container resources |
| `probes.readiness` / `probes.liveness` | `/api/health` on port 8000 | Same timings as `k8s/deployment.yaml` |
| `strategy` | RollingUpdate, maxSurge 1, maxUnavailable 0 | Zero-downtime rollout |
| `autoscaling.enabled` | `true` | CPU HPA, 2–10 replicas, 70% target |
| `persistence.enabled` / `size` / `storageClass` | `true` / `1Gi` / empty | PVC. Empty storage class uses the cluster default. `"-"` sets `storageClassName: ""`. `existingClaim` skips creating a PVC |
| `ingress.enabled` / `ingress.hosts` | `true` / `kanx.example.com` | nginx Ingress, 8m body, 60s read timeout |
| `env` | config path, checkpoint path, batch cap, TF and OpenMP settings | Container environment |
| `service` | ClusterIP port 80 → 8000 | Prometheus scrape annotations on port 8000 |

Set `persistence.enabled=false` to mount an emptyDir instead of a claim. The pod still starts, and the checkpoint does not survive a restart.

CPU HPA is a starter signal. For latency-sensitive inference, replace it with a custom metric. See `docs/deployment.md`.
