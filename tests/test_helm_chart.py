"""The Helm chart parameterizes k8s/, and the raw manifests still parse."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "charts" / "kanx"
K8S = ROOT / "k8s"


def _load(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_chart_metadata_and_defaults() -> None:
    chart = _load(CHART / "Chart.yaml")
    values = _load(CHART / "values.yaml")
    assert chart["apiVersion"] == "v2"
    assert chart["name"] == "kanx"
    assert chart["type"] == "application"
    assert chart["version"]
    assert chart["appVersion"] == "0.1.9"
    assert values["replicaCount"] == 2
    assert values["image"] == {
        "repository": "ghcr.io/mattral/kanx",
        "tag": "latest",
        "pullPolicy": "IfNotPresent",
    }
    assert values["resources"]["requests"]["cpu"] == "500m"
    assert values["resources"]["requests"]["memory"] == "1Gi"
    assert values["resources"]["limits"]["cpu"] == "2"
    assert values["resources"]["limits"]["memory"] == "4Gi"
    assert values["probes"]["readiness"]["path"] == "/api/health"
    assert values["probes"]["readiness"]["initialDelaySeconds"] == 10
    assert values["probes"]["liveness"]["path"] == "/api/health"
    assert values["probes"]["liveness"]["initialDelaySeconds"] == 30
    assert values["autoscaling"]["enabled"] is True
    assert values["autoscaling"]["minReplicas"] == 2
    assert values["autoscaling"]["maxReplicas"] == 10
    assert values["autoscaling"]["targetCPUUtilizationPercentage"] == 70
    assert values["persistence"]["enabled"] is True
    assert values["persistence"]["size"] == "1Gi"
    assert values["persistence"]["mountPath"] == "/mnt/models"
    assert values["persistence"]["readOnly"] is True
    assert values["ingress"]["enabled"] is True
    assert values["ingress"]["className"] == "nginx"
    assert values["ingress"]["hosts"][0]["host"] == "kanx.example.com"
    assert values["env"]["KANX_CHECKPOINT"] == "/mnt/models/kanx_model.keras"
    assert values["env"]["KANX_MAX_BATCH"] == "4096"


def test_templates_reference_parameterized_settings() -> None:
    names = [
        "deployment.yaml",
        "service.yaml",
        "pvc.yaml",
        "hpa.yaml",
        "ingress.yaml",
        "_helpers.tpl",
        "NOTES.txt",
    ]
    present = {path.name for path in (CHART / "templates").iterdir()}
    assert set(names) <= present
    deployment = (CHART / "templates" / "deployment.yaml").read_text(encoding="utf-8")
    assert ".Values.image.repository" in deployment
    assert ".Values.image.tag" in deployment
    assert ".Values.replicaCount" in deployment
    assert ".Values.resources" in deployment
    assert ".Values.probes.readiness" in deployment
    assert ".Values.probes.liveness" in deployment
    assert ".Values.persistence.mountPath" in deployment
    hpa = (CHART / "templates" / "hpa.yaml").read_text(encoding="utf-8")
    assert ".Values.autoscaling.minReplicas" in hpa
    assert ".Values.autoscaling.maxReplicas" in hpa
    assert ".Values.autoscaling.targetCPUUtilizationPercentage" in hpa
    pvc = (CHART / "templates" / "pvc.yaml").read_text(encoding="utf-8")
    assert ".Values.persistence.size" in pvc
    assert ".Values.persistence.storageClass" in pvc
    ingress = (CHART / "templates" / "ingress.yaml").read_text(encoding="utf-8")
    assert ".Values.ingress.hosts" in ingress
    assert ".Values.service.port" in ingress


def test_raw_manifests_still_parse() -> None:
    docs: list[dict] = []
    for path in sorted(K8S.glob("*.yaml")):
        loaded = yaml.safe_load_all(path.read_text(encoding="utf-8"))
        docs.extend(doc for doc in loaded if isinstance(doc, dict))
    by_kind = {doc["kind"]: doc for doc in docs}
    assert set(by_kind) == {
        "Deployment",
        "Service",
        "Ingress",
        "HorizontalPodAutoscaler",
        "PersistentVolumeClaim",
    }
    deploy = by_kind["Deployment"]
    container = deploy["spec"]["template"]["spec"]["containers"][0]
    assert deploy["metadata"]["name"] == "kanx-api"
    assert deploy["spec"]["replicas"] == 2
    assert container["image"] == "ghcr.io/mattral/kanx:latest"
    assert container["resources"]["requests"]["cpu"] == "500m"
    assert container["resources"]["limits"]["memory"] == "4Gi"
    assert container["readinessProbe"]["httpGet"]["path"] == "/api/health"
    assert container["livenessProbe"]["httpGet"]["path"] == "/api/health"
    volumes = deploy["spec"]["template"]["spec"]["volumes"]
    assert volumes[0]["persistentVolumeClaim"]["claimName"] == "kanx-models-pvc"
    assert by_kind["Service"]["spec"]["ports"][0]["port"] == 80
    assert by_kind["Service"]["spec"]["ports"][0]["targetPort"] == 8000
    pvc = by_kind["PersistentVolumeClaim"]
    assert pvc["metadata"]["name"] == "kanx-models-pvc"
    assert pvc["spec"]["resources"]["requests"]["storage"] == "1Gi"
    hpa = by_kind["HorizontalPodAutoscaler"]
    assert hpa["spec"]["minReplicas"] == 2
    assert hpa["spec"]["maxReplicas"] == 10
    assert hpa["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 70
    ingress = by_kind["Ingress"]
    assert ingress["spec"]["rules"][0]["host"] == "kanx.example.com"
    assert ingress["spec"]["ingressClassName"] == "nginx"


def _helm() -> str | None:
    return shutil.which("helm")


def _render(helm: str, *args: str) -> list[dict]:
    result = subprocess.run(
        [helm, "template", "kanx-api", str(CHART), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


@pytest.mark.skipif(_helm() is None, reason="helm is not installed")
def test_helm_lint_and_default_render() -> None:
    helm = _helm()
    assert helm is not None
    lint = subprocess.run([helm, "lint", str(CHART)], check=False, capture_output=True, text=True)
    assert lint.returncode == 0, lint.stdout + lint.stderr
    docs = _render(helm)
    by_kind = {doc["kind"]: doc for doc in docs}
    assert set(by_kind) == {
        "PersistentVolumeClaim",
        "Service",
        "Deployment",
        "HorizontalPodAutoscaler",
        "Ingress",
    }
    deploy = by_kind["Deployment"]
    container = deploy["spec"]["template"]["spec"]["containers"][0]
    assert deploy["metadata"]["name"] == "kanx-api"
    assert deploy["spec"]["replicas"] == 2
    assert container["image"] == "ghcr.io/mattral/kanx:latest"
    assert container["resources"]["requests"]["cpu"] == "500m"
    assert container["resources"]["requests"]["memory"] == "1Gi"
    assert str(container["resources"]["limits"]["cpu"]) == "2"
    assert container["resources"]["limits"]["memory"] == "4Gi"
    assert container["readinessProbe"]["httpGet"] == {"path": "/api/health", "port": 8000}
    assert container["livenessProbe"]["initialDelaySeconds"] == 30
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["KANX_CONFIG"] == "/app/configs/default.yaml"
    assert env["KANX_CHECKPOINT"] == "/mnt/models/kanx_model.keras"
    assert env["KANX_MAX_BATCH"] == "4096"
    claim = container["volumeMounts"][0]
    assert claim["mountPath"] == "/mnt/models"
    assert claim["readOnly"] is True
    pvc = by_kind["PersistentVolumeClaim"]
    assert pvc["metadata"]["name"] == "kanx-api-models"
    assert pvc["spec"]["resources"]["requests"]["storage"] == "1Gi"
    assert "storageClassName" not in pvc["spec"]
    hpa = by_kind["HorizontalPodAutoscaler"]
    assert hpa["spec"]["scaleTargetRef"]["name"] == "kanx-api"
    assert hpa["spec"]["minReplicas"] == 2
    assert hpa["spec"]["maxReplicas"] == 10
    assert hpa["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 70
    ingress = by_kind["Ingress"]
    assert ingress["spec"]["ingressClassName"] == "nginx"
    assert ingress["spec"]["rules"][0]["host"] == "kanx.example.com"
    backend = ingress["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]
    assert backend["name"] == "kanx-api"
    assert backend["port"]["number"] == 80
    service = by_kind["Service"]
    assert service["spec"]["ports"][0]["port"] == 80
    assert service["spec"]["ports"][0]["targetPort"] == 8000
    assert service["metadata"]["annotations"]["prometheus.io/path"] == "/metrics"


@pytest.mark.skipif(_helm() is None, reason="helm is not installed")
def test_helm_can_disable_objects_and_reuse_a_claim() -> None:
    helm = _helm()
    assert helm is not None
    off = _render(
        helm,
        "--set",
        "persistence.enabled=false",
        "--set",
        "ingress.enabled=false",
        "--set",
        "autoscaling.enabled=false",
    )
    assert {doc["kind"] for doc in off} == {"Service", "Deployment"}
    deploy = next(doc for doc in off if doc["kind"] == "Deployment")
    assert deploy["spec"]["replicas"] == 2
    assert deploy["spec"]["template"]["spec"]["volumes"][0]["emptyDir"] == {}

    reused = _render(helm, "--set", "persistence.existingClaim=my-claim")
    assert "PersistentVolumeClaim" not in {doc["kind"] for doc in reused}
    deploy = next(doc for doc in reused if doc["kind"] == "Deployment")
    claim = deploy["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]
    assert claim["claimName"] == "my-claim"

    forced = _render(helm, "--set", "persistence.storageClass=-")
    pvc = next(doc for doc in forced if doc["kind"] == "PersistentVolumeClaim")
    assert pvc["spec"]["storageClassName"] == ""
