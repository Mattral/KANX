# 🚀 Introducing kanx — Production-Ready Kolmogorov-Arnold Networks

KANs are one of the most interesting deep-learning ideas in years —
**learnable spline activations on edges** instead of fixed nonlinearities on
nodes. The paper (Liu et al., 2024) showed they can hit MLP-beating accuracy
with a fraction of the parameters.

But every implementation I found was a research prototype: notebooks, no
tests, no API, no Docker, no benchmarks anyone could rerun. So I built one.

**`kanx`** is a KAN library with:

✔️ **Two backends**: TensorFlow (primary) and PyTorch (`kanx.torch`)
✔️ **ONNX export** from both — deploy anywhere
✔️ **REST API** (FastAPI) with hot-swap checkpoint loading
✔️ **Docker + Kubernetes** manifests with HPA + readiness probes
✔️ **95 tests, 94% coverage**, including property-based + numerical-contract tests
✔️ **CI/CD**: lint + test matrix (py3.10/3.11/3.12) + Docker smoke + tag-triggered PyPI + GHCR + Pages release

On a smooth 2-D regression task:

| Model            | Params | Test MSE |
|------------------|-------:|---------:|
| **KAN[2,16,1]**  | **432**  | **2.14 × 10⁻⁵** |
| MLP[2,16,16,1]   | 337  | 1.60 × 10⁻³ |
| MLP[2,64,64,1]   | 4 417 | 5.51 × 10⁻⁴ |

**Honest, param-matched, 100 epochs.** Best-case for KANs (smooth, separable target).

Install:
```
pip install kanx                 # core
pip install "kanx[torch]"        # +PyTorch
pip install "kanx[all]"          # everything
```

Try it in Colab (2 minutes): https://colab.research.google.com/github/Mattral/KANX/blob/main/notebooks/quickstart.ipynb

Repo: https://github.com/Mattral/KANX
Docs: https://mattral.github.io/KANX/

Looking for feedback, contributors, and real-world use cases.

#MachineLearning #DeepLearning #MLOps #PyTorch #TensorFlow #ONNX
