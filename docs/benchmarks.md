# Benchmark: KAN vs MLP — fair, multi-baseline

> Reproduce with `python benchmarks/compare_mlp.py`.

## Setup

- **Target.** `y = sin(π·x₁) + cos(2π·x₂)` — *deliberately smooth & separable*; this is the regime where KANs are theoretically optimal (Liu et al. 2024). **Real-world targets are not this smooth.**
- **Data.** 4 096 train / 1 024 test, uniform on `[-1, 1]²`, seed=0.
- **Training.** Adam(lr=1e-2), batch=128, **100 epochs** (fixed).
- **Hardware.** aarch64 / Linux / Python 3.11.15 / TF 2.21.0, CPU.

## Results

| Model            | Params | Train (s) | Infer 4k (ms) | Train MSE | **Test MSE** |
|------------------|------:|---------:|-------------:|---------:|-------------:|
| KAN[2,16,1]      |   432 |    12.50 |       68.64 | 2.17e-05 | **2.14e-05** |
| KAN[2,32,1]      |   864 |    16.62 |       25.52 | 4.50e-04 | **4.44e-04** |
| MLP[2,32,1]      |   129 |     5.07 |        6.17 | 4.74e-01 | **4.61e-01** |
| MLP[2,16,16,1]   |   337 |     5.46 |        4.08 | 1.44e-03 | **1.60e-03** |
| MLP[2,64,64,1]   |  4417 |     6.00 |        5.74 | 4.76e-04 | **5.51e-04** |

## What this benchmark honestly shows

- On a **smooth separable 2-D regression**, parameter-matched KAN and MLP are roughly comparable, with KANs sometimes winning by a small margin.
- The previous headline claim ('265× lower MSE than MLP[2,64,64,1]') compared KAN[2,32,1] (864 params) against a deliberately **5× over-parameterised** MLP that was trained for only 30 epochs. That comparison was unfair on two axes.
- **Compute cost.** KAN inference is consistently ~3–5× slower than an equivalent-MSE MLP on CPU, because per-edge B-spline evaluation does more work per parameter than a matmul + activation.

## Caveats

- This benchmark is **best-case** for KANs (the target is exactly the kind of function the Kolmogorov-Arnold representation theorem applies to). On real tabular or vision data, the picture is far more nuanced.
- We do **not** claim KANs are universally better than MLPs.
- For non-smooth or high-dimensional targets, an MLP will typically beat a same-size KAN on both accuracy and throughput.

## Real-world tabular benchmarks (new)

We ran 5-fold cross-validated real-world tabular benchmarks on three UCI datasets (California Housing, Concrete Strength, Energy Efficiency) using the TensorFlow KAN implementation. PyTorch benchmarks are disabled on CPU-only environments due to TensorFlow/PyTorch CUDA initialization conflicts; see the benchmark artifact for full details.

Summary (median across folds):

| Dataset | Model | Params | Train (s) | RMSE mean ± std | R² mean ± std | CPU latency (ms) |
|---|---|---:|---:|---|---|---:|
| california | KAN_TF | 2592 | 3.19 | 0.46799 ± 0.01410 | 0.78077 ± 0.01257 | 17.85 |
| concrete | KAN_TF | 2592 | 1.37 | 0.38078 ± 0.03534 | 0.85409 ± 0.01255 | 17.99 |
| energy | KAN_TF | 2592 | 1.37 | 0.19694 ± 0.01527 | 0.96056 ± 0.00605 | 17.82 |

Full results (JSON) and system metadata are saved as an artifact in the repository:

- [benchmarks/results/real_world_results.json](benchmarks/results/real_world_results.json)

Notes:
- These experiments were run on a CPU-only environment, so the committed table is TensorFlow-only.
- On a GPU machine, `python benchmarks/real_world.py --device cuda` still trains the TensorFlow KAN in this process and trains the PyTorch KAN and MLP in `benchmarks/torch_worker.py`. The child does not import TensorFlow, which avoids the TensorFlow + PyTorch CUDA segfault.
- `benchmarks/compare_mlp.py` never imports PyTorch, so that comparison stays in-process.
- CPU runs and `--smoke` do not start the PyTorch child.

GPU timing note:
- The micro-benchmarks now record both CPU and GPU inference latencies where available. Run the canonical benchmark on a CUDA-enabled host to populate the `Infer 4k GPU (ms)` column:

```bash
PYTHONPATH=src python benchmarks/compare_mlp.py    # CPU-only or GPU if available
PYTHONPATH=src python benchmarks/compare_mlp.py --long  # convergence runs
```

On a CUDA runner, `benchmarks/real_world.py --device cuda` starts the PyTorch models in a separate process. `compare_mlp.py` only times the TensorFlow GPU when TensorFlow can see one.

## CPU smoke gate

GitHub Actions job `benchmark-smoke` runs a tiny CPU configuration so a broken training path fails CI without the 100-epoch suite:

```bash
python benchmarks/compare_mlp.py --smoke
python benchmarks/real_world.py --smoke
```

`--smoke` prints the metrics and, unless you pass `--output`, does not overwrite `benchmarks/results.md` or `benchmarks/results/real_world_results.json`. The real-world smoke loads the first 256 rows of California Housing and needs `scikit-learn`. The process exits non-zero when a metric is missing or non-finite.


