# Benchmark: KAN vs MLP — fair, multi-baseline

> Reproduce with `python benchmarks/compare_mlp.py`.

## Setup

- **Target.** `y = sin(π·x₁) + cos(2π·x₂)` — *deliberately smooth & separable*; this is the regime where KANs are theoretically optimal (Liu et al. 2024). **Real-world targets are not this smooth.**
- **Data.** 4 096 train / 1 024 test, uniform on `[-1, 1]²`, train seed 0 and test seed 1, shared across init seeds 0, 1, and 2.
- **Training.** Adam(lr=1e-2), batch=128, **100 epochs** (fixed).
- **Inference.** 10 untimed forwards, then the median of 20. Each sample ends in `.numpy()`. The ± is the population standard deviation across seeds.
- **Hardware.** Intel Core i9-9900 at 3.10 GHz / AMD64 / Windows / Python 3.11.17 / TF 2.21.0, CPU.

## Results

The canonical table is [`benchmarks/results.md`](https://github.com/Mattral/KANX/blob/main/benchmarks/results.md). Means from that run:

| Model | Params | Train (s) | Infer 4k (ms) | **Test MSE** |
|---|---:|---:|---:|---:|
| KAN[2,16,1] | 432 | 9.42 ± 0.12 | 26.72 ± 0.31 | 2.58×10⁻⁴ ± 1.71×10⁻⁴ |
| KAN[2,32,1] | 864 | 11.12 ± 0.18 | 29.15 ± 0.46 | **1.09×10⁻⁴ ± 6.34×10⁻⁵** |
| MLP[2,32,1] | 129 | 6.09 ± 0.07 | 2.06 ± 0.03 | 4.62×10⁻¹ ± 1.40×10⁻³ |
| MLP[2,16,16,1] | 337 | 6.78 ± 0.23 | 2.90 ± 0.05 | 1.44×10⁻³ ± 3.83×10⁻⁴ |
| MLP[2,64,64,1] | 4417 | 7.00 ± 0.07 | 3.53 ± 0.05 | 6.43×10⁻⁴ ± 4.02×10⁻⁴ |

## What this benchmark honestly shows

- On a **smooth separable 2-D regression**, the wider KAN has the lower mean test MSE. The smaller KAN's error moves by about as much as its mean across seeds.
- The previous headline claim ('265× lower MSE than MLP[2,64,64,1]') compared KAN[2,32,1] (864 params) against a deliberately **5× over-parameterised** MLP that was trained for only 30 epochs. That comparison was unfair on two axes.
- **Compute cost.** On this run, warmed inference for KAN[2,16,1] is 7.57× MLP[2,64,64,1]. That ratio is this machine and this protocol. The older ~3–5× line described an unwarmed single call and is withdrawn.

## Caveats

- This benchmark is **best-case** for KANs (the target is exactly the kind of function the Kolmogorov-Arnold representation theorem applies to). On real tabular or vision data, the picture is far more nuanced.
- We do **not** claim KANs are universally better than MLPs.
- For non-smooth or high-dimensional targets, an MLP will typically beat a same-size KAN on both accuracy and throughput.
- The 0.2.0 tables each published one unwarmed forward. Those columns disagreed with each other and are not averaged into this table.

## Real-world tabular benchmarks (new)

We ran 5-fold cross-validated real-world tabular benchmarks on three UCI datasets (California Housing, Concrete Strength, Energy Efficiency). CPU runs stay on the TensorFlow KAN. `python benchmarks/real_world.py --device cuda` trains the PyTorch KAN and MLP in a child process so TensorFlow and PyTorch do not initialize CUDA together.

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


