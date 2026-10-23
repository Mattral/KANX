# Benchmark: KAN vs MLP — fair, multi-baseline

> Reproduce with `python benchmarks/compare_mlp.py`.

## Setup

- **Target.** `y = sin(π·x₁) + cos(2π·x₂)` — *deliberately smooth & separable*; this is the regime where KANs are theoretically optimal (Liu et al. 2024). **Real-world targets are not this smooth.**
- **Data.** 4096 train / 1024 test, uniform on `[-1, 1]²`, train seed 0 and test seed 1, shared across init seeds [0, 1, 2].
- **Training.** Adam(lr=1e-2), batch=128, **100 epochs** (fixed).
- **Inference.** 10 untimed forwards, then the median of 20. Each sample ends in `.numpy()`, so the time includes finished compute. The spread is the population standard deviation (`ddof=0`) across init seeds.
- **Hardware.** Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz / AMD64 / Windows / Python 3.11.17 / TF 2.21.0. TF intra-op threads 0, inter-op threads 0. CPU.

## Results

| Model | Params | Train (s) | Infer 4k CPU (ms) | Infer 4k GPU (ms) | Train MSE | **Test MSE** |
|---|---:|---:|---:|---:|---:|---:|
| KAN[2,16,1] | 432 | 9.42 ± 0.12 | 26.72 ± 0.31 | N/A | 2.67e-04 ± 1.79e-04 | **2.58e-04 ± 1.71e-04** |
| KAN[2,32,1] | 864 | 11.12 ± 0.18 | 29.15 ± 0.46 | N/A | 1.06e-04 ± 6.16e-05 | **1.09e-04 ± 6.34e-05** |
| MLP[2,32,1] | 129 | 6.09 ± 0.07 | 2.06 ± 0.03 | N/A | 4.75e-01 ± 2.21e-03 | **4.62e-01 ± 1.40e-03** |
| MLP[2,16,16,1] | 337 | 6.78 ± 0.23 | 2.90 ± 0.05 | N/A | 1.35e-03 ± 3.93e-04 | **1.44e-03 ± 3.83e-04** |
| MLP[2,64,64,1] | 4417 | 7.00 ± 0.07 | 3.53 ± 0.05 | N/A | 6.44e-04 ± 4.29e-04 | **6.43e-04 ± 4.02e-04** |

## What this benchmark honestly shows

- On a **smooth separable 2-D regression**, parameter-matched KAN and MLP are roughly comparable, with KANs sometimes winning by a small margin.
- The previous headline claim ('265× lower MSE than MLP[2,64,64,1]') compared KAN[2,32,1] (864 params) against a deliberately **5× over-parameterised** MLP that was trained for only 30 epochs. That comparison was unfair on two axes.
- **Compute cost.** On this run, warmed inference for KAN[2,16,1] is 7.57× MLP[2,64,64,1]. That ratio is this machine and this protocol (3 init seed(s), 10 warmup forwards, median of 20). The older ~3–5× line described an unwarmed single call and is withdrawn.

## Caveats

- This benchmark is **best-case** for KANs (the target is exactly the kind of function the Kolmogorov-Arnold representation theorem applies to). On real tabular or vision data, the picture is far more nuanced.
- We do **not** claim KANs are universally better than MLPs.
- For non-smooth or high-dimensional targets, an MLP will typically beat a same-size KAN on both accuracy and throughput.
- The 0.2.0 tables each published one unwarmed forward. Those columns disagreed (README 68.64 ms vs 25.52 ms; the previous `benchmarks/results.md` 43.59 ms vs 31.61 ms) and are not averaged into this table.
