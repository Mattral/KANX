"""KAN vs MLP — fair, multi-baseline regression benchmark.

This script is the public-facing benchmark for `kanx`. It is designed to be
**honest**: same training budget, same optimizer, same target function,
**parameter-count-matched** MLP baselines (not a deliberately overparameterised
one), AND a convergence-matched run with patience-based early stopping.

Reproduce with::

    python benchmarks/compare_mlp.py             # 100 epochs, init seeds 0, 1 and 2
    python benchmarks/compare_mlp.py --long      # convergence run (1000 epochs + early-stop)
    python benchmarks/compare_mlp.py --smoke     # CPU smoke gate (does not rewrite results.md)

Outputs `benchmarks/results.md` with the full table + methodology.
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from kanx import KAN, set_global_seed


# ---------------------------------------------------------------------------
@dataclass
class Result:
    model: str
    params: int
    train_s: float
    infer_ms_4k_cpu: float
    infer_ms_4k_gpu: float | None
    train_mse: float
    test_mse: float
    train_s_std: float = 0.0
    infer_ms_4k_cpu_std: float = 0.0
    train_mse_std: float = 0.0
    test_mse_std: float = 0.0
    n_seeds: int = 1


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype("float32")
    y = (np.sin(np.pi * X[:, :1]) + np.cos(2 * np.pi * X[:, 1:2])).astype("float32")
    return X, y


def make_mlp(in_dim: int, hidden, out_dim: int = 1) -> tf.keras.Model:
    """`hidden` can be a single int or a tuple."""
    if isinstance(hidden, int):
        hidden = (hidden,)
    layers = [tf.keras.layers.Input(shape=(in_dim,))]
    for h in hidden:
        layers.append(tf.keras.layers.Dense(h, activation="silu"))
    layers.append(tf.keras.layers.Dense(out_dim))
    return tf.keras.Sequential(layers)


def n_params(model) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_variables))


def _population_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _cpu_name() -> str:
    name = platform.processor() or platform.machine()
    if platform.system() != "Windows":
        return name
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ],
            text=True,
            timeout=20,
        )
        line = next((item.strip() for item in out.splitlines() if item.strip()), "")
        if line:
            return line
    except (OSError, subprocess.SubprocessError):
        pass
    return name


def _synced_forward_ms(model, batch) -> float:
    """Time one forward through the NumPy copy, so the sample is finished work."""
    started = time.perf_counter()
    out = model(batch)
    _ = out.numpy()
    return (time.perf_counter() - started) * 1000.0


def _warmed_infer_ms(model, batch, *, warmup: int = 10, repeats: int = 20) -> float:
    for _ in range(warmup):
        _synced_forward_ms(model, batch)
    samples = [_synced_forward_ms(model, batch) for _ in range(repeats)]
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def fit_and_score(
    name: str, model: tf.keras.Model, X, y, X_test, y_test,
    *, epochs: int, lr: float, batch_size: int, early_stop_patience: int,
) -> Result:
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    callbacks = []
    if early_stop_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=early_stop_patience,
                restore_best_weights=True, verbose=0,
            )
        )
    t0 = time.perf_counter()
    model.fit(
        X, y, epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=callbacks,
    )
    train_s = time.perf_counter() - t0

    n_inf = min(4096, X_test.shape[0])
    infer_ms_cpu = _warmed_infer_ms(model, X_test[:n_inf])

    # GPU inference latency if GPU is available for TensorFlow
    infer_ms_gpu = None
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            with tf.device("/GPU:0"):
                # force eager sync by converting output to numpy
                for _ in range(10):
                    _ = model(X_test[:n_inf])
                t2 = time.perf_counter()
                out = model(X_test[:n_inf]).numpy() # noqa: F841
                infer_ms_gpu = (time.perf_counter() - t2) * 1000.0
    except Exception:
        infer_ms_gpu = None

    return Result(
        model=name,
        params=n_params(model),
        train_s=round(train_s, 3),
        infer_ms_4k_cpu=round(infer_ms_cpu, 3),
        infer_ms_4k_gpu=round(infer_ms_gpu, 3) if infer_ms_gpu is not None else None,
        train_mse=float(tf.reduce_mean((model(X)      - y     ) ** 2).numpy()),
        test_mse=float(tf.reduce_mean((model(X_test) - y_test) ** 2).numpy()),
    )


def _fmt_pm(mean: float, std: float, n_seeds: int, digits: int) -> str:
    if n_seeds <= 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _fmt_mse(mean: float, std: float, n_seeds: int) -> str:
    if n_seeds <= 1:
        return f"{mean:.2e}"
    return f"{mean:.2e} ± {std:.2e}"


def write_results(
    rows: list[Result],
    path: str,
    *,
    epochs: int,
    mode: str,
    seeds: tuple[int, ...] = (0,),
    n_train: int = 4096,
    n_test: int = 1024,
    batch_size: int = 128,
):
    if mode == "long":
        reproduce = "`python benchmarks/compare_mlp.py --long`"
    elif mode == "smoke":
        reproduce = "`python benchmarks/compare_mlp.py --smoke`"
    else:
        reproduce = "`python benchmarks/compare_mlp.py`"
    with open(path, "w") as f:
        f.write("# Benchmark: KAN vs MLP — fair, multi-baseline\n\n")
        if mode == "smoke":
            f.write("CPU smoke gate. This file is not the published benchmark table.\n\n")
        f.write(f"> Reproduce with {reproduce}.\n\n")
        f.write("## Setup\n\n")
        f.write("- **Target.** `y = sin(π·x₁) + cos(2π·x₂)` — *deliberately smooth & separable*; "
                "this is the regime where KANs are theoretically optimal (Liu et al. 2024). "
                "**Real-world targets are not this smooth.**\n")
        n_seeds = rows[0].n_seeds if rows else 1
        f.write(f"- **Data.** {n_train} train / {n_test} test, uniform on `[-1, 1]²`, "
                f"train seed 0 and test seed 1, shared across init seeds {list(seeds)}.\n")
        f.write(f"- **Training.** Adam(lr=1e-2), batch={batch_size}, **{epochs} epochs**"
                f"{' with EarlyStopping(patience=50)' if mode == 'long' else ' (fixed)'}.\n")
        f.write("- **Inference.** 10 untimed forwards, then the median of 20. "
                "Each sample ends in `.numpy()`, so the time includes finished compute. "
                "The spread is the population standard deviation (`ddof=0`) across init seeds.\n")
        intra = tf.config.threading.get_intra_op_parallelism_threads()
        inter = tf.config.threading.get_inter_op_parallelism_threads()
        f.write(f"- **Hardware.** {_cpu_name()} / {platform.machine()} / {platform.system()} / "
                f"Python {platform.python_version()} / TF {tf.__version__}. "
                f"TF intra-op threads {intra}, inter-op threads {inter}. CPU.\n\n")

        f.write("## Results\n\n")
        f.write("| Model | Params | Train (s) | Infer 4k CPU (ms) | Infer 4k GPU (ms) | Train MSE | **Test MSE** |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            gpu_str = (
                _fmt_pm(r.infer_ms_4k_gpu, 0.0, 1, 2)
                if r.infer_ms_4k_gpu is not None
                else "N/A"
            )
            f.write(
                f"| {r.model} | {r.params} | "
                f"{_fmt_pm(r.train_s, r.train_s_std, r.n_seeds, 2)} | "
                f"{_fmt_pm(r.infer_ms_4k_cpu, r.infer_ms_4k_cpu_std, r.n_seeds, 2)} | "
                f"{gpu_str} | {_fmt_mse(r.train_mse, r.train_mse_std, r.n_seeds)} | "
                f"**{_fmt_mse(r.test_mse, r.test_mse_std, r.n_seeds)}** |\n"
            )

        kan = next((r for r in rows if r.model == "KAN[2,16,1]"), None)
        mlp = next((r for r in rows if r.model == "MLP[2,64,64,1]"), None)
        if kan is not None and mlp is not None and mlp.infer_ms_4k_cpu > 0:
            ratio = kan.infer_ms_4k_cpu / mlp.infer_ms_4k_cpu
            cost = (
                f"- **Compute cost.** On this run, warmed inference for KAN[2,16,1] "
                f"is {ratio:.2f}× MLP[2,64,64,1]. That ratio is this machine and this protocol "
                f"({n_seeds} init seed(s), 10 warmup forwards, median of 20). "
                f"The older ~3–5× line described an unwarmed single call and is withdrawn.\n"
            )
        else:
            cost = (
                "- **Compute cost.** Inference in this file is a warmed median. "
                "Read the ratio from the table. The older ~3–5× line is withdrawn.\n"
            )

        f.write("\n## What this benchmark honestly shows\n\n")
        f.write("- On a **smooth separable 2-D regression**, parameter-matched KAN and MLP "
                "are roughly comparable, with KANs sometimes winning by a small margin.\n")
        f.write("- The previous headline claim ('265× lower MSE than MLP[2,64,64,1]') "
                "compared KAN[2,32,1] (864 params) against a deliberately **5× over-parameterised** "
                "MLP that was trained for only 30 epochs. That comparison was unfair on two axes.\n")
        f.write(cost + "\n")
        f.write("## Caveats\n\n")
        f.write("- This benchmark is **best-case** for KANs (the target is exactly the kind "
                "of function the Kolmogorov-Arnold representation theorem applies to). On real "
                "tabular or vision data, the picture is far more nuanced.\n")
        f.write("- We do **not** claim KANs are universally better than MLPs.\n")
        f.write("- For non-smooth or high-dimensional targets, an MLP will typically beat a "
                "same-size KAN on both accuracy and throughput.\n")
        if mode != "smoke":
            f.write("- The 0.2.0 tables each published one unwarmed forward. "
                    "Those columns disagreed (README 68.64 ms vs 25.52 ms; "
                    "the previous `benchmarks/results.md` 43.59 ms vs 31.61 ms) "
                    "and are not averaged into this table.\n")


def ensure_compare_gate(rows: list[Result]) -> None:
    """Fail the process when a synthetic benchmark result is unusable."""
    problems: list[str] = []
    if not rows:
        problems.append("no model results")
    for row in rows:
        if row.params <= 0:
            problems.append(f"{row.model}: no parameters")
        for label, value in (
            ("train_mse", row.train_mse),
            ("test_mse", row.test_mse),
            ("train_s", row.train_s),
            ("infer_ms_4k_cpu", row.infer_ms_4k_cpu),
        ):
            if not math.isfinite(value):
                problems.append(f"{row.model}: {label} is not finite ({value})")
    if problems:
        raise SystemExit("benchmark gate failed:\n- " + "\n- ".join(problems))


def _specs(smoke: bool) -> list[tuple[str, object]]:
    if smoke:
        return [
            ("KAN[2,4,1]", KAN([2, 4, 1])),
            ("MLP[2,8,1]", make_mlp(2, 8)),
        ]
    return [
        ("KAN[2,16,1]", KAN([2, 16, 1])),
        ("KAN[2,32,1]", KAN([2, 32, 1])),
        ("MLP[2,32,1]", make_mlp(2, 32)),
        ("MLP[2,16,16,1]", make_mlp(2, (16, 16))),
        ("MLP[2,64,64,1]", make_mlp(2, (64, 64))),
    ]


def _aggregate(groups: list[list[Result]]) -> list[Result]:
    if not groups:
        return []
    width = len(groups[0])
    aggregated: list[Result] = []
    for index in range(width):
        rows = [group[index] for group in groups]
        names = {row.model for row in rows}
        params = {row.params for row in rows}
        if len(names) != 1 or len(params) != 1:
            raise SystemExit(f"seed rows disagree at slot {index}")

        def mean_of(attr: str) -> float:
            return float(np.mean([getattr(row, attr) for row in rows]))

        def std_of(attr: str) -> float:
            return _population_std([getattr(row, attr) for row in rows])

        gpu_values = [row.infer_ms_4k_gpu for row in rows]
        gpu_mean = None
        if all(value is not None for value in gpu_values):
            gpu_mean = round(float(np.mean(gpu_values)), 3)
        base = rows[0]
        aggregated.append(
            Result(
                model=base.model,
                params=base.params,
                train_s=round(mean_of("train_s"), 3),
                infer_ms_4k_cpu=round(mean_of("infer_ms_4k_cpu"), 3),
                infer_ms_4k_gpu=gpu_mean,
                train_mse=mean_of("train_mse"),
                test_mse=mean_of("test_mse"),
                train_s_std=std_of("train_s"),
                infer_ms_4k_cpu_std=std_of("infer_ms_4k_cpu"),
                train_mse_std=std_of("train_mse"),
                test_mse_std=std_of("test_mse"),
                n_seeds=len(rows),
            )
        )
    return aggregated


def run_compare(
    *,
    epochs: int,
    patience: int,
    n_train: int,
    n_test: int,
    batch_size: int,
    smoke: bool,
    seeds: tuple[int, ...] = (0,),
) -> list[Result]:
    X, y = make_data(n_train, 0)
    X_test, y_test = make_data(n_test, 1)
    score_kwargs = {
        "epochs": epochs,
        "lr": 1e-2,
        "batch_size": batch_size,
        "early_stop_patience": patience,
    }

    groups: list[list[Result]] = []
    for seed in seeds:
        set_global_seed(seed)
        groups.append(
            [
                fit_and_score(name, model, X, y, X_test, y_test, **score_kwargs)
                for name, model in _specs(smoke)
            ]
        )
    rows = _aggregate(groups)
    ensure_compare_gate(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true",
                    help="convergence run: 1000 epochs + early-stopping(50)")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke gate: one epoch on a tiny KAN and MLP. Does not overwrite results.md.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Write the markdown table here. Defaults to benchmarks/results.md except for --smoke.",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated init seeds. Default is 0,1,2 for the 100-epoch run and 0 otherwise.",
    )
    args = ap.parse_args(argv)
    if args.long and args.smoke:
        ap.error("--smoke and --long cannot be combined")
    if args.seeds is None:
        seeds = (0,) if args.smoke or args.long else (0, 1, 2)
    else:
        try:
            seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip() != "")
        except ValueError:
            ap.error("--seeds must be a comma-separated list of integers")
        if not seeds:
            ap.error("--seeds must name at least one seed")

    if args.smoke:
        epochs, patience, mode = 1, 0, "smoke"
        n_train, n_test, batch_size = 128, 64, 32
    elif args.long:
        epochs, patience, mode = 1000, 50, "long"
        n_train, n_test, batch_size = 4096, 1024, 128
    else:
        epochs, patience, mode = 100, 0, "quick"
        n_train, n_test, batch_size = 4096, 1024, 128

    rows = run_compare(
        epochs=epochs,
        patience=patience,
        n_train=n_train,
        n_test=n_test,
        batch_size=batch_size,
        smoke=args.smoke,
        seeds=seeds,
    )

    for row in rows:
        print(row, flush=True)
    if args.output is not None:
        out = args.output
    elif args.smoke:
        out = None
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "results.md")
    if out is not None:
        write_results(
            rows,
            out,
            epochs=epochs,
            mode=mode,
            seeds=seeds,
            n_train=n_train,
            n_test=n_test,
            batch_size=batch_size,
        )
        print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
