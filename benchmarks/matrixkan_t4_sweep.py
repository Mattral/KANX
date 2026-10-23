# Colab: Runtime -> T4 GPU, then run this file as one cell.
# Prints META and ROWS. The 0.2.0 loop measurement already in
# benchmarks/results/matrixkan_t4_0.2.0.json did not copy weights,
# did not time backward, and used constructor defaults for grid size
# and spline order.

import json
import subprocess
import sys
import time

print("Installing kanx 0.2.0. This sweep times the Python-loop basis, not a GEMM.")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kanx[torch]==0.2.0"])

import torch
import kanx
from kanx.torch import KAN, MatrixKAN
from kanx.torch.layers import KANLinear
from kanx.torch.matrix_kan import MatrixKANLinear

assert torch.cuda.is_available(), "Select a GPU runtime (T4) and rerun."

GRID_SIZE = 5
ORDERS = (3, 5, 7)
WIDTHS = (32, 64, 128, 256, 512)
BATCHES = (1024, 4096, 8192, 16384)
FORWARD_WARMUP = 10
FORWARD_REPS = 50
BACKWARD_WARMUP = 5
BACKWARD_REPS = 20

meta = {
    "kanx": kanx.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_torch": torch.cuda.get_device_name(0),
    "nvidia_smi_L": subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
    "nvidia_smi_query": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip(),
    "shape": "[2, width, 1]",
    "grid_size": GRID_SIZE,
    "spline_orders": list(ORDERS),
    "dtype": "float32",
    "forward_warmup": FORWARD_WARMUP,
    "forward_reps": FORWARD_REPS,
    "backward_warmup": BACKWARD_WARMUP,
    "backward_reps": BACKWARD_REPS,
    "weights": "copied from KAN into MatrixKAN before the agreement check",
    "implementation": "0.2.0 Python loop over Cox-de Boor basis index",
}
print("META", json.dumps(meta))


def copy_weights(src: KAN, dst: MatrixKAN) -> None:
    sources = [m for m in src.modules() if isinstance(m, KANLinear)]
    targets = [m for m in dst.modules() if isinstance(m, MatrixKANLinear)]
    if len(sources) != len(targets):
        raise RuntimeError(f"layer count {len(sources)} != {len(targets)}")
    with torch.no_grad():
        for left, right in zip(sources, targets):
            right.base_weight.copy_(left.base_weight)
            right.spline_weight.copy_(left.spline_weight)
            right.grid.copy_(left.grid)


def time_forward(model, batch) -> float:
    torch.cuda.synchronize()
    for _ in range(FORWARD_WARMUP):
        with torch.no_grad():
            model(batch)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(FORWARD_REPS):
        with torch.no_grad():
            model(batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / FORWARD_REPS * 1000.0


def time_backward(model, batch) -> float:
    target = torch.zeros(batch.shape[0], 1, device=batch.device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(batch), target)
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    for _ in range(BACKWARD_WARMUP):
        step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(BACKWARD_REPS):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / BACKWARD_REPS * 1000.0


rows = []
for order in ORDERS:
    for width in WIDTHS:
        for batch in BATCHES:
            torch.manual_seed(0)
            data = torch.empty(batch, 2, device="cuda").uniform_(-1, 1)
            kan = matrix = None
            try:
                kan = KAN([2, width, 1], grid_size=GRID_SIZE, spline_order=order).to("cuda")
                matrix = MatrixKAN([2, width, 1], grid_size=GRID_SIZE, spline_order=order).to("cuda")
                copy_weights(kan, matrix)
                with torch.no_grad():
                    err = (kan(data[:32]) - matrix(data[:32])).abs().max().item()
                if err > 1e-4:
                    row = {
                        "spline_order": order,
                        "width": width,
                        "batch": batch,
                        "max_abs_error": err,
                        "error": "outputs disagree; timing skipped",
                    }
                else:
                    row = {
                        "spline_order": order,
                        "width": width,
                        "batch": batch,
                        "max_abs_error": err,
                        "kan_forward_ms": round(time_forward(kan, data), 3),
                        "matrix_forward_ms": round(time_forward(matrix, data), 3),
                        "kan_backward_ms": round(time_backward(kan, data), 3),
                        "matrix_backward_ms": round(time_backward(matrix, data), 3),
                    }
                    row["matrix_over_kan_forward"] = round(
                        row["matrix_forward_ms"] / row["kan_forward_ms"], 3
                    )
            except RuntimeError as exc:
                row = {
                    "spline_order": order,
                    "width": width,
                    "batch": batch,
                    "error": str(exc).splitlines()[-1],
                }
            print(row, flush=True)
            rows.append(row)
            del kan, matrix, data
            torch.cuda.empty_cache()

print("ROWS", json.dumps(rows))
