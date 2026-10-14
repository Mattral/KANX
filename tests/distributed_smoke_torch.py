"""One torch.distributed step for the PyTorch KAN.

Run ``python tests/distributed_smoke_torch.py``. Rank 0 prints a JSON object.
Two or more CUDA devices use NCCL. Otherwise two gloo processes run on CPU.
This process does not import TensorFlow.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_kan():
    """Import the PyTorch KAN without executing ``kanx/__init__.py``."""
    src = Path(__file__).resolve().parents[1] / "src"
    if "kanx" not in sys.modules:
        package = types.ModuleType("kanx")
        package.__path__ = [str(src / "kanx")]
        package.__package__ = "kanx"
        sys.modules["kanx"] = package
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from kanx.torch import KAN

    return KAN


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(rank: int, world: int, port: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["OMP_NUM_THREADS"] = "1"
    dist.init_process_group(
        backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    try:
        torch.manual_seed(0)
        kan_cls = _load_kan()
        model = kan_cls([2, 4, 1], grid_size=3, spline_order=2)
        if backend == "nccl":
            torch.cuda.set_device(rank)
            model.cuda(rank)
            replica = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            device = torch.device("cuda", rank)
        else:
            replica = torch.nn.parallel.DistributedDataParallel(model)
            device = torch.device("cpu")
        inputs = torch.linspace(-1, 1, 16, device=device).reshape(8, 2)
        target = torch.sin(torch.pi * inputs[:, :1])
        optimizer = torch.optim.SGD(replica.parameters(), lr=1e-2)
        optimizer.zero_grad()
        prediction = replica(inputs)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        finite = bool(torch.isfinite(loss).item())
        if not finite:
            raise RuntimeError(f"non-finite loss on rank {rank}")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "backend": "torch",
                        "distributed": backend,
                        "world_size": world,
                        "loss": float(loss.detach()),
                    }
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    world = 2
    backend = "nccl" if torch.cuda.device_count() >= world else "gloo"
    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    mp.spawn(run, args=(world, port, backend), nprocs=world, join=True)


if __name__ == "__main__":
    main()
