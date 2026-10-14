"""One MirroredStrategy step for the TensorFlow KAN.

Run ``python tests/distributed_smoke_tf.py``. The last stdout line is a JSON
object. With two or more GPUs the strategy uses those GPUs. Otherwise it uses
two logical CPUs. If that configuration is unavailable, it trains on one CPU
with ``OneDeviceStrategy``.
"""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf


def _strategy() -> tuple[tf.distribute.Strategy, str]:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) >= 2:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return tf.distribute.MirroredStrategy(), "gpu"
    cpus = tf.config.list_physical_devices("CPU")
    try:
        tf.config.set_logical_device_configuration(
            cpus[0],
            [tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()],
        )
        strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/cpu:1"])
        if strategy.num_replicas_in_sync >= 2:
            return strategy, "cpu-logical"
    except (RuntimeError, ValueError):
        pass
    return tf.distribute.OneDeviceStrategy("/cpu:0"), "single-device"


def main() -> None:
    strategy, mode = _strategy()
    from kanx import KAN

    rng = np.random.default_rng(0)
    features = rng.uniform(-1, 1, size=(8, 2)).astype("float32")
    target = (np.sin(np.pi * features[:, :1]) + features[:, 1:2] ** 2).astype("float32")
    with strategy.scope():
        model = KAN([2, 4, 1], grid_size=3, spline_order=2)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="mse")
    history = model.fit(features, target, epochs=1, batch_size=8, verbose=0)
    loss = float(history.history["loss"][-1])
    payload = {
        "ok": loss == loss and abs(loss) != float("inf"),
        "backend": "tensorflow",
        "strategy": type(strategy).__name__,
        "mode": mode,
        "replicas": int(strategy.num_replicas_in_sync),
        "loss": loss,
    }
    print(json.dumps(payload), flush=True)
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
