"""High-level KAN model.

`KAN` is a thin convenience wrapper around `tf.keras.Sequential` that wires
together a list of `KANLinear` layers from a list of widths
(e.g. ``[2, 64, 64, 1]``) or a list of per-layer config dicts.
"""
from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import huggingface_hub
import tensorflow as tf
import yaml
from huggingface_hub import HfApi

from .layers import KANLinear

LayersSpec = Sequence[int] | Sequence[dict]


@tf.keras.utils.register_keras_serializable(package="kanx")
class KAN(tf.keras.Sequential):
    """Sequential KAN.

    Two equivalent ways to build it:

    >>> KAN([2, 64, 64, 1])                                       # widths
    >>> KAN([                                                     # per-layer
    ...     {"in_features": 2,  "out_features": 64},
    ...     {"in_features": 64, "out_features": 64},
    ...     {"in_features": 64, "out_features": 1},
    ... ])
    """

    def __init__(
        self,
        layers: LayersSpec,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        regularization_factor: float = 0.0,
        grid_range=(-1.0, 1.0),
        name: str = "kan",
        **kwargs,
    ):
        super().__init__(name=name)
        self._layers_spec = list(layers)
        self._default_kwargs = {
            "grid_size": grid_size,
            "spline_order": spline_order,
            "base_activation": base_activation,
            "regularization_factor": regularization_factor,
            "grid_range": grid_range,
        }
        self._defaults = self._default_kwargs
        self._extra_kwargs = kwargs

        if len(self._layers_spec) == 0:
            raise ValueError("`layers` must not be empty")

        first = self._layers_spec[0]
        if isinstance(first, dict):
            for cfg in self._layers_spec:
                self.add(KANLinear(**{**self._default_kwargs, **cfg}))
        else:
            widths: list[int] = [int(w) for w in self._layers_spec]
            if len(widths) < 2:
                raise ValueError(
                    "When passing widths, provide at least 2 values "
                    "(input_dim, output_dim)."
                )
            for i in range(len(widths) - 1):
                self.add(
                    KANLinear(
                        in_features=widths[i],
                        out_features=widths[i + 1],
                        **self._default_kwargs,
                    )
                )

    # ---- convenience -------------------------------------------------------
    def fit(self, x=None, y=None, *args, **kwargs):
        """Zero-friction fit: auto-compiles with Adam(1e-3) + MSE if not compiled.

        This makes ``KAN([2, 32, 1]).fit(X, y, epochs=20)`` work out of the
        box without a separate ``model.compile(...)`` call.
        """
        if getattr(self, "optimizer", None) is None:
            self.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.MeanSquaredError(),
            )
        return super().fit(x, y, *args, **kwargs)

    def predict_tensor(self, x):
        """Tensor-in, tensor-out inference (no Keras progress bars / batching)."""
        return self(x, training=False)

    def update_grid_from_samples(self, x: tf.Tensor, margin: float = 0.01) -> None:
        """Update grids on all KANLinear layers from input samples.

        Each layer grid is updated directly from the raw input samples,
        without propagating transformed outputs through prior layers.
        This preserves the original feature dimensions and avoids
        shape mismatches during matmul.

        Args:
            x: Input tensor of shape (batch, in_features).
            margin: Margin applied to grid boundaries.
        """
        kan_layers = [layer for layer in self.layers if isinstance(layer, KANLinear)]
        if not kan_layers:
            return

        current_x = x
        for layer in kan_layers:
            layer.update_grid_from_samples(current_x, margin=margin)
            # propagate forward for next layer
            current_x = layer(current_x)

    def prune(
        self,
        *,
        threshold: float | None = None,
        relative: float | None = None,
        samples=None,
    ) -> dict[str, int]:
        """Prune low-magnitude edges on every ``KANLinear`` layer.

        ``threshold`` and ``relative`` are forwarded to each layer. When
        ``samples`` has shape ``(batch, in_features)``, each layer is scored
        on the activations it actually receives, and later layers see the
        pruned outputs of earlier ones.

        Returns total ``pruned``, ``total``, and ``active`` edge counts.
        """
        kan_layers = [layer for layer in self.layers if isinstance(layer, KANLinear)]
        if not kan_layers:
            return {"pruned": 0, "total": 0, "active": 0}
        if not self.built:
            width = int(kan_layers[0].in_features)
            self(tf.zeros((1, width), dtype=tf.float32), training=False)

        current = None
        if samples is not None:
            current = tf.convert_to_tensor(samples, dtype=tf.float32)
            if current.shape.rank != 2:
                raise ValueError(
                    f"samples must have shape (batch, in_features), got {tuple(current.shape)}"
                )

        pruned = 0
        total = 0
        for layer in kan_layers:
            stats = layer.prune(threshold=threshold, relative=relative, samples=current)
            pruned += stats["pruned"]
            total += stats["total"]
            if current is not None:
                current = layer(current, training=False)
        return {"pruned": pruned, "total": total, "active": total - pruned}

    def get_config(self):
        return {
            "layers": list(self._layers_spec),
            "grid_size": int(self._default_kwargs["grid_size"]),
            "spline_order": int(self._default_kwargs["spline_order"]),
            "base_activation": str(self._default_kwargs["base_activation"]),
            "regularization_factor": float(self._default_kwargs["regularization_factor"]),
            "grid_range": list(self._default_kwargs["grid_range"]),
            "name": self.name,
        }

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str = "main", **kwargs) -> KAN:
        model_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename="model.keras", revision=revision)
        config_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision)
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_cfg = config.get("model", {})
        layers = model_cfg["layers"]
        grid_size = model_cfg.get("grid_size", 5)
        spline_order = model_cfg.get("spline_order", 3)
        base_activation = model_cfg.get("base_activation", "silu")
        regularization_factor = model_cfg.get("regularization_factor", 0.0)
        grid_range = tuple(model_cfg.get("grid_range", (-1.0, 1.0)))

        model = cls(
            layers,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
            regularization_factor=regularization_factor,
            grid_range=grid_range,
            **kwargs,
        )
        model(tf.zeros((1, int(layers[0])), dtype=tf.float32))
        model.load_weights(model_path)
        return model

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload KANX model",
        private: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            model_path = temp_dir / "model.keras"
            config_path = temp_dir / "config.yaml"

            self.save(model_path)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"model": self.get_config()}, f)

            api = HfApi()
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            api.upload_folder(
                folder_path=str(temp_dir),
                path_in_repo="",
                repo_id=repo_id,
                commit_message=commit_message,
            )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


def build_kan(
    layers: Iterable[int],
    *,
    grid_size: int = 5,
    spline_order: int = 3,
    base_activation: str = "silu",
    regularization_factor: float = 0.0,
    grid_range=(-1.0, 1.0),
) -> KAN:
    """Functional builder, mirrors `KAN.__init__` arguments."""
    return KAN(
        list(layers),
        grid_size=grid_size,
        spline_order=spline_order,
        base_activation=base_activation,
        regularization_factor=regularization_factor,
        grid_range=grid_range,
    )
