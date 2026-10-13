"""High-level PyTorch KAN model + ergonomic builders."""
from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import huggingface_hub
import torch
import yaml
from huggingface_hub import HfApi
from torch import nn

from .layers import KANLinear

LayersSpec = Sequence[int] | Sequence[dict]


class KAN(nn.Sequential):
    """PyTorch KAN model.

    Two equivalent constructors::

        KAN([2, 64, 1])                                # widths
        KAN([{"in_features": 2, "out_features": 64},   # per-layer dicts
             {"in_features": 64, "out_features": 1}])
    """

    def __init__(
        self,
        layers: LayersSpec,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        grid_range=(-1.0, 1.0),
    ) -> None:
        modules: list[nn.Module] = []
        defaults = dict(
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
            grid_range=grid_range,
        )

        layers = list(layers)
        if len(layers) == 0:
            raise ValueError("`layers` must not be empty")
        first = layers[0]
        if isinstance(first, dict):
            for cfg in layers:
                modules.append(KANLinear(**{**defaults, **cfg}))
        else:
            widths = [int(w) for w in layers]
            if len(widths) < 2:
                raise ValueError("widths must contain at least input and output dims")
            for i in range(len(widths) - 1):
                modules.append(
                    KANLinear(widths[i], widths[i + 1], **defaults)
                )

        super().__init__(*modules)
        self._layers_spec = layers
        self._defaults = defaults

    # ---- convenience ergonomics --------------------------------------------
    def fit(
        self,
        X,
        y,
        *,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        optimizer: str = "adam",
        val_split: float = 0.0,
        early_stopping_patience: int = 0,
        verbose: int = 1,
        device: str | None = None,
        loss_fn=None,
        tensorboard: bool = False,
        log_dir: str = "logs/kanx",
    ):
        """Zero-friction fit: ``model.fit(X, y)`` — no optimizer or compile dance.

        Internally wraps :class:`kanx.torch.Trainer`; returns its
        ``TrainHistory`` so you get ``hist.loss`` and ``hist.val_loss``.
        """
        # Local import to avoid an import cycle.
        from .trainer import Trainer
        return Trainer(self, device=device, loss_fn=loss_fn).fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            val_split=val_split,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose,
            tensorboard=tensorboard,
            log_dir=log_dir,
        )

    @torch.no_grad()
    def predict(
        self,
        x,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Inference. Accepts list / numpy array / tensor. Returns a CPU tensor."""
        was_training = self.training
        self.eval()
        try:
            xt = _as_tensor(x)
            if batch_size is None or xt.shape[0] <= batch_size:
                return self(xt).cpu()
            chunks = []
            for i in range(0, xt.shape[0], batch_size):
                chunks.append(self(xt[i : i + batch_size]).cpu())
            return torch.cat(chunks, dim=0)
        finally:
            self.train(was_training)

    def update_grid_from_samples(self, x, margin: float = 0.01) -> None:
        """Update grids on all KANLinear layers from input samples.

        Each layer grid is updated directly from the raw input samples,
        without propagating transformed outputs through prior layers.
        This preserves the original feature dimensions and avoids
        shape mismatches during matmul.

        Args:
            x: Input data (batch, in_features); can be numpy array, list, or tensor.
            margin: Margin applied to grid boundaries.
        """
        xt = _as_tensor(x)
        kan_layers = [layer for layer in self.modules() if isinstance(layer, KANLinear)]
        if not kan_layers:
            return

        with torch.no_grad():
            current_x = xt
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

        ``threshold`` and ``relative`` match :meth:`kanx.torch.layers.KANLinear.prune`.
        When ``samples`` is provided, each layer is scored on the batch it
        receives after earlier layers have been pruned.
        """
        kan_layers = [layer for layer in self.modules() if isinstance(layer, KANLinear)]
        if not kan_layers:
            return {"pruned": 0, "total": 0, "active": 0}

        current = None if samples is None else _as_tensor(samples)
        if current is not None and current.ndim != 2:
            raise ValueError(
                f"samples must have shape (batch, in_features), got {tuple(current.shape)}"
            )

        pruned = 0
        total = 0
        with torch.no_grad():
            for layer in kan_layers:
                stats = layer.prune(threshold=threshold, relative=relative, samples=current)
                pruned += stats["pruned"]
                total += stats["total"]
                if current is not None:
                    current = layer(current)
        return {"pruned": pruned, "total": total, "active": total - pruned}

    def save(self, path: str) -> str:
        """Save state_dict + architecture spec to a single .pt file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "layers_spec": self._layers_spec,
                "defaults": self._defaults,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str, map_location="cpu") -> KAN:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ckpt["layers_spec"], **ckpt["defaults"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str = "main", **kwargs) -> KAN:
        model_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename="model.pt", revision=revision)
        config_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision)
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_cfg = config.get("model", {})
        layers = model_cfg["layers"]
        grid_size = model_cfg.get("grid_size", 5)
        spline_order = model_cfg.get("spline_order", 3)
        base_activation = model_cfg.get("base_activation", "silu")
        grid_range = tuple(model_cfg.get("grid_range", (-1.0, 1.0)))

        model = cls(
            layers,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
            grid_range=grid_range,
            **kwargs,
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu")["state_dict"])
        return model

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload KANX model",
        private: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            model_path = temp_dir / "model.pt"
            config_path = temp_dir / "config.yaml"

            self.save(model_path)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "model": {
                            **{
                                **self._defaults,
                                "grid_range": list(self._defaults["grid_range"]),
                            },
                            "layers": list(self._layers_spec),
                            "name": self.__class__.__name__,
                        }
                    },
                    f,
                )

            api = HfApi()
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            api.upload_folder(
                folder_path=str(temp_dir),
                path_in_repo="",
                repo_id=repo_id,
                commit_message=commit_message,
            )


def build_kan(layers: Iterable, **kwargs) -> KAN:
    """Functional builder mirroring `KAN.__init__`."""
    return KAN(list(layers), **kwargs)


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"Inputs must be rank-1 or rank-2; got shape {tuple(t.shape)}")
    return t.float()
