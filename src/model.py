"""Segmentation model factory."""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch


def build_model(
    architecture: str,
    encoder: str,
    num_classes: int,
    in_channels: int = 3,
    encoder_weights: str | None = "imagenet",
) -> torch.nn.Module:
    architecture_key = architecture.lower().replace("-", "").replace("_", "")

    common: dict[str, Any] = {
        "encoder_name": encoder,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": num_classes,
        "activation": None,
    }

    if architecture_key == "unet":
        return smp.Unet(**common)
    if architecture_key in {"unetplusplus", "unetpp"}:
        return smp.UnetPlusPlus(**common)
    if architecture_key in {"deeplabv3plus", "deeplabv3p"}:
        return smp.DeepLabV3Plus(**common)

    raise ValueError(
        f"Unsupported architecture '{architecture}'. "
        "Choose one of: unet, unetplusplus, deeplabv3plus."
    )


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    config: dict[str, Any],
    epoch: int,
    best_miou: float,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_miou": best_miou,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    model = build_model(
        architecture=model_cfg["architecture"],
        encoder=model_cfg["encoder"],
        num_classes=int(data_cfg["num_classes"]),
        in_channels=int(model_cfg.get("in_channels", 3)),
        encoder_weights=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
