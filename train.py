"""Train a multiclass FloodNet segmentation model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import apply_overrides, load_config
from src.dataset import FloodSegmentationDataset, build_transforms, estimate_class_weights
from src.engine import train_one_epoch, validate_one_epoch
from src.losses import CombinedSegmentationLoss
from src.model import build_model, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disaster segmentation model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--architecture", choices=["unet", "unetplusplus", "deeplabv3plus"])
    parser.add_argument("--encoder")
    parser.add_argument("--encoder-weights")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = apply_overrides(
        load_config(args.config),
        data_root=args.data_root,
        image_size=args.image_size,
        architecture=args.architecture,
        encoder=args.encoder,
        encoder_weights=args.encoder_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    set_seed(int(config.get("seed", 42)))

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_root = Path(data_cfg["root"])
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(data_cfg["image_size"])
    num_classes = int(data_cfg["num_classes"])
    train_ds = FloodSegmentationDataset(
        data_root,
        "train",
        image_size,
        num_classes,
        transform=build_transforms(image_size, train=True),
    )
    val_ds = FloodSegmentationDataset(
        data_root,
        "val",
        image_size,
        num_classes,
        transform=build_transforms(image_size, train=False),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=pin_memory,
        persistent_workers=int(data_cfg.get("num_workers", 2)) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=pin_memory,
        persistent_workers=int(data_cfg.get("num_workers", 2)) > 0,
    )

    model = build_model(
        architecture=model_cfg["architecture"],
        encoder=model_cfg["encoder"],
        num_classes=num_classes,
        in_channels=int(model_cfg.get("in_channels", 3)),
        encoder_weights=model_cfg.get("encoder_weights"),
    ).to(device)

    class_weight_setting = train_cfg.get("class_weights", "auto")
    if class_weight_setting == "auto":
        weight_dataset = FloodSegmentationDataset(
            data_root,
            "train",
            image_size,
            num_classes,
            transform=build_transforms(image_size, train=False),
        )
        class_weights = estimate_class_weights(weight_dataset, num_classes).to(device)
    elif isinstance(class_weight_setting, list):
        class_weights = torch.tensor(class_weight_setting, dtype=torch.float32, device=device)
    else:
        class_weights = None

    criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
        cross_entropy_weight=float(train_cfg.get("cross_entropy_weight", 1.0)),
        dice_weight=float(train_cfg.get("dice_weight", 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
    )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = make_scaler(use_amp)
    best_miou = -1.0
    patience_counter = 0
    history: list[dict[str, float]] = []
    best_path = output_dir / "best_model.pth"

    print(f"Device: {device}")
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")
    print(f"Model: {model_cfg['architecture']} + {model_cfg['encoder']}")

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes,
            scaler,
            use_amp,
            int(train_cfg.get("gradient_accumulation_steps", 1)),
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, num_classes, use_amp
        )
        val_miou = float(val_metrics["mean_iou"])
        scheduler.step(val_miou)

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": float(train_metrics["loss"]),
            "train_miou": float(train_metrics["mean_iou"]),
            "train_dice": float(train_metrics["mean_dice"]),
            "val_loss": float(val_metrics["loss"]),
            "val_miou": val_miou,
            "val_dice": float(val_metrics["mean_dice"]),
            "val_pixel_accuracy": float(val_metrics["pixel_accuracy"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train loss {row['train_loss']:.4f} | "
            f"val loss {row['val_loss']:.4f} | val mIoU {val_miou:.4f} | "
            f"val Dice {row['val_dice']:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            save_checkpoint(str(best_path), model, config, epoch, best_miou)
            with (output_dir / "best_validation_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(val_metrics, handle, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= int(train_cfg.get("patience", 7)):
                print("Early stopping triggered.")
                break

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

    save_checkpoint(str(output_dir / "last_model.pth"), model, config, epoch, best_miou)
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
