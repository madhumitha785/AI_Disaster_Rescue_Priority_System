"""Evaluate a trained checkpoint on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import FloodSegmentationDataset, build_transforms
from src.engine import validate_one_epoch
from src.losses import CombinedSegmentationLoss
from src.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="outputs/test_metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    data_cfg = config["data"]
    data_root = args.data_root or data_cfg["root"]
    image_size = int(data_cfg["image_size"])
    num_classes = int(data_cfg["num_classes"])

    dataset = FloodSegmentationDataset(
        data_root,
        args.split,
        image_size,
        num_classes,
        transform=build_transforms(image_size, train=False),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    criterion = CombinedSegmentationLoss()
    metrics = validate_one_epoch(
        model, loader, criterion, device, num_classes, amp=device.type == "cuda"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
