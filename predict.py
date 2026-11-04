"""Command-line inference for one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.inference import predict_mask
from src.model import load_checkpoint
from src.priority import calculate_damage_report
from src.visual import create_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict damage from one aerial image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/prediction_overlay.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    image = Image.open(args.image).convert("RGB")
    mask, original = predict_mask(
        model, image, int(config["data"]["image_size"]), device
    )
    report = calculate_damage_report(mask, config.get("priority", {}))
    overlay = create_overlay(original, mask)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"Saved overlay to {output_path}")


if __name__ == "__main__":
    main()
