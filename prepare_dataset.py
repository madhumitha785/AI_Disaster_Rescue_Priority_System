"""Normalize an unpacked FloodNet-style dataset into a simple project layout.

Output:
    data/processed/
        train/images, train/masks
        val/images, val/masks
        test/images, test/masks

The scanner supports common FloodNet names such as ``*_lab.png`` and folders
containing Mask, Label, ColorMask, Annotation, or GroundTruth.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.dataset import SUPPORTED_SUFFIXES, decode_mask

MASK_HINTS = {
    "mask",
    "masks",
    "label",
    "labels",
    "lab",
    "annotation",
    "annotations",
    "segmentation",
    "colormask",
    "colormasks",
    "groundtruth",
    "ground_truth",
    "gt",
}


def normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(?:_lab|_mask|_masks|_label|_labels|_gt|_annotation)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def is_mask_path(path: Path) -> bool:
    tokens = [re.sub(r"[^a-z0-9]+", "", part.lower()) for part in path.parts]
    stem = path.stem.lower()
    if re.search(r"(?:_lab|_mask|_label|_gt|_annotation)$", stem):
        return True
    exact_hints = {re.sub(r"[^a-z0-9]+", "", hint) for hint in MASK_HINTS}
    for token in tokens:
        if token in exact_hints:
            return True
        if "mask" in token or "groundtruth" in token or "annotation" in token:
            return True
    return False


def infer_split(path: Path) -> str | None:
    tokens = [re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_") for part in path.parts]
    for token in tokens:
        if re.search(r"(?:^|_)(train|training|trainset|train_set)(?:_|$)", token):
            return "train"
        if re.search(r"(?:^|_)(validation|valid|val|validationset|validation_set|valset)(?:_|$)", token):
            return "val"
        if re.search(r"(?:^|_)(test|testing|testset|test_set)(?:_|$)", token):
            return "test"
    return None


def discover_pairs(raw_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    images: dict[tuple[str, str], list[Path]] = defaultdict(list)
    masks: dict[tuple[str, str], list[Path]] = defaultdict(list)
    warnings: list[str] = []

    for path in files:
        split = infer_split(path)
        if split is None:
            continue
        key = (split, normalized_stem(path))
        if is_mask_path(path):
            masks[key].append(path)
        else:
            images[key].append(path)

    pairs: list[tuple[str, Path, Path]] = []
    for key, image_candidates in images.items():
        mask_candidates = masks.get(key, [])
        if not mask_candidates:
            continue
        if len(image_candidates) > 1:
            warnings.append(f"Multiple images for {key}; using {image_candidates[0]}")
        if len(mask_candidates) > 1:
            warnings.append(f"Multiple masks for {key}; using {mask_candidates[0]}")
        pairs.append((key[0], image_candidates[0], mask_candidates[0]))
    return sorted(pairs, key=lambda item: (item[0], item[1].name)), warnings


def convert_mask(
    mask_path: Path,
    output_path: Path,
    num_classes: int,
    color_map: dict[tuple[int, int, int], int] | None,
) -> None:
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    decoded = decode_mask(raw, num_classes, color_map=color_map)
    if not cv2.imwrite(str(output_path), decoded.astype(np.uint8)):
        raise RuntimeError(f"Could not write converted mask: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a FloodNet-style dataset.")
    parser.add_argument("--raw-dir", required=True, help="Unpacked raw dataset directory")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--color-map",
        help='Optional JSON mapping like {"0,0,0": 0, "255,0,0": 1} for true RGB masks',
    )
    parser.add_argument("--copy", action="store_true", help="Copy images instead of making links")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    color_map = None
    if args.color_map:
        with Path(args.color_map).open("r", encoding="utf-8") as handle:
            raw_map = json.load(handle)
        color_map = {}
        for key, value in raw_map.items():
            rgb = tuple(int(part.strip()) for part in key.split(","))
            if len(rgb) != 3:
                raise ValueError(f"Invalid RGB key in color map: {key}")
            color_map[rgb] = int(value)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
        )
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    pairs, warnings = discover_pairs(raw_dir)
    if not pairs:
        raise RuntimeError(
            "No image-mask pairs were found. Check the raw folder and DATASET_GUIDE.md."
        )

    counts: Counter[str] = Counter()
    manifest: list[dict[str, str]] = []
    for index, (split, image_path, mask_path) in enumerate(tqdm(pairs, desc="Preparing")):
        safe_stem = f"{normalized_stem(image_path)}_{index:05d}"
        image_out = output_dir / split / "images" / f"{safe_stem}{image_path.suffix.lower()}"
        mask_out = output_dir / split / "masks" / f"{safe_stem}.png"

        if args.copy:
            shutil.copy2(image_path, image_out)
        else:
            try:
                image_out.symlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, image_out)
        convert_mask(mask_path, mask_out, args.num_classes, color_map)
        counts[split] += 1
        manifest.append(
            {
                "split": split,
                "image": str(image_out.relative_to(output_dir)),
                "mask": str(mask_out.relative_to(output_dir)),
                "source_image": str(image_path),
                "source_mask": str(mask_path),
            }
        )

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"counts": counts, "pairs": manifest, "warnings": warnings}, handle, indent=2)

    print("Dataset prepared successfully.")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counts[split]} paired samples")
    if warnings:
        print(f"  warnings: {len(warnings)} (see manifest.json)")


if __name__ == "__main__":
    main()
