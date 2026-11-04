"""Dataset loading and augmentation for FloodNet-style segmentation data."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_mask(
    mask: np.ndarray,
    num_classes: int,
    color_map: dict[tuple[int, int, int], int] | None = None,
) -> np.ndarray:
    """Convert common indexed or RGB FloodNet masks to HxW class indices.

    Official and common repackaged FloodNet masks are often stored as either:
    - one-channel class IDs 0..9, or
    - three equal channels such as (5, 5, 5) for class 5.
    """
    if mask.ndim == 2:
        decoded = mask
    elif mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        if np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(
            mask[:, :, 1], mask[:, :, 2]
        ):
            decoded = mask[:, :, 0]
        else:
            if color_map is None:
                colors = np.unique(mask.reshape(-1, 3), axis=0)[:20]
                raise ValueError(
                    "This mask uses a true RGB color palette rather than class IDs. "
                    "Provide an explicit color-map JSON to prepare_dataset.py. "
                    f"Observed colors include: {colors.tolist()}"
                )
            decoded = np.full(mask.shape[:2], -1, dtype=np.int64)
            # cv2 reads color masks as BGR, so convert each pixel to RGB before mapping.
            rgb_mask = mask[:, :, ::-1]
            for rgb, class_id in color_map.items():
                matches = np.all(rgb_mask == np.asarray(rgb, dtype=np.uint8), axis=2)
                decoded[matches] = class_id
            if np.any(decoded < 0):
                missing = np.unique(rgb_mask[decoded < 0].reshape(-1, 3), axis=0)[:20]
                raise ValueError(
                    "The color map does not cover every mask color. "
                    f"Missing RGB colors include: {missing.tolist()}"
                )
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    decoded = decoded.astype(np.int64)
    invalid = (decoded < 0) | (decoded >= num_classes)
    if invalid.any():
        values = np.unique(decoded[invalid])[:20]
        raise ValueError(f"Mask contains invalid class IDs: {values.tolist()}")
    return decoded


def build_transforms(image_size: int, train: bool) -> A.Compose:
    transforms: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=12,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.4,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.35
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.10),
            ]
        )
    transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return A.Compose(transforms)


class FloodSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        num_classes: int,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        self.num_classes = num_classes
        self.transform = transform or build_transforms(image_size, train=split == "train")

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected '{self.image_dir}' and '{self.mask_dir}'. "
                "Run prepare_dataset.py first or arrange the dataset in the documented format."
            )

        images = [p for p in self.image_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES]
        masks_by_stem = {
            p.stem: p for p in self.mask_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
        }
        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(images):
            mask_path = masks_by_stem.get(image_path.stem)
            if mask_path is not None:
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No matching image-mask pairs found for split '{split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, mask_path = self.samples[index]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask_raw is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = decode_mask(mask_raw, self.num_classes)
        transformed = self.transform(image=image, mask=mask)
        image_tensor = torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(transformed["mask"]).long()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": image_path.name,
        }


def estimate_class_weights(dataset: FloodSegmentationDataset, num_classes: int) -> torch.Tensor:
    """Estimate inverse-frequency weights from resized masks without augmentation."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, mask_path in dataset.samples:
        raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        mask = decode_mask(raw, num_classes)
        counts += np.bincount(mask.ravel(), minlength=num_classes)

    counts = np.maximum(counts, 1.0)
    frequencies = counts / counts.sum()
    weights = 1.0 / np.log(1.02 + frequencies)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
