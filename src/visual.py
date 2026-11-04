"""Visualization helpers for masks and overlays."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .constants import PALETTE


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.asarray(PALETTE, dtype=np.uint8)
    clipped = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[clipped]


def create_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color_mask = colorize_mask(mask)
    if color_mask.shape[:2] != image_rgb.shape[:2]:
        color_mask = np.asarray(
            Image.fromarray(color_mask).resize(
                (image_rgb.shape[1], image_rgb.shape[0]), Image.Resampling.NEAREST
            )
        )
    overlay = (image_rgb.astype(np.float32) * (1 - alpha) + color_mask * alpha).clip(0, 255)
    return overlay.astype(np.uint8)


def image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()
