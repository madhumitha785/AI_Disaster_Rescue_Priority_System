"""Model and demonstration-mode inference utilities."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def preprocess_image(image: Image.Image, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"))
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - np.asarray(IMAGENET_MEAN)) / np.asarray(IMAGENET_STD)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, rgb


@torch.inference_mode()
def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor, original = preprocess_image(image, image_size)
    logits = model(tensor.to(device))
    prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    prediction = cv2.resize(
        prediction,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return prediction, original


def demonstration_mask(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Create a clearly labelled, non-ML mask only for testing the UI."""
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

    # Approximate vegetation and water colors.
    vegetation = (h >= 30) & (h <= 90) & (s > 45)
    water = (h >= 85) & (h <= 135) & (s > 35)
    gray_surface = (s < 45) & (v > 55) & (v < 205)
    bright_structure = (s < 70) & (v >= 180)

    mask[vegetation] = 6
    mask[water] = 5
    mask[gray_surface] = 4
    mask[bright_structure] = 2

    # Areas close to detected water are marked as flooded only to preview UI behavior.
    water_u8 = (water.astype(np.uint8) * 255)
    near_water = cv2.dilate(water_u8, np.ones((21, 21), np.uint8), iterations=1) > 0
    mask[near_water & bright_structure] = 1
    mask[near_water & gray_surface] = 3
    return mask, rgb
