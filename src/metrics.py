"""Confusion-matrix based segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred = predictions.detach().cpu().numpy().astype(np.int64).ravel()
        true = targets.detach().cpu().numpy().astype(np.int64).ravel()
        valid = (true >= 0) & (true < self.num_classes)
        indices = self.num_classes * true[valid] + pred[valid]
        self.confusion += np.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        support = cm.sum(axis=1)

        iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
        dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

        present = support > 0
        mean_iou = float(iou[present].mean()) if present.any() else 0.0
        mean_dice = float(dice[present].mean()) if present.any() else 0.0
        pixel_accuracy = float(tp.sum() / cm.sum()) if cm.sum() else 0.0

        return {
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }
