"""Loss functions for imbalanced multiclass segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dims)
        denominator = torch.sum(probabilities + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        cross_entropy_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = MulticlassDiceLoss()
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.cross_entropy(logits, targets)
        dice = self.dice(logits, targets)
        return self.cross_entropy_weight * ce + self.dice_weight * dice
