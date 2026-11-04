"""Training and validation loops."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from tqdm import tqdm

from .metrics import SegmentationMeter


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    scaler,
    amp: bool,
    accumulation_steps: int = 1,
) -> dict[str, object]:
    model.train()
    meter = SegmentationMeter(num_classes)
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Train", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with _autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, masks) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % accumulation_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_loss = float(loss.item() * accumulation_steps)
        total_loss += batch_loss
        predictions = logits.argmax(dim=1)
        meter.update(predictions, masks)
        progress.set_postfix(loss=f"{batch_loss:.4f}")

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.inference_mode()
def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    amp: bool,
) -> dict[str, object]:
    model.eval()
    meter = SegmentationMeter(num_classes)
    total_loss = 0.0

    progress = tqdm(loader, desc="Validation", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with _autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += float(loss.item())
        meter.update(logits.argmax(dim=1), masks)

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics
