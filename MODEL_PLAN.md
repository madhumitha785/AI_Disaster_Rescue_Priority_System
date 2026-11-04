# Model plan

## Recommended experiment

Train and compare two models on the same split and preprocessing:

### Model A — deployment model

- Architecture: U-Net
- Encoder: MobileNetV2 pretrained on ImageNet
- Input: 512 × 512 RGB
- Purpose: smaller checkpoint and faster CPU inference for hosting

### Model B — research comparison

- Architecture: DeepLabV3+
- Encoder: ResNet50 pretrained on ImageNet
- Input: 512 × 512 RGB
- Purpose: stronger multiscale contextual segmentation, but heavier deployment

## Training setup

- Loss: weighted Cross-Entropy + multiclass Dice
- Optimizer: AdamW
- Initial learning rate: 1e-4
- Epochs: up to 40
- Early stopping: validation mIoU patience 7
- Scheduler: ReduceLROnPlateau on validation mIoU
- Metrics: pixel accuracy, per-class IoU, mean IoU, Dice, precision and recall
- Augmentation: flips, small rotations, scale/shift, brightness/contrast and light blur

## Final selection

Use validation mIoU to select the checkpoint. Report test metrics once after final model selection. Do not select the model using test performance.

For a college demonstration, host the U-Net MobileNetV2 checkpoint and include the DeepLabV3+ comparison in the report.
