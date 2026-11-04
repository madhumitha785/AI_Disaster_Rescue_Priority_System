# Drone Disaster Damage Assessment

A complete PyTorch + Streamlit project for semantic segmentation of post-flood drone images. The system predicts flooded buildings, non-flooded buildings, flooded roads, non-flooded roads, water and surrounding scene classes, then calculates an interpretable Low/Medium/High response-priority score.

## Included

- Flexible FloodNet dataset preparation
- U-Net, U-Net++ and DeepLabV3+ model choices
- MobileNetV2 or ResNet encoders through `segmentation-models-pytorch`
- Transfer learning with ImageNet weights
- Albumentations-based paired image/mask augmentation
- Weighted Cross-Entropy + multiclass Dice loss
- Early stopping and best-checkpoint saving by validation mIoU
- Test evaluation with per-class IoU, Dice, precision and recall
- Batch Streamlit UI with masks, overlays, class coverage and CSV reports
- Optional coordinates and map display
- Streamlit Community Cloud and Docker deployment files
- Clearly labelled non-AI preview mode for testing the interface before training

## Important limitation

The ZIP does not contain the FloodNet dataset or trained model weights because they are large and have separate licensing/storage requirements. The application is complete, but real AI inference begins only after you train a checkpoint and place it at:

```text
models/best_model.pth
```

## 1. Environment setup

Python 3.10 or 3.11 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, double-click `scripts/run_windows.bat` to create the environment and start the preview UI.

## 2. Download the dataset

Use **FloodNet-Supervised v1.0** from the official BinaLab repository. Read `DATASET_GUIDE.md` before training.

Unzip it outside the Git repository, for example:

```text
D:\Datasets\FloodNet-Supervised_v1.0
```

## 3. Prepare the dataset

```powershell
python prepare_dataset.py `
  --raw-dir "D:\Datasets\FloodNet-Supervised_v1.0" `
  --output-dir data/processed `
  --copy `
  --overwrite
```

Expected output:

```text
data/processed/train/images
data/processed/train/masks
data/processed/val/images
data/processed/val/masks
data/processed/test/images
data/processed/test/masks
```

Review `data/processed/manifest.json` and manually inspect several image-mask pairs.

## 4. Train the host-friendly model

```powershell
python train.py `
  --config config.yaml `
  --architecture unet `
  --encoder mobilenet_v2 `
  --output-dir outputs/unet_mobilenetv2
```

This is the recommended model for the live hosted UI.

## 5. Train the stronger comparison model

```powershell
python train.py `
  --config config.yaml `
  --architecture deeplabv3plus `
  --encoder resnet50 `
  --batch-size 2 `
  --output-dir outputs/deeplabv3plus_resnet50
```

Reduce `image_size` to 384 or 256 in `config.yaml` if GPU memory is insufficient.

## 6. Evaluate on the test split

```powershell
python evaluate.py `
  --checkpoint outputs/unet_mobilenetv2/best_model.pth `
  --split test `
  --output outputs/unet_test_metrics.json
```

Use the test split only after choosing the final model using validation mIoU.

## 7. Add the model to the UI

```powershell
Copy-Item outputs\unet_mobilenetv2\best_model.pth models\best_model.pth
```

## 8. Run the application

```powershell
streamlit run app.py
```

The browser UI supports multiple images and provides:

- Original image
- Segmentation mask
- Transparent overlay
- Flooded-building percentage
- Flooded-road percentage
- Water coverage
- Priority score
- Class-coverage table and graph
- Overlay PNG download
- Combined CSV report

## Priority calculation

The default project rule is:

```text
Priority score =
0.50 × flooded-building percentage
+ 0.35 × flooded-road percentage
+ 0.15 × water-coverage percentage
```

Thresholds:

- Low: score ≤ 30
- Medium: 30 < score ≤ 60
- High: score > 60

This is a transparent academic decision-support rule, not an official emergency standard. You can change it in `config.yaml`.

## Project structure

```text
.
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── prepare_dataset.py
├── config.yaml
├── requirements.txt
├── Dockerfile
├── DATASET_GUIDE.md
├── DEPLOYMENT.md
├── MODEL_PLAN.md
├── sample_data/
├── models/
├── outputs/
├── scripts/
├── tests/
└── src/
    ├── config.py
    ├── constants.py
    ├── dataset.py
    ├── engine.py
    ├── inference.py
    ├── losses.py
    ├── metrics.py
    ├── model.py
    ├── priority.py
    └── visual.py
```

## Suggested final project presentation

1. Problem and SDG 11/13 relevance
2. FloodNet dataset and 10 classes
3. Image-mask preparation and leakage-safe official split
4. U-Net MobileNetV2 baseline/deployment model
5. DeepLabV3+ ResNet50 comparison model
6. Combined loss and class imbalance
7. Validation mIoU model selection
8. Per-class test metrics
9. Streamlit damage dashboard
10. Limitations and future extension to pre/post-disaster change detection

## Responsible-use note

Predictions can fail because of unseen locations, lighting, altitude, camera angle, water reflections or dataset shift. The application is for academic decision support and demonstrations. Emergency decisions require human validation and appropriate operational data.
