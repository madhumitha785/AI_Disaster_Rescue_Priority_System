# Dataset guide

## Recommended dataset: FloodNet-Supervised v1.0

Use the supervised FloodNet release for the first complete version of the project. It contains high-resolution UAV images from post-Hurricane-Harvey scenes and 10 semantic-segmentation classes:

0. Background
1. Building Flooded
2. Building Non-Flooded
3. Road Flooded
4. Road Non-Flooded
5. Water
6. Tree
7. Vehicle
8. Pool
9. Grass

Official source: `BinaLab/FloodNet-Supervised_v1.0` on GitHub. Follow its dataset download link and license/citation instructions.

## What the model needs

Each training sample requires:

- One RGB drone/aerial image.
- One pixel-level mask of exactly the same scene.
- Each mask pixel must represent one of the class IDs 0–9.

The preparation script supports grayscale class-ID masks and RGB masks where all three channels contain the same class ID, such as `(5, 5, 5)` for Water.

## Standard project layout

After preparation, the project expects:

```text
data/processed/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Every image and its mask must have the same filename stem:

```text
train/images/scene_001.jpg
train/masks/scene_001.png
```

## Automatic preparation

1. Download and unzip FloodNet.
2. From the project root, run:

```bash
python prepare_dataset.py --raw-dir "PATH/TO/UNZIPPED/FLOODNET" --output-dir data/processed --copy --overwrite
```

3. Inspect `data/processed/manifest.json`.
4. Open a few image-mask pairs before training. Incorrect mask IDs will invalidate the experiment.

## Recommended split policy

Prefer FloodNet's official train/validation/test split. Do not randomly mix neighboring frames from the same drone flight across splits, because visually near-duplicate scenes can create leakage.

## Optional future datasets

For earthquake/cyclone building damage, use a pre-disaster and post-disaster paired-image dataset such as xBD. That requires a different change-detection architecture and is intentionally not mixed into this first flood model.

## True RGB palette masks

When the mask uses actual RGB colors rather than equal-channel class IDs, create a verified mapping JSON and pass it explicitly:

```bash
python prepare_dataset.py --raw-dir "PATH/TO/FLOODNET" --output-dir data/processed --color-map color_map_example.json --copy --overwrite
```

`color_map_example.json` contains the app's display palette only. Do not assume it matches a dataset's annotation palette. Replace the RGB values using the dataset's official mapping before conversion.
