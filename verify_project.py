"""Check dependencies, project files and optional dataset/model availability."""

from __future__ import annotations

import importlib
from pathlib import Path

REQUIRED_MODULES = [
    "streamlit",
    "numpy",
    "pandas",
    "PIL",
    "yaml",
    "cv2",
    "torch",
    "torchvision",
    "segmentation_models_pytorch",
    "albumentations",
    "tqdm",
]

REQUIRED_FILES = [
    "app.py",
    "train.py",
    "evaluate.py",
    "prepare_dataset.py",
    "config.yaml",
    "requirements.txt",
]


def main() -> None:
    failed = False
    print("Dependency check")
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"  [OK] {module_name}: {version}")
        except Exception as exc:
            failed = True
            print(f"  [MISSING] {module_name}: {exc}")

    print("\nProject files")
    for filename in REQUIRED_FILES:
        exists = Path(filename).exists()
        failed |= not exists
        print(f"  [{'OK' if exists else 'MISSING'}] {filename}")

    checkpoint = Path("models/best_model.pth")
    print(f"\nModel checkpoint: {'available' if checkpoint.exists() else 'not added yet'}")
    for split in ("train", "val", "test"):
        image_dir = Path("data/processed") / split / "images"
        mask_dir = Path("data/processed") / split / "masks"
        image_count = len(list(image_dir.glob("*"))) if image_dir.exists() else 0
        mask_count = len(list(mask_dir.glob("*"))) if mask_dir.exists() else 0
        print(f"Dataset {split}: {image_count} images, {mask_count} masks")

    if failed:
        raise SystemExit("Some required dependencies or files are missing.")
    print("\nProject verification passed.")


if __name__ == "__main__":
    main()
