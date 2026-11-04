from pathlib import Path

import numpy as np

from src.priority import calculate_damage_report
from src.visual import colorize_mask, create_overlay


def test_priority_and_visualization():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:8, :8] = 1
    mask[8:16, :8] = 2
    mask[:8, 8:16] = 3
    mask[8:16, 8:16] = 4
    mask[16:, :] = 5
    report = calculate_damage_report(mask, {})
    assert report.flooded_building_percent == 50.0
    assert report.flooded_road_percent == 50.0
    assert colorize_mask(mask).shape == (32, 32, 3)
    overlay = create_overlay(np.zeros((32, 32, 3), dtype=np.uint8), mask)
    assert overlay.shape == (32, 32, 3)


def test_required_files_exist():
    for name in ["app.py", "train.py", "prepare_dataset.py", "config.yaml"]:
        assert Path(name).exists()
