"""Convert segmentation masks into interpretable damage and priority scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .constants import (
    FLOODED_BUILDING,
    FLOODED_ROAD,
    NON_FLOODED_BUILDING,
    NON_FLOODED_ROAD,
    WATER,
)


@dataclass
class DamageReport:
    flooded_building_percent: float
    flooded_road_percent: float
    water_coverage_percent: float
    priority_score: float
    priority_level: str
    flooded_building_pixels: int
    flooded_road_pixels: int
    water_pixels: int
    total_pixels: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_percent(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole > 0 else 0.0


def calculate_damage_report(mask: np.ndarray, priority_config: dict) -> DamageReport:
    total = int(mask.size)
    building_flooded = int(np.count_nonzero(mask == FLOODED_BUILDING))
    building_non_flooded = int(np.count_nonzero(mask == NON_FLOODED_BUILDING))
    road_flooded = int(np.count_nonzero(mask == FLOODED_ROAD))
    road_non_flooded = int(np.count_nonzero(mask == NON_FLOODED_ROAD))
    water = int(np.count_nonzero(mask == WATER))

    building_percent = _safe_percent(
        building_flooded, building_flooded + building_non_flooded
    )
    road_percent = _safe_percent(road_flooded, road_flooded + road_non_flooded)
    water_percent = _safe_percent(water, total)

    score = (
        float(priority_config.get("building_weight", 0.50)) * building_percent
        + float(priority_config.get("road_weight", 0.35)) * road_percent
        + float(priority_config.get("water_weight", 0.15)) * water_percent
    )
    low_max = float(priority_config.get("low_max", 30.0))
    medium_max = float(priority_config.get("medium_max", 60.0))
    if score <= low_max:
        level = "Low"
    elif score <= medium_max:
        level = "Medium"
    else:
        level = "High"

    return DamageReport(
        flooded_building_percent=round(building_percent, 2),
        flooded_road_percent=round(road_percent, 2),
        water_coverage_percent=round(water_percent, 2),
        priority_score=round(score, 2),
        priority_level=level,
        flooded_building_pixels=building_flooded,
        flooded_road_pixels=road_flooded,
        water_pixels=water,
        total_pixels=total,
    )
