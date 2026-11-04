"""Project-wide class definitions and visualization palette."""

CLASS_NAMES = [
    "Background",
    "Building Flooded",
    "Building Non-Flooded",
    "Road Flooded",
    "Road Non-Flooded",
    "Water",
    "Tree",
    "Vehicle",
    "Pool",
    "Grass",
]

# RGB colors used only for visualization. Training masks remain class-index masks.
PALETTE = [
    (20, 20, 20),       # Background
    (220, 20, 60),      # Building flooded
    (255, 165, 0),      # Building non-flooded
    (255, 215, 0),      # Road flooded
    (160, 160, 160),    # Road non-flooded
    (30, 144, 255),     # Water
    (34, 139, 34),      # Tree
    (148, 0, 211),      # Vehicle
    (0, 255, 255),      # Pool
    (124, 252, 0),      # Grass
]

FLOODED_BUILDING = 1
NON_FLOODED_BUILDING = 2
FLOODED_ROAD = 3
NON_FLOODED_ROAD = 4
WATER = 5
