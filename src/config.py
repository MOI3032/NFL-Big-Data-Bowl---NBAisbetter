"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "nfl-big-data-bowl-regional-event-data-main" / "data"
TRACKING_DIR = RAW_DIR / "tracking"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

FIELD_X = 120.0
FIELD_Y = 53.3
FPS = 10.0

# Real event names found in the tracking files (no "pass_shovel" in this release).
SNAP_EVENT = "ball_snap"
END_EVENTS = ["pass_forward", "pass_shovel", "qb_sack", "qb_strip_sack"]
END_FALLBACK = "autoevent_passforward"  # used only when no manual end event exists

# Real pff_role labels (capitalised differently from the brief).
ROLE_QB = "Pass"
ROLE_RUSH = "Pass Rush"
ROLE_BLOCK = "Pass Block"

KEEP_PASS_RESULTS = ["C", "I", "S", "IN"]

PRESSURE_RADIUS = 2.0
RADII = [1.5, 2.0, 3.0]

TTT_MIN, TTT_MAX = 0.5, 8.0
MIN_TEAM_PLAYS = 60

# PIS weights on z-scores: cushion, clean-time share, mean pocket area.
PIS_WEIGHTS = {"cushion": 0.45, "clean_time_pct": 0.40, "pocket_area": 0.15}

RUSH_ORDER = ["≤3", "4", "5", "6+"]
BOX_ORDER = ["≤5", "6", "7", "8+"]
