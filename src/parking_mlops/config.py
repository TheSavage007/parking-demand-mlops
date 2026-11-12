"""Central settings so paths and thresholds live in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "parking_occupancy.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "occupancy_rate"
TEST_DAYS = 7          # hold out the final week for evaluation
RANDOM_STATE = 42
# Quality gate: the model must beat the seasonal baseline by at least this much MAE
MIN_IMPROVEMENT = 0.5
