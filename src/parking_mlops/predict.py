"""Load the trained artifact and score requests."""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from . import config
from .features import build_features


@lru_cache(maxsize=1)
def load_model(path: str = str(config.MODEL_PATH)) -> dict:
    return joblib.load(path)


def predict(lot_code: str, timestamp, artifact: dict | None = None) -> dict:
    artifact = artifact or load_model()
    meta = artifact["metadata"]
    if lot_code not in meta["capacities"]:
        raise KeyError(f"Unknown lot: {lot_code}")
    capacity = meta["capacities"][lot_code]
    frame = pd.DataFrame(
        [{"lot_code": lot_code, "lot_capacity": capacity, "timestamp": pd.Timestamp(timestamp)}]
    )
    rate = float(artifact["pipeline"].predict(build_features(frame))[0])
    rate = max(0.0, min(rate, 110.0))
    expected_cars = round(capacity * rate / 100)
    return {
        "lot_code": lot_code,
        "timestamp": pd.Timestamp(timestamp).isoformat(),
        "predicted_occupancy_rate": round(rate, 1),
        "capacity": capacity,
        "predicted_open_spaces": max(0, capacity - expected_cars),
        "model_version": meta["model_version"],
    }


def recommend(timestamp, artifact: dict | None = None) -> list[dict]:
    """Every lot ranked by predicted open spaces, most first."""
    artifact = artifact or load_model()
    preds = [predict(lot, timestamp, artifact) for lot in artifact["metadata"]["lots"]]
    return sorted(preds, key=lambda p: p["predicted_open_spaces"], reverse=True)
