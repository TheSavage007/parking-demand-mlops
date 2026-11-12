"""Feature engineering shared by training and serving.

Keeping one function for both paths prevents training and serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["lot_code"]
NUMERIC = [
    "lot_capacity",
    "day_of_week",
    "is_weekend",
    "hour",
    "minute_of_day",
    "tod_sin",
    "tod_cos",
]
FEATURES = CATEGORICAL + NUMERIC


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"lot_code", "lot_capacity", "lot_occupancy", "timestamp", "occupancy_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.sort_values("timestamp").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar and lot features from lot_code, lot_capacity, and timestamp."""
    ts = pd.to_datetime(df["timestamp"])
    out = pd.DataFrame(index=df.index)
    out["lot_code"] = df["lot_code"].astype(str)
    out["lot_capacity"] = df["lot_capacity"].astype(float)
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour"] = ts.dt.hour
    out["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    angle = 2 * np.pi * out["minute_of_day"] / 1440
    out["tod_sin"] = np.sin(angle)
    out["tod_cos"] = np.cos(angle)
    return out[FEATURES]


def time_split(df: pd.DataFrame, test_days: int):
    """Train on the earlier period, test on the final test_days. No shuffling."""
    cutoff = df["timestamp"].max().normalize() - pd.Timedelta(days=test_days - 1)
    return df[df["timestamp"] < cutoff], df[df["timestamp"] >= cutoff]
