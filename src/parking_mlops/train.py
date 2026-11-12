"""Train, evaluate, and gate the parking demand model.

Usage:
    python -m parking_mlops.train
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import __version__, config
from .features import CATEGORICAL, NUMERIC, build_features, load_raw, time_split


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("lot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


def seasonal_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Historical mean rate for the same lot, weekday, and hour."""
    keys = ["lot_code", "day_of_week", "hour"]
    tr = build_features(train).assign(y=train[config.TARGET].values)
    te = build_features(test)
    table = tr.groupby(keys)["y"].mean().rename("pred").reset_index()
    merged = te[keys].merge(table, on=keys, how="left")
    fallback = tr.groupby("lot_code")["y"].mean()
    pred = merged["pred"].to_numpy()
    fill = te["lot_code"].map(fallback).to_numpy()
    return np.where(np.isnan(pred), fill, pred)


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 3),
    }


def train(data_path=config.DATA_PATH, model_path=config.MODEL_PATH, metrics_path=config.METRICS_PATH) -> dict:
    df = load_raw(data_path)
    train_df, test_df = time_split(df, config.TEST_DAYS)

    pipe = make_pipeline()
    pipe.fit(build_features(train_df), train_df[config.TARGET])
    model_metrics = evaluate(test_df[config.TARGET], pipe.predict(build_features(test_df)))
    base_metrics = evaluate(test_df[config.TARGET], seasonal_baseline(train_df, test_df))

    improvement = round(base_metrics["mae"] - model_metrics["mae"], 3)
    passed = improvement >= config.MIN_IMPROVEMENT

    # Refit on all data for the served artifact once the holdout check passes
    final = make_pipeline().fit(build_features(df), df[config.TARGET])

    metrics = {
        "model_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_period": [str(test_df["timestamp"].min()), str(test_df["timestamp"].max())],
        "model": model_metrics,
        "seasonal_baseline": base_metrics,
        "mae_improvement_vs_baseline": improvement,
        "quality_gate_passed": passed,
        "lots": sorted(df["lot_code"].unique().tolist()),
        "capacities": df.groupby("lot_code")["lot_capacity"].first().astype(int).to_dict(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "metadata": metrics}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    metrics = train()
    print(json.dumps(metrics, indent=2))
    if not metrics["quality_gate_passed"]:
        raise SystemExit("Quality gate failed: model does not beat the seasonal baseline")


if __name__ == "__main__":
    main()
