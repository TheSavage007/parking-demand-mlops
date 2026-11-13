"""REST API that serves parking occupancy forecasts.

Run locally:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from parking_mlops import __version__
from parking_mlops.predict import load_model, predict, recommend

app = FastAPI(
    title="Parking Demand Forecast API",
    version=__version__,
    description="Forecasts occupancy and open spaces for nine plaza parking lots.",
)


class ForecastRequest(BaseModel):
    lot_code: str = Field(examples=["Lot02"])
    timestamp: datetime = Field(examples=["2022-12-14T12:30:00"])


class Forecast(BaseModel):
    lot_code: str
    timestamp: str
    predicted_occupancy_rate: float
    capacity: int
    predicted_open_spaces: int
    model_version: str


@app.get("/health")
def health() -> dict:
    meta = load_model()["metadata"]
    return {"status": "ok", "model_version": meta["model_version"], "trained_at": meta["trained_at"]}


@app.get("/lots")
def lots() -> dict:
    return load_model()["metadata"]["capacities"]


@app.get("/metrics")
def metrics() -> dict:
    meta = load_model()["metadata"]
    return {k: meta[k] for k in ("model", "seasonal_baseline", "mae_improvement_vs_baseline", "test_period")}


@app.post("/predict", response_model=Forecast)
def predict_one(req: ForecastRequest) -> dict:
    try:
        return predict(req.lot_code, req.timestamp)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.get("/recommend", response_model=list[Forecast])
def recommend_lots(timestamp: datetime) -> list[dict]:
    """All lots ranked by predicted open spaces at the given time."""
    return recommend(timestamp)
