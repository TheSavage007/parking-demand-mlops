import joblib
import pytest
from fastapi.testclient import TestClient

from parking_mlops import predict as predict_module


@pytest.fixture
def client(trained, monkeypatch):
    artifact = joblib.load(trained["model_path"])
    predict_module.load_model.cache_clear()
    monkeypatch.setattr(predict_module, "load_model", lambda *a, **k: artifact)
    from app import main

    monkeypatch.setattr(main, "load_model", lambda *a, **k: artifact)
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_bounded_forecast(client):
    r = client.post("/predict", json={"lot_code": "Lot02", "timestamp": "2022-12-14T12:30:00"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["predicted_occupancy_rate"] <= 110
    assert 0 <= body["predicted_open_spaces"] <= body["capacity"]


def test_unknown_lot_is_404(client):
    r = client.post("/predict", json={"lot_code": "Lot99", "timestamp": "2022-12-14T12:30:00"})
    assert r.status_code == 404


def test_recommend_ranks_quiet_lots_first_at_weekday_peak(client):
    ranked = client.get("/recommend", params={"timestamp": "2022-12-14T13:00:00"}).json()
    assert len(ranked) == 9
    assert ranked[0]["predicted_open_spaces"] >= ranked[-1]["predicted_open_spaces"]
    assert ranked[-1]["lot_code"] in {"Lot01", "Lot02", "Lot06"}
