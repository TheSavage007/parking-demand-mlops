# Parking Demand MLOps

An end to end machine learning service that forecasts parking lot occupancy and recommends the lot with the most open spaces. It covers the full lifecycle: feature pipeline, time based evaluation against a baseline, an automated quality gate, a FastAPI service, a Docker image, and CI that tests, trains, builds, and smoke tests on every push.

![CI](https://github.com/TheSavage007/parking-demand-mlops/actions/workflows/ci.yml/badge.svg)

## Why This Exists

The companion analysis, [parking-occupancy-analysis](https://github.com/TheSavage007/parking-occupancy-analysis), found that the plaza's parking problem is distribution, not capacity. At the weekday midday peak one lot has about 6 open spaces while another has more than 500. This service turns that finding into something a sign, an app, or a property dashboard can call.

## Architecture

```
data/parking_occupancy.csv
        │
        ▼
features.py ── shared by training and serving (no skew)
        │
        ▼
train.py ── time split ─► model vs seasonal baseline ─► quality gate ─► models/model.joblib + metrics.json
        │
        ▼
app/main.py (FastAPI) ─► /predict  /recommend  /lots  /metrics  /health
        │
        ▼
Dockerfile ─► image trains at build time and serves on port 8000
        │
        ▼
GitHub Actions ─► ruff ─► pytest ─► train with gate ─► docker build ─► container smoke test
```

## Results

Trained on three weeks, tested on the held out final week (1,114 readings).

| Model | MAE (rate points) | RMSE |
|---|---|---|
| Gradient boosting | **4.86** | **6.56** |
| Seasonal baseline | 5.55 | 7.68 |

The build fails if the model stops beating the baseline by at least 0.5 MAE points. Details and limitations are in the [model card](models/model_card.md).

## API

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/health` | Status, model version, training time |
| GET | `/lots` | Every lot and its capacity |
| GET | `/metrics` | Holdout metrics for the served model |
| POST | `/predict` | Forecast for one lot at one time |
| GET | `/recommend?timestamp=` | All lots ranked by predicted open spaces |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"lot_code": "Lot02", "timestamp": "2022-12-14T12:30:00"}'
```

```json
{"lot_code": "Lot02", "timestamp": "2022-12-14T12:30:00", "predicted_occupancy_rate": 99.9,
 "capacity": 387, "predicted_open_spaces": 0, "model_version": "1.0.0"}
```

`/recommend` for Wednesday at 13:00 sends drivers to Lot04 (about 531 open), then Lots 03 and 08, and puts Lot02 last.

Interactive docs are at `localhost:8000/docs` once the service is running.

## Run It

```bash
make install     # dependencies
make test        # lint free, 9 tests: features, data quality, training gate, API
make train       # writes models/model.joblib and models/metrics.json
make serve       # http://localhost:8000/docs
make docker      # build and run the container
```

## Project Structure

```
parking-demand-mlops/
├── app/main.py                 FastAPI service
├── src/parking_mlops/
│   ├── config.py               paths and quality gate threshold
│   ├── features.py             shared feature pipeline
│   ├── train.py                training, evaluation, gate
│   └── predict.py              inference and lot ranking
├── models/                     metrics.json and model card
├── tests/                      pytest suite
├── Dockerfile
└── .github/workflows/ci.yml
```

## Tech Stack

Python, pandas, scikit-learn, FastAPI, Pydantic, Docker, GitHub Actions, pytest, Ruff

## Author

Randy O. Powell | [GitHub](https://github.com/TheSavage007) | [LinkedIn](https://www.linkedin.com/in/randy-powell-224a92bb)
