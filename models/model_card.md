# Model Card: Parking Demand Forecaster v1.0.0

## Purpose

Forecast the occupancy rate and open spaces for each of nine plaza parking lots at a given date and time, so drivers and signage can be steered toward lots with room.

## Model

Scikit-learn pipeline: one hot encoded lot code plus calendar features (day of week, weekend flag, hour, minute of day, and a cyclical time of day encoding) feeding a `HistGradientBoostingRegressor`.

## Training Data

4,138 readings from nine lots, November 20 to December 17, 2022, roughly every 30 minutes from 8:00 to 16:30. Source: plaza parking occupancy data from Bradenton, Florida, which I analyzed for the plaza owner.

## Evaluation

Time based holdout: trained on the first three weeks, tested on the final week (December 11 to 17, 1,114 readings). No shuffling, so the test set never leaks into training.

| Model | MAE (rate points) | RMSE |
|---|---|---|
| Gradient boosting | 4.86 | 6.56 |
| Seasonal baseline (same lot, weekday, hour mean) | 5.55 | 7.68 |

The model cuts error 12.5 percent against the seasonal baseline. Training fails the build if the model does not beat the baseline by at least 0.5 MAE points.

## Limitations

1. Four weeks of data from late fall. Holidays, summer, and special events are not represented.
2. Calendar features only. The model does not see live readings, weather, or events, so it forecasts typical demand, not today's surprises.
3. Readings only cover 8:00 to 16:30. Forecasts outside that window are extrapolation.
4. Rated capacity can be exceeded, so predictions are capped at 110 percent.

## Next Steps

Add live occupancy lags for short horizon forecasts, weather and event features, and drift monitoring that compares live error to the holdout MAE.
