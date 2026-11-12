import pandas as pd

from parking_mlops import config
from parking_mlops.features import FEATURES, build_features, load_raw, time_split


def test_build_features_columns_and_values():
    frame = pd.DataFrame(
        [{"lot_code": "Lot02", "lot_capacity": 387, "timestamp": pd.Timestamp("2022-12-17 12:30")}]
    )
    feats = build_features(frame)
    assert list(feats.columns) == FEATURES
    row = feats.iloc[0]
    assert row["day_of_week"] == 5 and row["is_weekend"] == 1
    assert row["minute_of_day"] == 750


def test_time_split_has_no_overlap():
    df = load_raw(config.DATA_PATH)
    train, test = time_split(df, config.TEST_DAYS)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(test) == len(df)


def test_data_quality():
    df = load_raw(config.DATA_PATH)
    assert df["lot_code"].nunique() == 9
    assert df[["lot_occupancy", "lot_capacity"]].ge(0).all().all()
    assert not df.isna().any().any()
