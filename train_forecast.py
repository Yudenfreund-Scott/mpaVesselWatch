"""Train the 24-hour MPA fishing-entry forecast model.

LightGBM classifier on data/training_set.parquet with a strict
calendar-time split: train on the first 42 days (June 1 - July 12),
test on the held-out final 30 days (July 13 - August 11). A random
split would let the model see later patterns while "predicting"
earlier days, silently inflating every metric.

Saves {model, feature list, split date} to models/forecast_model.pkl.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "training_set.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "forecast_model.pkl"

SPLIT_DATE = "2024-07-13"  # train strictly before, test at/after

FEATURES = [
    # current state at T
    "sog", "cog", "heading", "dist_to_mpa_m", "heads_toward_mpa",
    # vessel history strictly before T
    "has_history", "prior_mpa_entries", "prior_fishing_tracks",
    "n_prior_mpas_visited", "peak_hour", "in_usual_hours",
    # environment at T
    "tide_height_m", "tide_rising", "wind_speed_kn", "wind_dir_deg",
    "wave_height_m", "traffic_within_10km",
    # calendar / seasons
    "month", "day_of_week", "n_open_seasons",
    "salmon_open", "crab_open", "lobster_open", "rockfish_open",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    # AIS uses 511 as "heading unavailable"; LightGBM handles NaN natively
    X.loc[X["heading"] == 511, "heading"] = np.nan
    for col in X.select_dtypes("bool"):
        X[col] = X[col].astype(int)
    return X


def main() -> None:
    data = pd.read_parquet(DATA_PATH)
    split = pd.Timestamp(SPLIT_DATE)
    train = data[data["base_date_time"] < split]
    test = data[data["base_date_time"] >= split]

    X_train, y_train = prepare(train), train["label"]
    X_test, y_test = prepare(test), test["label"]

    pos = y_train.sum()
    neg = len(y_train) - pos
    print(f"Train: {len(train):,} rows ({pos:,} pos, rate {pos/len(train):.2%}), "
          f"{train['base_date_time'].dt.date.nunique()} days")
    print(f"Test:  {len(test):,} rows ({y_test.sum():,} pos, "
          f"rate {y_test.mean():.2%}), "
          f"{test['base_date_time'].dt.date.nunique()} days")

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        scale_pos_weight=neg / pos,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    print("\nTop feature importances:")
    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]
    )
    for name, imp in importances[:10]:
        print(f"  {name:<24} {imp}")

    joblib.dump(
        {"model": model, "features": FEATURES, "split_date": SPLIT_DATE},
        MODEL_PATH,
    )
    print(f"\nSaved model to {MODEL_PATH} "
          f"(trained through {SPLIT_DATE}, exclusive)")


if __name__ == "__main__":
    main()
