"""Train a fishing / non-fishing Random Forest on the MPA track features.

Evaluates with 5-fold stratified group cross-validation — grouped by
MMSI so a vessel's tracks never span train and test folds — then fits
on all rows and saves the model to models/rf_classifier.pkl.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "labeled_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_classifier.pkl"

FEATURES = [
    "mean_speed",
    "speed_std",
    "mean_heading_change_rate",
    "time_in_mpa_minutes",
    "max_stationary_minutes",
    "n_direction_reversals",
    "n_pings",
]
RANDOM_STATE = 42


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["label"]
    print(f"{len(data)} tracks: {y.sum()} fishing, {(y == 0).sum()} non-fishing\n")

    model = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    )

    groups = data["mmsi"]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model, X, y, cv=cv, groups=groups, scoring=("precision", "recall", "f1")
    )

    print("5-fold stratified group CV, grouped by MMSI "
          "(positive class = fishing):")
    for metric in ("precision", "recall", "f1"):
        vals = scores[f"test_{metric}"]
        print(f"  {metric:<10} {vals.mean():.3f} +/- {vals.std():.3f}   "
              f"folds: {[round(float(v), 2) for v in vals]}")

    # Out-of-fold predictions for every track, for a pooled confusion matrix
    y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print("\nConfusion matrix summed across all 5 folds:")
    print(f"  True positives:  {tp:>3}   (fishing correctly flagged)")
    print(f"  False positives: {fp:>3}   (non-fishing wrongly flagged)")
    print(f"  True negatives:  {tn:>3}   (non-fishing correctly passed)")
    print(f"  False negatives: {fn:>3}   (fishing missed)")

    model.fit(X, y)
    print("\nFeature importances (full-data fit):")
    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]
    )
    for name, imp in importances:
        print(f"  {name:<26} {imp:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel fit on all {len(data)} rows saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
