"""Score all MPA vessel tracks and flag enforcement candidates.

Runs the trained Random Forest over every track in
data/labeled_features.csv, attaches a fishing probability, and assigns
a risk flag based on the MPA's designation:

  high_risk - fishing_probability > 0.75 inside a zone where all take
              is illegal: SMR (State Marine Reserve) or no-take SMCA
  review    - fishing_probability > 0.75 inside a regular SMCA (State
              Marine Conservation Area), where some take may be legal
  clean     - everything else

Saves the scored tracks, sorted by fishing probability, to
outputs/flagged_vessels.csv.
"""

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "labeled_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_classifier.pkl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "flagged_vessels.csv"

PROBABILITY_THRESHOLD = 0.75


def assign_flag(row: pd.Series) -> str:
    if row["fishing_probability"] <= PROBABILITY_THRESHOLD:
        return "clean"
    # Fishing-like movement where any take is illegal
    if row["mpa_type"] in ("SMR", "SMCA (No-Take)"):
        return "high_risk"
    if row["mpa_type"].startswith("SMCA"):
        return "review"
    return "clean"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    X = data[list(model.feature_names_in_)]
    data["fishing_probability"] = model.predict_proba(X)[:, 1]

    data["flag"] = data.apply(assign_flag, axis=1)

    data = data.sort_values("fishing_probability", ascending=False)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data)} scored tracks to {OUTPUT_PATH}\n")

    print("Flag breakdown (tracks):")
    print(data["flag"].value_counts().reindex(
        ["high_risk", "review", "clean"], fill_value=0).to_string())

    # High-probability tracks in zone types outside the SMR/SMCA rule
    # (e.g. SMP, SMRMA, Special Closure) end up "clean" by definition;
    # surface them so they aren't silently lost.
    other = data[
        (data["fishing_probability"] > PROBABILITY_THRESHOLD)
        & (data["flag"] == "clean")
    ]
    if not other.empty:
        print(f"\nNote: {len(other)} high-probability track(s) in other "
              f"zone types (flagged clean by rule):")
        print(other[["mmsi", "mpa_name", "mpa_type",
                     "fishing_probability"]].to_string(index=False))

    print("\nFlagged tracks:")
    flagged = data[data["flag"] != "clean"]
    print(flagged[["mmsi", "mpa_name", "mpa_type", "fishing_probability",
                   "label", "flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
