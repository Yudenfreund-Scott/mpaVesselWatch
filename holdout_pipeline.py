"""Score a held-out day of AIS data with the trained classifier.

Runs the full pipeline on data/ais_holdout/ (NOAA MarineCadastre daily
file, August 15 2024 — never seen in training): California bounding box
filter, spatial join to MPA boundaries, per-track feature extraction,
scoring with models/rf_classifier.pkl, and risk flagging. Results go to
outputs/flagged_vessels_holdout.csv.
"""

from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd

from ais_mpa_join import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, find_file, join_ais_to_mpas
from flag_vessels import assign_flag
from vessel_features import track_features

PROJECT_ROOT = Path(__file__).parent
HOLDOUT_DIR = PROJECT_ROOT / "data" / "ais_holdout"
MPA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_classifier.pkl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "flagged_vessels_holdout.csv"
INSIDE_PINGS_PATH = PROJECT_ROOT / "data" / "ais_holdout_inside_mpas.csv"

# NOAA MarineCadastre column names -> the schema the pipeline expects
NOAA_RENAME = {
    "MMSI": "mmsi",
    "BaseDateTime": "base_date_time",
    "LAT": "latitude",
    "LON": "longitude",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "VesselName": "vessel_name",
    "VesselType": "vessel_type",
}


def load_holdout_california(csv_path: Path) -> pd.DataFrame:
    print(f"Loading {csv_path.name} ({csv_path.stat().st_size / 1e6:.0f} MB)")
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(
        csv_path, usecols=list(NOAA_RENAME), chunksize=1_000_000
    ):
        chunk = chunk.rename(columns=NOAA_RENAME)
        total_rows += len(chunk)
        chunks.append(chunk[
            chunk["latitude"].between(LAT_MIN, LAT_MAX)
            & chunk["longitude"].between(LON_MIN, LON_MAX)
        ])
    ais = pd.concat(chunks, ignore_index=True)
    print(f"Total AIS rows: {total_rows:,}")
    print(f"Rows in California bbox: {len(ais):,}\n")
    return ais


def main() -> None:
    ais = load_holdout_california(find_file(HOLDOUT_DIR, "*.csv"))
    mpas = gpd.read_file(find_file(MPA_DIR, "*.shp"))

    inside = join_ais_to_mpas(ais, mpas)
    print(f"Pings inside MPAs: {len(inside):,}")

    inside = inside.rename(columns={"NAME": "mpa_name", "Type": "mpa_type"})
    # Keep the raw pings so downstream tools (e.g. the anomaly map) can
    # draw actual vessel tracks without re-reading the 1+ GB source
    inside.drop(columns=["geometry", "index_right"]).to_csv(
        INSIDE_PINGS_PATH, index=False
    )
    features = (
        inside.groupby(["mmsi", "mpa_name", "mpa_type"])
        .apply(track_features, include_groups=False)
        .reset_index()
    )
    print(f"Tracks before cleaning: {len(features)}")

    # Same cleaning as training: drop short tracks, fill NaNs with 0
    features = features[features["n_pings"] >= 3].fillna(0)
    print(f"Tracks after cleaning (n_pings >= 3): {len(features)}, "
          f"{features['mmsi'].nunique()} unique vessels\n")

    model = joblib.load(MODEL_PATH)
    X = features[list(model.feature_names_in_)]
    features["fishing_probability"] = model.predict_proba(X)[:, 1]
    features["flag"] = features.apply(assign_flag, axis=1)

    features = features.sort_values("fishing_probability", ascending=False)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(features)} scored tracks to {OUTPUT_PATH}\n")

    print("Flag breakdown (tracks):")
    print(features["flag"].value_counts().reindex(
        ["high_risk", "review", "clean"], fill_value=0).to_string())

    flagged = features[features["flag"] != "clean"]
    if not flagged.empty:
        print("\nFlagged tracks:")
        print(flagged[["mmsi", "mpa_name", "mpa_type",
                       "fishing_probability", "flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
