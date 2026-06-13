"""Build a binary fishing / non-fishing labeled set for MPA vessel tracks.

Labels every track in data/vessel_features.csv:
  1 (fishing)     - MMSI appears in the GFW fishing-vessels-v3 registry,
                    OR the vessel self-reports a fishing vessel_type in
                    its raw AIS messages
  0 (non-fishing) - neither condition holds

Writes the result to data/labeled_features.csv.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
GFW_PATH = PROJECT_ROOT / "data" / "gfw" / "fishing-vessels-v3.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "vessel_features.csv"
AIS_PATH = PROJECT_ROOT / "data" / "ais_inside_mpas.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "labeled_features.csv"

# AIS vessel type codes that indicate fishing: 30 is the ITU standard
# "fishing" code; 1001/1002 are legacy US codes for commercial fishing
# and fish processing used in some NOAA-sourced feeds.
FISHING_VESSEL_TYPES = {30, 1001, 1002}


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    print(f"Feature matrix: {len(features)} tracks, "
          f"{features['mmsi'].nunique()} unique vessels")

    # Pass 1: vessel appears in the GFW fishing vessel registry
    gfw_mmsis = set(
        pd.read_csv(GFW_PATH, usecols=["mmsi"])["mmsi"].unique()
    )
    in_gfw = features["mmsi"].isin(gfw_mmsis)

    # Pass 2: vessel self-reports a fishing type in its AIS messages.
    # vessel_type can vary across a vessel's pings, so flag the MMSI if
    # any of its pings carries a fishing code.
    ais_types = pd.read_csv(AIS_PATH, usecols=["mmsi", "vessel_type"])
    fishing_type_mmsis = set(
        ais_types.loc[
            ais_types["vessel_type"].isin(FISHING_VESSEL_TYPES), "mmsi"
        ].unique()
    )
    ais_says_fishing = features["mmsi"].isin(fishing_type_mmsis)

    features["label"] = (in_gfw | ais_says_fishing).astype(int)
    features["label_source"] = ""
    features.loc[in_gfw, "label_source"] = "gfw"
    features.loc[ais_says_fishing & ~in_gfw, "label_source"] = "ais_vessel_type"
    features.loc[ais_says_fishing & in_gfw, "label_source"] = "gfw+ais_vessel_type"

    n_vessels = features["mmsi"].nunique()
    fishing_vessels = features.loc[features["label"] == 1, "mmsi"].nunique()
    print(f"\nLabel breakdown (tracks):")
    print(features["label"].value_counts().rename(
        {1: "1 (fishing)", 0: "0 (non-fishing)"}).to_string())
    print(f"\nLabel source breakdown (tracks):")
    print(features.loc[features["label"] == 1, "label_source"]
          .value_counts().to_string())
    print(f"\nVessels: {fishing_vessels} fishing / "
          f"{n_vessels - fishing_vessels} non-fishing of {n_vessels} total")

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
