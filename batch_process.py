"""Spatial join + feature extraction for every day in data/ais_ca/.

For each daily parquet: join pings to MPA boundaries (reusing
ais_mpa_join.join_ais_to_mpas), extract per-track movement features
(reusing vessel_features.track_features), and tag each track with its
date. All days are concatenated into data/all_mpa_tracks.parquet.

A track's is_fishing_vessel flag uses the same rule as training:
MMSI in the GFW fishing-vessels-v3 registry, or the vessel broadcast a
fishing AIS vessel type at any point in the period.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ais_mpa_join import find_file, join_ais_to_mpas
from build_labeled_set import FISHING_VESSEL_TYPES
from vessel_features import track_features

PROJECT_ROOT = Path(__file__).parent
AIS_CA_DIR = PROJECT_ROOT / "data" / "ais_ca"
GFW_PATH = PROJECT_ROOT / "data" / "gfw" / "fishing-vessels-v3.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "all_mpa_tracks.parquet"
# Ping-level times inside MPAs, kept for the 24h-lookahead labeler
MPA_PINGS_DIR = PROJECT_ROOT / "data" / "mpa_pings"


def main() -> None:
    mpas = gpd.read_file(find_file(PROJECT_ROOT / "data", "*.shp"))
    daily_files = sorted(AIS_CA_DIR.glob("*.parquet"))
    print(f"{len(daily_files)} daily files to process")

    all_tracks = []
    ais_fishing_mmsis: set[int] = set()
    for path in daily_files:
        ais = pd.read_parquet(path)

        ais_fishing_mmsis.update(
            ais.loc[ais["vessel_type"].isin(FISHING_VESSEL_TYPES), "mmsi"]
        )

        inside = join_ais_to_mpas(ais, mpas)
        if inside.empty:
            print(f"{path.stem}: no pings inside MPAs", flush=True)
            continue
        inside = inside.rename(columns={"NAME": "mpa_name", "Type": "mpa_type"})

        MPA_PINGS_DIR.mkdir(exist_ok=True)
        inside[["mmsi", "base_date_time", "mpa_name", "mpa_type"]].to_parquet(
            MPA_PINGS_DIR / f"{path.stem}.parquet", index=False
        )

        tracks = (
            inside.groupby(["mmsi", "mpa_name", "mpa_type"])
            .apply(track_features, include_groups=False)
            .reset_index()
        )
        tracks["date"] = path.stem
        all_tracks.append(tracks)
        print(f"{path.stem}: {len(inside):,} pings inside MPAs, "
              f"{len(tracks)} tracks", flush=True)

    tracks = pd.concat(all_tracks, ignore_index=True)

    gfw_mmsis = set(pd.read_csv(GFW_PATH, usecols=["mmsi"])["mmsi"].unique())
    tracks["is_fishing_vessel"] = tracks["mmsi"].isin(
        gfw_mmsis | ais_fishing_mmsis
    )

    tracks.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(tracks):,} tracks to {OUTPUT_PATH}")
    print(f"Unique vessels: {tracks['mmsi'].nunique():,}")
    print(f"Fishing-flagged tracks: {tracks['is_fishing_vessel'].sum():,} "
          f"({tracks.loc[tracks['is_fishing_vessel'], 'mmsi'].nunique():,} "
          f"unique fishing vessels)")


if __name__ == "__main__":
    main()
