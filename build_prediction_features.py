"""Leakage-safe feature rows for every decision point.

For each decision point at time T, builds features using ONLY
information available at or before T:

  Current state (from the decision point itself, observed at T):
    - speed (sog), course (cog), distance to nearest MPA boundary
    - heads_toward_mpa: whether the current course points at the
      nearest MPA (bearing to the nearest boundary point vs cog,
      within 30 degrees)

  Vessel history (from vessel_profiles.parquet):
    - profile snapshot as of the END OF THE DAY BEFORE T's date -
      same-day snapshots include pings from later in T's own day, so
      they are excluded by construction
    - prior_mpa_entries, prior_fishing_tracks, n_prior_mpas_visited,
      peak_hour, in_usual_hours (T's local hour within +/-2h of the
      vessel's historical peak hour)
    - has_history: False for vessels never seen before T's date
      (history features are 0/-1 filled)

Asserts that every joined profile row predates its decision point's
date, then saves to data/prediction_features.parquet.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points

from ais_mpa_join import find_file
from build_vessel_profiles import UTC_TO_LOCAL_HOURS

PROJECT_ROOT = Path(__file__).parent
DECISIONS_PATH = PROJECT_ROOT / "data" / "decision_points.parquet"
PROFILES_PATH = PROJECT_ROOT / "data" / "vessel_profiles.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "prediction_features.parquet"

PROJECTED_CRS = "EPSG:3310"
TOWARD_TOLERANCE_DEG = 30
USUAL_HOURS_TOLERANCE = 2


def bearings_to_mpas(decisions: pd.DataFrame,
                     mpas: gpd.GeoDataFrame) -> np.ndarray:
    """Compass bearing from each decision point to its nearest MPA."""
    geoms = dict(zip(mpas["NAME"], mpas.geometry))
    points = gpd.GeoSeries(
        gpd.points_from_xy(decisions["longitude"], decisions["latitude"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)

    bearings = np.empty(len(decisions))
    for i, (point, mpa_name) in enumerate(
        zip(points.values, decisions["nearest_mpa"])
    ):
        target = nearest_points(point, geoms[mpa_name].boundary)[1]
        # Planar approximation is fine at <=20 km range
        bearings[i] = np.degrees(
            np.arctan2(target.x - point.x, target.y - point.y)
        ) % 360
    return bearings


def build_features(decisions: pd.DataFrame, profiles: pd.DataFrame,
                   mpas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Build leakage-safe feature rows. `mpas` must be in EPSG:3310."""
    # --- current-state features (observed at T itself) ---
    print("Computing bearings to nearest MPA boundaries...", flush=True)
    bearing = bearings_to_mpas(decisions, mpas)
    course_offset = np.abs((decisions["cog"] - bearing + 180) % 360 - 180)
    decisions["heads_toward_mpa"] = course_offset <= TOWARD_TOLERANCE_DEG
    # On or inside the boundary, direction is moot - already there
    decisions.loc[
        decisions["dist_to_mpa_m"] == 0, "heads_toward_mpa"
    ] = True

    # --- history features (strictly before T's date) ---
    decisions["decision_date"] = decisions["base_date_time"].dt.normalize()
    profiles["profile_date"] = pd.to_datetime(profiles["date"])
    # Profile rows snapshot end-of-day; joining on date < decision_date
    # means we only ever use full days strictly before T
    cutoff = decisions["decision_date"] - pd.Timedelta(days=1)

    merged = pd.merge_asof(
        decisions.assign(cutoff=cutoff).sort_values("cutoff"),
        profiles.sort_values("profile_date"),
        left_on="cutoff",
        right_on="profile_date",
        by="mmsi",
        direction="backward",
    )

    merged["has_history"] = merged["profile_date"].notna()
    merged["prior_mpa_entries"] = merged["cum_mpa_entries"].fillna(0)
    merged["prior_fishing_tracks"] = merged["cum_fishing_tracks"].fillna(0)
    merged["n_prior_mpas_visited"] = (
        merged["mpas_visited"].fillna("")
        .map(lambda s: len(s.split(";")) if s else 0)
    )
    merged["peak_hour"] = merged["peak_hour"].fillna(-1).astype(int)

    local_hour = (
        merged["base_date_time"].dt.hour + UTC_TO_LOCAL_HOURS
    ) % 24
    hour_offset = np.minimum(
        (local_hour - merged["peak_hour"]) % 24,
        (merged["peak_hour"] - local_hour) % 24,
    )
    merged["in_usual_hours"] = merged["has_history"] & (
        hour_offset <= USUAL_HOURS_TOLERANCE
    )

    # --- leakage assertion ---
    with_history = merged[merged["has_history"]]
    leaks = (with_history["profile_date"]
             >= with_history["decision_date"]).sum()
    assert leaks == 0, f"{leaks} rows use profile data from T's date or later"
    print(f"ASSERTION PASSED: all {len(with_history):,} joined profiles "
          f"predate their decision date (0 leaks); "
          f"{(~merged['has_history']).sum():,} points had no prior history")

    feature_cols = [
        "mmsi", "base_date_time", "latitude", "longitude",
        "sog", "cog", "heading", "dist_to_mpa_m", "nearest_mpa",
        "heads_toward_mpa",
        "has_history", "prior_mpa_entries", "prior_fishing_tracks",
        "n_prior_mpas_visited", "peak_hour", "in_usual_hours",
    ]
    return merged[feature_cols]


def main() -> None:
    decisions = pd.read_parquet(DECISIONS_PATH)
    profiles = pd.read_parquet(PROFILES_PATH)
    mpas = gpd.read_file(find_file(PROJECT_ROOT / "data", "*.shp"))
    mpas = mpas.to_crs(PROJECTED_CRS)
    print(f"{len(decisions):,} decision points, "
          f"{len(profiles):,} profile snapshots")

    features = build_features(decisions, profiles, mpas)
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(features):,} feature rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
