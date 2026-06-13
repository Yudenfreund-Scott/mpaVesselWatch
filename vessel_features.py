"""Per-vessel movement features for AIS tracks inside MPAs.

Reads data/ais_inside_mpas.csv, groups pings into tracks (one vessel in
one MPA), computes movement features for each track, and writes the
feature matrix to data/vessel_features.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
INPUT_PATH = PROJECT_ROOT / "data" / "ais_inside_mpas.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "vessel_features.csv"

STATIONARY_KNOTS = 0.5
# Course change (degrees) between consecutive pings that counts as a
# direction reversal
REVERSAL_DEGREES = 150.0


def angular_diff(courses: pd.Series) -> pd.Series:
    """Signed change between consecutive courses, wrapped to [-180, 180]."""
    diff = courses.diff()
    return (diff + 180) % 360 - 180


def max_consecutive_stationary_minutes(
    stationary: pd.Series, minutes_elapsed: pd.Series
) -> float:
    """Longest unbroken stretch of stationary pings, in minutes.

    `minutes_elapsed[i]` is the gap between ping i-1 and ping i; it is
    counted toward a stretch when both endpoints are stationary.
    """
    longest = current = 0.0
    for i in range(1, len(stationary)):
        if stationary.iloc[i] and stationary.iloc[i - 1]:
            current += minutes_elapsed.iloc[i]
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def track_features(track: pd.DataFrame) -> pd.Series:
    """Compute movement features for one vessel's pings inside one MPA.

    Expects columns: base_date_time, sog, cog. Pings are sorted by time
    before computing anything sequential.
    """
    track = track.sort_values("base_date_time")
    times = pd.to_datetime(track["base_date_time"])
    minutes_elapsed = times.diff().dt.total_seconds() / 60

    sog = track["sog"]
    course_change = angular_diff(track["cog"])

    # Degrees per minute between consecutive pings; guard against
    # duplicate timestamps (zero elapsed time)
    valid_gap = minutes_elapsed > 0
    change_rate = (course_change.abs()[valid_gap] / minutes_elapsed[valid_gap])

    stationary = sog < STATIONARY_KNOTS

    return pd.Series(
        {
            "n_pings": len(track),
            "mean_speed": sog.mean(),
            "speed_std": sog.std(ddof=0),
            "mean_heading_change_rate": change_rate.mean(),
            "time_in_mpa_minutes": (times.iloc[-1] - times.iloc[0]).total_seconds() / 60,
            "max_stationary_minutes": max_consecutive_stationary_minutes(
                stationary, minutes_elapsed
            ),
            "n_direction_reversals": int(
                (course_change.abs() > REVERSAL_DEGREES).sum()
            ),
        }
    )


def main() -> None:
    ais = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(ais):,} pings, {ais['mmsi'].nunique()} unique vessels")

    features = (
        ais.groupby(["mmsi", "mpa_name", "mpa_type"])
        .apply(track_features, include_groups=False)
        .reset_index()
    )

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(features)} vessel-MPA tracks to {OUTPUT_PATH}\n")
    print(features.describe().round(2))


if __name__ == "__main__":
    main()
