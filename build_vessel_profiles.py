"""Per-vessel historical profiles, stored as running daily cumulatives.

For every MMSI, walks the 92 days in chronological order and emits one
profile row per (vessel, active day) holding that vessel's cumulative
history THROUGH THE END OF that day:

  - cum_mpa_entries:    vessel-MPA tracks recorded so far
  - cum_fishing_tracks: those tracks made while the vessel was known to
                        be a fishing vessel (GFW registry, which predates
                        the study window, or a fishing AIS vessel type
                        seen on or before that day - never future data)
  - mpas_visited:       distinct MPAs entered so far (";"-joined)
  - peak_hour:          most common operating hour-of-day so far
                        (local time, Pacific = UTC-7 in summer)
  - home_lat/home_lon:  center of the 0.05-degree grid cell where the
                        vessel has spent the most nighttime (20:00-06:00
                        local) stationary (<0.5 kn) pings so far

Because each row is a snapshot as of its date, downstream code can ask
"what did we know about this vessel on day D?" by taking the latest row
with date < D - no future behavior ever leaks into a past prediction.

Saves to data/vessel_profiles.parquet.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_labeled_set import FISHING_VESSEL_TYPES

PROJECT_ROOT = Path(__file__).parent
AIS_CA_DIR = PROJECT_ROOT / "data" / "ais_ca"
TRACKS_PATH = PROJECT_ROOT / "data" / "all_mpa_tracks.parquet"
GFW_PATH = PROJECT_ROOT / "data" / "gfw" / "fishing-vessels-v3.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "vessel_profiles.parquet"

UTC_TO_LOCAL_HOURS = -7          # Pacific Daylight Time, valid Jun-Aug
NIGHT_HOURS = set(range(20, 24)) | set(range(0, 6))
STATIONARY_KNOTS = 0.5
GRID_DEG = 0.05


def main() -> None:
    tracks = pd.read_parquet(TRACKS_PATH)
    tracks_by_date = dict(tuple(tracks.groupby("date")))

    # GFW registry covers 2012-2024, compiled before the study window,
    # so it is safe as-of any date in the window
    gfw_mmsis = set(pd.read_csv(GFW_PATH, usecols=["mmsi"])["mmsi"].unique())

    hour_counts: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(24))
    night_cells: dict[int, Counter] = defaultdict(Counter)
    cum_entries: Counter = Counter()
    cum_fishing: Counter = Counter()
    mpas_visited: dict[int, set] = defaultdict(set)
    seen_fishing_type: set[int] = set()

    rows = []
    for path in sorted(AIS_CA_DIR.glob("*.parquet")):
        day = path.stem
        pings = pd.read_parquet(
            path, columns=["mmsi", "base_date_time", "latitude",
                           "longitude", "sog", "vessel_type"]
        )

        seen_fishing_type.update(
            pings.loc[pings["vessel_type"].isin(FISHING_VESSEL_TYPES), "mmsi"]
        )

        local_hour = (pings["base_date_time"].dt.hour + UTC_TO_LOCAL_HOURS) % 24
        hour_hist = (
            pd.DataFrame({"mmsi": pings["mmsi"], "hour": local_hour})
            .groupby(["mmsi", "hour"]).size()
        )
        for (mmsi, hour), n in hour_hist.items():
            hour_counts[mmsi][hour] += n

        night_stationary = pings[
            local_hour.isin(NIGHT_HOURS) & (pings["sog"] < STATIONARY_KNOTS)
        ]
        cells = (
            night_stationary.assign(
                cell_lat=(night_stationary["latitude"] / GRID_DEG).round()
                * GRID_DEG,
                cell_lon=(night_stationary["longitude"] / GRID_DEG).round()
                * GRID_DEG,
            )
            .groupby(["mmsi", "cell_lat", "cell_lon"]).size()
        )
        for (mmsi, cell_lat, cell_lon), n in cells.items():
            night_cells[mmsi][(cell_lat, cell_lon)] += n

        day_tracks = tracks_by_date.get(day)
        if day_tracks is not None:
            for mmsi, n in day_tracks.groupby("mmsi").size().items():
                cum_entries[mmsi] += n
                if mmsi in gfw_mmsis or mmsi in seen_fishing_type:
                    cum_fishing[mmsi] += n
            for mmsi, names in day_tracks.groupby("mmsi")["mpa_name"]:
                mpas_visited[mmsi].update(names)

        # Snapshot for every vessel active today, as of end of today
        for mmsi in pings["mmsi"].unique():
            counts = hour_counts[mmsi]
            home = night_cells[mmsi].most_common(1)
            rows.append({
                "mmsi": mmsi,
                "date": day,
                "cum_mpa_entries": cum_entries[mmsi],
                "cum_fishing_tracks": cum_fishing[mmsi],
                "mpas_visited": ";".join(sorted(mpas_visited[mmsi])),
                "peak_hour": int(counts.argmax()),
                "home_lat": home[0][0][0] if home else np.nan,
                "home_lon": home[0][0][1] if home else np.nan,
            })
        print(f"{day}: {pings['mmsi'].nunique():,} active vessels", flush=True)

    profiles = pd.DataFrame(rows)
    profiles.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(profiles):,} daily profile snapshots for "
          f"{profiles['mmsi'].nunique():,} vessels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
