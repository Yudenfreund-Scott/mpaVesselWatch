"""Generate forecast decision points from the daily California AIS files.

A decision point is a moment we'd want to predict whether a vessel is
about to enter (or misbehave in) an MPA: any ping within 20 km of an
MPA boundary, sampled at most once per vessel per 6-hour window (the
first qualifying ping in each window).

Each decision point records: MMSI, timestamp, lat/lon, speed, course,
heading, distance to the nearest MPA boundary (meters), and which MPA
that is. Saves to data/decision_points.parquet.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ais_mpa_join import find_file

PROJECT_ROOT = Path(__file__).parent
AIS_CA_DIR = PROJECT_ROOT / "data" / "ais_ca"
OUTPUT_PATH = PROJECT_ROOT / "data" / "decision_points.parquet"

# California Albers - meters, accurate statewide
PROJECTED_CRS = "EPSG:3310"
NEAR_DISTANCE_M = 20_000
WINDOW = pd.Timedelta(hours=6)


def load_boundaries(mpas: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {"mpa_name": mpas["NAME"]},
        geometry=mpas.to_crs(PROJECTED_CRS).geometry.boundary,
        crs=PROJECTED_CRS,
    )


def extract_decision_points(pings: pd.DataFrame,
                            boundaries: "gpd.GeoDataFrame") -> pd.DataFrame:
    """Decision points for one day of pings: first ping per vessel per
    6h window that is within 20 km of an MPA boundary."""
    points = gpd.GeoDataFrame(
        pings,
        geometry=gpd.points_from_xy(pings["longitude"], pings["latitude"],
                                    crs="EPSG:4326"),
    ).to_crs(PROJECTED_CRS)

    near = gpd.sjoin_nearest(
        points, boundaries,
        max_distance=NEAR_DISTANCE_M,
        distance_col="dist_to_mpa_m",
    ).drop(columns=["geometry", "index_right"])
    # A ping equidistant to two boundaries joins twice; keep one
    near = near[~near.index.duplicated(keep="first")]

    # At most one decision point per vessel per 6-hour window:
    # keep the first qualifying ping in each window
    near = near.sort_values("base_date_time")
    window = near["base_date_time"].dt.floor(WINDOW)
    near = near.groupby(["mmsi", window]).head(1)
    return near.rename(columns={"mpa_name": "nearest_mpa"})


def main() -> None:
    mpas = gpd.read_file(find_file(PROJECT_ROOT / "data", "*.shp"))
    boundaries = load_boundaries(mpas)

    all_points = []
    for path in sorted(AIS_CA_DIR.glob("*.parquet")):
        pings = pd.read_parquet(
            path, columns=["mmsi", "base_date_time", "latitude", "longitude",
                           "sog", "cog", "heading"]
        )
        near = extract_decision_points(pings, boundaries)
        all_points.append(near)
        print(f"{path.stem}: {len(near):,} decision points", flush=True)

    decision_points = pd.concat(all_points, ignore_index=True)
    decision_points.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(decision_points):,} decision points "
          f"({decision_points['mmsi'].nunique():,} vessels) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
