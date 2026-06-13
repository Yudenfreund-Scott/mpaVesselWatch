"""Find AIS pings inside California MPA boundaries.

Loads the AIS CSV from data/ais/, filters to a California bounding box,
converts the pings to a GeoDataFrame, and spatially joins them against
the MPA boundary polygons. Pings that fall inside an MPA are written to
data/ais_inside_mpas.csv along with the MPA name and designation type.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
AIS_DIR = PROJECT_ROOT / "data" / "ais"
MPA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ais_inside_mpas.csv"

# California coastal bounding box
LAT_MIN, LAT_MAX = 32.5, 42.0
LON_MIN, LON_MAX = -124.5, -117.0


def find_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} found in {directory}")
    return matches[0]


def load_ais_california(csv_path: Path) -> pd.DataFrame:
    """Load AIS pings, keeping only rows inside the California bbox."""
    print(f"Loading {csv_path.name} ({csv_path.stat().st_size / 1e6:.0f} MB)")

    # Stream in chunks so the 700+ MB file never sits in memory whole
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=1_000_000):
        total_rows += len(chunk)
        in_bbox = chunk[
            chunk["latitude"].between(LAT_MIN, LAT_MAX)
            & chunk["longitude"].between(LON_MIN, LON_MAX)
        ]
        chunks.append(in_bbox)

    ais = pd.concat(chunks, ignore_index=True)
    print(f"Total AIS rows: {total_rows:,}")
    print(f"Rows in California bbox: {len(ais):,} "
          f"({len(ais) / total_rows:.1%})\n")
    return ais


def join_ais_to_mpas(
    ais: pd.DataFrame, mpas: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatially join AIS pings to MPA polygons.

    Returns only the pings that fall inside an MPA boundary, with the
    MPA's name and designation type attached.
    """
    points = gpd.GeoDataFrame(
        ais,
        geometry=gpd.points_from_xy(ais["longitude"], ais["latitude"]),
        crs="EPSG:4326",
    )
    mpas = mpas.to_crs(points.crs)

    keep_cols = [c for c in ("NAME", "Type") if c in mpas.columns]
    inside = gpd.sjoin(
        points,
        mpas[keep_cols + ["geometry"]],
        how="inner",
        predicate="within",
    )

    # A ping on a boundary between adjacent MPAs can match twice;
    # count each ping once
    inside = inside[~inside.index.duplicated(keep="first")]
    return inside


def main() -> None:
    ais_path = find_file(AIS_DIR, "*.csv")
    mpa_path = find_file(MPA_DIR, "*.shp")

    ais = load_ais_california(ais_path)
    mpas = gpd.read_file(mpa_path)

    inside = join_ais_to_mpas(ais, mpas)
    outside_count = len(ais) - len(inside)

    print(f"Pings inside MPAs:  {len(inside):,} ({len(inside) / len(ais):.1%})")
    print(f"Pings outside MPAs: {outside_count:,}")

    result = inside.drop(columns=["geometry", "index_right"]).rename(
        columns={"NAME": "mpa_name", "Type": "mpa_type"}
    )
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
