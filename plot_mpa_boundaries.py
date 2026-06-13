"""Plot California no-take MPA boundaries on an interactive Folium map.

Loads the California MPA boundary shapefile from data/, filters to
no-take designations (State Marine Reserves and no-take SMCAs), and
writes an interactive map to outputs/mpa_boundaries.html.
"""

from pathlib import Path

import folium
import geopandas as gpd

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "mpa_boundaries.html"

# Center of California, roughly midway along the coast
CALIFORNIA_CENTER = (36.8, -121.5)

# CDFW designation types that prohibit all take. "SMR" is a State Marine
# Reserve (fully no-take); some State Marine Conservation Areas are
# explicitly designated no-take.
NO_TAKE_TYPES = {"SMR", "SMCA (No-Take)"}


def find_shapefile(data_dir: Path) -> Path:
    shapefiles = sorted(data_dir.rglob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(
            f"No .shp file found in {data_dir}. Download the California MPA "
            "boundaries (CDFW ds582) and place the shapefile there."
        )
    if len(shapefiles) > 1:
        print(f"Multiple shapefiles found, using the first: {shapefiles[0].name}")
    return shapefiles[0]


def find_type_column(gdf: gpd.GeoDataFrame) -> str:
    """Locate the MPA designation column, which varies across releases."""
    for candidate in ("Type", "TYPE", "type", "DESIG", "Designation"):
        if candidate in gdf.columns:
            return candidate
    raise KeyError(
        f"Could not find an MPA designation column. Available columns: "
        f"{list(gdf.columns)}"
    )


def find_name_column(gdf: gpd.GeoDataFrame) -> str | None:
    for candidate in ("NAME", "Name", "name", "SHORTNAME", "FULLNAME"):
        if candidate in gdf.columns:
            return candidate
    return None


def main() -> None:
    shapefile = find_shapefile(DATA_DIR)
    print(f"Loading {shapefile}")
    gdf = gpd.read_file(shapefile)

    # Folium expects WGS84 lat/lon
    gdf = gdf.to_crs(epsg=4326)

    type_col = find_type_column(gdf)
    print(f"Designation breakdown:\n{gdf[type_col].value_counts()}\n")

    no_take = gdf[gdf[type_col].isin(NO_TAKE_TYPES)].copy()
    if no_take.empty:
        raise ValueError(
            f"No features matched no-take types {sorted(NO_TAKE_TYPES)} in "
            f"column '{type_col}'. Values present: "
            f"{sorted(gdf[type_col].dropna().unique())}"
        )
    print(f"{len(no_take)} no-take zones out of {len(gdf)} total MPAs")

    m = folium.Map(location=CALIFORNIA_CENTER, zoom_start=6, tiles="cartodbpositron")

    name_col = find_name_column(no_take)
    tooltip = folium.GeoJsonTooltip(fields=[c for c in (name_col, type_col) if c])

    folium.GeoJson(
        no_take,
        name="No-take MPAs",
        style_function=lambda _: {
            "fillColor": "#d62728",
            "color": "#a31515",
            "weight": 1.5,
            "fillOpacity": 0.45,
        },
        tooltip=tooltip,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    m.save(OUTPUT_PATH)
    print(f"Map saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
