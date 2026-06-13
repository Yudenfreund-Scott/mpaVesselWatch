"""Interactive map of flagged vessel tracks from the holdout run.

Draws all California MPA boundaries (colored by protection level), then
overlays the high_risk and review vessel tracks from
outputs/flagged_vessels_holdout.csv, with popups describing each flag.
Saves to outputs/vessel_anomaly_map.html.
"""

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
FLAGGED_PATH = PROJECT_ROOT / "outputs" / "flagged_vessels_holdout.csv"
PINGS_PATH = PROJECT_ROOT / "data" / "ais_holdout_inside_mpas.csv"
MPA_PATH = next(iter(sorted((PROJECT_ROOT / "data").glob("*.shp"))))
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "vessel_anomaly_map.html"

CALIFORNIA_CENTER = (36.8, -121.5)

MPA_COLORS = {
    "SMR": "#8b0000",             # dark red: full no-take reserve
    "SMCA (No-Take)": "#e67700",  # orange: no-take conservation area
}
MPA_DEFAULT_COLOR = "#1f6fb5"     # blue: all other designations

TRACK_COLORS = {"high_risk": "#ff0000", "review": "#ffcc00"}

TITLE_HTML = """
<div style="position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
            z-index: 9999; background: white; padding: 8px 18px;
            border: 2px solid #444; border-radius: 6px;
            font-size: 16px; font-weight: bold; font-family: sans-serif;">
  MPA Vessel Anomaly Detection &mdash; August 15 2024 Holdout
</div>
"""

LEGEND_HTML = """
<div style="position: fixed; bottom: 24px; left: 12px; z-index: 9999;
            background: white; padding: 10px 14px; border: 2px solid #444;
            border-radius: 6px; font-size: 13px; font-family: sans-serif;">
  <b>Legend</b><br>
  <span style="color:#8b0000;">&#9632;</span> SMR boundary (no take allowed)<br>
  <span style="color:#e67700;">&#9632;</span> SMCA No-Take boundary (no take allowed)<br>
  <span style="color:#1f6fb5;">&#9632;</span> Other MPA boundary<br>
  <span style="color:#ff0000;">&#9473;</span> High-risk vessel track<br>
  <span style="color:#ffcc00;">&#9473;</span> Review vessel track
</div>
"""


def flag_reason(row: pd.Series) -> str:
    if row["flag"] == "high_risk":
        return (f"Fishing-like movement (p={row['fishing_probability']:.2f}) "
                f"inside {row['mpa_type']} where all take is prohibited")
    return (f"Fishing-like movement (p={row['fishing_probability']:.2f}) "
            f"inside {row['mpa_type']} — verify against zone regulations")


def add_mpa_layer(m: folium.Map, mpas: gpd.GeoDataFrame) -> None:
    def style(feature):
        color = MPA_COLORS.get(
            feature["properties"]["Type"], MPA_DEFAULT_COLOR
        )
        return {"fillColor": color, "color": color,
                "weight": 1.5, "fillOpacity": 0.25}

    folium.GeoJson(
        mpas[["NAME", "Type", "geometry"]],
        name="MPA boundaries",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["NAME", "Type"]),
    ).add_to(m)


def add_track(m: folium.Map, track: pd.DataFrame, row: pd.Series) -> None:
    track = track.sort_values("base_date_time")
    coords = list(zip(track["latitude"], track["longitude"]))
    popup = folium.Popup(
        f"<b>MMSI:</b> {row['mmsi']}<br>"
        f"<b>MPA:</b> {row['mpa_name']}<br>"
        f"<b>Fishing probability:</b> {row['fishing_probability']:.2f}<br>"
        f"<b>Flag reason:</b> {flag_reason(row)}",
        max_width=320,
    )
    folium.PolyLine(
        coords,
        color=TRACK_COLORS[row["flag"]],
        weight=4,
        opacity=0.9,
        popup=popup,
    ).add_to(m)
    # Mark the start of the track so direction is readable
    folium.CircleMarker(
        coords[0], radius=4, color=TRACK_COLORS[row["flag"]],
        fill=True, fill_opacity=1,
    ).add_to(m)


def main() -> None:
    flagged = pd.read_csv(FLAGGED_PATH)
    flagged = flagged[flagged["flag"] != "clean"]
    pings = pd.read_csv(PINGS_PATH)
    mpas = gpd.read_file(MPA_PATH).to_crs(epsg=4326)

    m = folium.Map(
        location=CALIFORNIA_CENTER, zoom_start=6, tiles="cartodbpositron"
    )
    add_mpa_layer(m, mpas)

    for _, row in flagged.iterrows():
        track = pings[
            (pings["mmsi"] == row["mmsi"])
            & (pings["mpa_name"] == row["mpa_name"])
        ]
        if track.empty:
            print(f"WARNING: no pings found for MMSI {row['mmsi']} "
                  f"in {row['mpa_name']}")
            continue
        add_track(m, track, row)

    m.get_root().html.add_child(folium.Element(TITLE_HTML))
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))
    folium.LayerControl().add_to(m)

    m.save(OUTPUT_PATH)
    print(f"Map saved to {OUTPUT_PATH}")
    print(f"Tracks drawn: {len(flagged)} "
          f"({(flagged['flag'] == 'high_risk').sum()} high_risk, "
          f"{(flagged['flag'] == 'review').sum()} review)")


if __name__ == "__main__":
    main()
