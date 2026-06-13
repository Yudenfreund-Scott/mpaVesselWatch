"""Daily MPA fishing-entry forecast: ranked vessel list + map.

Usage:
    python daily_forecast.py 2024-08-31

Pulls that day's AIS (downloading from NOAA if not already cached),
runs the full feature pipeline, scores every vessel near an MPA with
models/forecast_model.pkl, and writes:

  outputs/forecast_YYYY-MM-DD.csv      - all vessels ranked by 24h
        MPA fishing-entry risk, with key risk factors per vessel and
        a plain-English summary at the top
  outputs/forecast_map_YYYY-MM-DD.html - top 10 vessels with position,
        heading arrow, and the MPA each is approaching
"""

import sys
from pathlib import Path

import folium
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from ais_mpa_join import find_file
from build_prediction_features import PROJECTED_CRS, build_features
from decision_points import extract_decision_points, load_boundaries
from download_ais_range import TMP_DIR, download, filter_to_california
from enrich_features import enrich
from train_forecast import prepare

PROJECT_ROOT = Path(__file__).parent
AIS_CA_DIR = PROJECT_ROOT / "data" / "ais_ca"
PROFILES_PATH = PROJECT_ROOT / "data" / "vessel_profiles.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "forecast_model.pkl"
SUPPRESSION_PATH = PROJECT_ROOT / "data" / "suppression_list.csv"

TOP_K = 10
ARROW_LENGTH_DEG = 0.03  # ~3 km heading arrow on the map


def ensure_day(day: str) -> pd.DataFrame:
    path = AIS_CA_DIR / f"{day}.parquet"
    if not path.exists():
        print(f"{day} not cached, downloading from NOAA...")
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = download(pd.Timestamp(day).date())
        if zip_path is None:
            raise SystemExit(f"NOAA has no AIS file for {day}")
        ca = filter_to_california(zip_path)
        ca.to_parquet(path, index=False)
        zip_path.unlink()
    return pd.read_parquet(path)


def risk_factors(model, X: pd.DataFrame, top_n: int = 3) -> list[str]:
    """Top positive feature contributions per row, via LightGBM's
    built-in SHAP values."""
    contribs = model.booster_.predict(X, pred_contrib=True)[:, :-1]
    factors = []
    for row in contribs:
        order = np.argsort(row)[::-1][:top_n]
        factors.append("; ".join(
            X.columns[i] for i in order if row[i] > 0
        ))
    return factors


def build_map(top: pd.DataFrame, mpas: gpd.GeoDataFrame,
              day: str, out_path: Path) -> None:
    center = (top["latitude"].mean(), top["longitude"].mean())
    m = folium.Map(location=center, zoom_start=7, tiles="cartodbpositron")

    target_mpas = mpas[mpas["NAME"].isin(top["nearest_mpa"])]
    folium.GeoJson(
        target_mpas[["NAME", "Type", "geometry"]].to_crs(epsg=4326),
        style_function=lambda _: {"fillColor": "#d62728",
                                  "color": "#a31515",
                                  "weight": 2, "fillOpacity": 0.25},
        tooltip=folium.GeoJsonTooltip(fields=["NAME", "Type"]),
    ).add_to(m)

    for rank, (_, v) in enumerate(top.iterrows(), start=1):
        popup = folium.Popup(
            f"<b>#{rank} — MMSI {v['mmsi']}</b><br>"
            f"Risk score: {v['risk']:.2f}<br>"
            f"Approaching: {v['nearest_mpa']} "
            f"({v['dist_to_mpa_m'] / 1000:.1f} km away)<br>"
            f"Speed: {v['sog']:.1f} kn, course {v['cog']:.0f}&deg;<br>"
            f"Risk factors: {v['factors']}",
            max_width=320,
        )
        folium.CircleMarker(
            (v["latitude"], v["longitude"]), radius=7, color="#d62728",
            fill=True, fill_opacity=0.9, popup=popup,
        ).add_to(m)
        # Heading arrow: glyph points east by default, rotate to course
        if pd.notna(v["cog"]):
            folium.Marker(
                (v["latitude"], v["longitude"]),
                icon=folium.DivIcon(html=(
                    f'<div style="font-size:22px; color:#d62728; '
                    f'transform: rotate({v["cog"] - 90:.0f}deg); '
                    f'transform-origin: center;">&#10148;</div>'
                )),
            ).add_to(m)

    m.get_root().html.add_child(folium.Element(
        f'<div style="position: fixed; top: 12px; left: 50%; '
        f'transform: translateX(-50%); z-index: 9999; background: white; '
        f'padding: 8px 18px; border: 2px solid #444; border-radius: 6px; '
        f'font-size: 15px; font-weight: bold; font-family: sans-serif;">'
        f'24-Hour MPA Fishing-Entry Forecast &mdash; {day}</div>'
    ))
    m.save(out_path)


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else None
    if day is None:
        raise SystemExit("Usage: python daily_forecast.py YYYY-MM-DD")

    pings = ensure_day(day)
    mpas = gpd.read_file(find_file(PROJECT_ROOT / "data", "*.shp"))
    boundaries = load_boundaries(mpas)
    profiles = pd.read_parquet(PROFILES_PATH)
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    decisions = extract_decision_points(
        pings[["mmsi", "base_date_time", "latitude", "longitude",
               "sog", "cog", "heading"]].copy(),
        boundaries,
    )
    print(f"{len(decisions):,} decision points for "
          f"{decisions['mmsi'].nunique():,} vessels")

    features = build_features(decisions, profiles, mpas.to_crs(PROJECTED_CRS))
    features = enrich(features)

    X = prepare(features)
    features["risk"] = model.predict_proba(X)[:, 1]

    # Known-legitimate operators (lifeguards, pilots, fireboats,
    # excursion boats) never belong on a patrol-priority list
    if SUPPRESSION_PATH.exists():
        suppressed_mmsis = set(
            pd.read_csv(SUPPRESSION_PATH)["mmsi"]
        )
        n_before = features["mmsi"].nunique()
        features = features[~features["mmsi"].isin(suppressed_mmsis)]
        print(f"Suppressed {n_before - features['mmsi'].nunique()} "
              f"known-legitimate vessels (data/suppression_list.csv)")

    # One row per vessel: its highest-risk decision point of the day
    vessels = (
        features.sort_values("risk", ascending=False)
        .drop_duplicates("mmsi")
        .reset_index(drop=True)
    )
    vessels["factors"] = risk_factors(model, prepare(vessels))

    names = pings.dropna(subset=["vessel_name"]).drop_duplicates("mmsi")
    vessels = vessels.merge(
        names[["mmsi", "vessel_name"]], on="mmsi", how="left"
    )

    top = vessels.head(TOP_K)
    leader = top.iloc[0]
    summary = (
        f"24-hour MPA fishing-entry forecast for {day}. "
        f"{len(vessels)} vessels were evaluated near California MPAs. "
        f"The highest-risk vessel is MMSI {leader['mmsi']}"
        + (f" ({leader['vessel_name']})" if pd.notna(leader["vessel_name"])
           else "")
        + f", {leader['dist_to_mpa_m'] / 1000:.1f} km from "
        f"{leader['nearest_mpa']} with a risk score of "
        f"{leader['risk']:.2f}, driven by: {leader['factors']}."
    )

    csv_path = PROJECT_ROOT / "outputs" / f"forecast_{day}.csv"
    out_cols = ["mmsi", "vessel_name", "risk", "nearest_mpa",
                "dist_to_mpa_m", "sog", "cog", "heads_toward_mpa",
                "prior_mpa_entries", "prior_fishing_tracks", "factors",
                "base_date_time", "latitude", "longitude"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {summary}\n")
        vessels[out_cols].round(3).to_csv(f, index=False)
    print(f"\nRanked forecast saved to {csv_path}")

    map_path = PROJECT_ROOT / "outputs" / f"forecast_map_{day}.html"
    build_map(top, mpas, day, map_path)
    print(f"Top-{TOP_K} map saved to {map_path}\n")
    print(summary)


if __name__ == "__main__":
    main()
