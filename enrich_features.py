"""Add environmental context to every prediction feature row.

New columns on data/prediction_features.parquet:

  tide_height_m, tide_rising   - interpolated from hourly NOAA CO-OPS
                                 predictions at the nearest CA tide station
  wind_speed_kn, wind_dir_deg,
  wave_height_m                - Open-Meteo historical archive + marine
                                 APIs, fetched per 0.5-degree grid cell
                                 and joined on (cell, hour)
  month, day_of_week           - calendar features (0 = Monday)
  traffic_within_10km          - other vessels with a ping in the same
                                 10-minute window within 10 km
  open_seasons, n_open_seasons,
  salmon_open, crab_open,
  lobster_open, rockfish_open  - hardcoded California fishing seasons
                                 keyed by month

API responses are cached under data/env_cache/ so reruns are free.
"""

import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).parent
FEATURES_PATH = PROJECT_ROOT / "data" / "prediction_features.parquet"
AIS_CA_DIR = PROJECT_ROOT / "data" / "ais_ca"
CACHE_DIR = PROJECT_ROOT / "data" / "env_cache"

# NOAA CO-OPS tide stations spanning the California coast
TIDE_STATIONS = {
    "9410170": ("San Diego", 32.714, -117.174),
    "9410660": ("Los Angeles", 33.720, -118.273),
    "9411340": ("Santa Barbara", 34.405, -119.692),
    "9412110": ("Port San Luis", 35.177, -120.760),
    "9413450": ("Monterey", 36.605, -121.888),
    "9414290": ("San Francisco", 37.806, -122.466),
    "9415020": ("Point Reyes", 37.996, -122.977),
    "9416841": ("Arena Cove", 38.913, -123.708),
    "9418767": ("North Spit (Humboldt)", 40.767, -124.217),
    "9419750": ("Crescent City", 41.745, -124.183),
}

# California fishing seasons by month (simplified; ocean sport/commercial
# blend). Sources: CDFW season summaries.
FISHING_SEASONS = {
    1:  ["dungeness_crab", "spiny_lobster", "sanddab"],
    2:  ["dungeness_crab", "spiny_lobster", "sanddab"],
    3:  ["dungeness_crab", "spiny_lobster", "sanddab"],
    4:  ["dungeness_crab", "rockfish", "salmon", "halibut"],
    5:  ["dungeness_crab", "rockfish", "salmon", "halibut", "squid"],
    6:  ["dungeness_crab", "rockfish", "salmon", "halibut", "squid",
         "albacore"],
    7:  ["rockfish", "salmon", "halibut", "squid", "albacore"],
    8:  ["rockfish", "salmon", "halibut", "squid", "albacore"],
    9:  ["rockfish", "salmon", "halibut", "squid", "albacore"],
    10: ["rockfish", "spiny_lobster", "halibut", "squid"],
    11: ["dungeness_crab", "spiny_lobster", "rockfish", "squid"],
    12: ["dungeness_crab", "spiny_lobster", "sanddab"],
}

WEATHER_GRID_DEG = 0.5
TRAFFIC_RADIUS_M = 10_000
TRAFFIC_BIN = "10min"
# Equirectangular approximation, accurate to a few percent at 10 km
M_PER_DEG_LAT = 110_540
M_PER_DEG_LON = 111_320 * np.cos(np.radians(37.0))


def fetch_tides(start: str, end: str) -> pd.DataFrame:
    """Hourly tide predictions for all CA stations, cached."""
    cache = CACHE_DIR / f"tides_{start}_{end}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    frames = []
    for station_id, (name, _, _) in TIDE_STATIONS.items():
        r = requests.get(
            "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
            params={
                "product": "predictions", "application": "mpa-vessel-watch",
                "begin_date": start.replace("-", ""),
                "end_date": end.replace("-", ""),
                "datum": "MLLW", "station": station_id, "time_zone": "gmt",
                "units": "metric", "interval": "h", "format": "json",
            },
            timeout=120,
        )
        r.raise_for_status()
        preds = pd.DataFrame(r.json()["predictions"])
        frames.append(pd.DataFrame({
            "station": station_id,
            "time": pd.to_datetime(preds["t"]),
            "height_m": preds["v"].astype(float),
        }))
        print(f"  tides: {name} ({len(preds)} hours)", flush=True)
        time_mod.sleep(0.5)

    tides = pd.concat(frames, ignore_index=True)
    CACHE_DIR.mkdir(exist_ok=True)
    tides.to_parquet(cache, index=False)
    return tides


def fetch_weather(cells: list[tuple[float, float]],
                  start: str, end: str) -> pd.DataFrame:
    """Hourly wind + wave per grid cell from Open-Meteo, cached."""
    cache = CACHE_DIR / f"weather_{start}_{end}.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        have = set(map(tuple, cached[["cell_lat", "cell_lon"]].values))
        if all(c in have for c in cells):
            return cached

    frames = []
    for i, (lat, lon) in enumerate(cells):
        wind = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": lat, "longitude": lon,
                    "start_date": start, "end_date": end,
                    "hourly": "wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "kn", "timezone": "UTC"},
            timeout=120,
        ).json()
        wave = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={"latitude": lat, "longitude": lon,
                    "start_date": start, "end_date": end,
                    "hourly": "wave_height", "timezone": "UTC"},
            timeout=120,
        ).json()
        hours = pd.to_datetime(wind["hourly"]["time"])
        frames.append(pd.DataFrame({
            "cell_lat": lat, "cell_lon": lon, "time": hours,
            "wind_speed_kn": wind["hourly"]["wind_speed_10m"],
            "wind_dir_deg": wind["hourly"]["wind_direction_10m"],
            "wave_height_m": (wave.get("hourly", {}).get("wave_height")
                              or [np.nan] * len(hours)),
        }))
        if (i + 1) % 10 == 0:
            print(f"  weather: {i + 1}/{len(cells)} cells", flush=True)
        time_mod.sleep(0.3)

    weather = pd.concat(frames, ignore_index=True)
    CACHE_DIR.mkdir(exist_ok=True)
    weather.to_parquet(cache, index=False)
    return weather


def add_tide_features(df: pd.DataFrame, tides: pd.DataFrame) -> pd.DataFrame:
    """Nearest station, then linear interpolation to the decision time."""
    station_ids = list(TIDE_STATIONS)
    coords = np.array([(lat, lon) for _, lat, lon in TIDE_STATIONS.values()])
    nearest = cKDTree(coords).query(df[["latitude", "longitude"]].values)[1]
    df["tide_station"] = [station_ids[i] for i in nearest]

    heights = np.full(len(df), np.nan)
    rising = np.zeros(len(df), dtype=bool)
    for station, idx in df.groupby("tide_station").groups.items():
        series = (tides[tides["station"] == station]
                  .set_index("time")["height_m"].sort_index())
        t = df.loc[idx, "base_date_time"]
        pos = np.clip(series.index.searchsorted(t) - 1, 0, len(series) - 2)
        t0 = series.index[pos]
        h0, h1 = series.values[pos], series.values[pos + 1]
        frac = ((t - t0).dt.total_seconds() / 3600).clip(0, 1).values
        heights[df.index.get_indexer(idx)] = h0 + (h1 - h0) * frac
        rising[df.index.get_indexer(idx)] = h1 > h0
    df["tide_height_m"] = heights
    df["tide_rising"] = rising
    return df.drop(columns=["tide_station"])


def add_weather_features(df: pd.DataFrame,
                         weather: pd.DataFrame) -> pd.DataFrame:
    df["cell_lat"] = (df["latitude"] / WEATHER_GRID_DEG).round() * WEATHER_GRID_DEG
    df["cell_lon"] = (df["longitude"] / WEATHER_GRID_DEG).round() * WEATHER_GRID_DEG
    df["hour_key"] = df["base_date_time"].dt.floor("h")
    merged = df.merge(
        weather.rename(columns={"time": "hour_key"}),
        on=["cell_lat", "cell_lon", "hour_key"],
        how="left",
    )
    # Ocean cells slightly inland of the marine grid return null waves
    # (sometimes as Python None, which makes the column object-dtype);
    # coerce to float and fill from the regional median
    for col in ("wind_speed_kn", "wind_dir_deg", "wave_height_m"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["wave_height_m"] = merged["wave_height_m"].fillna(
        merged["wave_height_m"].median()
    )
    return merged.drop(columns=["cell_lat", "cell_lon", "hour_key"])


def add_traffic_density(df: pd.DataFrame, ais_dir: Path) -> pd.DataFrame:
    """Count other vessels pinging within 10 km in the same 10-min bin."""
    counts = np.zeros(len(df), dtype=int)
    df = df.reset_index(drop=True)
    for day, day_idx in df.groupby(
        df["base_date_time"].dt.date.astype(str)
    ).groups.items():
        path = ais_dir / f"{day}.parquet"
        if not path.exists():
            continue
        pings = pd.read_parquet(
            path, columns=["mmsi", "base_date_time", "latitude", "longitude"]
        )
        pings["bin"] = pings["base_date_time"].dt.floor(TRAFFIC_BIN)
        by_bin = dict(tuple(pings.groupby("bin")))

        for bin_t, bin_idx in df.loc[day_idx].groupby(
            df.loc[day_idx, "base_date_time"].dt.floor(TRAFFIC_BIN)
        ).groups.items():
            bin_pings = by_bin.get(bin_t)
            if bin_pings is None:
                continue
            # One position per vessel in this window
            uniq = bin_pings.drop_duplicates("mmsi")
            tree = cKDTree(np.column_stack([
                uniq["longitude"] * M_PER_DEG_LON,
                uniq["latitude"] * M_PER_DEG_LAT,
            ]))
            query = np.column_stack([
                df.loc[bin_idx, "longitude"] * M_PER_DEG_LON,
                df.loc[bin_idx, "latitude"] * M_PER_DEG_LAT,
            ])
            neighbors = tree.query_ball_point(query, r=TRAFFIC_RADIUS_M)
            mmsis = uniq["mmsi"].values
            for row_pos, (i, nb) in enumerate(zip(bin_idx, neighbors)):
                own = df.at[i, "mmsi"]
                counts[i] = np.sum(mmsis[nb] != own)
    df["traffic_within_10km"] = counts
    return df


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = df["base_date_time"].dt.month
    df["day_of_week"] = df["base_date_time"].dt.dayofweek
    df["open_seasons"] = df["month"].map(
        lambda m: ";".join(FISHING_SEASONS[m])
    )
    df["n_open_seasons"] = df["month"].map(lambda m: len(FISHING_SEASONS[m]))
    for fishery in ("salmon", "dungeness_crab", "spiny_lobster", "rockfish"):
        col = fishery.split("_")[-1] + "_open"
        df[col] = df["month"].map(lambda m: fishery in FISHING_SEASONS[m])
    return df


def enrich(df: pd.DataFrame, ais_dir: Path = AIS_CA_DIR) -> pd.DataFrame:
    """Add all environmental features. Importable by daily_forecast."""
    start = df["base_date_time"].min().strftime("%Y-%m-%d")
    end = df["base_date_time"].max().strftime("%Y-%m-%d")

    print("Fetching tides...", flush=True)
    tides = fetch_tides(start, end)
    df = add_tide_features(df, tides)

    cells = sorted(set(zip(
        (df["latitude"] / WEATHER_GRID_DEG).round() * WEATHER_GRID_DEG,
        (df["longitude"] / WEATHER_GRID_DEG).round() * WEATHER_GRID_DEG,
    )))
    print(f"Fetching weather for {len(cells)} grid cells...", flush=True)
    weather = fetch_weather(cells, start, end)
    df = add_weather_features(df, weather)

    print("Computing traffic density...", flush=True)
    df = add_traffic_density(df, ais_dir)

    return add_season_features(df)


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    print(f"{len(df):,} feature rows to enrich")
    df = enrich(df)
    df.to_parquet(FEATURES_PATH, index=False)
    new_cols = ["tide_height_m", "tide_rising", "wind_speed_kn",
                "wind_dir_deg", "wave_height_m", "month", "day_of_week",
                "traffic_within_10km", "n_open_seasons"]
    print(f"\nSaved enriched file to {FEATURES_PATH}")
    print(df[new_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
