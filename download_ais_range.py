"""Download NOAA MarineCadastre daily AIS files, keeping only California.

For each day in the range: download the national zip, filter to the
California bounding box (lat 32.5-42.0, lon -124.5 to -117.0), write
data/ais_ca/YYYY-MM-DD.parquet, and delete the raw download before
moving to the next day. Days whose parquet already exists are skipped,
so an interrupted run can simply be restarted.

While one day is being filtered, the next day's file downloads in the
background, so the two phases overlap.

Usage:
    python download_ais_range.py --start 2024-06-01 --end 2024-08-31
"""

import argparse
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.csv as pacsv
import requests

PROJECT_ROOT = Path(__file__).parent
OUT_DIR = PROJECT_ROOT / "data" / "ais_ca"
TMP_DIR = OUT_DIR / "_tmp"

URL_TEMPLATE = (
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/"
    "{d.year}/AIS_{d.year}_{d.month:02d}_{d.day:02d}.zip"
)

LAT_MIN, LAT_MAX = 32.5, 42.0
LON_MIN, LON_MAX = -124.5, -117.0

# NOAA column name -> the schema the rest of the pipeline expects
NOAA_RENAME = {
    "MMSI": "mmsi",
    "BaseDateTime": "base_date_time",
    "LAT": "latitude",
    "LON": "longitude",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "VesselName": "vessel_name",
    "VesselType": "vessel_type",
}


def parquet_path(day: date) -> Path:
    return OUT_DIR / f"{day.isoformat()}.parquet"


def download(day: date) -> Path | None:
    """Stream one day's zip to the temp dir. Returns None on 404."""
    url = URL_TEMPLATE.format(d=day)
    dest = TMP_DIR / f"{day.isoformat()}.zip"
    with requests.get(url, stream=True, timeout=120) as r:
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with open(dest, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
    return dest


def filter_to_california(zip_path: Path) -> pd.DataFrame:
    """Read the CSV inside the zip and keep only California-bbox rows.

    Streams the CSV in record batches so memory stays bounded
    regardless of the (1+ GB) file size.
    """
    kept = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            reader = pacsv.open_csv(
                f,
                convert_options=pacsv.ConvertOptions(
                    include_columns=list(NOAA_RENAME)
                ),
            )
            for batch in reader:
                chunk = batch.to_pandas().rename(columns=NOAA_RENAME)
                kept.append(chunk[
                    chunk["latitude"].between(LAT_MIN, LAT_MAX)
                    & chunk["longitude"].between(LON_MIN, LON_MAX)
                ])
    return pd.concat(kept, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)
    for stale in TMP_DIR.glob("*.zip"):  # leftovers from an aborted run
        stale.unlink()

    days = []
    day = args.start
    while day <= args.end:
        days.append(day)
        day += timedelta(days=1)

    pending = [d for d in days if not parquet_path(d).exists()]
    print(f"{len(days)} days requested, {len(days) - len(pending)} already "
          f"downloaded, {len(pending)} to fetch")

    downloaded_bytes = 0
    missing = []
    # Two download workers so the next file arrives while the current
    # one is being filtered
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}

        def submit_ahead(start_idx: int) -> None:
            for d in pending[start_idx:start_idx + 2]:
                if d not in futures:
                    futures[d] = pool.submit(download, d)

        for i, d in enumerate(pending):
            submit_ahead(i)
            # Up to 3 attempts per day; a transient failure must not
            # kill a multi-hour run
            for attempt in range(1, 4):
                try:
                    if attempt == 1:
                        zip_path = futures.pop(d).result()
                    else:
                        time.sleep(30)
                        zip_path = download(d)
                    if zip_path is None:
                        print(f"{d}: not available on NOAA server, skipping")
                        missing.append(d)
                        break
                    raw_size = zip_path.stat().st_size

                    ca = filter_to_california(zip_path)
                    ca.to_parquet(parquet_path(d), index=False)
                    zip_path.unlink()  # raw file deleted immediately
                    downloaded_bytes += raw_size

                    print(f"{d}: {raw_size / 1e6:.0f} MB raw -> "
                          f"{len(ca):,} CA rows -> "
                          f"{parquet_path(d).stat().st_size / 1e6:.1f} MB "
                          f"parquet", flush=True)
                    break
                except Exception as exc:
                    print(f"{d}: attempt {attempt} failed: {exc!r}",
                          flush=True)
                    if attempt == 3:
                        missing.append(d)

    kept_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet"))
    print(f"\nDone. Downloaded {downloaded_bytes / 1e9:.1f} GB raw "
          f"(all deleted); {kept_bytes / 1e9:.2f} GB of filtered parquet "
          f"kept in {OUT_DIR}")
    if missing:
        print(f"Missing days ({len(missing)}): "
              f"{', '.join(str(m) for m in missing)}")


if __name__ == "__main__":
    main()
