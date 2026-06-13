"""Quick look at the AIS data: columns, sample rows, and unique vessel count.

Loads the AIS CSV from data/ais/, prints the column names and first 5
rows, and reports the number of unique vessel IDs (MMSI).
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
AIS_DIR = PROJECT_ROOT / "data" / "ais"


def find_csv(ais_dir: Path) -> Path:
    csvs = sorted(ais_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv file found in {ais_dir}")
    if len(csvs) > 1:
        print(f"Multiple CSVs found, using the first: {csvs[0].name}")
    return csvs[0]


def main() -> None:
    csv_path = find_csv(AIS_DIR)
    print(f"Loading {csv_path.name} ({csv_path.stat().st_size / 1e6:.0f} MB)\n")

    # The file is large, so only read the vessel-id column in full;
    # the preview needs just the first few rows.
    preview = pd.read_csv(csv_path, nrows=5)

    print("Columns:")
    for col in preview.columns:
        print(f"  {col}")

    print("\nFirst 5 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(preview)

    mmsi = pd.read_csv(csv_path, usecols=["mmsi"])["mmsi"]
    print(f"\nTotal AIS messages: {len(mmsi):,}")
    print(f"Unique vessel IDs (MMSI): {mmsi.nunique():,}")


if __name__ == "__main__":
    main()
