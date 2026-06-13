"""Build the known-legitimate-operator suppression list.

Scans vessel names across the daily AIS parquets for official-vessel
name patterns (lifeguard, patrol, pilot, research, etc.) and adds known
excursion operators by MMSI. The output, data/suppression_list.csv, is
a plain CSV the coordinator can edit by hand — vessels on it are
excluded from daily forecasts and the dashboard.
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
AIS_CA_DIR = PROJECT_ROOT / "data" / "ais_ca"
OUTPUT_PATH = PROJECT_ROOT / "data" / "suppression_list.csv"

# Word-boundary patterns so e.g. "PATRIOT" does not match PATROL
NAME_PATTERNS = {
    "BAYWATCH": "lifeguard",
    "LIFEGUARD": "lifeguard",
    "PATROL": "patrol/enforcement",
    "ENFORCE": "patrol/enforcement",
    "SHERIFF": "law enforcement",
    "POLICE": "law enforcement",
    "RESCUE": "search and rescue",
    "FIREBOAT": "fire department",
    "FIRE BOAT": "fire department",
    "PILOT": "harbor pilot",
    "RESEARCH": "research vessel",
    "TOWBOAT": "commercial tow",
    "VESSEL ASSIST": "commercial tow",
}

# Known legitimate operators identified by MMSI (name patterns alone
# would not catch these)
MANUAL_ENTRIES = [
    (366899260, "ISLAND ADVENTURE", "Channel Islands excursion operator"),
    (366813530, "ISLANDER", "Channel Islands excursion operator"),
]


def main() -> None:
    names = []
    for path in sorted(AIS_CA_DIR.glob("*.parquet")):
        pings = pd.read_parquet(path, columns=["mmsi", "vessel_name"])
        names.append(pings.dropna(subset=["vessel_name"])
                     .drop_duplicates("mmsi"))
    vessels = (pd.concat(names, ignore_index=True)
               .drop_duplicates("mmsi"))
    print(f"{len(vessels):,} named vessels across "
          f"{len(list(AIS_CA_DIR.glob('*.parquet')))} days")

    rows = []
    for pattern, reason in NAME_PATTERNS.items():
        regex = re.compile(rf"\b{re.escape(pattern)}", re.IGNORECASE)
        hits = vessels[vessels["vessel_name"].str.contains(regex)]
        for _, v in hits.iterrows():
            rows.append((v["mmsi"], v["vessel_name"].strip(), reason))

    rows.extend(MANUAL_ENTRIES)
    suppression = (
        pd.DataFrame(rows, columns=["mmsi", "vessel_name", "reason"])
        .drop_duplicates("mmsi")
        .sort_values("vessel_name")
    )
    suppression.to_csv(OUTPUT_PATH, index=False)
    print(f"Suppression list: {len(suppression)} vessels -> {OUTPUT_PATH}\n")
    print(suppression.to_string(index=False))


if __name__ == "__main__":
    main()
