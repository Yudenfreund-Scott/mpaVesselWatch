"""Clean the vessel feature matrix for modeling.

Drops tracks with fewer than 3 pings (their sequential features are
meaningless), fills remaining NaNs with 0, and overwrites
data/vessel_features.csv in place.
"""

from pathlib import Path

import pandas as pd

PATH = Path(__file__).parent / "data" / "vessel_features.csv"

features = pd.read_csv(PATH)
print(f"Loaded {len(features)} rows")

cleaned = features[features["n_pings"] >= 3].copy()
print(f"After dropping n_pings < 3: {len(cleaned)} rows")

nan_count = cleaned.isna().sum().sum()
cleaned = cleaned.fillna(0)
print(f"Filled {nan_count} remaining NaN values with 0")

cleaned.to_csv(PATH, index=False)
print(f"Saved cleaned features to {PATH}")
