"""24-hour lookahead labels for every decision point.

For each decision point at time T: label = 1 if, within (T, T+24h],
the vessel had pings inside any MPA polygon AND its track in that MPA
that day scored as fishing — rf_classifier.pkl probability >= 0.5 for
tracks with >= 3 pings. Tracks under 3 pings cannot be scored and are
conservatively treated as not-fishing (they are nearly always brief
transits clipping an MPA corner).

Decision points in the final 24 hours of the dataset have no full
lookahead window and are dropped rather than mislabeled.

Joins labels onto data/prediction_features.parquet and saves the
result as data/training_set.parquet.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
FEATURES_PATH = PROJECT_ROOT / "data" / "prediction_features.parquet"
TRACKS_PATH = PROJECT_ROOT / "data" / "all_mpa_tracks.parquet"
MPA_PINGS_DIR = PROJECT_ROOT / "data" / "mpa_pings"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_classifier.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_set.parquet"

LOOKAHEAD = pd.Timedelta(hours=24)
FISHING_PROB_THRESHOLD = 0.5


def fishing_track_keys() -> set[tuple]:
    """(mmsi, mpa_name, date) for every track that scored as fishing."""
    tracks = pd.read_parquet(TRACKS_PATH)
    model = joblib.load(MODEL_PATH)

    scorable = tracks[tracks["n_pings"] >= 3].fillna(0)
    probs = model.predict_proba(
        scorable[list(model.feature_names_in_)]
    )[:, 1]
    fishing = scorable[probs >= FISHING_PROB_THRESHOLD]
    print(f"{len(fishing):,} of {len(tracks):,} MPA tracks scored as "
          f"fishing behavior")
    return set(zip(fishing["mmsi"], fishing["mpa_name"], fishing["date"]))


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    fishing_keys = fishing_track_keys()

    # Times of every ping that belongs to a fishing-scored track,
    # collected per vessel
    ping_files = sorted(MPA_PINGS_DIR.glob("*.parquet"))
    fishing_pings = []
    for path in ping_files:
        pings = pd.read_parquet(path)
        keys = list(zip(pings["mmsi"], pings["mpa_name"],
                        [path.stem] * len(pings)))
        pings = pings[[k in fishing_keys for k in keys]]
        fishing_pings.append(pings[["mmsi", "base_date_time"]])
    fishing_pings = pd.concat(fishing_pings, ignore_index=True)
    times_by_mmsi = {
        mmsi: np.sort(g["base_date_time"].values)
        for mmsi, g in fishing_pings.groupby("mmsi")
    }

    # Drop decision points whose 24h window runs past the data
    data_end = max(
        pd.Timestamp(p.stem) for p in ping_files
    ) + pd.Timedelta(days=1)
    has_window = features["base_date_time"] + LOOKAHEAD <= data_end
    dropped = (~has_window).sum()
    features = features[has_window].copy()

    labels = np.zeros(len(features), dtype=int)
    for i, (mmsi, t) in enumerate(
        zip(features["mmsi"], features["base_date_time"])
    ):
        times = times_by_mmsi.get(mmsi)
        if times is None:
            continue
        lo = np.searchsorted(times, np.datetime64(t), side="right")
        hi = np.searchsorted(times, np.datetime64(t + LOOKAHEAD),
                             side="right")
        labels[i] = int(hi > lo)
    features["label"] = labels

    features.to_parquet(OUTPUT_PATH, index=False)
    pos = labels.sum()
    print(f"Dropped {dropped:,} decision points without a full 24h window")
    print(f"Training set: {len(features):,} rows, {pos:,} positives "
          f"(positive rate {pos / len(features):.2%})")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
