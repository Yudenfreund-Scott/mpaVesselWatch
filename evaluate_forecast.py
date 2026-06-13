"""Backtest the forecast model on the held-out final month.

Operational framing: each test day, rank vessels by their highest
predicted risk and imagine a coordinator following up on the top 10.

Reports precision@10 / recall@10 per day, lift over the daily base
rate, a calibration table (predicted probability vs observed
frequency), and saves a precision-recall curve to
outputs/forecast_pr_curve.png.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from train_forecast import prepare

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "training_set.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "forecast_model.pkl"
CURVE_PATH = PROJECT_ROOT / "outputs" / "forecast_pr_curve.png"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"

TOP_K = 10


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model, split = bundle["model"], pd.Timestamp(bundle["split_date"])

    data = pd.read_parquet(DATA_PATH)
    test = data[data["base_date_time"] >= split].copy()
    test["prob"] = model.predict_proba(prepare(test))[:, 1]
    test["day"] = test["base_date_time"].dt.date
    print(f"Held-out test: {test['day'].nunique()} days, "
          f"{len(test):,} decision points, "
          f"{test['mmsi'].nunique():,} vessels\n")

    # --- daily top-10 vessel metrics ---
    vessel_day = (
        test.groupby(["day", "mmsi"])
        .agg(prob=("prob", "max"), label=("label", "max"))
        .reset_index()
    )
    p_at_k, r_at_k, base_rates = [], [], []
    for _, day_df in vessel_day.groupby("day"):
        top = day_df.nlargest(TOP_K, "prob")
        positives = day_df["label"].sum()
        base_rates.append(day_df["label"].mean())
        p_at_k.append(top["label"].mean())
        if positives > 0:
            r_at_k.append(top["label"].sum() / positives)

    precision_at_10 = sum(p_at_k) / len(p_at_k)
    # r_at_k only includes days that had at least one true positive
    recall_at_10 = sum(r_at_k) / len(r_at_k) if r_at_k else float("nan")
    base_rate = sum(base_rates) / len(base_rates)
    lift = precision_at_10 / base_rate if base_rate else float("nan")
    print(f"({len(r_at_k)} of {len(p_at_k)} test days had at least one "
          f"true fishing entry)")

    print(f"precision@{TOP_K}: {precision_at_10:.3f}  "
          f"(of the {TOP_K} vessels flagged daily, this fraction "
          f"entered an MPA and fished within 24h)")
    print(f"recall@{TOP_K}:    {recall_at_10:.3f}  "
          f"(this fraction of each day's true fishing entries was "
          f"in the top {TOP_K})")
    print(f"base rate:    {base_rate:.3f}  "
          f"(vessel-days that fished in an MPA)")
    print(f"lift:         {lift:.1f}x over random patrol\n")

    # --- calibration ---
    test["bin"] = pd.qcut(test["prob"], 10, duplicates="drop")
    calib = test.groupby("bin", observed=True).agg(
        mean_predicted=("prob", "mean"),
        actual_rate=("label", "mean"),
        n=("label", "size"),
    )
    print("Calibration (decile bins of predicted probability):")
    print(calib.round(3).to_string(), "\n")

    # --- PR curve ---
    precision, recall, _ = precision_recall_curve(test["label"], test["prob"])
    ap = average_precision_score(test["label"], test["prob"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#1f6fb5")
    ax.axhline(test["label"].mean(), ls="--", color="gray",
               label=f"base rate {test['label'].mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"24h MPA fishing-entry forecast — held-out month "
                 f"(AP = {ap:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=150)
    print(f"PR curve saved to {CURVE_PATH}")

    # Persist headline numbers for downstream consumers (dashboard)
    METRICS_PATH.write_text(json.dumps({
        "precision_at_10": round(precision_at_10, 3),
        "recall_at_10": round(recall_at_10, 3),
        "base_rate": round(base_rate, 4),
        "lift": round(lift, 1),
        "average_precision": round(ap, 3),
        "test_days": int(test["day"].nunique()),
        "test_start": str(test["day"].min()),
        "test_end": str(test["day"].max()),
    }, indent=2))
    print(f"Metrics saved to {METRICS_PATH}")

    print(
        f"\nOPERATIONAL READ: if a coordinator checked the model's top "
        f"{TOP_K} vessels each morning during the held-out month, about "
        f"{precision_at_10:.0%} of those checks would have found a vessel "
        f"that went on to fish inside an MPA within 24 hours — versus "
        f"{base_rate:.0%} by picking vessels at random ({lift:.1f}x more "
        f"efficient). Those {TOP_K} daily checks would have caught "
        f"{recall_at_10:.0%} of all fishing entries."
    )


if __name__ == "__main__":
    main()
