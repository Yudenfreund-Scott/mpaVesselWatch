"""Feature importance chart for the fishing-vessel Random Forest.

Loads models/rf_classifier.pkl, plots the 7 features as a horizontal
bar chart with the top 3 highlighted, and saves it to
outputs/feature_importance.png.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "rf_classifier.pkl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "feature_importance.png"

TOP_COLOR = "#d62728"
REST_COLOR = "#9ecae1"

# Plain-English names for the chart
DISPLAY_NAMES = {
    "mean_heading_change_rate": "Course change rate",
    "speed_std": "Speed variability",
    "mean_speed": "Average speed",
    "n_pings": "Number of position reports",
    "time_in_mpa_minutes": "Time spent inside MPA",
    "n_direction_reversals": "Direction reversals",
    "max_stationary_minutes": "Longest stationary stretch",
}


def main() -> None:
    model = joblib.load(MODEL_PATH)
    ranked = sorted(
        zip(model.feature_names_in_, model.feature_importances_),
        key=lambda t: t[1],
    )  # ascending so the most important ends up at the top of the chart
    names = [DISPLAY_NAMES.get(n, n) for n, _ in ranked]
    values = [v for _, v in ranked]
    colors = [REST_COLOR] * (len(ranked) - 3) + [TOP_COLOR] * 3

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Importance (share of model's decisions)")
    ax.set_title(
        "What the model looks at to spot fishing vessels inside MPAs",
        fontsize=13,
    )
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Chart saved to {OUTPUT_PATH}\n")

    print("Top features, in plain English:")
    for name, value in sorted(ranked, key=lambda t: -t[1])[:3]:
        print(f"  {DISPLAY_NAMES[name]} ({value:.3f})")
    interpretations = {
        "mean_heading_change_rate": (
            "Course change rate: how often and how sharply a vessel turns. "
            "Boats that are fishing weave, circle, and double back over the "
            "same spot; boats just passing through hold a straight line."
        ),
        "speed_std": (
            "Speed variability: how much a vessel's speed fluctuates. "
            "Fishing involves constant speeding up and slowing down (setting "
            "gear, hauling nets), while transiting vessels cruise at one "
            "steady speed."
        ),
        "mean_speed": (
            "Average speed: how fast the vessel moves overall. Fishing "
            "happens at low speeds (typically 2-5 knots); cargo ships and "
            "ferries pass through MPAs much faster."
        ),
    }
    print()
    for name, _ in sorted(ranked, key=lambda t: -t[1])[:3]:
        print(f"- {interpretations[name]}\n")


if __name__ == "__main__":
    main()
