"""Coordinator dashboard: one HTML page with map, activity, and metrics.

Usage:
    python build_dashboard.py [YYYY-MM-DD]   (default 2024-08-12)

Builds outputs/dashboard.html with three sections:
  1. Daily forecast map  - vessels near MPAs colored by risk score
  2. 30-day activity     - top 15 MPAs by fishing-flagged tracks
  3. Model performance   - plain-English metric cards

The daily forecast CSV must already exist for the chosen date (run
daily_forecast.py first); it is suppression-filtered at the source.
"""

import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px

from build_labels import fishing_track_keys

PROJECT_ROOT = Path(__file__).parent
TRACKS_PATH = PROJECT_ROOT / "data" / "all_mpa_tracks.parquet"
DECISIONS_PATH = PROJECT_ROOT / "data" / "decision_points.parquet"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"
AIS_CA_DIR = PROJECT_ROOT / "data" / "ais_ca"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "dashboard.html"

HIGH_RISK = 0.75
REVIEW = 0.40

ZONE_COLORS = {"SMR": "#8b0000", "SMCA (No-Take)": "#e67700"}
ZONE_DEFAULT = "#1f6fb5"

CALIFORNIA_CENTER = (33.8, -119.2)


def marker_color(risk: float) -> str:
    if risk > HIGH_RISK:
        return "#d62728"
    if risk >= REVIEW:
        return "#e6b800"
    return "#9a9a9a"


def build_forecast_map(day: str) -> tuple[str, dict]:
    forecast = pd.read_csv(
        PROJECT_ROOT / "outputs" / f"forecast_{day}.csv", comment="#"
    )

    # Entries in the 30 days before the forecast date, per vessel
    tracks = pd.read_parquet(
        TRACKS_PATH, columns=["mmsi", "date"]
    )
    window_start = (pd.Timestamp(day) - pd.Timedelta(days=30)).strftime(
        "%Y-%m-%d"
    )
    recent = tracks[(tracks["date"] > window_start) & (tracks["date"] <= day)]
    entries_30d = recent.groupby("mmsi").size()
    forecast["entries_30d"] = (
        forecast["mmsi"].map(entries_30d).fillna(0).astype(int)
    )

    m = folium.Map(location=CALIFORNIA_CENTER, zoom_start=7,
                   tiles="cartodbpositron")
    # Draw low-risk first so high-risk markers sit on top
    for _, v in forecast.sort_values("risk").iterrows():
        name = (v["vessel_name"] if pd.notna(v["vessel_name"])
                else "(no name broadcast)")
        factors = "; ".join(str(v["factors"]).split("; ")[:2])
        popup = folium.Popup(
            f"<b>MMSI {v['mmsi']}</b> — {name}<br>"
            f"Risk score: <b>{v['risk']:.2f}</b><br>"
            f"Nearest MPA: {v['nearest_mpa']} "
            f"({v['dist_to_mpa_m'] / 1000:.1f} km)<br>"
            f"Top factors: {factors}<br>"
            f"MPA entries, past 30 days: {v['entries_30d']}",
            max_width=300,
        )
        high = v["risk"] > HIGH_RISK
        folium.CircleMarker(
            (v["latitude"], v["longitude"]),
            radius=7 if high else 4,
            color=marker_color(v["risk"]),
            fill=True, fill_opacity=0.85 if high else 0.5,
            weight=1, popup=popup,
        ).add_to(m)

    counts = {
        "high": int((forecast["risk"] > HIGH_RISK).sum()),
        "review": int(forecast["risk"].between(REVIEW, HIGH_RISK).sum()),
        "clean": int((forecast["risk"] < REVIEW).sum()),
    }
    return m.get_root()._repr_html_(), counts


def build_activity_chart() -> str:
    tracks = pd.read_parquet(TRACKS_PATH)
    keys = fishing_track_keys()
    fishing = pd.DataFrame(
        list(keys), columns=["mmsi", "mpa_name", "date"]
    )
    fishing = fishing[fishing["date"].between("2024-07-13", "2024-08-11")]

    zone_types = tracks.drop_duplicates("mpa_name").set_index(
        "mpa_name")["mpa_type"]
    top = (
        fishing.groupby("mpa_name").size()
        .nlargest(15).rename("flagged_tracks").reset_index()
    )
    top["zone_type"] = top["mpa_name"].map(zone_types)
    top["color"] = top["zone_type"].map(ZONE_COLORS).fillna(ZONE_DEFAULT)

    fig = px.bar(
        top.sort_values("flagged_tracks"),
        x="flagged_tracks", y="mpa_name", orientation="h",
        labels={"flagged_tracks": "Fishing-flagged vessel tracks",
                "mpa_name": ""},
        hover_data={"zone_type": True, "color": False},
    )
    fig.update_traces(marker_color=top.sort_values("flagged_tracks")["color"])
    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=13),
        xaxis=dict(gridcolor="#eee"),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def metric_cards() -> str:
    metrics = json.loads(METRICS_PATH.read_text())
    n_vessels = pd.read_parquet(
        DECISIONS_PATH, columns=["mmsi"]
    )["mmsi"].nunique()
    n_days = len(list(AIS_CA_DIR.glob("*.parquet")))

    cards = [
        ("22x", "Better than random patrol",
         "The top 10 highest-risk vessels the model flags each morning are"
         " 22x more likely to fish inside a protected zone within the next"
         " 24 hours than a randomly chosen vessel near an MPA."),
        (f"{n_days}", "Days of AIS data",
         "June 1 – August 12, 2024"),
        (f"{n_vessels:,}", "Vessels monitored",
         "near California MPAs"),
    ]
    html = ""
    for value, title, sub in cards:
        html += (
            f'<div class="card"><div class="card-value">{value}</div>'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-sub">{sub}</div></div>'
        )
    return html


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else "2024-08-12"

    print("Building forecast map...")
    map_html, counts = build_forecast_map(day)
    print(f"  {counts['high']} high-risk, {counts['review']} review, "
          f"{counts['clean']} clean")
    print("Building activity chart...")
    chart_html = build_activity_chart()
    print("Building metric cards...")
    cards_html = metric_cards()

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MPA Vessel Watch Dashboard (Made by Scott Y)</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #fafafa;
         color: #222; margin: 0; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px 60px; }}
  header {{ border-bottom: 3px solid #1f6fb5; padding-bottom: 12px;
            margin-bottom: 28px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  .subtitle {{ color: #666; font-size: 14px; }}
  h2 {{ font-size: 19px; margin: 36px 0 6px; }}
  .desc {{ color: #555; font-size: 14px; margin: 0 0 14px; }}
  .legend span {{ display: inline-block; margin-right: 18px;
                  font-size: 13px; color: #444; }}
  .dot {{ display: inline-block; width: 11px; height: 11px;
          border-radius: 50%; margin-right: 5px; vertical-align: -1px; }}
  .cards {{ display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 16px; margin-top: 14px; }}
  .card {{ background: white; border: 1px solid #e2e2e2; border-radius: 8px;
           padding: 18px 16px; text-align: center; }}
  .card-value {{ font-size: 30px; font-weight: 700; color: #1f6fb5; }}
  .card-title {{ font-size: 14px; font-weight: 600; margin-top: 4px; }}
  .card-sub {{ font-size: 12px; color: #777; margin-top: 4px; }}
  .note {{ font-size: 13px; color: #555; margin-top: 18px; }}
  .mapbox {{ background: white; border: 1px solid #e2e2e2;
             border-radius: 8px; padding: 8px; }}
  @media (max-width: 700px) {{ .cards {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>MPA Vessel Watch Dashboard (Made by Scott Y)</h1>
  <div class="subtitle">ML Model trained with 73 Days of AIS data (June 1 &ndash; August 12, 2024) &nbsp;|&nbsp;
  Known lifeguard, pilot, fire, enforcement, and excursion vessels
  are filtered out</div>
</header>

<h2>1. Summer Risk Map</h2>
<p class="desc">Every vessel evaluated near an MPA on {day}, colored by
24-hour fishing-entry risk. Click a marker for vessel details and
recent history.</p>
<div class="legend">
  <span><span class="dot" style="background:#d62728"></span>
        High risk (&gt; 0.75) — {counts['high']} vessels</span>
  <span><span class="dot" style="background:#e6b800"></span>
        Review (0.40 – 0.75) — {counts['review']} vessels</span>
  <span><span class="dot" style="background:#9a9a9a"></span>
        Clean (&lt; 0.40) — {counts['clean']} vessels</span>
</div>
<div class="mapbox">{map_html}</div>

<h2>2. Where Fishing Activity Concentrated (Jul 13 – Aug 11)</h2>
<p class="desc">MPAs ranked by the number of vessel tracks whose
movement scored as fishing behavior over the past 30 days.
<span style="color:#8b0000; font-weight:600;">Dark red</span> = State
Marine Reserve (no take allowed),
<span style="color:#e67700; font-weight:600;">orange</span> = no-take
SMCA, <span style="color:#1f6fb5; font-weight:600;">blue</span> = other
designations. Use this to direct volunteer survey effort.</p>
{chart_html}

<h2>3. How Much to Trust This</h2>
<p class="desc">Backtested on a held-out month (July 13 – August 11)
the model never saw during training.</p>
<div class="cards">{cards_html}</div>
<div class="note">Risk Score displays a risk percentile, not probability.</div>
</div>
</body>
</html>"""

    OUTPUT_PATH.write_text(page, encoding="utf-8")
    print(f"\nDashboard saved to {OUTPUT_PATH} "
          f"({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
