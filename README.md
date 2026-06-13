# MPA Vessel Watch

A machine-learning pipeline that forecasts which vessels are most likely to fish inside California's Marine Protected Areas within the next 24 hours. Built to help MPA Watch coordinators and California wildlife wardens direct limited patrol resources more efficiently.

**Live demo dashboard:** [outputs/dashboard.html](outputs/dashboard.html) — trained on 73 days of summer 2024 AIS data (June 1 – August 12, 2024).

## What it does

Vessel tracking data (AIS) covers the entire 1,100-mile California coastline 24/7 — including offshore areas, nighttime, and fog that shore-based observers can't reach. This pipeline:

1. Downloads daily AIS vessel positions from NOAA (free, public data)
2. Identifies vessels approaching California MPAs
3. Scores each vessel with a LightGBM model using movement patterns, vessel history, tides, wind/wave conditions, and fishing season calendars
4. Produces a ranked daily forecast: the top 10 flagged vessels are **22x more likely** to fish inside a protected zone within 24 hours than a randomly chosen vessel nearby
5. Publishes a coordinator dashboard with a risk map, 30-day MPA activity ranking, and plain-English model performance metrics

Known lifeguard, fire, pilot, enforcement, and excursion vessels are filtered from results automatically.

## Backtest results (held-out month: Jul 13 – Aug 11, 2024)

| Metric | Value |
|--------|-------|
| Precision@10 | 57% |
| Lift over random patrol | 22x |
| Base rate (vessels that fished) | 2.5% |
| Vessels monitored | 6,706 |

## Important caveats

- **AIS coverage gap:** AIS transponders are federally required only on commercial vessels ≥65 feet. Smaller boats are invisible to this system.
- **"Fished" means movement that looks like fishing.** Ground truth labels are derived from a movement classifier (slow speed, direction reversals, stationary time), not direct observation.
- **Backtest covers one summer.** Results should be revalidated each season as vessel behavior shifts.
- **Scores are rankings, not proof.** Treat alerts as patrol priorities, not evidence of violation.

## Data sources (all free)

| Source | Used for | URL |
|--------|----------|-----|
| NOAA MarineCadastre | Daily AIS vessel positions | marinecadastre.gov |
| CDFW ds582 | California MPA boundaries | data.cnra.ca.gov |
| GFW fishing-vessels-v3 | Fishing vessel registry for labels | Zenodo record 14982712 |
| NOAA CO-OPS | Hourly tide levels (10 CA stations) | tidesandcurrents.noaa.gov |
| Open-Meteo | Wind speed/direction, wave height | open-meteo.com |

No API keys required. All services are publicly accessible.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Running the pipeline

Download AIS data for a date range (skips dates already cached):
```bash
python download_ais_range.py
```

Run the full training pipeline (one-time):
```bash
python batch_process.py
python build_vessel_profiles.py
python decision_points.py
python build_prediction_features.py
python enrich_features.py
python build_labels.py
python train_forecast.py
python evaluate_forecast.py
```

Generate a daily forecast and dashboard:
```bash
python daily_forecast.py 2024-08-12
python build_dashboard.py 2024-08-12
```

## Project structure

```
mpa-vessel-watch/
├── data/
│   ├── California_Marine_Protected_Areas_[ds582].*   # MPA boundaries (CDFW)
│   ├── suppression_list.csv                          # Known-legitimate vessels
│   ├── labeled_features.csv                          # Phase-1 training set (84 tracks)
│   └── vessel_features.csv                           # Computed movement features
├── models/                    # Trained models (generated, not committed)
├── outputs/
│   ├── dashboard.html         # Coordinator dashboard (demo, Aug 12 2024)
│   ├── metrics.json           # Backtest results
│   ├── feature_importance.png # Model feature chart
│   └── model_card.txt         # Plain-English model documentation
├── *.py                       # Pipeline scripts
└── requirements.txt
```

## Pipeline scripts (in order)

| Script | What it does |
|--------|-------------|
| `download_ais_range.py` | Download daily NOAA AIS ZIPs, filter to CA bbox, save as parquet |
| `batch_process.py` | Spatial join: which vessels were inside MPAs each day |
| `build_vessel_profiles.py` | Cumulative per-vessel history snapshots (for leakage-safe joins) |
| `decision_points.py` | First ping per vessel per 6h window within 20 km of an MPA boundary |
| `build_prediction_features.py` | Movement + vessel history features at each decision point |
| `enrich_features.py` | Add tides, weather, traffic density, fishing season flags |
| `build_labels.py` | 24h lookahead: did this vessel fish in an MPA the next day? |
| `train_forecast.py` | Train LightGBM on temporal split (train Jun 1–Jul 12, test Jul 13–Aug 11) |
| `evaluate_forecast.py` | Backtest metrics: precision@10, recall@10, lift, calibration |
| `daily_forecast.py` | Score today's vessels, apply suppression filter, rank and export |
| `build_dashboard.py` | Build the coordinator HTML dashboard |
| `make_suppression_list.py` | Scan vessel names for lifeguards, pilots, fire boats, etc. |

## License

Code: MIT  
AIS data: U.S. Government public domain (NOAA/USCG)  
MPA boundaries: California State public domain (CDFW)
