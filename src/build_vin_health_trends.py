"""
build_vin_health_trends.py

What this script does:
- Reads data/analysis/vin_event_scores.csv (one row per braking event)
- Computes monthly health metrics per VIN:
    - effective_rate: fraction of events with min_Act_RetardPctTorqExh < -60
    - deep_rate:      fraction of events with min_Act_RetardPctTorqExh < -90
    - n_events:       number of candidate events that month
    - median/min of min_Act_RetardPctTorqExh (optional useful stats)
- Joins inspection-based labels from data/analysis/vin_exhaust_valve_labels.csv
- Writes: data/analysis/vin_health_trends.csv
- Writes: data/analysis/vin_health_trends_errors.csv (if any)
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(".")
EVENTS_CSV = PROJECT_ROOT / "data" / "analysis" / "vin_event_scores.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "analysis" / "vin_exhaust_valve_labels.csv"
OUT_CSV = PROJECT_ROOT / "data" / "analysis" / "vin_health_trends.csv"
ERR_CSV = PROJECT_ROOT / "data" / "analysis" / "vin_health_trends_errors.csv"

# Thresholds (keep consistent with your plots)
TH_EFFECTIVE = -60
TH_DEEP = -90

def main():
    errors = []

    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"Missing: {EVENTS_CSV}")

    df = pd.read_csv(EVENTS_CSV)

    # Required columns
    req = {"vin", "event_start_time", "min_Act_RetardPctTorqExh"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"vin_event_scores.csv missing columns: {sorted(missing)}")

    # Parse time and build year-month
    df["event_start_time"] = pd.to_datetime(df["event_start_time"], errors="coerce")
    bad_time = df["event_start_time"].isna().sum()
    if bad_time:
        errors.append({"issue": "bad_event_start_time", "count": int(bad_time)})

    df = df.dropna(subset=["event_start_time"])
    df["year_month"] = df["event_start_time"].dt.to_period("M").astype(str)

    # Boolean event indicators
    x = pd.to_numeric(df["min_Act_RetardPctTorqExh"], errors="coerce")
    df["is_effective"] = (x < TH_EFFECTIVE).astype(int)
    df["is_deep"] = (x < TH_DEEP).astype(int)

    # Aggregate monthly metrics per VIN
    agg = (
        df.groupby(["vin", "year_month"], as_index=False)
          .agg(
              n_events=("vin", "size"),
              effective_rate=("is_effective", "mean"),
              deep_rate=("is_deep", "mean"),
              median_min_retard=("min_Act_RetardPctTorqExh", "median"),
              min_min_retard=("min_Act_RetardPctTorqExh", "min"),
          )
    )

    # Join labels (optional)
    if LABELS_CSV.exists():
        labels = pd.read_csv(LABELS_CSV)
        if "vin" in labels.columns and "valve_label" in labels.columns:
            labels = labels.rename(columns={"valve_label": "label"})
            agg = agg.merge(labels[["vin", "label"]], on="vin", how="left")
        else:
            errors.append({"issue": "labels_missing_cols", "detail": "need vin,valve_label"})
            agg["label"] = None
    else:
        agg["label"] = None

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(ERR_CSV, index=False)
        print(f"⚠️ Wrote errors: {ERR_CSV}")

    print(f"✅ Wrote: {OUT_CSV}  (rows={len(agg)})")

if __name__ == "__main__":
    main()
