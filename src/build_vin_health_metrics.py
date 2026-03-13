"""
Build monthly exhaust-brake health metrics per VIN.

Metrics:
- deep_retard_rate: fraction of events achieving strong exhaust braking
- effective_rate: fraction of events with meaningful retard response

Purpose:
- Visualize degradation over time
- Compare GOOD vs BAD VIN behavior
- Support physics-based failure detection
"""

import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
EVENT_CSV = Path("data/analysis/vin_event_scores.csv")
OUT_CSV = Path("data/analysis/vin_health_metrics_monthly.csv")

DEEP_RETARD_THRESH = -90
EFFECTIVE_RETARD_THRESH = -60
MIN_EVENTS_PER_MONTH = 5
# ---------------------------------------


def main():
    df = pd.read_csv(EVENT_CSV, parse_dates=["event_start_time"])

    # Keep labeled VINs only
    df = df[df["label"].isin(["GOOD", "BAD"])].copy()

    # Time bucket
    df["year_month"] = df["event_start_time"].dt.to_period("M").astype(str)

    # Metrics
    df["deep_retard_flag"] = df["min_Act_RetardPctTorqExh"] < DEEP_RETARD_THRESH
    df["effective_flag"] = df["min_Act_RetardPctTorqExh"] < EFFECTIVE_RETARD_THRESH

    # Aggregate
    agg = (
        df.groupby(["vin", "year_month", "label"])
        .agg(
            n_events=("vin", "size"),
            deep_retard_rate=("deep_retard_flag", "mean"),
            effective_rate=("effective_flag", "mean"),
        )
        .reset_index()
    )

    # Filter sparse months
    agg = agg[agg["n_events"] >= MIN_EVENTS_PER_MONTH]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False)

    print(f"✅ Wrote: {OUT_CSV}")
    print("Rows:", len(agg))


if __name__ == "__main__":
    main()
