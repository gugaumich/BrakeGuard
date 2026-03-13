"""
Plot exhaust-brake health metrics over time for a single VIN.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

VIN = sys.argv[1] if len(sys.argv) > 1 else "VIN02764"

DATA = Path("data/analysis/vin_health_metrics_monthly.csv")
OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA)
    vin_df = df[df["vin"] == VIN]

    if vin_df.empty:
        print(f"No data for {VIN}")
        return

    vin_df = vin_df.sort_values("year_month")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(
        vin_df["year_month"],
        vin_df["deep_retard_rate"],
        marker="o",
        label="Deep Retard Rate (< -90)",
    )

    ax1.plot(
        vin_df["year_month"],
        vin_df["effective_rate"],
        marker="s",
        label="Effective Rate (< -60)",
    )

    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Rate")
    ax1.set_xlabel("Year-Month")
    ax1.set_title(f"{VIN} — Exhaust Brake Health Over Time")
    ax1.legend()
    ax1.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    outpath = OUTDIR / f"{VIN}_health_trends.png"
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"✅ Saved: {outpath}")


if __name__ == "__main__":
    main()
