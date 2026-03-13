"""
Compare average exhaust brake health trends for GOOD vs BAD VINs.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/analysis/vin_health_trends.csv")

df = df[df["label"].isin(["GOOD", "BAD"])]

pop = (
    df.groupby(["label", "year_month"])
    .agg(
        effective_rate=("effective_rate", "mean"),
        deep_rate=("deep_rate", "mean"),
    )
    .reset_index()
)

plt.figure(figsize=(12, 6))

for label, g in pop.groupby("label"):
    plt.plot(g["year_month"], g["effective_rate"], marker="o", label=f"{label} Effective Rate")

plt.xticks(rotation=45)
plt.ylabel("Effective Braking Rate (< -60)")
plt.title("Population Exhaust Brake Health Trend")
plt.legend()
plt.tight_layout()
plt.show()
