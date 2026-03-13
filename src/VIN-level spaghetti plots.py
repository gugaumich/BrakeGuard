import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/analysis/vin_health_trends.csv")

for label in ["GOOD", "BAD"]:
    sub = df[df["label"] == label]

    plt.figure(figsize=(12, 6))
    for vin, g in sub.groupby("vin"):
        plt.plot(g["year_month"], g["effective_rate"], alpha=0.3)

    plt.title(f"{label} VINs — Effective Exhaust Brake Rate (< -60)")
    plt.ylabel("Effective Rate")
    plt.xlabel("Year-Month")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
