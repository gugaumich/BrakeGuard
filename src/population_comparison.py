import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/analysis/vin_health_trends.csv")
df = df[df["label"].isin(["GOOD", "BAD"])]

agg = (
    df.groupby(["label", "year_month"])
    .agg(
        mean_rate=("effective_rate", "mean"),
        std_rate=("effective_rate", "std"),
        n=("effective_rate", "count"),
    )
    .reset_index()
)

plt.figure(figsize=(12, 6))

for label, g in agg.groupby("label"):
    plt.plot(g["year_month"], g["mean_rate"], label=label)
    plt.fill_between(
        g["year_month"],
        g["mean_rate"] - g["std_rate"],
        g["mean_rate"] + g["std_rate"],
        alpha=0.2
    )

plt.title("Population Exhaust Brake Effectiveness Over Time")
plt.ylabel("Effective Rate (< -60)")
plt.xlabel("Year-Month")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
