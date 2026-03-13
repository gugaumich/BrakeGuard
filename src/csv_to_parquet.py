import pandas as pd
from pathlib import Path

csv_gz = Path("data/analysis/vin_event_samples.csv.gz")
parquet = Path("data/analysis/vin_event_samples.parquet")

print("Reading:", csv_gz)
df = pd.read_csv(csv_gz, compression="gzip")
print("Rows:", len(df))

print("Writing:", parquet)
df.to_parquet(parquet, engine="pyarrow", index=False)

print("DONE")
print("Parquet size (bytes):", parquet.stat().st_size)
