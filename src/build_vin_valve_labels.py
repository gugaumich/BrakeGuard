"""
Build VIN-level exhaust valve health labels by merging
manual inspection results with available VIN data.

Inputs:
- data/visual_inspection.csv
- data/analysis/VIN*/timeseries.parquet

Output:
- data/analysis/vin_exhaust_valve_labels.csv
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSPECTION_FILE = PROJECT_ROOT / "data" / "visual_inspection.csv"
VIN_DATA_ROOT = PROJECT_ROOT / "data" / "analysis"
OUTPUT_FILE = VIN_DATA_ROOT / "vin_exhaust_valve_labels.csv"


def load_available_vins():
    """Find VINs that actually exist in processed data."""
    vins = []
    for p in VIN_DATA_ROOT.glob("VIN*/timeseries.parquet"):
        vins.append(p.parent.name)
    return sorted(set(vins))


def main():
    # -----------------------------
    # Load inspection table
    # -----------------------------
    df = pd.read_csv(INSPECTION_FILE)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Expect at least: vin, model, inspection_status (or remarks)
    if "vin" not in df.columns:
        raise ValueError("visual_inspection.csv must contain a 'vin' column")

    # Try to find inspection status column
    status_col = None
    for c in df.columns:
        if "remark" in c or "status" in c:
            status_col = c
            break

    if status_col is None:
        raise ValueError("Could not find inspection status column in visual_inspection.csv")

    df = df.rename(columns={status_col: "inspection_status"})
    df["inspection_status"] = df["inspection_status"].str.strip().str.upper()

    # -----------------------------
    # Load available VINs
    # -----------------------------
    available_vins = load_available_vins()
    df["vin"] = df["vin"].str.strip().str.upper()

    df = df[df["vin"].isin(available_vins)].copy()

    print(f"VINs with inspection data: {len(df)}")
    print(f"VINs available in dataset: {len(available_vins)}")
    print(f"VINs in both: {df['vin'].nunique()}")

    # -----------------------------
    # Map inspection status → valve label
    # -----------------------------
    def map_label(x):
        if x == "GOOD":
            return "GOOD"
        if "STUCK" in x:
            return "BAD"
        return "EXCLUDED"

    df["valve_label"] = df["inspection_status"].apply(map_label)

    # Exclude NO_GO / ambiguous
    df_final = df[df["valve_label"] != "EXCLUDED"].copy()

    # Keep only essential columns
    keep_cols = ["vin"]
    if "model" in df_final.columns:
        keep_cols.append("model")

    keep_cols += ["inspection_status", "valve_label"]

    df_final = df_final[keep_cols].sort_values("vin")

    # -----------------------------
    # Write output
    # -----------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Wrote labels to: {OUTPUT_FILE}")
    print(df_final["valve_label"].value_counts())


if __name__ == "__main__":
    main()
