"""
build_correlations.py

What this script does
- Reads your SLIM event-samples dataset:
    data/analysis/vin_event_samples_slim.parquet (partitioned by vin=VINxxxxx)
- Applies a strict "right condition" gating filter (configurable)
- Computes:
    1) Population correlation matrix (Pearson + Spearman)
    2) Per-VIN correlation summary (corr of Act_RetardPctTorqExh vs key signals)
- Writes:
    data/analysis/correlations_population_pearson.csv
    data/analysis/correlations_population_spearman.csv
    data/analysis/correlations_by_vin.csv
- Optionally saves a few population scatter plots as PNG (requires kaleido)

Run
  python src/build_correlations.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SLIM_DS = Path("data/analysis/vin_event_samples_slim.parquet")
OUT_DIR = Path("data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Core signals you care about (only keep those that exist in the dataset)
SIG_CANDIDATES = [
    "Act_RetardPctTorqExh",
    "EngSpeed",
    "VehSpeedEng",
    "TrRgAttai",
    "FuelRate",
    "EngPctTorq",
    "EngDmdPctTorq",
    "BoostPres",
]

# Gating signals
GATE_CANDIDATES = [
    "EngRetarderStat_1587",
    "TransTorqConvLockupEngaged",
    "AccelPedalPos",
    "AccelPedalPos_1587",
    "BrakeSwitch",
    "ABS_BrkCtl_1587",
    "ABS_RetCont_1587",
]

def accel_series(df: pd.DataFrame) -> pd.Series:
    if "AccelPedalPos" in df.columns:
        return df["AccelPedalPos"]
    if "AccelPedalPos_1587" in df.columns:
        return df["AccelPedalPos_1587"]
    return pd.Series(np.nan, index=df.index)

def build_gate_mask(df: pd.DataFrame) -> pd.Series:
    eng = df["EngSpeed"] if "EngSpeed" in df.columns else pd.Series(np.nan, index=df.index)
    acc = accel_series(df)

    mask = (eng > 1000) & (acc < 7)

    # Lockup if present
    if "TransTorqConvLockupEngaged" in df.columns:
        mask &= (df["TransTorqConvLockupEngaged"] == 1)

    # Retarder ON if present (best)
    if "EngRetarderStat_1587" in df.columns:
        mask &= (df["EngRetarderStat_1587"] == 1)

    # Also require actual negative retard to avoid padded zeros
    if "Act_RetardPctTorqExh" in df.columns:
        mask &= (df["Act_RetardPctTorqExh"] < -10)

    # BrakeSwitch optional (only apply if present and not mostly missing)
    if "BrakeSwitch" in df.columns:
        if df["BrakeSwitch"].notna().mean() > 0.2:
            mask &= (df["BrakeSwitch"] == 1)

    # Exclude ABS active windows if present
    if "ABS_BrkCtl_1587" in df.columns:
        mask &= (df["ABS_BrkCtl_1587"] != 1)
    if "ABS_RetCont_1587" in df.columns:
        mask &= (df["ABS_RetCont_1587"] != 1)

    return mask.fillna(False)

def load_all_partitions(ds_root: Path) -> pd.DataFrame:
    # Reads whole dataset (may take time but should be manageable for slim)
    return pd.read_parquet(ds_root)

def main():
    if not SLIM_DS.exists():
        raise FileNotFoundError(f"Missing dataset: {SLIM_DS}")

    print(f"Reading: {SLIM_DS}")
    df = load_all_partitions(SLIM_DS)
    print("Rows total:", len(df), "| Cols:", len(df.columns))

    # Keep only numeric correlation signals that exist
    keep = [c for c in (SIG_CANDIDATES + GATE_CANDIDATES + ["vin"]) if c in df.columns]
    df = df[keep].copy()

    # Coerce numeric where possible
    for c in [c for c in keep if c != "vin"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply gating
    mask = build_gate_mask(df)
    dfg = df[mask].copy()
    print("Rows after gating:", len(dfg))

    corr_cols = [c for c in SIG_CANDIDATES if c in dfg.columns]
    dfg_corr = dfg[corr_cols].dropna(how="any")
    print("Rows for correlation (complete cases):", len(dfg_corr))
    if len(dfg_corr) < 1000:
        print("WARNING: Very few rows. Consider relaxing gating slightly.")

    pearson = dfg_corr.corr(method="pearson")
    spearman = dfg_corr.corr(method="spearman")

    pearson.to_csv(OUT_DIR / "correlations_population_pearson.csv")
    spearman.to_csv(OUT_DIR / "correlations_population_spearman.csv")

    # Per-VIN correlation summary vs retard proxy
    target = "Act_RetardPctTorqExh"
    by_vin_rows = []
    if "vin" in dfg.columns and target in dfg.columns:
        for vin, g in dfg.groupby("vin"):
            row = {"vin": vin, "n_rows": len(g)}
            for c in corr_cols:
                if c == target:
                    continue
                gg = g[[target, c]].dropna()
                if len(gg) >= 200:
                    row[f"pearson_{c}"] = gg[target].corr(gg[c], method="pearson")
                    row[f"spearman_{c}"] = gg[target].corr(gg[c], method="spearman")
                else:
                    row[f"pearson_{c}"] = np.nan
                    row[f"spearman_{c}"] = np.nan
            by_vin_rows.append(row)

        by_vin = pd.DataFrame(by_vin_rows).sort_values("n_rows", ascending=False)
        by_vin.to_csv(OUT_DIR / "correlations_by_vin.csv", index=False)

    print("✅ Wrote:")
    print(" - data/analysis/correlations_population_pearson.csv")
    print(" - data/analysis/correlations_population_spearman.csv")
    print(" - data/analysis/correlations_by_vin.csv")

if __name__ == "__main__":
    main()
