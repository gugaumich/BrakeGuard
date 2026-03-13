# src/build_correlation_plots.py
"""
Correlation plots for Exhaust Brake analysis (Population + VIN-specific)

Outputs (under data/analysis):
- corr_population_matrix.csv            : correlation matrix (population, filtered)
- corr_population_pairs.csv             : per-pair correlation (population)
- plots_correlations/                   : scatter plots + VIN rolling correlation plots
- vin_rolling_corr.csv                  : long table of rolling correlations (VIN/time)

What it does
1) Loads vin_event_samples.parquet (row-level samples in candidate events)
2) Applies conservative filters so correlation is computed in the “right condition”
3) Computes:
   - population correlation matrix
   - population scatter plots (Act_RetardPctTorqExh vs key signals)
   - VIN-specific rolling correlations over time (monthly), per VIN

Run:
  python src/build_correlation_plots.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INPUT_PARQUET = Path("data/analysis/vin_event_samples.parquet")
OUT_DIR = Path("data/analysis")
PLOTS_DIR = OUT_DIR / "plots_correlations"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---- Choose signals to correlate (edit here) ----
TARGET_Y = "Act_RetardPctTorqExh"

CANDIDATE_X = [
    "EngSpeed",
    "VehSpeedEng",
    "FuelRate",
    "EngPctTorq",
    "BoostPres",
    "TransTorqConvLockupEngaged",
    "EngRetarderStat_1587",
    "AccelPedalPos",
    "AccelPedalPos_1587",
    "TrRgAttai",
]

# Filters to ensure “right condition”
DEFAULT_FILTERS = {
    "min_engspeed": 1000.0,
    "max_accel": 7.0,
    "require_lockup": True,
    "require_retarder_on": False,  # set True if you want only retarder-on periods
    "min_speed": 5.0,
    "max_speed": 80.0,
    "drop_null_fraction_min": 0.90,  # keep only columns with >=90% non-null in filtered data
    "downsample_max_rows": 500_000,   # scatter plots become huge otherwise
}


def pick_accel_col(df: pd.DataFrame) -> str | None:
    if "AccelPedalPos" in df.columns:
        return "AccelPedalPos"
    if "AccelPedalPos_1587" in df.columns:
        return "AccelPedalPos_1587"
    return None


def apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    # Must have timestamps
    if "timestamp" not in df.columns:
        # build_vin_event_samples.py likely wrote timestamp column.
        # If not, try event time
        if "event_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["event_time"], errors="coerce")
        else:
            raise ValueError("Expected a timestamp column in vin_event_samples.parquet")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Speed
    if "VehSpeedEng" in df.columns:
        df = df[(df["VehSpeedEng"] >= cfg["min_speed"]) & (df["VehSpeedEng"] <= cfg["max_speed"])]

    # Engine speed
    if "EngSpeed" in df.columns:
        df = df[df["EngSpeed"] >= cfg["min_engspeed"]]

    # Accel off
    accel_col = pick_accel_col(df)
    if accel_col:
        df = df[df[accel_col] <= cfg["max_accel"]]

    # Lockup
    if cfg["require_lockup"] and "TransTorqConvLockupEngaged" in df.columns:
        df = df[df["TransTorqConvLockupEngaged"] == 1]

    # Retarder on
    if cfg["require_retarder_on"] and "EngRetarderStat_1587" in df.columns:
        df = df[df["EngRetarderStat_1587"] == 1]

    # Keep only numeric columns for correlation
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric["timestamp"] = df["timestamp"]
    numeric["vin"] = df["vin"] if "vin" in df.columns else "UNKNOWN"

    return numeric


def safe_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols2 = [c for c in cols if c in df.columns]
    if TARGET_Y not in cols2:
        cols2 = [TARGET_Y] + cols2
    cols2 = [c for c in cols2 if c in df.columns]
    corr = df[cols2].corr(method="pearson")
    return corr


def save_population_outputs(df_f: pd.DataFrame, cols: list[str]) -> None:
    # Drop columns with too many nulls
    keep = []
    for c in [TARGET_Y] + cols:
        if c in df_f.columns:
            nonnull = df_f[c].notna().mean()
            if nonnull >= DEFAULT_FILTERS["drop_null_fraction_min"]:
                keep.append(c)

    corr = df_f[keep].corr(method="pearson")
    corr.to_csv(OUT_DIR / "corr_population_matrix.csv")

    # Pairwise correlations with counts
    rows = []
    for x in keep:
        if x == TARGET_Y:
            continue
        s = df_f[[TARGET_Y, x]].dropna()
        if len(s) < 50:
            continue
        r = float(s[TARGET_Y].corr(s[x]))
        rows.append({"x": x, "y": TARGET_Y, "pearson_r": r, "n": len(s)})

    pairs = pd.DataFrame(rows).sort_values("pearson_r")
    pairs.to_csv(OUT_DIR / "corr_population_pairs.csv", index=False)

    # Scatter plots (downsample for speed)
    plot_df = df_f[[TARGET_Y] + [x for x in keep if x != TARGET_Y]].dropna()
    if len(plot_df) > DEFAULT_FILTERS["downsample_max_rows"]:
        plot_df = plot_df.sample(DEFAULT_FILTERS["downsample_max_rows"], random_state=7)

    for x in [c for c in keep if c != TARGET_Y]:
        if x not in plot_df.columns:
            continue
        plt.figure()
        plt.scatter(plot_df[x].values, plot_df[TARGET_Y].values, s=2)
        plt.xlabel(x)
        plt.ylabel(TARGET_Y)
        plt.title(f"Population scatter: {TARGET_Y} vs {x}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"population_scatter__{TARGET_Y}__vs__{x}.png", dpi=200)
        plt.close()


def rolling_corr_by_month(df_f: pd.DataFrame, x: str) -> pd.DataFrame:
    """
    For each VIN and month, compute correlation between TARGET_Y and x.
    """
    if x not in df_f.columns or TARGET_Y not in df_f.columns:
        return pd.DataFrame()

    g = df_f[["vin", "timestamp", TARGET_Y, x]].dropna().copy()
    g["year_month"] = g["timestamp"].dt.to_period("M").astype(str)

    out = []
    for (vin, ym), sub in g.groupby(["vin", "year_month"]):
        if len(sub) < 50:
            continue
        r = float(sub[TARGET_Y].corr(sub[x]))
        out.append({"vin": vin, "year_month": ym, "x": x, "pearson_r": r, "n": len(sub)})

    return pd.DataFrame(out)


def plot_vin_rolling_corr(df_roll: pd.DataFrame, vin: str, x: str) -> None:
    sub = df_roll[(df_roll["vin"] == vin) & (df_roll["x"] == x)].sort_values("year_month")
    if sub.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(sub["year_month"], sub["pearson_r"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(-1.0, 1.0)
    plt.title(f"{vin} monthly corr: {TARGET_Y} vs {x}")
    plt.xlabel("year_month")
    plt.ylabel("pearson_r")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"vin_rollcorr__{vin}__{TARGET_Y}__vs__{x}.png", dpi=200)
    plt.close()


def main():
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Missing: {INPUT_PARQUET}")

    print(f"Reading: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)

    # Basic existence checks
    need = ["vin", TARGET_Y, "timestamp"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in vin_event_samples.parquet")

    df_f = apply_filters(df, DEFAULT_FILTERS)
    print(f"Filtered rows: {len(df_f):,}")

    # Population
    save_population_outputs(df_f, CANDIDATE_X)
    print(f"✅ Wrote population correlation outputs to: {OUT_DIR} and {PLOTS_DIR}")

    # VIN rolling correlations
    roll_all = []
    for x in CANDIDATE_X:
        rc = rolling_corr_by_month(df_f, x)
        if not rc.empty:
            roll_all.append(rc)

    if roll_all:
        roll = pd.concat(roll_all, ignore_index=True)
        roll.to_csv(OUT_DIR / "vin_rolling_corr.csv", index=False)
        print(f"✅ Wrote: {OUT_DIR / 'vin_rolling_corr.csv'}  (rows={len(roll):,})")

        # Plot a subset (top VINs) to avoid 96*signals explosion
        vins = sorted(roll["vin"].unique())
        top_vins = vins[:10]  # change if you want more
        for vin in top_vins:
            for x in ["EngSpeed", "VehSpeedEng", "FuelRate", "EngPctTorq"]:
                plot_vin_rolling_corr(roll, vin, x)
        print(f"✅ Wrote example VIN rolling-corr plots (first 10 VINs) into: {PLOTS_DIR}")
    else:
        print("⚠️ No rolling correlations produced (maybe too strict filters / too many nulls).")


if __name__ == "__main__":
    main()
