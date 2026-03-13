"""
build_correlations_v2.py

Fixes ArrowTypeError (vin string vs dictionary) by reading partitions one-by-one.

Outputs:
- data/analysis/correlations_population_pearson.csv
- data/analysis/correlations_population_spearman.csv
- data/analysis/correlations_by_vin.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SLIM_DS = Path("data/analysis/vin_event_samples_slim.parquet")
OUT_DIR = Path("data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def accel_series(df: pd.DataFrame) -> pd.Series:
    if "AccelPedalPos" in df.columns:
        return df["AccelPedalPos"]
    if "AccelPedalPos_1587" in df.columns:
        return df["AccelPedalPos_1587"]
    return pd.Series(np.nan, index=df.index)

def gated_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # coerce numerics (keep vin as string-ish)
    for c in df.columns:
        if c != "vin":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    eng = df["EngSpeed"] if "EngSpeed" in df.columns else pd.Series(np.nan, index=df.index)
    acc = accel_series(df)

    m = (eng > 1000) & (acc < 7)

    if "TransTorqConvLockupEngaged" in df.columns:
        m &= (df["TransTorqConvLockupEngaged"] == 1)

    if "EngRetarderStat_1587" in df.columns:
        m &= (df["EngRetarderStat_1587"] == 1)

    if "Act_RetardPctTorqExh" in df.columns:
        m &= (df["Act_RetardPctTorqExh"] < -10)

    # Optional: BrakeSwitch
    if "BrakeSwitch" in df.columns and df["BrakeSwitch"].notna().mean() > 0.2:
        m &= (df["BrakeSwitch"] == 1)

    # Optional: exclude ABS-active
    if "ABS_BrkCtl_1587" in df.columns:
        m &= (df["ABS_BrkCtl_1587"] != 1)
    if "ABS_RetCont_1587" in df.columns:
        m &= (df["ABS_RetCont_1587"] != 1)

    return df[m.fillna(False)].copy()

def list_vin_partitions(ds_root: Path) -> list[Path]:
    return sorted([p for p in ds_root.iterdir() if p.is_dir() and p.name.startswith("vin=")])

def main():
    if not SLIM_DS.exists():
        raise FileNotFoundError(f"Missing dataset root: {SLIM_DS}")

    parts = list_vin_partitions(SLIM_DS)
    if not parts:
        raise RuntimeError(f"No vin= partitions found in {SLIM_DS}")

    print(f"Found {len(parts)} VIN partitions under {SLIM_DS}")

    # For population correlation: keep a capped sample so RAM stays bounded
    POP_MAX_ROWS = 500_000
    pop_chunks = []
    pop_rows = 0

    by_vin_rows = []

    for i, part in enumerate(parts, start=1):
        vin = part.name.split("vin=")[-1]

        try:
            df = pd.read_parquet(part)
        except Exception as e:
            print(f"⚠️  Skip {vin}: read failed: {e}")
            continue

        # Ensure vin column exists and is plain string
        if "vin" not in df.columns:
            df["vin"] = vin
        else:
            df["vin"] = df["vin"].astype(str)

        dfg = gated_rows(df)

        # signals available in this VIN
        avail = [c for c in SIG_CANDIDATES if c in dfg.columns]
        if "Act_RetardPctTorqExh" not in avail or len(avail) < 2:
            continue

        # Per-VIN correlations vs retard proxy
        target = "Act_RetardPctTorqExh"
        row = {"vin": vin, "n_rows": int(len(dfg))}
        for c in avail:
            if c == target:
                continue
            gg = dfg[[target, c]].dropna()
            if len(gg) >= 200:
                row[f"pearson_{c}"] = float(gg[target].corr(gg[c], method="pearson"))
                row[f"spearman_{c}"] = float(gg[target].corr(gg[c], method="spearman"))
            else:
                row[f"pearson_{c}"] = np.nan
                row[f"spearman_{c}"] = np.nan
        by_vin_rows.append(row)

        # Add to population sample (cap)
        if pop_rows < POP_MAX_ROWS:
            use = dfg[avail].dropna()
            if len(use) > 0:
                remaining = POP_MAX_ROWS - pop_rows
                if len(use) > remaining:
                    use = use.sample(remaining, random_state=7)
                pop_chunks.append(use)
                pop_rows += len(use)

        if i % 10 == 0 or i == len(parts):
            print(f"...processed {i}/{len(parts)} VINs | pop_rows={pop_rows:,} | by_vin={len(by_vin_rows):,}")

    # Write by-VIN summary
    by_vin = pd.DataFrame(by_vin_rows).sort_values("n_rows", ascending=False)
    by_vin.to_csv(OUT_DIR / "correlations_by_vin.csv", index=False)

    # Population correlation on sampled gated rows
    if pop_chunks:
        pop = pd.concat(pop_chunks, ignore_index=True)
        pearson = pop.corr(method="pearson")
        spearman = pop.corr(method="spearman")
        pearson.to_csv(OUT_DIR / "correlations_population_pearson.csv")
        spearman.to_csv(OUT_DIR / "correlations_population_spearman.csv")
        print(f"✅ Population corr computed on {len(pop):,} rows (sampled)")
    else:
        print("⚠️ No population rows after gating; population corr files not written.")

    print("✅ Wrote:")
    print(" - data/analysis/correlations_by_vin.csv")
    if pop_chunks:
        print(" - data/analysis/correlations_population_pearson.csv")
        print(" - data/analysis/correlations_population_spearman.csv")

if __name__ == "__main__":
    main()
