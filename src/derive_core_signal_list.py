
#Derive Tier-1 core signal list automatically

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_CATALOG = PROJECT_ROOT / "data" / "processed" / "signal_catalog.csv"
OUT_CORE = PROJECT_ROOT / "data" / "processed" / "core_signal_list.csv"

# Thresholds (we chose VIN coverage = 80%)
VIN_COVERAGE_MIN = 0.80
FILE_COVERAGE_MIN = 0.50
EMPTY_RATE_MAX = 0.10

def main():
    if not SIGNAL_CATALOG.exists():
        raise FileNotFoundError(f"Missing: {SIGNAL_CATALOG}")

    df = pd.read_csv(SIGNAL_CATALOG)

    # Core rule set (Tier-1)
    core = df[
        (df["pct_vins_present"] >= VIN_COVERAGE_MIN) &
        (df["pct_files_present_opened"] >= FILE_COVERAGE_MIN) &
        (df["pct_files_empty_among_present"] <= EMPTY_RATE_MAX)
    ].copy()

    core = core.sort_values("signal_name").reset_index(drop=True)

    OUT_CORE.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(OUT_CORE, index=False)

    print(">>> derive_core_signal_list.py DONE <<<")
    print("Input:", SIGNAL_CATALOG)
    print("Output:", OUT_CORE)
    print("Core signals:", len(core))
    print("Thresholds:",
          f"pct_vins_present>={VIN_COVERAGE_MIN}, "
          f"pct_files_present_opened>={FILE_COVERAGE_MIN}, "
          f"pct_files_empty_among_present<={EMPTY_RATE_MAX}")

if __name__ == "__main__":
    main()
