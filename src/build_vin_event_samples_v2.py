"""
build_vin_event_samples_v2.py

Purpose
-------
Build event-sample datasets from:
- data/analysis/vin_event_scores.csv        (event windows per VIN)
- data/analysis/VINxxxx/timeseries.parquet  (raw timeseries per VIN)

Outputs
-------
Writes partitioned Parquet datasets (schema-flexible, per VIN partition):
A) data/analysis/vin_event_samples_full.parquet     (optional)
B) data/analysis/vin_event_samples_slim.parquet     (recommended)

Also writes:
- data/analysis/vin_event_samples_failures.csv      (log of failures)

Why this version is robust & faster
-----------------------------------
- NO global schema casting (VINs can have different columns)
- NO global VIN_DIR mutation (vin_dir passed explicitly)
- Atomic writes: write to local temp, validate size > 0, then move into the final output path
- SLIM mode reads only needed columns from timeseries parquet (much faster)

Run
---
python src/build_vin_event_samples_v2.py

Common options
--------------
- Slim only (fast):                 python src/build_vin_event_samples_v2.py
- Slim + full (slow/huge):          python src/build_vin_event_samples_v2.py --write-full
- Clean outputs before writing:     python src/build_vin_event_samples_v2.py --clean
- Change padding / cap events:      python src/build_vin_event_samples_v2.py --pad 20 --max-events-per-vin 500
"""

from __future__ import annotations

import os
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Defaults (NO GLOBAL MUTATION)
# -----------------------------
DEFAULT_EVENTS_CSV = Path("data/analysis/vin_event_scores.csv")
DEFAULT_VIN_DIR = Path("data/analysis")

DEFAULT_OUT_FULL = Path("data/analysis/vin_event_samples_full.parquet")
DEFAULT_OUT_SLIM = Path("data/analysis/vin_event_samples_slim.parquet")
DEFAULT_FAIL_LOG = Path("data/analysis/vin_event_samples_failures.csv")
DEFAULT_TMP_ROOT = Path(tempfile.gettempdir()) / "vin_event_samples_tmp"

# Minimal metadata columns to carry forward for ML + audit trail
META_COLS = [
    "vin",
    "event_id",
    "event_start_time",
    "event_end_time",
    "duration_s",
    "speed_start",
    "speed_end",
    "decel_rate",
    "min_Act_RetardPctTorqExh",
    "median_Act_RetardPctTorqExh",
    "retarder_on_duration_s",
    "speed_band",
    "label",
    "source_file",
    "date_yyyymmdd",
]

# Slim signal shortlist for correlation/ML (plus time)
SLIM_SIGNALS = [
    "UTC_1Hz",  # preferred time col
    "timestamp",  # fallback
    "Act_RetardPctTorqExh",
    "EngSpeed",
    "VehSpeedEng",
    "FuelRate",
    "EngPctTorq",
    "BoostPres",
    "TrRgAttai",
    "AccelPedalPos",
    "AccelPedalPos_1587",
    "TransTorqConvLockupEngaged",
    "EngRetarderStat_1587",
]


# -----------------------------
# Utilities
# -----------------------------
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def safe_to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


def detect_time_col(cols: List[str]) -> Optional[str]:
    if "UTC_1Hz" in cols:
        return "UTC_1Hz"
    if "timestamp" in cols:
        return "timestamp"
    for c in cols:
        lc = c.lower()
        if "utc" in lc or "timestamp" in lc or lc == "time":
            return c
    return None


def vin_parquet_path(vin_dir: Path, vin: str) -> Path:
    return vin_dir / vin / "timeseries.parquet"


def atomic_write_parquet_table(table: pa.Table, final_path: Path, tmp_root: Path) -> None:
    """
    Atomic write:
      1) write to tmp (local)
      2) validate size > 0
      3) move into final destination
    """
    ensure_parent(final_path)
    tmp_root.mkdir(parents=True, exist_ok=True)

    tmp_path = tmp_root / (final_path.name + f".tmp_{os.getpid()}_{np.random.randint(0, 1_000_000)}")
    pq.write_table(table, tmp_path)

    if (not tmp_path.exists()) or tmp_path.stat().st_size == 0:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Atomic write failed (0 bytes): {final_path}")

    shutil.move(str(tmp_path), str(final_path))


class PartitionedWriter:
    """
    Schema-flexible partitioned Parquet dataset writer.
    Writes one file per write call under: out_dir/vin=VINxxxxx/part_*.parquet
    """

    def __init__(self, out_dir: Path, tmp_root: Path):
        self.out_dir = out_dir
        self.tmp_root = tmp_root
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._writes = 0

    def write_df(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        if "vin" not in df.columns:
            raise ValueError("write_df expects a 'vin' column")

        vin = str(df["vin"].iloc[0])
        part_dir = self.out_dir / f"vin={vin}"
        part_dir.mkdir(parents=True, exist_ok=True)

        fname = f"part_{os.getpid()}_{self._writes}_{np.random.randint(0, 1_000_000)}.parquet"
        final_path = part_dir / fname

        table = pa.Table.from_pandas(df, preserve_index=False)
        atomic_write_parquet_table(table, final_path, self.tmp_root)
        self._writes += 1


def read_events(events_csv: Path) -> pd.DataFrame:
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events file: {events_csv}")

    df = pd.read_csv(events_csv)

    for c in ["event_start_time", "event_end_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    must = ["vin", "event_id", "event_start_time", "event_end_time"]
    missing = [c for c in must if c not in df.columns]
    if missing:
        raise ValueError(f"vin_event_scores.csv missing required columns: {missing}")

    return df


def read_timeseries_columns(parquet_path: Path) -> List[str]:
    # Cheap schema read (OK even if large)
    return pd.read_parquet(parquet_path, engine="pyarrow").columns.tolist()


def load_timeseries(parquet_path: Path, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if cols is None:
        return pd.read_parquet(parquet_path)

    seen = set()
    cols2 = []
    for c in cols:
        if c not in seen:
            cols2.append(c)
            seen.add(c)

    return pd.read_parquet(parquet_path, columns=cols2)


def slice_event_window(df: pd.DataFrame, time_col: str, start: pd.Timestamp, end: pd.Timestamp, pad_s: int) -> pd.DataFrame:
    if df.empty:
        return df
    start2 = start - pd.Timedelta(seconds=pad_s)
    end2 = end + pd.Timedelta(seconds=pad_s)
    t = df[time_col]
    m = (t >= start2) & (t <= end2)
    return df.loc[m].copy()


def write_failures(failures: List[Dict], out_csv: Path) -> None:
    ensure_parent(out_csv)
    if not failures:
        pd.DataFrame(columns=["vin", "event_id", "reason"]).to_csv(out_csv, index=False)
        return
    pd.DataFrame(failures).to_csv(out_csv, index=False)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=str, default=str(DEFAULT_EVENTS_CSV))
    ap.add_argument("--vin-dir", type=str, default=str(DEFAULT_VIN_DIR))
    ap.add_argument("--out-slim", type=str, default=str(DEFAULT_OUT_SLIM))
    ap.add_argument("--out-full", type=str, default=str(DEFAULT_OUT_FULL))
    ap.add_argument("--fail-log", type=str, default=str(DEFAULT_FAIL_LOG))

    ap.add_argument("--pad", type=int, default=20, help="seconds padding around event window")
    ap.add_argument("--max-events-per-vin", type=int, default=0, help="0 = no cap; else limit events per VIN for speed")
    ap.add_argument("--write-full", action="store_true", help="Also write FULL raw (huge). Default writes SLIM only.")
    ap.add_argument("--clean", action="store_true", help="Delete output dirs before writing.")
    ap.add_argument(
        "--tmp-root",
        type=str,
        default=str(DEFAULT_TMP_ROOT),
        help="Local temp folder for atomic writes.",
    )
    args = ap.parse_args()

    events_csv = Path(args.events)
    vin_dir = Path(args.vin_dir)

    out_slim = Path(args.out_slim)
    out_full = Path(args.out_full)
    fail_log = Path(args.fail_log)
    tmp_root = Path(args.tmp_root)

    print(f"Reading events: {events_csv}")
    events = read_events(events_csv)

    if args.clean:
        print("Cleaning output dirs...")
        safe_rmtree(out_slim)
        if args.write_full:
            safe_rmtree(out_full)

    out_slim.mkdir(parents=True, exist_ok=True)
    if args.write_full:
        out_full.mkdir(parents=True, exist_ok=True)
    ensure_parent(fail_log)

    writer_slim = PartitionedWriter(out_slim, tmp_root)
    writer_full = PartitionedWriter(out_full, tmp_root) if args.write_full else None

    failures: List[Dict] = []
    total_rows_slim = 0
    total_rows_full = 0

    vins = sorted(events["vin"].dropna().unique().tolist())
    print(f"Unique VINs in events: {len(vins)}")

    for i, vin in enumerate(vins, start=1):
        vin_path = vin_parquet_path(vin_dir, vin)
        if not vin_path.exists():
            failures.append({"vin": vin, "event_id": None, "reason": f"Missing timeseries parquet: {vin_path}"})
            continue

        ev = events[events["vin"] == vin].copy()
        if args.max_events_per_vin and args.max_events_per_vin > 0:
            ev = ev.head(args.max_events_per_vin)

        # Read schema to pick time column and available columns
        try:
            cols_all = read_timeseries_columns(vin_path)
        except Exception as e:
            failures.append({"vin": vin, "event_id": None, "reason": f"Schema read failed: {e}"})
            continue

        time_col = detect_time_col(cols_all)
        if time_col is None:
            failures.append({"vin": vin, "event_id": None, "reason": f"No time column found in {vin_path.name}"})
            continue

        # SLIM cols (only those that exist)
        slim_cols = [c for c in SLIM_SIGNALS if c in cols_all]
        if time_col not in slim_cols:
            slim_cols = [time_col] + slim_cols

        try:
            df_slim = load_timeseries(vin_path, cols=slim_cols)
        except Exception as e:
            failures.append({"vin": vin, "event_id": None, "reason": f"SLIM load failed: {e}"})
            continue

        df_slim[time_col] = safe_to_datetime(df_slim[time_col])
        df_slim = df_slim.sort_values(time_col)

        df_full = None
        if args.write_full:
            try:
                df_full = load_timeseries(vin_path, cols=None)
                df_full[time_col] = safe_to_datetime(df_full[time_col])
                df_full = df_full.sort_values(time_col)
            except Exception as e:
                failures.append({"vin": vin, "event_id": None, "reason": f"FULL load failed: {e}"})
                df_full = None

        out_rows_slim: List[pd.DataFrame] = []
        out_rows_full: List[pd.DataFrame] = []

        for _, r in ev.iterrows():
            try:
                event_id = int(r["event_id"])
                start = r["event_start_time"]
                end = r["event_end_time"]

                if pd.isna(start) or pd.isna(end):
                    failures.append({"vin": vin, "event_id": event_id, "reason": "NaT start/end"})
                    continue

                w_slim = slice_event_window(df_slim, time_col, start, end, args.pad)
                if w_slim.empty:
                    continue

                # Attach metadata columns (replicated on each row)
                for c in META_COLS:
                    w_slim[c] = r[c] if c in r.index else np.nan

                # Ensure key identifiers exist
                w_slim["vin"] = vin
                w_slim["event_id"] = event_id
                w_slim["event_start_time"] = start
                w_slim["event_end_time"] = end

                out_rows_slim.append(w_slim)

                if args.write_full and df_full is not None and writer_full is not None:
                    w_full = slice_event_window(df_full, time_col, start, end, args.pad)
                    if not w_full.empty:
                        for c in META_COLS:
                            w_full[c] = r[c] if c in r.index else np.nan
                        w_full["vin"] = vin
                        w_full["event_id"] = event_id
                        w_full["event_start_time"] = start
                        w_full["event_end_time"] = end
                        out_rows_full.append(w_full)

                # Flush chunks to limit RAM
                if len(out_rows_slim) >= 50:
                    chunk = pd.concat(out_rows_slim, ignore_index=True)
                    writer_slim.write_df(chunk)
                    total_rows_slim += len(chunk)
                    out_rows_slim.clear()

                if args.write_full and writer_full is not None and len(out_rows_full) >= 20:
                    chunkf = pd.concat(out_rows_full, ignore_index=True)
                    writer_full.write_df(chunkf)
                    total_rows_full += len(chunkf)
                    out_rows_full.clear()

            except Exception as e:
                failures.append({"vin": vin, "event_id": r.get("event_id", None), "reason": f"Event processing failed: {e}"})

        # Flush remainder
        if out_rows_slim:
            chunk = pd.concat(out_rows_slim, ignore_index=True)
            writer_slim.write_df(chunk)
            total_rows_slim += len(chunk)

        if args.write_full and writer_full is not None and out_rows_full:
            chunkf = pd.concat(out_rows_full, ignore_index=True)
            writer_full.write_df(chunkf)
            total_rows_full += len(chunkf)

        if i % 5 == 0 or i == len(vins):
            msg = f"...processed {vin} ({len(ev)} events) | slim_rows={total_rows_slim}"
            if args.write_full:
                msg += f" | full_rows={total_rows_full}"
            print(msg)

    write_failures(failures, fail_log)

    print("\n✅ DONE")
    print(f"✅ SLIM dataset: {out_slim}  (rows written across partitions ≈ {total_rows_slim})")
    if args.write_full:
        print(f"✅ FULL dataset: {out_full}  (rows written across partitions ≈ {total_rows_full})")
    print(f"⚠️ Failures log: {fail_log}  (n={len(failures)})")

    print("\nTip: Read a single VIN partition like:")
    print(r'  python -c "import pandas as pd; print(pd.read_parquet('
          r'\'data/analysis/vin_event_samples_slim.parquet/vin=VIN02756\').head())"')


if __name__ == "__main__":
    main()
