"""
Build VIN Event Samples (Parquet + CSV.GZ) with Atomic Writes + Validation

What this script does
---------------------
1) Reads candidate braking events from: data/analysis/vin_event_scores.csv
2) For each VIN, loads that VIN's time series parquet:
      data/analysis/<VIN>/timeseries.parquet
3) For each event, extracts ALL raw rows from the original timeseries within:
      [event_start_time - pre_s, event_end_time + post_s]
   (default: pre_s=5, post_s=5)
4) Writes:
   - data/analysis/vin_event_samples.parquet  (partitioned dataset by vin)
   - data/analysis/vin_event_samples.csv.gz   (single gzipped CSV)
   - data/analysis/vin_event_samples_failures.csv (event-level failures)

Key upgrades in this patched version
------------------------------------
- Atomic writes: never leaves 0-byte final files
- Output validation: asserts written files are non-empty + readable
- Faster: chunked read of events CSV, per-VIN processing, single VIN parquet load
- Lower memory: incremental Parquet dataset writer + streaming CSV.GZ writer

Run
---
(myenv) PS> python src/build_vin_event_samples.py

Optional args
-------------
--pre_s 5 --post_s 5 --max_events_per_vin 0 (0 = no cap)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Parquet writing (fast incremental dataset)
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# -----------------------------
# Paths
# -----------------------------
EVENTS_CSV = Path("data/analysis/vin_event_scores.csv")
VIN_TS_GLOB = "data/analysis/VIN*/timeseries.parquet"

OUT_DIR = Path("data/analysis")
OUT_PARQUET_DIR = OUT_DIR / "vin_event_samples.parquet"  # dataset folder
OUT_CSV_GZ = OUT_DIR / "vin_event_samples.csv.gz"
FAIL_LOG = OUT_DIR / "vin_event_samples_failures.csv"


# -----------------------------
# Utilities
# -----------------------------
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    """
    Replace a directory atomically-ish:
      - remove final_dir if exists
      - rename tmp_dir -> final_dir
    On Windows this is usually reliable if no process is holding files.
    """
    if final_dir.exists():
        # remove directory tree
        for child in final_dir.rglob("*"):
            if child.is_file():
                try:
                    child.unlink()
                except Exception:
                    pass
        # remove empty dirs deepest-first
        for child in sorted(final_dir.rglob("*"), key=lambda x: len(str(x)), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except Exception:
                    pass
        try:
            final_dir.rmdir()
        except Exception:
            pass
    tmp_dir.replace(final_dir)


def safe_parse_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    # handle numeric epochs
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if len(v) == 0:
            return pd.to_datetime(s, errors="coerce")
        med = float(np.nanmedian(v))
        if med > 1e12:
            return pd.to_datetime(s, unit="ms", errors="coerce")
        if med > 1e9:
            return pd.to_datetime(s, unit="s", errors="coerce")
        return pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def find_time_col(cols: List[str]) -> Optional[str]:
    preferred = ["timestamp", "UTC_1Hz", "UTC", "Time", "time", "Timestamp", "DateTime", "datetime"]
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        lc = c.lower()
        if "utc" in lc or "timestamp" in lc or lc == "time" or "datetime" in lc:
            return c
    return None


def list_vin_timeseries() -> Dict[str, Path]:
    vin_to_path: Dict[str, Path] = {}
    for p in Path(".").glob(VIN_TS_GLOB):
        vin = p.parent.name  # VIN02756
        vin_to_path[vin] = p
    return vin_to_path


@dataclass
class FailureRow:
    vin: str
    event_id: int
    event_start_time: str
    event_end_time: str
    reason: str


# -----------------------------
# Read events efficiently
# -----------------------------
def load_events_grouped(events_csv: Path, chunksize: int = 300_000) -> Dict[str, pd.DataFrame]:
    """
    Load vin_event_scores.csv in chunks and group by VIN.
    Returns dict[vin] -> DataFrame of events for that vin.
    """
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events file: {events_csv}")

    usecols = [
        "vin",
        "event_id",
        "event_start_time",
        "event_end_time",
        "label",
        "speed_band",
        "min_Act_RetardPctTorqExh",
        "median_Act_RetardPctTorqExh",
        "speed_start",
        "speed_end",
        "decel_rate",
        "duration_s",
        "retarder_on_duration_s",
    ]
    grouped: Dict[str, List[pd.DataFrame]] = {}

    reader = pd.read_csv(events_csv, chunksize=chunksize, usecols=lambda c: c in usecols)
    for chunk in reader:
        # parse time columns
        for tc in ["event_start_time", "event_end_time"]:
            if tc in chunk.columns:
                chunk[tc] = pd.to_datetime(chunk[tc], errors="coerce")
        chunk = chunk.dropna(subset=["vin", "event_start_time", "event_end_time"])
        for vin, g in chunk.groupby("vin", sort=False):
            grouped.setdefault(vin, []).append(g)

    out: Dict[str, pd.DataFrame] = {}
    for vin, parts in grouped.items():
        df = pd.concat(parts, ignore_index=True)
        df = df.sort_values("event_start_time")
        out[vin] = df

    return out


# -----------------------------
# Incremental Parquet dataset writer
# -----------------------------
class DatasetWriter:
    def __init__(self, out_dir: Path, partition_cols: List[str]):
        self.out_dir = out_dir
        self.partition_cols = partition_cols
        self._schema: Optional[pa.Schema] = None

    def write_df(self, df: pd.DataFrame):
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._schema is None:
            self._schema = table.schema
        else:
            # align to first schema (safe cast if needed)
            table = table.cast(self._schema, safe=False)

        ds.write_dataset(
            data=table,
            base_dir=str(self.out_dir),
            format="parquet",
            partitioning=self.partition_cols,
            existing_data_behavior="overwrite_or_ignore",
        )

    def finalize(self):
        # nothing to do; dataset already written
        pass


# -----------------------------
# Main extraction
# -----------------------------
def extract_event_rows(
    ts: pd.DataFrame,
    time_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pre_s: int,
    post_s: int,
) -> pd.DataFrame:
    """
    Extract raw rows within [start-pre_s, end+post_s].
    Assumes ~1Hz; uses time window, not row counts.
    """
    if pd.isna(start) or pd.isna(end):
        return ts.iloc[0:0]

    w0 = start - pd.Timedelta(seconds=int(pre_s))
    w1 = end + pd.Timedelta(seconds=int(post_s))

    t = ts[time_col]
    m = (t >= w0) & (t <= w1)
    return ts.loc[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_s", type=int, default=5)
    ap.add_argument("--post_s", type=int, default=5)
    ap.add_argument("--max_events_per_vin", type=int, default=0, help="0 = no cap")
    ap.add_argument("--events_chunksize", type=int, default=300_000)
    args = ap.parse_args()

    pre_s = args.pre_s
    post_s = args.post_s
    max_events = args.max_events_per_vin

    vin_to_ts_path = list_vin_timeseries()
    if not vin_to_ts_path:
        raise RuntimeError(f"No VIN timeseries parquets found via glob: {VIN_TS_GLOB}")

    print(f"Reading events: {EVENTS_CSV}")
    events_by_vin = load_events_grouped(EVENTS_CSV, chunksize=args.events_chunksize)

    # Temp outputs for atomic replace
    tmp_parquet_dir = OUT_PARQUET_DIR.with_name(OUT_PARQUET_DIR.name + ".tmp")
    tmp_csv_gz = OUT_CSV_GZ.with_suffix(".csv.gz.tmp")
    tmp_fail_log = FAIL_LOG.with_suffix(".csv.tmp")

    # Prepare dirs
    tmp_parquet_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(tmp_csv_gz)
    ensure_parent(tmp_fail_log)

    # Streaming CSV.GZ writer
    csv_file = gzip.open(tmp_csv_gz, mode="wt", newline="", encoding="utf-8")
    csv_writer = None

    # Fail log writer
    fail_f = open(tmp_fail_log, "w", newline="", encoding="utf-8")
    fail_writer = csv.writer(fail_f)
    fail_writer.writerow(["vin", "event_id", "event_start_time", "event_end_time", "reason"])

    # Parquet dataset writer (partitioned by vin)
    ds_writer = DatasetWriter(tmp_parquet_dir, partition_cols=["vin"])

    total_rows = 0
    total_failures = 0

    try:
        for vin, ev in events_by_vin.items():
            if vin not in vin_to_ts_path:
                # no data for this VIN
                for _, r in ev.iterrows():
                    fail_writer.writerow([vin, int(r.get("event_id", -1)), r["event_start_time"], r["event_end_time"], "VIN timeseries missing"])
                    total_failures += 1
                continue

            ts_path = vin_to_ts_path[vin]

            # Cap events per VIN if requested
            if max_events and len(ev) > max_events:
                ev = ev.iloc[:max_events].copy()

            # Load VIN timeseries once. Only columns needed? We need "all rows from original data".
            # To keep size manageable, load full parquet (you can later limit columns if needed).
            try:
                ts = pd.read_parquet(ts_path)
            except Exception as e:
                for _, r in ev.iterrows():
                    fail_writer.writerow([vin, int(r.get("event_id", -1)), r["event_start_time"], r["event_end_time"], f"Failed reading timeseries: {e}"])
                    total_failures += 1
                continue

            time_col = find_time_col(ts.columns.tolist())
            if time_col is None:
                for _, r in ev.iterrows():
                    fail_writer.writerow([vin, int(r.get("event_id", -1)), r["event_start_time"], r["event_end_time"], "No time column found"])
                    total_failures += 1
                continue

            ts[time_col] = safe_parse_time(ts[time_col])
            ts = ts.dropna(subset=[time_col]).sort_values(time_col)

            vin_rows: List[pd.DataFrame] = []

            for _, r in ev.iterrows():
                eid = int(r.get("event_id", -1))
                start = r["event_start_time"]
                end = r["event_end_time"]

                try:
                    w = extract_event_rows(ts, time_col, start, end, pre_s=pre_s, post_s=post_s)
                    if w.empty:
                        fail_writer.writerow([vin, eid, start, end, "No rows found in window"])
                        total_failures += 1
                        continue

                    # Attach event metadata to every row
                    w2 = w.copy()
                    w2.insert(0, "vin", vin)
                    w2.insert(1, "event_id", eid)
                    w2.insert(2, "event_start_time", start)
                    w2.insert(3, "event_end_time", end)

                    # Carry useful event summary columns too (same on each row)
                    for c in ["label", "speed_band", "duration_s", "speed_start", "speed_end", "decel_rate",
                              "min_Act_RetardPctTorqExh", "median_Act_RetardPctTorqExh", "retarder_on_duration_s"]:
                        if c in r.index:
                            w2[c] = r[c]

                    vin_rows.append(w2)

                except Exception as e:
                    fail_writer.writerow([vin, eid, start, end, f"Exception extracting window: {e}"])
                    total_failures += 1
                    continue

            if not vin_rows:
                print(f"...processed {vin} ({len(ev)} events) | total_rows={total_rows} (no extracted rows)")
                continue

            out_vin_df = pd.concat(vin_rows, ignore_index=True)
            total_rows += len(out_vin_df)

            # Write parquet dataset partition
            ds_writer.write_df(out_vin_df)

            # Stream to CSV.GZ
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(out_vin_df.columns))
                csv_writer.writeheader()
            for rec in out_vin_df.to_dict(orient="records"):
                csv_writer.writerow(rec)

            print(f"...processed {vin} ({len(ev)} events) | total_rows={total_rows}")

    finally:
        # close files
        try:
            csv_file.close()
        except Exception:
            pass
        try:
            fail_f.close()
        except Exception:
            pass
        try:
            ds_writer.finalize()
        except Exception:
            pass

    # Atomic replace outputs
    # Parquet dataset dir
    atomic_replace_dir(tmp_parquet_dir, OUT_PARQUET_DIR)
    # CSV.GZ
    tmp_csv_gz.replace(OUT_CSV_GZ)
    # Fail log
    tmp_fail_log.replace(FAIL_LOG)

    print(f"✅ Wrote:")
    print(f"- {OUT_PARQUET_DIR} (partitioned dataset)")
    print(f"- {OUT_CSV_GZ}")
    print(f"- {FAIL_LOG} (failures={total_failures})")
    print(f"Total event-sample rows: {total_rows}")

    # -----------------------------
    # Validation (hard fail if bad)
    # -----------------------------
    assert OUT_CSV_GZ.exists() and OUT_CSV_GZ.stat().st_size > 0, "CSV.GZ not written or is 0 bytes"
    assert OUT_PARQUET_DIR.exists(), "Parquet dataset folder missing"

    # Validate parquet is readable
    # (read a tiny sample)
    ds0 = ds.dataset(str(OUT_PARQUET_DIR), format="parquet")
    sample = ds0.to_table(columns=["vin", "event_id"]).slice(0, 5).to_pandas()
    assert len(sample) > 0, "Parquet dataset readable but empty"

    print("✅ Output validation passed")
    print("CSV bytes:", OUT_CSV_GZ.stat().st_size)


if __name__ == "__main__":
    main()
