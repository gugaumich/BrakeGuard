from __future__ import annotations

from pathlib import Path
import json
import time
import pickle
from dataclasses import dataclass, field
from typing import Dict, Set, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from spacepy import pycdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "file_schema_catalog.parquet"

OUT_CSV = PROJECT_ROOT / "data" / "processed" / "signal_catalog.csv"
OUT_ERRORS = PROJECT_ROOT / "data" / "processed" / "signal_catalog_errors.csv"
CHECKPOINT = PROJECT_ROOT / "data" / "processed" / "signal_catalog_checkpoint.pkl"

# How often to checkpoint progress
CHECKPOINT_EVERY_FILES = 200

# Limit for a quick test; set to None for full run
MAX_FILES = None  # e.g., 200 for testing; None for all files


@dataclass
class SignalStats:
    n_files_present: int = 0
    n_files_empty: int = 0
    first_seen_date: Optional[str] = None  # YYYYMMDD
    last_seen_date: Optional[str] = None   # YYYYMMDD
    vins: Set[str] = field(default_factory=set)
    min_non_empty_length: Optional[int] = None
    max_non_empty_length: Optional[int] = None


@dataclass
class State:
    processed_paths: Set[str] = field(default_factory=set)
    signals: Dict[str, SignalStats] = field(default_factory=dict)
    n_files_catalog_total: int = 0
    n_files_opened_ok: int = 0
    n_files_open_failed: int = 0
    unique_vins_total: Set[str] = field(default_factory=set)
    errors: List[Dict[str, Any]] = field(default_factory=list)


def save_checkpoint(state: State) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "wb") as f:
        pickle.dump(state, f)


def load_checkpoint() -> Optional[State]:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "rb") as f:
            return pickle.load(f)
    return None


def is_empty_value(arr: Any) -> Tuple[bool, int, str]:
    """
    Option A emptiness:
      - zero length => empty
      - all missing/NaN OR fully masked => empty
    All-zeros is NOT treated as empty.

    Returns: (is_empty, length, reason)
    """
    # Some pycdf variables may not behave like ndarray until sliced
    try:
        data = np.asanyarray(arr)
    except Exception:
        # If conversion fails, we can't judge; treat as non-empty unknown
        return (False, -1, "unreadable_as_array")

    # length/size checks
    try:
        size = data.size
    except Exception:
        return (False, -1, "unknown_size")

    if size == 0:
        return (True, 0, "zero_length")

    # Handle masked arrays
    if np.ma.isMaskedArray(data):
        mask = np.ma.getmaskarray(data)
        # If fully masked, empty
        if mask.all():
            return (True, int(size), "fully_masked")

        # If float/cfloat masked array, consider NaN among unmasked
        if data.dtype.kind in ("f", "c"):
            filled = data.filled(np.nan)
            # Only consider unmasked entries
            unmasked = ~mask
            vals = filled[unmasked]
            if vals.size == 0:
                return (True, int(size), "no_unmasked_values")
            if np.isnan(vals).all():
                return (True, int(size), "all_nan_unmasked")
            return (False, int(size), "has_some_values")

        # Non-float masked array: if not fully masked, treat as non-empty
        return (False, int(size), "has_unmasked_values")

    # Non-masked arrays
    kind = data.dtype.kind

    # Floats/complex: empty if all NaN
    if kind in ("f", "c"):
        try:
            if np.isnan(data).all():
                return (True, int(size), "all_nan")
        except Exception:
            # If isnan fails, assume non-empty
            return (False, int(size), "nan_check_failed")
        return (False, int(size), "has_some_values")

    # Object / strings: empty if all None/"" after stripping
    if kind in ("O", "U", "S"):
        try:
            flat = data.ravel()
            def _is_blank(x):
                if x is None:
                    return True
                if isinstance(x, (bytes, bytearray)):
                    try:
                        x = x.decode(errors="ignore")
                    except Exception:
                        return False
                if isinstance(x, str):
                    return len(x.strip()) == 0
                return False
            if all(_is_blank(x) for x in flat):
                return (True, int(size), "all_blank_or_none")
            return (False, int(size), "has_some_values")
        except Exception:
            return (False, int(size), "string_check_failed")

    # Integers/bools/etc: cannot be NaN; if size>0 treat as non-empty
    return (False, int(size), "non_nan_dtype")


def get_or_create(stats_map: Dict[str, SignalStats], signal: str) -> SignalStats:
    if signal not in stats_map:
        stats_map[signal] = SignalStats()
    return stats_map[signal]


def update_first_last_date(s: SignalStats, date_yyyymmdd: str) -> None:
    if s.first_seen_date is None or date_yyyymmdd < s.first_seen_date:
        s.first_seen_date = date_yyyymmdd
    if s.last_seen_date is None or date_yyyymmdd > s.last_seen_date:
        s.last_seen_date = date_yyyymmdd


def update_non_empty_lengths(s: SignalStats, length: int) -> None:
    if length < 0:
        return
    if s.min_non_empty_length is None or length < s.min_non_empty_length:
        s.min_non_empty_length = length
    if s.max_non_empty_length is None or length > s.max_non_empty_length:
        s.max_non_empty_length = length


def main():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Missing catalog: {CATALOG_PATH}\n"
            f"Run: python src/schema_catalog.py"
        )

    # Resume if checkpoint exists
    state = load_checkpoint()
    if state is None:
        state = State()

    df = pd.read_parquet(CATALOG_PATH)
    df = df.sort_values(["vin", "file_ts"]).reset_index(drop=True)

    if MAX_FILES is not None:
        df = df.head(MAX_FILES).copy()

    state.n_files_catalog_total = len(df)
    state.unique_vins_total |= set(df["vin"].unique().tolist())

    print(">>> build_signal_catalog.py START <<<")
    print("Catalog rows (this run):", state.n_files_catalog_total)
    print("Unique VINs:", len(state.unique_vins_total))
    print("Checkpoint found:", CHECKPOINT.exists())
    print("Already processed files in checkpoint:", len(state.processed_paths))

    t0 = time.time()

    for idx, r in df.iterrows():
        path = str(r["path"])
        vin = str(r["vin"])
        date_yyyymmdd = str(r["date_yyyymmdd"])

        # Skip if already processed in checkpoint
        if path in state.processed_paths:
            continue

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"...scanned row {idx+1}/{len(df)} (opened_ok={state.n_files_opened_ok}, failed={state.n_files_open_failed}) elapsed={elapsed:,.1f}s")

        try:
            with pycdf.CDF(path) as cdf:
                keys = list(cdf.keys())

                # For each signal in this file
                for sig in keys:
                    st = get_or_create(state.signals, sig)

                    # Presence counts
                    st.n_files_present += 1
                    st.vins.add(vin)
                    update_first_last_date(st, date_yyyymmdd)

                    # Emptiness check (read values)
                    try:
                        # Materialize values. NOTE: This can be heavy for huge arrays,
                        # but is required to compute % empty accurately.
                        arr = cdf[sig][...]
                        empty, length, reason = is_empty_value(arr)

                        if empty:
                            st.n_files_empty += 1
                        else:
                            update_non_empty_lengths(st, length)

                    except Exception as e_sig:
                        state.errors.append({
                            "error_scope": "signal_read",
                            "vin": vin,
                            "date_yyyymmdd": date_yyyymmdd,
                            "signal_name": sig,
                            "path": path,
                            "error_message": str(e_sig),
                        })

            state.n_files_opened_ok += 1

        except Exception as e_file:
            state.n_files_open_failed += 1
            state.errors.append({
                "error_scope": "file_open",
                "vin": vin,
                "date_yyyymmdd": date_yyyymmdd,
                "signal_name": "",
                "path": path,
                "error_message": str(e_file),
            })

        state.processed_paths.add(path)

        # checkpoint periodically
        if len(state.processed_paths) % CHECKPOINT_EVERY_FILES == 0:
            save_checkpoint(state)
            print(f"✅ checkpoint saved ({len(state.processed_paths)} files processed total)")

    # Final checkpoint
    save_checkpoint(state)

    # Build output table
    total_opened = max(state.n_files_opened_ok, 1)
    total_catalog = max(state.n_files_catalog_total, 1)
    total_vins = max(len(state.unique_vins_total), 1)

    rows = []
    for sig, st in state.signals.items():
        n_present = st.n_files_present
        n_empty = st.n_files_empty

        rows.append({
            "signal_name": sig,
            "n_files_present": n_present,
            "pct_files_present_opened": n_present / total_opened,
            "pct_files_present_catalog": n_present / total_catalog,
            "n_unique_vins": len(st.vins),
            "pct_vins_present": len(st.vins) / total_vins,
            "first_seen_date": st.first_seen_date,
            "last_seen_date": st.last_seen_date,
            "n_files_empty": n_empty,
            "pct_files_empty_among_present": (n_empty / n_present) if n_present else 0.0,
            "min_non_empty_length": st.min_non_empty_length,
            "max_non_empty_length": st.max_non_empty_length,
        })

    out = pd.DataFrame(rows).sort_values("signal_name").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Errors
    if state.errors:
        pd.DataFrame(state.errors).to_csv(OUT_ERRORS, index=False)

    elapsed = time.time() - t0
    print("\n✅ Wrote:")
    print("-", OUT_CSV)
    if state.errors:
        print("-", OUT_ERRORS)
    print("\nRun summary:")
    print("Catalog rows considered:", state.n_files_catalog_total)
    print("Files opened OK:", state.n_files_opened_ok)
    print("Files failed open:", state.n_files_open_failed)
    print("Distinct signals:", len(out))
    print(f"Elapsed: {elapsed:,.1f}s")
    print(">>> build_signal_catalog.py DONE <<<")


if __name__ == "__main__":
    main()
