from __future__ import annotations
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from spacepy import pycdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PARQUET = PROJECT_ROOT / "data" / "processed" / "file_schema_catalog.parquet"
CORE_LIST = PROJECT_ROOT / "data" / "processed" / "core_signal_list.csv"

OUT_BASE = PROJECT_ROOT / "data" / "analysis"
ERROR_LOG = PROJECT_ROOT / "data" / "processed" / "vin_parquet_errors.csv"

VIN_STABLE_PCT = 0.95  # Tier-2 rule
PREFERRED_TIME_VARS = ["UTC_1Hz", "UTC", "Epoch", "Time", "timestamp"]


def pick_time_var(keys: List[str]) -> Optional[str]:
    for k in PREFERRED_TIME_VARS:
        if k in keys:
            return k
    # fallback: first key that looks like time
    lower = [k.lower() for k in keys]
    for k, kl in zip(keys, lower):
        if "utc" in kl or "epoch" in kl or kl == "time" or "timestamp" in kl:
            return k
    return None


def load_core_signals() -> List[str]:
    if not CORE_LIST.exists():
        raise FileNotFoundError(f"Missing core list: {CORE_LIST}. Run derive_core_signal_list.py first.")
    df = pd.read_csv(CORE_LIST)
    return df["signal_name"].astype(str).tolist()


# def compute_vin_stable_signals(df_vin: pd.DataFrame) -> List[str]:
#     """
#     df_vin contains one row per file with a 'signals' column holding list[str].
#     """
#     n_files = len(df_vin)
#     if n_files == 0:
#         return []

#     # Count presence per signal (within VIN)
#     counts: Dict[str, int] = {}
#     for sig_list in df_vin["signals"]:
#         for s in sig_list:
#             counts[s] = counts.get(s, 0) + 1

#     stable = [s for s, c in counts.items() if (c / n_files) >= VIN_STABLE_PCT]
#     stable.sort()
#     return stable

def compute_vin_stable_signals_from_files(df_vin: pd.DataFrame, vin: str) -> List[str]:
    """
    Compute VIN-stable signals by scanning CDF headers (keys only).
    Stable = appears in >= VIN_STABLE_PCT of files for this VIN.
    """
    n_files = len(df_vin)
    if n_files == 0:
        return []

    counts: Dict[str, int] = {}
    ok_files = 0

    for _, r in df_vin.sort_values("file_ts").iterrows():
        path = str(r["path"])
        try:
            with pycdf.CDF(path) as cdf:
                keys = list(cdf.keys())
                ok_files += 1
                for s in keys:
                    counts[s] = counts.get(s, 0) + 1
        except Exception:
            # skip bad file; errors will be logged in pass 2 anyway
            continue

    denom = max(ok_files, 1)
    stable = [s for s, c in counts.items() if (c / denom) >= VIN_STABLE_PCT]
    stable.sort()

    print(f"    {vin}: stable signals computed from headers: {len(stable)} (ok_files={ok_files}/{n_files})")
    return stable


def safe_to_1d_array(x: Any) -> np.ndarray:
    arr = np.asanyarray(x)
    # many CDF vars come as shape (N,) already; if multi-d, flatten for now
    return arr.ravel()


def build_one_vin(vin: str, df_vin: pd.DataFrame, core_signals: List[str]) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []

    vin_out_dir = OUT_BASE / vin
    vin_out_dir.mkdir(parents=True, exist_ok=True)

    # Tier-2 stable signals for this VIN
    vin_stable = compute_vin_stable_signals_from_files(df_vin, vin)

    # columns to extract = core + vin_stable + time var (discovered per file)
    # NOTE: a signal may not exist in a particular file => NaN
    target_signals = sorted(set(core_signals).union(vin_stable))

    all_frames = []
    years = set()

    for _, r in df_vin.sort_values("file_ts").iterrows():
        path = str(r["path"])
        date_yyyymmdd = str(r.get("date_yyyymmdd", ""))
        year_folder = r.get("year_folder", None)
        if pd.notna(year_folder):
            try:
                years.add(int(year_folder))
            except Exception:
                pass

        try:
            with pycdf.CDF(path) as cdf:
                keys = list(cdf.keys())
                tvar = pick_time_var(keys)
                if tvar is None:
                    errors.append({
                        "vin": vin, "path": path, "error_scope": "missing_time_var",
                        "error_message": f"No time var found. Keys include: {keys[:10]}..."
                    })
                    continue

                try:
                    t = safe_to_1d_array(cdf[tvar][...])
                except Exception as e:
                    errors.append({
                        "vin": vin, "path": path, "error_scope": "time_read_error",
                        "error_message": str(e)
                    })
                    continue

                n = t.size
                if n == 0:
                    errors.append({
                        "vin": vin, "path": path, "error_scope": "zero_length_time",
                        "error_message": f"{tvar} has length 0"
                    })
                    continue

                data = {"timestamp": t}

                # extract each signal if present; otherwise fill NaN
                for s in target_signals:
                    if s == tvar:
                        continue
                    if s in cdf:
                        try:
                            v = safe_to_1d_array(cdf[s][...])
                            # align length: if different length, pad/truncate
                            if v.size == n:
                                data[s] = v
                            elif v.size == 0:
                                data[s] = np.full(n, np.nan)
                            else:
                                # conservative: pad/truncate to match time axis
                                if v.size > n:
                                    data[s] = v[:n]
                                else:
                                    pad = np.full(n - v.size, np.nan)
                                    data[s] = np.concatenate([v, pad])
                        except Exception as e:
                            errors.append({
                                "vin": vin, "path": path, "signal_name": s,
                                "error_scope": "signal_read_error",
                                "error_message": str(e)
                            })
                            data[s] = np.full(n, np.nan)
                    else:
                        data[s] = np.full(n, np.nan)

                df_file = pd.DataFrame(data)
                df_file["source_file"] = Path(path).name
                df_file["date_yyyymmdd"] = date_yyyymmdd

                all_frames.append(df_file)

        except Exception as e:
            errors.append({
                "vin": vin, "path": path, "error_scope": "file_open_error",
                "error_message": str(e)
            })

    if not all_frames:
        return False, {"vin": vin, "n_rows": 0, "n_files": len(df_vin)}, errors

    out_df = pd.concat(all_frames, ignore_index=True)

    # sort by timestamp within VIN (and stable secondary key)
    out_df = out_df.sort_values(["timestamp", "date_yyyymmdd", "source_file"]).reset_index(drop=True)

    # write parquet
    out_parquet = vin_out_dir / "timeseries.parquet"
    out_df.to_parquet(out_parquet, index=False)

    # write signal_manifest
    manifest_rows = []
    for s in target_signals:
        tier = "core" if s in core_signals else "vin_stable"
        manifest_rows.append({"signal_name": s, "tier": tier})
    pd.DataFrame(manifest_rows).sort_values(["tier", "signal_name"]).to_csv(vin_out_dir / "signal_manifest.csv", index=False)

    # metadata
    meta = {
        "vin": vin,
        "n_files": int(len(df_vin)),
        "n_rows": int(len(out_df)),
        "years_covered": sorted(list(years)) if years else [],
        "core_signal_count": int(len(core_signals)),
        "vin_stable_signal_count": int(len(vin_stable)),
        "total_signal_count_written": int(len(target_signals)),
        "parquet_path": str(out_parquet),
    }
    with open(vin_out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return True, meta, errors


def main():
    if not CATALOG_PARQUET.exists():
        raise FileNotFoundError(f"Missing: {CATALOG_PARQUET}")
    core_signals = load_core_signals()

    df = pd.read_parquet(CATALOG_PARQUET)

    # Expect columns: group, model, vin, year_folder, date_yyyymmdd, time_hhmm, filename, path, file_ts, signals, schema_hash
   # 'signals' column is optional; if missing we will compute VIN-stable signals from file headers


    vins = sorted(df["vin"].astype(str).unique().tolist())

    print(">>> build_vin_parquet.py START <<<")
    print("VINs:", len(vins))
    print("Core signals:", len(core_signals))
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    all_errors: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, vin in enumerate(vins, 1):
        df_vin = df[df["vin"].astype(str) == vin].copy()
        ok, meta, errs = build_one_vin(vin, df_vin, core_signals)

        if not ok:
            print(f"[{i}/{len(vins)}] {vin}: ❌ no output rows")
        else:
            print(f"[{i}/{len(vins)}] {vin}: ✅ rows={meta['n_rows']:,} files={meta['n_files']} signals={meta['total_signal_count_written']}")

        all_errors.extend(errs)

    if all_errors:
        pd.DataFrame(all_errors).to_csv(ERROR_LOG, index=False)
        print("⚠️ Errors written:", ERROR_LOG)

    print(f"Elapsed: {time.time()-t0:,.1f}s")
    print(">>> build_vin_parquet.py DONE <<<")


if __name__ == "__main__":
    main()
