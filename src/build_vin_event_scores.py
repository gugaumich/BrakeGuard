"""
Build VIN-level exhaust-brake event scores.

Output:
- data/analysis/vin_event_scores.csv  (one row per candidate exhaust-braking event)
- data/analysis/vin_event_scores_failures.csv (files/VINs with issues)

Logic (configurable):
We define a candidate event when (per sample):
  - Retarder is ON (EngRetarderStat_1587 or similar)  [required if present]
  - Accelerator is low (AccelPedalPos or AccelPedalPos_1587)              [required]
  - Optional: BrakeSwitch == 1                                            [if present + enabled]
  - Optional: Torque converter lockup engaged == 1                        [if present + enabled]
  - Optional: EngSpeed > 1000                                             [if present + enabled]
We then segment contiguous True samples into events, and compute per-event metrics.

Metrics per event:
  - event_start_time, event_end_time, duration_s
  - speed_start, speed_end, decel_rate
  - min_Act_RetardPctTorqExh, median_Act_RetardPctTorqExh
  - retarder_on_duration_s
  - speed_band
  - label (GOOD/BAD) from data/analysis/vin_exhaust_valve_labels.csv if available
"""

from __future__ import annotations

from pathlib import Path
import traceback
import pandas as pd
import numpy as np


# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIN_PARQUET_GLOB = PROJECT_ROOT / "data" / "analysis" / "VIN*" / "timeseries.parquet"
LABELS_CSV = PROJECT_ROOT / "data" / "analysis" / "vin_exhaust_valve_labels.csv"

OUT_SCORES = PROJECT_ROOT / "data" / "analysis" / "vin_event_scores.csv"
OUT_FAIL = PROJECT_ROOT / "data" / "analysis" / "vin_event_scores_failures.csv"

# event segmentation
MIN_EVENT_LEN_SAMPLES = 5          # ~5 seconds if your data is 1Hz; adjust if not
PAD_SAMPLES = 0                    # optional: include padding around events (for later plotting)

# thresholds
ACCEL_MAX = 3.0                    # pedal less than this means "foot off"
ENGSPEED_MIN = 1000.0              # only if EngSpeed exists & enabled below

# optional gates (set True/False)
REQUIRE_BRAKESWITCH = False         # set True if BrakeSwitch is reliable and you want it
REQUIRE_LOCKUP = False             # set True if lockup signal exists and is reliable
REQUIRE_ENGSPEED = True            # only applies if an EngSpeed column exists

# speed bands for comparability
SPEED_BANDS = [
    (5, 15, "5-15"),
    (20, 35, "20-35"),
    (40, 60, "40-60"),
]


# ----------------------------
# Column detection helpers
# ----------------------------
def pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def detect_columns(cols):
    """
    Returns a dict of canonical roles -> actual column names (or None).
    """
    cols = list(cols)

    time_col = pick_first_existing(cols, ["timestamp", "UTC_1Hz", "UTC", "utc", "Time", "time", "Timestamp"])

    accel_col = pick_first_existing(cols, ["AccelPedalPos", "AccelPedalPos_1587"])

    # Retarder ON status (your dataset likely has EngRetarderStat_1587; not in first 50 but may exist later)
    retarder_col = pick_first_existing(cols, [
        "EngRetarderStat_1587",
        "EngRetarderStat",
        "EngineRetarderStatus",
        "Cylinder2EngRetStat_1587",  # not ideal, but better than nothing
    ])

    # Achieved retard torque percent (this is your core response variable)
    retard_pct_col = pick_first_existing(cols, ["Act_RetardPctTorqExh"])

    # Vehicle speed
    speed_col = pick_first_existing(cols, [
        "VehSpeedEng",
        "VehSpeed",
        "VehicleSpeed",
        "WheelBasedVehicleSpeed",
        "VehSpd_1587",
        "VehicleSpeed_1587",
    ])

    # Engine speed
    engspeed_col = pick_first_existing(cols, ["EngSpeed", "EngineSpeed", "EngSpeed_1587"])

    # Torque converter lockup
    lockup_col = pick_first_existing(cols, [
        "TransTorqConvLockupEngaged",
        "TrnsTorqConvLockupEngaged",
        "TransTorqConvLockupEngaged_1587",
    ])

    brakeswitch_col = pick_first_existing(cols, ["BrakeSwitch"])

    return {
        "time": time_col,
        "accel": accel_col,
        "retarder": retarder_col,
        "retard_pct": retard_pct_col,
        "speed": speed_col,
        "engspeed": engspeed_col,
        "lockup": lockup_col,
        "brakeswitch": brakeswitch_col,
    }


def to_datetime_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    # if numeric epoch-like
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


def find_true_segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """
    mask: boolean array
    returns inclusive (start, end) segments where mask is True
    """
    segs = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            start = i
        elif (not v) and in_seg:
            end = i - 1
            if end - start + 1 >= min_len:
                segs.append((start, end))
            in_seg = False
    if in_seg:
        end = len(mask) - 1
        if end - start + 1 >= min_len:
            segs.append((start, end))
    return segs


def assign_speed_band(speed_start: float | None) -> str:
    if speed_start is None or np.isnan(speed_start):
        return "unknown"
    for lo, hi, name in SPEED_BANDS:
        if lo <= speed_start <= hi:
            return name
    return "other"


# ----------------------------
# Main scoring
# ----------------------------
def build_events_for_vin(vin: str, parquet_path: Path) -> tuple[list[dict], dict]:
    """
    Returns: (rows, metadata)
    metadata includes detected columns and counters
    """
    # load schema to choose columns
    cols = pd.read_parquet(parquet_path, engine="pyarrow").columns.tolist()
    role = detect_columns(cols)

    # required columns
    if role["time"] is None:
        raise ValueError("No time column found (expected 'timestamp').")
    if role["accel"] is None:
        raise ValueError("No accelerator column found (AccelPedalPos / AccelPedalPos_1587).")
    if role["retard_pct"] is None:
        raise ValueError("No Act_RetardPctTorqExh column found.")

    # load only needed columns (fast)
    need = [c for c in [
        role["time"], role["accel"], role["retard_pct"],
        role["retarder"], role["speed"], role["engspeed"],
        role["lockup"], role["brakeswitch"]
    ] if c is not None]

    df = pd.read_parquet(parquet_path, columns=need)

    # normalize + sort time
    df[role["time"]] = to_datetime_safe(df[role["time"]])
    df = df.dropna(subset=[role["time"]]).sort_values(role["time"]).reset_index(drop=True)

    # build condition mask
    accel = df[role["accel"]].astype(float)

    cond = (accel < ACCEL_MAX)

    # retarder ON if available
    if role["retarder"] is not None:
        ret_on = df[role["retarder"]]
        # handle booleans/ints
        ret_on = pd.to_numeric(ret_on, errors="coerce").fillna(0)
        cond = cond & (ret_on > 0)
    else:
        # if no retarder column, we still allow events based on accel only
        ret_on = pd.Series(np.nan, index=df.index)

    # optional brakeswitch
    if REQUIRE_BRAKESWITCH and role["brakeswitch"] is not None:
        bs = pd.to_numeric(df[role["brakeswitch"]], errors="coerce").fillna(0)
        cond = cond & (bs > 0)

    # optional lockup
    if REQUIRE_LOCKUP and role["lockup"] is not None:
        lk = pd.to_numeric(df[role["lockup"]], errors="coerce").fillna(0)
        cond = cond & (lk > 0)

    # optional engspeed threshold
    if REQUIRE_ENGSPEED and role["engspeed"] is not None:
        es = pd.to_numeric(df[role["engspeed"]], errors="coerce")
        cond = cond & (es > ENGSPEED_MIN)

    mask = cond.fillna(False).to_numpy(dtype=bool)

    segs = find_true_segments(mask, min_len=MIN_EVENT_LEN_SAMPLES)
    rows = []

    t = df[role["time"]]

    for idx, (s, e) in enumerate(segs):
        s2 = max(0, s - PAD_SAMPLES)
        e2 = min(len(df) - 1, e + PAD_SAMPLES)

        w = df.iloc[s2:e2+1]
        t0 = w[role["time"]].iloc[0]
        t1 = w[role["time"]].iloc[-1]
        dur_s = (t1 - t0).total_seconds()

        # speed metrics
        speed_start = speed_end = decel_rate = np.nan
        if role["speed"] is not None:
            sp = pd.to_numeric(w[role["speed"]], errors="coerce")
            speed_start = float(sp.iloc[0]) if pd.notna(sp.iloc[0]) else np.nan
            speed_end = float(sp.iloc[-1]) if pd.notna(sp.iloc[-1]) else np.nan
            if dur_s > 0 and pd.notna(speed_start) and pd.notna(speed_end):
                decel_rate = float((speed_start - speed_end) / dur_s)

        # retard torque stats
        rp = pd.to_numeric(w[role["retard_pct"]], errors="coerce")
        min_rp = float(np.nanmin(rp.to_numpy())) if rp.notna().any() else np.nan
        med_rp = float(np.nanmedian(rp.to_numpy())) if rp.notna().any() else np.nan

        # retarder duration (if retarder present)
        ret_dur = np.nan
        if role["retarder"] is not None:
            r = pd.to_numeric(w[role["retarder"]], errors="coerce").fillna(0)
            # assume ~1Hz; if not 1Hz, this is approximate. Good enough for now.
            ret_dur = float((r > 0).sum())

        band = assign_speed_band(speed_start)

        rows.append({
            "vin": vin,
            "event_id": idx,
            "event_start_time": t0,
            "event_end_time": t1,
            "duration_s": dur_s,
            "speed_start": speed_start,
            "speed_end": speed_end,
            "decel_rate": decel_rate,
            "min_Act_RetardPctTorqExh": min_rp,
            "median_Act_RetardPctTorqExh": med_rp,
            "retarder_on_duration_s": ret_dur,
            "speed_band": band,
        })

    meta = {
        "vin": vin,
        "n_rows": len(df),
        "n_events": len(rows),
        **{f"col_{k}": v for k, v in role.items()},
    }
    return rows, meta


def main():
    vin_paths = sorted(Path().glob(str(VIN_PARQUET_GLOB.relative_to(PROJECT_ROOT))))
    if not vin_paths:
        raise SystemExit(f"No VIN parquets found at {VIN_PARQUET_GLOB}")

    # load labels if present
    labels = None
    if LABELS_CSV.exists():
        labels = pd.read_csv(LABELS_CSV)
        labels["vin"] = labels["vin"].astype(str).str.strip().str.upper()
        labels = labels[["vin", "valve_label"]].drop_duplicates()
    else:
        labels = pd.DataFrame(columns=["vin", "valve_label"])

    all_rows = []
    failures = []

    print(f"Found VIN parquets: {len(vin_paths)}")

    for i, p in enumerate(vin_paths, 1):
        vin = p.parent.name.upper()
        try:
            rows, meta = build_events_for_vin(vin, p)
            all_rows.extend(rows)
            if i % 10 == 0 or i == len(vin_paths):
                print(f"...processed {i}/{len(vin_paths)} VINs | events so far: {len(all_rows)}")
        except Exception as e:
            failures.append({
                "vin": vin,
                "parquet_path": str(p),
                "error": str(e),
                "traceback": traceback.format_exc(limit=5),
            })

    df_scores = pd.DataFrame(all_rows)
    if df_scores.empty:
        raise SystemExit("No events found. Try lowering thresholds or disabling REQUIRE_ENGSPEED/REQUIRE_LOCKUP.")

    # join labels
    df_scores["vin"] = df_scores["vin"].astype(str).str.upper()
    df_scores = df_scores.merge(labels, how="left", on="vin")
    df_scores = df_scores.rename(columns={"valve_label": "label"})
    df_scores["label"] = df_scores["label"].fillna("UNKNOWN")

    # write outputs
    OUT_SCORES.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(OUT_SCORES, index=False)

    df_fail = pd.DataFrame(failures)
    if not df_fail.empty:
        df_fail.to_csv(OUT_FAIL, index=False)

    print(f"\n✅ Wrote: {OUT_SCORES}  (rows={len(df_scores)})")
    if not df_fail.empty:
        print(f"⚠️ Failures: {len(df_fail)}  -> {OUT_FAIL}")
    else:
        print("✅ No failures.")


if __name__ == "__main__":
    main()
