# src/exhbrake_failure_scan.py
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
ACCEL_THRESH = 1.0          # percent; switch to 7.0 if needed
MIN_EVENT_SEC = 3           # minimum enable window duration
NEG_TORQ_THRESH = -0.1      # treat anything < 0 as braking; can set to -1.0 to be stricter

HIGH_BAND = (40, 60)
LOW_BAND  = (5, 10)

SIGNALS = [
    "AccelPedalPos",
    "AccelPedalPos_1587",
    "Act_RetardPctTorqExh",
    "EngRetarderStat_1587",
    "TransTorqConvLockupEngaged",
    "TrRgAttai",
    # plus: engine speed and vehicle speed names you have
    "EngSpeed",
    "VehSpeed",
    # optional:
    "BrakeSwitch",
]

OUT_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")

# ----------------------------
# Helpers
# ----------------------------
def mph_to_mps(x_mph: pd.Series) -> pd.Series:
    return x_mph * 0.44704

def speed_band(mph: float) -> str:
    if HIGH_BAND[0] <= mph <= HIGH_BAND[1]:
        return "HIGH_40_60"
    if LOW_BAND[0] <= mph <= LOW_BAND[1]:
        return "LOW_5_10"
    return "OTHER"

def contiguous_true_runs(mask: pd.Series) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx) inclusive runs where mask is True."""
    mask = mask.fillna(False).astype(bool).values
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs

def pick_time_col(df: pd.DataFrame) -> str:
    # adapt if your parquet uses a specific column name
    for c in ["timestamp", "time", "UTC_1Hz", "UTC"]:
        if c in df.columns:
            return c
    # fallback: first datetime-like column
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    raise ValueError("No timestamp column found (expected timestamp/UTC_1Hz/UTC).")

def safe_get(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    return df[col] if col in df.columns else None

# ----------------------------
# Event extraction + scoring
# ----------------------------
@dataclass
class Event:
    vin: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_sec: float
    speed_mph_med: float
    band: str
    trg_mode: Optional[float]
    min_retard: float
    pct_neg: float
    delay_to_neg: Optional[float]
    decel_mps2_med: Optional[float]
    rpm_decay_med: Optional[float]
    is_clear_failure: bool
    severity: float

def compute_event_features(vin: str, df: pd.DataFrame, time_col: str, i0: int, i1: int) -> Event:
    seg = df.iloc[i0:i1+1].copy()

    t0 = seg[time_col].iloc[0]
    t1 = seg[time_col].iloc[-1]
    duration = (t1 - t0).total_seconds() if pd.notnull(t0) and pd.notnull(t1) else float(i1 - i0)

    vs = safe_get(seg, "VehSpeed")
    speed_med = float(np.nanmedian(vs)) if vs is not None else float("nan")
    band = speed_band(speed_med) if np.isfinite(speed_med) else "UNKNOWN"

    trg = safe_get(seg, "TrRgAttai")
    trg_mode = None
    if trg is not None and trg.notna().any():
        trg_mode = float(trg.mode().iloc[0])

    retard = safe_get(seg, "Act_RetardPctTorqExh")
    if retard is None:
        min_retard = float("nan")
        pct_neg = float("nan")
        delay_to_neg = None
    else:
        min_retard = float(np.nanmin(retard))
        neg_mask = retard < NEG_TORQ_THRESH
        pct_neg = float(np.nanmean(neg_mask)) if neg_mask.notna().any() else 0.0
        if neg_mask.any():
            first_idx = int(np.argmax(neg_mask.values))
            delay_to_neg = float(first_idx)  # 1 Hz assumption; if timestamps irregular, compute from time_col
        else:
            delay_to_neg = None

    # Decel: compute derivative of speed
    decel_med = None
    if vs is not None and len(seg) >= 3:
        v_mps = mph_to_mps(vs.astype(float))
        # 1 Hz: dv/dt approx diff
        dv = v_mps.diff()
        # decel positive means slowing down -> -dv
        decel = (-dv).clip(lower=0)
        decel_med = float(np.nanmedian(decel))

    # RPM decay
    rpm_decay_med = None
    es = safe_get(seg, "EngSpeed")
    if es is not None and len(seg) >= 3:
        des = es.astype(float).diff()
        # rpm decay positive means slowing down -> -dRPM
        rpm_decay = (-des).clip(lower=0)
        rpm_decay_med = float(np.nanmedian(rpm_decay))

    # Clear failure rule (conservative):
    # enabled window long enough, commanded ON, but no negative torque AND weak decel (speed-band aware)
    # decel thresholds are placeholders; we will calibrate from fleet distributions.
    weak_decel = True
    if decel_med is not None:
        if band == "HIGH_40_60":
            weak_decel = decel_med < 0.05  # m/s^2 (placeholder; calibrate)
        elif band == "LOW_5_10":
            weak_decel = decel_med < 0.02
        else:
            weak_decel = decel_med < 0.03

    no_torque = (retard is not None) and (pct_neg == 0.0)
    is_clear_failure = (duration >= MIN_EVENT_SEC) and no_torque and weak_decel

    # Severity (simple):
    # bigger if longer duration, weaker torque, weaker decel
    sev = 0.0
    if np.isfinite(duration):
        sev += min(duration / 10.0, 3.0)
    if retard is not None and np.isfinite(min_retard):
        sev += 1.0 if min_retard > -1 else 0.0
    if decel_med is not None:
        sev += 1.0 if weak_decel else 0.0

    return Event(
        vin=vin,
        start_time=t0,
        end_time=t1,
        duration_sec=float(duration),
        speed_mph_med=speed_med,
        band=band,
        trg_mode=trg_mode,
        min_retard=min_retard,
        pct_neg=pct_neg,
        delay_to_neg=delay_to_neg,
        decel_mps2_med=decel_med,
        rpm_decay_med=rpm_decay_med,
        is_clear_failure=bool(is_clear_failure),
        severity=float(sev),
    )

def build_enable_mask(df: pd.DataFrame) -> pd.Series:
    tc = safe_get(df, "TransTorqConvLockupEngaged")
    ret = safe_get(df, "EngRetarderStat_1587")
    a1  = safe_get(df, "AccelPedalPos")
    a2  = safe_get(df, "AccelPedalPos_1587")

    if tc is None or ret is None or (a1 is None and a2 is None):
        return pd.Series(False, index=df.index)

    accel = a1 if a1 is not None else a2
    # prefer raw AccelPedalPos but fall back to _1587 if missing
    if a1 is not None and a2 is not None:
        accel = a1.fillna(a2)

    mask = (tc == 1) & (ret == 1) & (accel < ACCEL_THRESH)
    return mask

# ----------------------------
# Plotting
# ----------------------------
def plot_good_vs_bad(vin: str, df: pd.DataFrame, time_col: str, good_evt: Event, bad_evt: Event, out_path: Path):
    def slice_evt(evt: Event):
        m = (df[time_col] >= evt.start_time) & (df[time_col] <= evt.end_time)
        seg = df.loc[m, [time_col] + [c for c in SIGNALS if c in df.columns]].copy()
        # relative time axis in seconds
        seg["t_sec"] = (seg[time_col] - seg[time_col].iloc[0]).dt.total_seconds()
        return seg

    g = slice_evt(good_evt)
    b = slice_evt(bad_evt)

    rows = [
        "AccelPedalPos",
        "EngRetarderStat_1587",
        "TransTorqConvLockupEngaged",
        "Act_RetardPctTorqExh",
        "TrRgAttai",
        "EngSpeed",
        "VehSpeed",
    ]
    rows = [r for r in rows if r in df.columns]

    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 2.2 * len(rows)), sharex="col")
    if len(rows) == 1:
        axes = np.array([axes])

    for i, sig in enumerate(rows):
        axL = axes[i, 0]
        axR = axes[i, 1]
        axL.plot(g["t_sec"], g[sig])
        axR.plot(b["t_sec"], b[sig])
        axL.set_ylabel(sig)
        axR.set_ylabel(sig)

        if i == 0:
            axL.set_title(f"{vin} GOOD (band={good_evt.band}, dur={good_evt.duration_sec:.1f}s)")
            axR.set_title(f"{vin} BAD (band={bad_evt.band}, dur={bad_evt.duration_sec:.1f}s)")

    axes[-1, 0].set_xlabel("seconds from event start")
    axes[-1, 1].set_xlabel("seconds from event start")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    vin_dirs = sorted(glob.glob("data/analysis/VIN*/timeseries.parquet"))
    all_events: List[Event] = []
    logs: List[Dict] = []

    for p in vin_dirs:
        vin = Path(p).parent.name.replace("VIN", "")
        try:
            df = pd.read_parquet(p)
            time_col = pick_time_col(df)
            df = df.sort_values(time_col).reset_index(drop=True)

            enable = build_enable_mask(df)
            runs = contiguous_true_runs(enable)

            # Build events
            for i0, i1 in runs:
                # duration filter using sample count at 1Hz
                if (i1 - i0 + 1) < MIN_EVENT_SEC:
                    continue
                evt = compute_event_features(vin, df, time_col, i0, i1)
                all_events.append(evt)

        except Exception as e:
            logs.append({"vin": vin, "path": str(p), "error": repr(e)})

    # Write event catalog
    ev_df = pd.DataFrame([e.__dict__ for e in all_events])
    ev_path = OUT_DIR / "exhbrake_event_catalog.csv"
    ev_df.to_csv(ev_path, index=False)

    fail_df = ev_df[ev_df["is_clear_failure"] == True].copy()
    fail_path = OUT_DIR / "exhbrake_failure_events.csv"
    fail_df.to_csv(fail_path, index=False)

    # VIN summary
    if len(ev_df) > 0:
        vin_sum = (ev_df.groupby("vin")
                   .agg(n_events=("vin", "size"),
                        n_fail=("is_clear_failure", "sum"),
                        fail_rate=("is_clear_failure", "mean"),
                        worst_severity=("severity", "max"),
                        first_fail=("start_time", "min"),
                        last_fail=("end_time", "max"))
                   .reset_index())
        vin_sum.to_csv(OUT_DIR / "vin_failure_summary.csv", index=False)

    # Log
    if logs:
        pd.DataFrame(logs).to_csv(OUT_DIR / "exhbrake_failures_log.csv", index=False)

    # Produce good-vs-bad plots for top failing VINs
    if len(fail_df) > 0:
        top = (fail_df.sort_values("severity", ascending=False)
               .groupby("vin").head(1).head(20))

        for _, r in top.iterrows():
            vin = r["vin"]
            parquet_path = f"data/analysis/VIN{vin}/timeseries.parquet"
            df = pd.read_parquet(parquet_path)
            time_col = pick_time_col(df)
            df = df.sort_values(time_col).reset_index(drop=True)

            # find best GOOD event in same speed band
            v_events = ev_df[ev_df["vin"] == vin].copy()
            bad_evt = Event(**{k: r[k] for k in Event.__annotations__.keys()})
            same_band = v_events[v_events["band"] == bad_evt.band]
            good_candidates = same_band[same_band["pct_neg"] > 0.5].sort_values("min_retard")  # more negative = stronger
            if len(good_candidates) == 0:
                continue
            g = good_candidates.iloc[0]
            good_evt = Event(**{k: g[k] for k in Event.__annotations__.keys()})

            out = FIG_DIR / f"VIN{vin}_{bad_evt.band}_good_vs_bad.png"
            plot_good_vs_bad(vin, df, time_col, good_evt, bad_evt, out)

    print(f"✅ Wrote:\n- {ev_path}\n- {fail_path}\n- {OUT_DIR / 'vin_failure_summary.csv'}\n- {OUT_DIR / 'exhbrake_failures_log.csv'}\n✅ Plots in: {FIG_DIR}")

if __name__ == "__main__":
    main()
