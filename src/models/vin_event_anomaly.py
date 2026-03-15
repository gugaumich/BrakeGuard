from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - environment dependent
    XGBRegressor = None
    HAS_XGBOOST = False


DEFAULT_PLOT_COLS = [
    "Act_RetardPctTorqExh",
    "EngSpeed",
    "VehSpeedEng",
    "VehSpeed",
    "AccelPedalPos",
]

OPTIONAL_PLOT_COLS = [
    "BoostPres",
    "BrakeSwitch",
    "ClutchSwtch",
]

DEFAULT_MAX_ANOMALY_DISPLAY = 200.0


@dataclass(frozen=True)
class ResidualStats:
    mean: float
    std: float
    median: float
    mad: float


def _validate_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _to_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _boolish_numeric(df: pd.DataFrame, col: str) -> pd.Series | None:
    series = _to_numeric_series(df, col)
    if series is None:
        return None
    return series.fillna(0.0)


def _build_elapsed_seconds(timestamps: pd.Series) -> np.ndarray:
    delta = timestamps - timestamps.iloc[0]
    return delta.dt.total_seconds().to_numpy(dtype=float)


def _value_at_seconds(values: np.ndarray, elapsed_seconds: np.ndarray, seconds: float) -> float:
    valid = np.isfinite(values) & np.isfinite(elapsed_seconds)
    if not valid.any():
        return np.nan
    idx = np.where(valid & (elapsed_seconds <= seconds))[0]
    if len(idx) == 0:
        return np.nan
    return float(values[idx[-1]])


def _safe_slope(elapsed_seconds: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(elapsed_seconds)
    if valid.sum() < 2:
        return np.nan
    x = elapsed_seconds[valid]
    y = values[valid]
    if np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _safe_auc(elapsed_seconds: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(elapsed_seconds)
    if valid.sum() < 2:
        return np.nan
    x = elapsed_seconds[valid]
    y = values[valid]
    order = np.argsort(x)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y[order], x[order]))
    return float(np.trapz(y[order], x[order]))


def _normalized_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _merge_short_gaps(mask: np.ndarray, merge_gap: int) -> np.ndarray:
    if merge_gap <= 0 or mask.size == 0:
        return mask.copy()

    merged = mask.copy()
    true_idx = np.flatnonzero(mask)
    if true_idx.size <= 1:
        return merged

    for left, right in zip(true_idx[:-1], true_idx[1:]):
        gap = right - left - 1
        if 0 < gap <= merge_gap:
            merged[left + 1 : right] = True
    return merged


def segment_precomputed_mask_events(
    df: pd.DataFrame,
    mask_col: str,
    time_col: str = "timestamp",
    min_len: int = 3,
    merge_gap: int = 1,
) -> pd.DataFrame:
    """
    Segment consecutive True values from a precomputed event mask.

    Parameters
    ----------
    df:
        Raw per-VIN time-series.
    mask_col:
        Boolean-ish event mask already present in `df`.
    time_col:
        Timestamp column used for event timing.
    min_len:
        Minimum number of masked samples required to keep an event.
    merge_gap:
        Merge neighboring event spans separated by <= this many unmasked samples.
    """
    _validate_columns(df, [mask_col, time_col], "segment_precomputed_mask_events")
    if min_len < 1:
        raise ValueError("min_len must be at least 1.")
    if merge_gap < 0:
        raise ValueError("merge_gap cannot be negative.")

    work = df.copy()
    work[time_col] = _to_datetime_series(work[time_col])
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "start_idx",
                "end_idx",
                "start_time",
                "end_time",
                "duration_sec",
                "n_samples",
            ]
        )

    mask = work[mask_col].fillna(False).astype(bool).to_numpy()
    mask = _merge_short_gaps(mask, merge_gap=merge_gap)

    starts: list[int] = []
    ends: list[int] = []
    in_event = False
    start_idx = 0

    for idx, flag in enumerate(mask):
        if flag and not in_event:
            start_idx = idx
            in_event = True
        elif not flag and in_event:
            end_idx = idx - 1
            if (end_idx - start_idx + 1) >= min_len:
                starts.append(start_idx)
                ends.append(end_idx)
            in_event = False

    if in_event:
        end_idx = len(mask) - 1
        if (end_idx - start_idx + 1) >= min_len:
            starts.append(start_idx)
            ends.append(end_idx)

    rows: list[dict[str, Any]] = []
    times = work[time_col]
    for event_id, (start, end) in enumerate(zip(starts, ends), start=1):
        start_time = times.iloc[start]
        end_time = times.iloc[end]
        rows.append(
            {
                "event_id": event_id,
                "start_idx": int(start),
                "end_idx": int(end),
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": float((end_time - start_time).total_seconds()),
                "n_samples": int(end - start + 1),
            }
        )

    return pd.DataFrame(rows)


def segment_existing_event_id_groups(
    df: pd.DataFrame,
    event_id_col: str = "event_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Build event spans from an existing per-row event identifier.

    This matches the repository's slim event-sample datasets, which already carry
    one `event_id` per pre-extracted event window.
    """
    _validate_columns(df, [event_id_col, time_col], "segment_existing_event_id_groups")

    work = df.copy()
    work[time_col] = _to_datetime_series(work[time_col])
    work = work.dropna(subset=[time_col]).sort_values([time_col, event_id_col]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "start_idx",
                "end_idx",
                "start_time",
                "end_time",
                "duration_sec",
                "n_samples",
            ]
        )

    rows: list[dict[str, Any]] = []
    grouped = work.groupby(event_id_col, sort=False, dropna=True)
    for _, group in grouped:
        start_idx = int(group.index.min())
        end_idx = int(group.index.max())
        start_time = group[time_col].iloc[0]
        end_time = group[time_col].iloc[-1]
        rows.append(
            {
                "event_id": group[event_id_col].iloc[0],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": float((end_time - start_time).total_seconds()),
                "n_samples": int(len(group)),
            }
        )

    return pd.DataFrame(rows).sort_values(["start_time", "event_id"]).reset_index(drop=True)


def extract_event_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, str],
) -> pd.DataFrame:
    """
    Compute event-level dynamic braking features from raw time-series windows.
    """
    time_col = config.get("time_col", "timestamp")
    retard_col = config.get("retard_col", "Act_RetardPctTorqExh")
    _validate_columns(df, [time_col], "extract_event_features input dataframe")
    _validate_columns(events, ["event_id", "start_idx", "end_idx"], "extract_event_features events dataframe")

    work = df.copy()
    work[time_col] = _to_datetime_series(work[time_col])
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if work.empty:
        raise ValueError("extract_event_features received an empty dataframe after timestamp parsing.")
    if retard_col not in work.columns:
        raise ValueError(f"Required retard column '{retard_col}' was not found in the input dataframe.")

    eng_speed_col = config.get("eng_speed_col") or _first_existing_column(work, ["EngSpeed", "EngineSpeed", "EngSpeed_1587"])
    veh_speed_col = config.get("veh_speed_col") or _first_existing_column(work, ["VehSpeedEng", "VehSpeed", "VehicleSpeed", "VehSpd_1587"])
    accel_col = config.get("accel_col") or _first_existing_column(work, ["AccelPedalPos", "AccelPedalPos_1587"])
    boost_col = config.get("boost_col", "BoostPres")
    baro_col = config.get("baro_col", "BarPres_Eng_1587")
    coolant_col = config.get("coolant_col", "EngCoolantTemp")
    gear_col = config.get("gear_col", "TransCurGear")
    lockup_col = config.get("lockup_col", "TransTorqConvLockupEngaged")
    brake_col = config.get("brake_col", "BrakeSwitch")
    clutch_col = config.get("clutch_col", "ClutchSwtch")
    abs_ret_col = config.get("abs_ret_col", "ABS_RetCont_1587")

    feature_rows: list[dict[str, Any]] = []

    for event in events.itertuples(index=False):
        start_idx = int(event.start_idx)
        end_idx = int(event.end_idx)
        if start_idx < 0 or end_idx >= len(work):
            raise IndexError(
                f"Event {event.event_id} references rows outside the dataframe bounds: "
                f"start_idx={start_idx}, end_idx={end_idx}, len(df)={len(work)}."
            )

        event_slice = work.iloc[start_idx : end_idx + 1].copy()
        event_slice = event_slice.dropna(subset=[time_col]).reset_index(drop=True)
        if event_slice.empty:
            continue

        times = event_slice[time_col]
        elapsed = _build_elapsed_seconds(times)

        retard = _to_numeric_series(event_slice, retard_col)
        if retard is None:
            raise ValueError(f"Required retard column '{retard_col}' was not found in event data.")
        retard_values = retard.to_numpy(dtype=float)
        abs_retard = np.abs(retard_values)

        peak_idx = int(np.nanargmax(abs_retard)) if np.isfinite(abs_retard).any() else 0
        peak_abs_retard = float(np.nanmax(abs_retard)) if np.isfinite(abs_retard).any() else np.nan

        eng_speed = _to_numeric_series(event_slice, eng_speed_col)
        veh_speed = _to_numeric_series(event_slice, veh_speed_col)
        accel = _to_numeric_series(event_slice, accel_col)
        boost = _to_numeric_series(event_slice, boost_col)
        baro = _to_numeric_series(event_slice, baro_col)
        coolant = _to_numeric_series(event_slice, coolant_col)
        gear = _to_numeric_series(event_slice, gear_col)
        lockup = _boolish_numeric(event_slice, lockup_col)
        brake = _boolish_numeric(event_slice, brake_col)
        clutch = _boolish_numeric(event_slice, clutch_col)
        abs_ret = _boolish_numeric(event_slice, abs_ret_col)

        eng_values = eng_speed.to_numpy(dtype=float) if eng_speed is not None else np.array([], dtype=float)
        veh_values = veh_speed.to_numpy(dtype=float) if veh_speed is not None else np.array([], dtype=float)
        accel_values = accel.to_numpy(dtype=float) if accel is not None else np.array([], dtype=float)

        initial_retard = float(retard.iloc[0]) if len(retard) else np.nan
        initial_eng_speed = float(eng_speed.iloc[0]) if eng_speed is not None and len(eng_speed) else np.nan
        initial_veh_speed = float(veh_speed.iloc[0]) if veh_speed is not None and len(veh_speed) else np.nan
        initial_accel = float(accel.iloc[0]) if accel is not None and len(accel) else np.nan

        eng_at_1s = _value_at_seconds(eng_values, elapsed, 1.0) if eng_speed is not None else np.nan
        eng_at_2s = _value_at_seconds(eng_values, elapsed, 2.0) if eng_speed is not None else np.nan
        veh_at_1s = _value_at_seconds(veh_values, elapsed, 1.0) if veh_speed is not None else np.nan
        veh_at_2s = _value_at_seconds(veh_values, elapsed, 2.0) if veh_speed is not None else np.nan

        row: dict[str, Any] = {
            "event_id": int(event.event_id),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": getattr(event, "start_time", times.iloc[0]),
            "end_time": getattr(event, "end_time", times.iloc[-1]),
            "event_order": int(event.event_id),
            "event_duration": float(elapsed[-1]) if len(elapsed) else 0.0,
            "n_samples": int(len(event_slice)),
            "initial_retard_torque": initial_retard,
            "peak_abs_retard_torque": peak_abs_retard,
            "mean_abs_retard_torque": float(np.nanmean(abs_retard)) if np.isfinite(abs_retard).any() else np.nan,
            "retard_torque_auc": _safe_auc(elapsed, abs_retard),
            "time_to_peak_retard": float(elapsed[peak_idx]) if len(elapsed) else np.nan,
            "initial_engine_speed": initial_eng_speed,
            "initial_vehicle_speed": initial_veh_speed,
            "rpm_drop_1s": float(initial_eng_speed - eng_at_1s) if np.isfinite(initial_eng_speed) and np.isfinite(eng_at_1s) else np.nan,
            "rpm_drop_2s": float(initial_eng_speed - eng_at_2s) if np.isfinite(initial_eng_speed) and np.isfinite(eng_at_2s) else np.nan,
            "veh_speed_drop_1s": float(initial_veh_speed - veh_at_1s) if np.isfinite(initial_veh_speed) and np.isfinite(veh_at_1s) else np.nan,
            "veh_speed_drop_2s": float(initial_veh_speed - veh_at_2s) if np.isfinite(initial_veh_speed) and np.isfinite(veh_at_2s) else np.nan,
            "rpm_decay_slope": _safe_slope(elapsed, eng_values) if eng_speed is not None else np.nan,
            "veh_decel_slope": _safe_slope(elapsed, veh_values) if veh_speed is not None else np.nan,
            "initial_accel_pedal": initial_accel,
            "normalized_peak_retard_by_initial_rpm": _normalized_ratio(peak_abs_retard, initial_eng_speed),
            "normalized_peak_retard_by_initial_speed": _normalized_ratio(peak_abs_retard, initial_veh_speed),
        }

        if boost is not None:
            row["initial_boost_pressure"] = float(boost.iloc[0]) if len(boost) else np.nan
            row["mean_boost_pressure"] = float(boost.mean(skipna=True))
        if baro is not None:
            row["initial_baro_pressure"] = float(baro.iloc[0]) if len(baro) else np.nan
        if coolant is not None:
            row["initial_coolant_temp"] = float(coolant.iloc[0]) if len(coolant) else np.nan
        if gear is not None:
            row["initial_gear"] = float(gear.iloc[0]) if len(gear) else np.nan
            row["median_gear"] = float(gear.median(skipna=True))
        if lockup is not None:
            row["lockup_fraction"] = float(lockup.mean(skipna=True))
            row["lockup_initial"] = float(lockup.iloc[0]) if len(lockup) else np.nan
        if brake is not None:
            row["brake_fraction"] = float(brake.mean(skipna=True))
            row["brake_initial"] = float(brake.iloc[0]) if len(brake) else np.nan
        if clutch is not None:
            row["clutch_fraction"] = float(clutch.mean(skipna=True))
            row["clutch_initial"] = float(clutch.iloc[0]) if len(clutch) else np.nan
        if abs_ret is not None:
            row["abs_ret_fraction"] = float(abs_ret.mean(skipna=True))
            row["abs_ret_initial"] = float(abs_ret.iloc[0]) if len(abs_ret) else np.nan

        feature_rows.append(row)

    if not feature_rows:
        raise ValueError("No event features were extracted. Check the event indices and source dataframe.")

    feature_df = pd.DataFrame(feature_rows).sort_values(["start_time", "event_id"]).reset_index(drop=True)
    feature_df["event_order"] = np.arange(1, len(feature_df) + 1)
    return feature_df


def split_early_late_events(
    event_df: pd.DataFrame,
    early_fraction: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split where early-life events are treated as mostly healthy.
    """
    if not 0 < early_fraction < 1:
        raise ValueError("early_fraction must be between 0 and 1.")
    _validate_columns(event_df, ["start_time"], "split_early_late_events")
    if len(event_df) < 3:
        raise ValueError("Need at least 3 events to create a meaningful early/late split.")

    ordered = event_df.sort_values(["start_time", "event_id"]).reset_index(drop=True)
    split_idx = max(1, int(np.floor(len(ordered) * early_fraction)))
    split_idx = min(split_idx, len(ordered) - 1)
    train_df = ordered.iloc[:split_idx].copy()
    test_df = ordered.iloc[split_idx:].copy()
    return train_df, test_df


def _default_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    exclude = {
        "event_id",
        "start_idx",
        "end_idx",
        "start_time",
        "end_time",
        "event_order",
        "predicted_target",
        "residual",
        "abs_residual",
        "zscore_residual",
        "anomaly_score",
        target_col,
    }
    return [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def train_expected_response_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "peak_abs_retard_torque",
    model_type: str = "xgb",
) -> tuple[Pipeline, ResidualStats]:
    """
    Train a per-VIN regression model for expected healthy event response.
    """
    _validate_columns(train_df, feature_cols + [target_col], "train_expected_response_model")
    usable_train = train_df.dropna(subset=[target_col]).copy()
    if len(usable_train) < 5:
        raise ValueError(
            f"Need at least 5 training events with non-null '{target_col}' to fit the model; got {len(usable_train)}."
        )

    resolved_model_type = model_type.lower()
    if resolved_model_type == "xgb" and HAS_XGBOOST:
        regressor: Any = XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
        )
    else:
        regressor = GradientBoostingRegressor(
            random_state=42,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
        )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", regressor),
        ]
    )

    X_train = usable_train[feature_cols]
    y_train = usable_train[target_col].astype(float)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    residuals = y_train.to_numpy(dtype=float) - train_pred
    median = float(np.nanmedian(residuals))
    mad = float(np.nanmedian(np.abs(residuals - median)))
    std = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    stats = ResidualStats(
        mean=float(np.nanmean(residuals)),
        std=std,
        median=median,
        mad=mad,
    )
    return model, stats


def score_event_anomalies(
    model: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_residual_stats: ResidualStats,
) -> pd.DataFrame:
    """
    Score event anomalies using residual error from the expected-response model.
    """
    _validate_columns(df, feature_cols + [target_col], "score_event_anomalies")
    scored = df.copy()
    scored["predicted_target"] = model.predict(scored[feature_cols])
    scored["residual"] = scored[target_col].astype(float) - scored["predicted_target"]
    scored["abs_residual"] = scored["residual"].abs()

    robust_scale = 1.4826 * train_residual_stats.mad
    fallback_scale = train_residual_stats.std if np.isfinite(train_residual_stats.std) and train_residual_stats.std > 0 else np.nan
    scale = robust_scale if np.isfinite(robust_scale) and robust_scale > 0 else fallback_scale
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    scored["zscore_residual"] = (scored["residual"] - train_residual_stats.median) / scale
    scored["anomaly_score"] = scored["zscore_residual"].abs()
    return scored


def plot_event_model_diagnostics(
    scored_df: pd.DataFrame,
    target_col: str,
    max_anomaly_display: float = DEFAULT_MAX_ANOMALY_DISPLAY,
) -> None:
    """
    Plot predicted-vs-actual and anomaly diagnostics for event-level scoring.
    """
    _validate_columns(
        scored_df,
        ["predicted_target", "residual", "anomaly_score", target_col],
        "plot_event_model_diagnostics",
    )

    if max_anomaly_display <= 0:
        raise ValueError("max_anomaly_display must be positive.")

    plot_df = scored_df.sort_values(["event_order", "start_time", "event_id"]).reset_index(drop=True)
    use_dates = "start_time" in plot_df.columns and plot_df["start_time"].notna().any()
    if use_dates:
        x_axis = pd.to_datetime(plot_df["start_time"], errors="coerce")
        x_label = "Event Date"
    else:
        x_axis = plot_df.get("event_order", pd.Series(np.arange(1, len(plot_df) + 1)))
        x_label = "Event Order"
    clipped_anomaly = plot_df["anomaly_score"].clip(upper=max_anomaly_display)
    n_clipped = int((plot_df["anomaly_score"] > max_anomaly_display).sum())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(plot_df["predicted_target"], plot_df[target_col], alpha=0.8)
    lim_min = float(np.nanmin([plot_df["predicted_target"].min(), plot_df[target_col].min()]))
    lim_max = float(np.nanmax([plot_df["predicted_target"].max(), plot_df[target_col].max()]))
    axes[0, 0].plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black", linewidth=1)
    axes[0, 0].set_title("Predicted vs Actual")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    axes[0, 1].plot(x_axis, plot_df["residual"], marker="o", linewidth=1)
    axes[0, 1].axhline(0.0, linestyle="--", color="black", linewidth=1)
    axes[0, 1].set_title("Residual by Event Date" if use_dates else "Residual by Event Order")
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel("Residual")

    axes[1, 0].plot(x_axis, clipped_anomaly, marker="o", linewidth=1, color="tab:red")
    axes[1, 0].set_title("Anomaly Score by Event Date" if use_dates else "Anomaly Score by Event Order")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("Anomaly Score")
    axes[1, 0].set_ylim(0, max_anomaly_display)
    if n_clipped:
        axes[1, 0].text(
            0.99,
            0.98,
            f"{n_clipped} event(s) clipped at {max_anomaly_display:g}",
            transform=axes[1, 0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes[1, 1].hist(clipped_anomaly.dropna(), bins=20, color="tab:orange", edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("Anomaly Score Distribution (clipped)" if n_clipped else "Anomaly Score Distribution")
    axes[1, 1].set_xlabel("Anomaly Score")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_xlim(0, max_anomaly_display)

    if use_dates:
        fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    plt.show()


def plot_top_anomaly_events(
    df_raw: pd.DataFrame,
    events_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    top_n: int = 5,
    cols: list[str] | None = None,
) -> None:
    """
    Plot aligned raw event traces for the highest-scoring anomaly events.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    time_col = _first_existing_column(df_raw, ["timestamp", "UTC_1Hz", "UTC", "time", "Time"])
    if time_col is None:
        raise ValueError("plot_top_anomaly_events could not find a timestamp column in the raw dataframe.")
    _validate_columns(events_df, ["event_id", "start_idx", "end_idx"], "plot_top_anomaly_events events dataframe")
    _validate_columns(scored_df, ["event_id", "anomaly_score"], "plot_top_anomaly_events scored dataframe")

    raw = df_raw.copy()
    raw[time_col] = _to_datetime_series(raw[time_col])
    raw = raw.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    merged = scored_df.merge(
        events_df[["event_id", "start_idx", "end_idx"]],
        on="event_id",
        how="left",
        suffixes=("", "_event"),
    )
    plot_cols = cols or [col for col in DEFAULT_PLOT_COLS + OPTIONAL_PLOT_COLS if col in raw.columns]
    if not plot_cols:
        raise ValueError("No requested plotting columns were found in the raw dataframe.")

    top_events = merged.sort_values("anomaly_score", ascending=False).head(top_n)
    for row in top_events.itertuples(index=False):
        if pd.isna(row.start_idx) or pd.isna(row.end_idx):
            continue
        start_idx = int(row.start_idx)
        end_idx = int(row.end_idx)
        event_slice = raw.iloc[start_idx : end_idx + 1].copy()
        if event_slice.empty:
            continue

        elapsed = _build_elapsed_seconds(event_slice[time_col])
        n_rows = len(plot_cols)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.8 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        for ax, col in zip(axes, plot_cols):
            values = _to_numeric_series(event_slice, col)
            if values is None:
                ax.set_visible(False)
                continue
            ax.plot(elapsed, values, linewidth=1.5)
            ax.set_ylabel(col)
            ax.grid(alpha=0.3)

        axes[0].set_title(
            f"Event {row.event_id} | anomaly_score={row.anomaly_score:.2f} | "
            f"actual={getattr(row, 'peak_abs_retard_torque', np.nan):.2f} | "
            f"pred={getattr(row, 'predicted_target', np.nan):.2f}"
        )
        axes[-1].set_xlabel("Seconds from event start")
        fig.tight_layout()
        plt.show()


def run_vin_event_anomaly_pipeline(
    df: pd.DataFrame,
    mask_col: str,
    config: dict[str, str],
    target_col: str = "peak_abs_retard_torque",
    early_fraction: float = 0.4,
) -> dict[str, Any]:
    """
    End-to-end per-VIN event anomaly workflow.
    """
    time_col = config.get("time_col", "timestamp")
    if time_col not in df.columns:
        raise ValueError(f"Configured time column '{time_col}' was not found in the dataframe.")
    raw = df.copy()
    raw[time_col] = _to_datetime_series(raw[time_col])
    raw = raw.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if raw.empty:
        raise ValueError("Input dataframe is empty after timestamp parsing.")

    event_id_col = config.get("event_id_col", "event_id")
    if mask_col in raw.columns:
        events = segment_precomputed_mask_events(
            raw,
            mask_col=mask_col,
            time_col=time_col,
            min_len=int(config.get("min_len", 3)),
            merge_gap=int(config.get("merge_gap", 1)),
        )
    elif event_id_col in raw.columns:
        events = segment_existing_event_id_groups(
            raw,
            event_id_col=event_id_col,
            time_col=time_col,
        )
    else:
        raise ValueError(
            f"Neither mask column '{mask_col}' nor event id column '{event_id_col}' was found in the dataframe. "
            "For raw VIN timeseries, provide a precomputed mask. For vin_event_samples datasets, include event_id."
        )
    if events.empty:
        raise ValueError(
            "No events were found. Check the precomputed mask, event_id column, or event segmentation settings."
        )

    event_features = extract_event_features(raw, events, config)
    if target_col not in event_features.columns:
        raise ValueError(f"Target column '{target_col}' was not generated during event feature extraction.")

    feature_cols = config.get("feature_cols")
    if feature_cols is None:
        feature_cols = _default_feature_columns(event_features, target_col=target_col)
    else:
        missing = [col for col in feature_cols if col not in event_features.columns]
        if missing:
            raise ValueError(f"Configured feature columns were not found in event features: {missing}")

    if not feature_cols:
        raise ValueError("No feature columns are available for model training.")

    train_df, test_df = split_early_late_events(event_features, early_fraction=early_fraction)
    model, train_stats = train_expected_response_model(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_type=str(config.get("model_type", "xgb")),
    )

    scored_events = score_event_anomalies(
        model=model,
        df=event_features,
        feature_cols=feature_cols,
        target_col=target_col,
        train_residual_stats=train_stats,
    )
    scored_events["dataset_split"] = np.where(scored_events["event_id"].isin(train_df["event_id"]), "train", "test")

    plot_event_model_diagnostics(
        scored_events,
        target_col=target_col,
        max_anomaly_display=float(config.get("max_anomaly_display", DEFAULT_MAX_ANOMALY_DISPLAY)),
    )

    return {
        "model": model,
        "events": events,
        "event_features": event_features,
        "scored_events": scored_events,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "train_events": train_df,
        "test_events": test_df,
        "train_residual_stats": train_stats,
    }


NOTEBOOK_USAGE_EXAMPLE = """
from src.models.vin_event_anomaly import run_vin_event_anomaly_pipeline

vin_df = df_all[df_all["vin"] == "VIN001"].copy()

config = {
    "time_col": "timestamp",
    "retard_col": "Act_RetardPctTorqExh",
    "eng_speed_col": "EngSpeed",
    "veh_speed_col": "VehSpeedEng",
    "accel_col": "AccelPedalPos",
    "boost_col": "BoostPres",
    "baro_col": "BarPres_Eng_1587",
    "coolant_col": "EngCoolantTemp",
    "gear_col": "TransCurGear",
    "lockup_col": "TransTorqConvLockupEngaged",
    "brake_col": "BrakeSwitch",
    "clutch_col": "ClutchSwtch",
    "abs_ret_col": "ABS_RetCont_1587",
    "event_id_col": "event_id",
}

result = run_vin_event_anomaly_pipeline(
    vin_df,
    mask_col="exh_brake_mask",
    config=config,
    target_col="peak_abs_retard_torque",
    early_fraction=0.4,
)

result["scored_events"].sort_values("anomaly_score", ascending=False).head(20)
""".strip()
