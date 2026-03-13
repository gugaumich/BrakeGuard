# src/app_vin_exhbrake_dashboard.py
"""
VIN Exhaust Brake Dashboard (Local Streamlit) - UPDATED for slim dataset + precomputed correlations

Includes:
1) VIN Visual Scanner (from data/analysis/VIN*/timeseries.parquet)
2) Health Trend (from data/analysis/vin_health_trends.csv if present)
3) Correlations:
   - Population correlation heatmap (from correlations_population_*.csv)
   - VIN-specific correlation heatmap (from correlations_by_vin.csv)
   - Scatter plots using slim event-samples dataset partitions:
       data/analysis/vin_event_samples_slim.parquet/vin=VINxxxx/part_*.parquet

Run:
  pip install streamlit plotly pandas numpy pyarrow
  streamlit run .\src\app_vin_exhbrake_dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------
# Streamlit config (must be first Streamlit call)
# -------------------------
st.set_page_config(page_title="VIN Exhaust Brake Dashboard", layout="wide")
st.title("VIN Exhaust Brake Dashboard (Local) — Slim + Correlations")


# -------------------------
# Paths
# -------------------------
DATA_GLOB = "data/analysis/VIN*/timeseries.parquet"

SLIM_DS_ROOT = Path("data/analysis/vin_event_samples_slim.parquet")  # dataset directory
HEALTH_TRENDS = Path("data/analysis/vin_health_trends.csv")
LABELS_CSV = Path("data/analysis/vin_exhaust_valve_labels.csv")

CORR_POP_PEARSON = Path("data/analysis/correlations_population_pearson.csv")
CORR_POP_SPEARMAN = Path("data/analysis/correlations_population_spearman.csv")
CORR_BY_VIN = Path("data/analysis/correlations_by_vin.csv")


# -------------------------
# Signal sets
# -------------------------
TARGET_Y = "Act_RetardPctTorqExh"  # your exhaust backpressure/retard proxy

PRIMARY_SIGNALS = [
    TARGET_Y,
    "EngSpeed",
    "VehSpeedEng",
    "TrRgAttai",
    "TransTorqConvLockupEngaged",
    "AccelPedalPos",
    "AccelPedalPos_1587",
    "FuelRate",
    "EngPctTorq",
    "EngDmdPctTorq",
    "EngRetarderStat_1587",
    "BoostPres",
    "BrakeSwitch",
]

# What we try to read from slim samples for correlation/scatter
SLIM_READ_COLS = [
    "timestamp", "UTC_1Hz",
    "vin", "event_id", "event_start_time", "event_end_time",
    "label", "speed_band",
] + PRIMARY_SIGNALS


# -------------------------
# Helpers
# -------------------------
def list_vin_parquets() -> list[Path]:
    return sorted(Path(".").glob(DATA_GLOB))


def guess_time_col(cols: list[str]) -> Optional[str]:
    preferred = ["UTC_1Hz", "timestamp", "UTC", "Time", "Datetime", "DateTime"]
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        lc = c.lower()
        if "utc" in lc or "time" in lc or "timestamp" in lc:
            return c
    return None


def to_datetime_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
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


def find_segments(mask: pd.Series, min_len: int = 2) -> list[tuple[int, int]]:
    arr = mask.to_numpy(dtype=bool)
    if arr.size == 0:
        return []
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(arr):
        if v and not in_seg:
            in_seg = True
            start = i
        elif (not v) and in_seg:
            end = i - 1
            if (end - start + 1) >= min_len:
                segments.append((start, end))
            in_seg = False
    if in_seg:
        end = len(arr) - 1
        if (end - start + 1) >= min_len:
            segments.append((start, end))
    return segments


def stacked_plot(df: pd.DataFrame, time_col: str, signals: list[str], title: str) -> go.Figure:
    n = len(signals)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=signals)
    x = df[time_col]
    for i, sig in enumerate(signals, start=1):
        if sig not in df.columns:
            continue
        fig.add_trace(go.Scatter(x=x, y=df[sig], mode="lines", name=sig), row=i, col=1)
    fig.update_layout(
        height=min(250 * n, 1800),
        title=title,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text=time_col, row=n, col=1)
    return fig


@st.cache_data
def load_labels() -> dict[str, str]:
    if not LABELS_CSV.exists():
        return {}
    lab = pd.read_csv(LABELS_CSV)
    if "vin" not in lab.columns or "valve_label" not in lab.columns:
        return {}
    return dict(zip(lab["vin"], lab["valve_label"]))


def vin_display(v: str, labels: dict[str, str]) -> str:
    lab = labels.get(v, "Unknown")
    if lab == "GOOD":
        return f"✅ {v} (GOOD)"
    if lab == "BAD":
        return f"❌ {v} (BAD)"
    return f"⚪ {v} (Unknown)"


@st.cache_data
def load_health_trends() -> Optional[pd.DataFrame]:
    if not HEALTH_TRENDS.exists():
        return None
    return pd.read_csv(HEALTH_TRENDS)


def accel_col(df: pd.DataFrame) -> Optional[str]:
    if "AccelPedalPos" in df.columns:
        return "AccelPedalPos"
    if "AccelPedalPos_1587" in df.columns:
        return "AccelPedalPos_1587"
    return None


def normalize_sample_time(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "timestamp" in d.columns:
        d["timestamp"] = to_datetime_safe(d["timestamp"])
        return d
    if "UTC_1Hz" in d.columns:
        d["timestamp"] = to_datetime_safe(d["UTC_1Hz"])
        return d
    # no time col present, leave as-is
    return d


def filter_for_correlation(
    df: pd.DataFrame,
    min_engspeed: float,
    max_accel: float,
    require_lockup: bool,
    require_retarder_on: bool,
    min_speed: float,
    max_speed: float,
    max_engpcttorq: Optional[float],
) -> pd.DataFrame:
    d = normalize_sample_time(df)

    if "EngSpeed" in d.columns:
        d = d[d["EngSpeed"] >= min_engspeed]

    ac = accel_col(d)
    if ac:
        d = d[d[ac] <= max_accel]

    if "VehSpeedEng" in d.columns:
        d = d[(d["VehSpeedEng"] >= min_speed) & (d["VehSpeedEng"] <= max_speed)]

    if max_engpcttorq is not None:
        # torque request close to 0 (helps isolate “decel / no throttle” condition)
        if "EngPctTorq" in d.columns:
            d = d[d["EngPctTorq"].abs() <= max_engpcttorq]
        elif "EngDmdPctTorq" in d.columns:
            d = d[d["EngDmdPctTorq"].abs() <= max_engpcttorq]

    if require_lockup and "TransTorqConvLockupEngaged" in d.columns:
        d = d[d["TransTorqConvLockupEngaged"] == 1]

    if require_retarder_on and "EngRetarderStat_1587" in d.columns:
        d = d[d["EngRetarderStat_1587"] == 1]

    return d


def corr_heatmap_from_matrix(cmat: pd.DataFrame, title: str) -> go.Figure:
    if cmat is None or cmat.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # ensure numeric
    c = cmat.copy()
    for col in c.columns:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(axis=0, how="all").dropna(axis=1, how="all")

    fig = go.Figure(
        data=go.Heatmap(
            z=c.values,
            x=c.columns.tolist(),
            y=c.index.tolist(),
            zmin=-1,
            zmax=1,
            colorbar=dict(title="corr"),
        )
    )
    fig.update_layout(title=title, height=540, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def load_corr_matrix(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Robust loader:
    - If CSV is a square matrix with an index column -> returns matrix
    - If CSV is long format (x,y,r or similar) -> pivots into matrix
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)

    # long format possibilities
    lower_cols = [c.lower() for c in df.columns]
    if set(["x", "y"]).issubset(lower_cols):
        # find r column
        rcol = None
        for cand in ["r", "pearson_r", "spearman_r", "corr", "value"]:
            if cand in lower_cols:
                rcol = df.columns[lower_cols.index(cand)]
                break
        if rcol is None:
            # fallback: third column
            if df.shape[1] >= 3:
                rcol = df.columns[2]
            else:
                return None
        xcol = df.columns[lower_cols.index("x")]
        ycol = df.columns[lower_cols.index("y")]
        mat = df.pivot(index=ycol, columns=xcol, values=rcol)
        # make symmetric-ish if missing
        return mat

    # matrix format: first col is index-like
    if df.shape[1] >= 2:
        first = df.columns[0]
        # common case: unnamed index column
        if first.lower() in ["", "unnamed: 0", "index"]:
            df = df.rename(columns={first: "var"}).set_index("var")
            return df
        # if diagonal names match headers, treat first column as index
        if first not in df.columns[1:]:
            df2 = df.set_index(first)
            # if it looks numeric-ish
            return df2

    return None


def load_corr_by_vin_long(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # expected columns usually: vin, method, x, y, r
    return df


def vin_partition_path(vin: str) -> Path:
    return SLIM_DS_ROOT / f"vin={vin}"


@st.cache_data(show_spinner=True)
def load_slim_vin_samples(vin: str, cols: List[str]) -> Optional[pd.DataFrame]:
    p = vin_partition_path(vin)
    if not p.exists():
        return None

    # read only available columns; pandas will error if column doesn't exist in a part,
    # so we try a two-step: read schema from one part
    parts = sorted(p.glob("*.parquet"))
    if not parts:
        return None

    schema_cols = pd.read_parquet(parts[0], engine="pyarrow").columns.tolist()
    cols2 = [c for c in cols if c in schema_cols]
    if not cols2:
        # at least read something
        cols2 = schema_cols

    d = pd.read_parquet(str(p), columns=cols2)
    d = normalize_sample_time(d)
    return d


def load_population_sample(cols: List[str], max_vins: int, max_rows: int, seed: int = 7) -> pd.DataFrame:
    """
    Sample across VIN partitions WITHOUT reading the dataset root
    (avoids Arrow schema merge issues).
    """
    if not SLIM_DS_ROOT.exists():
        return pd.DataFrame()

    vin_parts = sorted([p for p in SLIM_DS_ROOT.glob("vin=*") if p.is_dir()])
    if not vin_parts:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    pick = vin_parts if len(vin_parts) <= max_vins else list(rng.choice(vin_parts, size=max_vins, replace=False))

    out = []
    remaining = max_rows

    for vp in pick:
        # vp.name like "vin=VIN02756"
        vin = vp.name.split("=", 1)[-1]
        d = load_slim_vin_samples(vin, cols)
        if d is None or d.empty:
            continue
        if len(d) > remaining:
            d = d.sample(remaining, random_state=seed)
        out.append(d)
        remaining -= len(d)
        if remaining <= 0:
            break

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def scatter_xy(df: pd.DataFrame, x: str, y: str, title: str, max_points: int = 200_000) -> go.Figure:
    d = df[[x, y]].dropna()
    if len(d) > max_points:
        d = d.sample(max_points, random_state=7)
    fig = go.Figure(data=go.Scattergl(x=d[x], y=d[y], mode="markers"))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=520, margin=dict(l=40, r=20, t=60, b=40))
    return fig


# -------------------------
# VIN selection
# -------------------------
parquets = list_vin_parquets()
if not parquets:
    st.error(f"No VIN parquets found. Expected: {DATA_GLOB}")
    st.stop()

vin_labels = [p.parent.name for p in parquets]  # VIN02764
vin_to_path = {p.parent.name: p for p in parquets}

labels = load_labels()

selected_vin = st.selectbox(
    "Select VIN",
    options=vin_labels,
    index=0,
    format_func=lambda v: vin_display(v, labels),
)

st.caption(f"Selected: **{selected_vin}**  | file: `{vin_to_path[selected_vin]}`")


# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["VIN Visual Scanner", "Health Trend", "Correlations"])


# ============================================================
# TAB 1: VIN Visual Scanner
# ============================================================
with tab1:
    parquet_path = vin_to_path[selected_vin]
    schema_cols = pd.read_parquet(parquet_path, engine="pyarrow").columns.tolist()
    time_col = guess_time_col(schema_cols)
    if time_col is None:
        st.error("Could not find a time column (UTC/time/timestamp).")
        st.stop()

    # Show a minimal useful default set if present
    default_signals = [
        "VehSpeedEng",
        "EngSpeed",
        "FuelRate",
        "EngPctTorq",
        "BoostPres",
        "TrRgAttai",
        "AccelPedalPos",
        "AccelPedalPos_1587",
        "EngRetarderStat_1587",
        "TransTorqConvLockupEngaged",
        TARGET_Y,
    ]
    available_defaults = [s for s in default_signals if s in schema_cols]
    cols_to_load = [time_col] + [c for c in available_defaults if c != time_col]

    @st.cache_data(show_spinner=True)
    def load_vin_timeseries(path_str: str, cols: list[str], time_col_name: str) -> pd.DataFrame:
        d = pd.read_parquet(path_str, columns=cols)
        d[time_col_name] = to_datetime_safe(d[time_col_name])
        d = d.sort_values(time_col_name)
        return d

    df = load_vin_timeseries(str(parquet_path), cols_to_load, time_col)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        min_seg_len = st.number_input(
            "Min segment length (seconds/samples)",
            min_value=1,
            max_value=600,
            value=3,
            step=1,
            key="candidate_min_seg_len",
        )
    with col2:
        pad = st.number_input(
            "Padding around segment (seconds/samples)",
            min_value=0,
            max_value=600,
            value=20,
            step=5,
            key="candidate_pad",
        )
    with col3:
        require_lockup = st.checkbox("Require TransTorqConvLockupEngaged == 1", value=True)

    st.subheader("Candidate Window Filter (adjustable)")
    c1, c2, c3 = st.columns(3)
    with c1:
        eng_min = st.number_input(
            "EngSpeed >= ",
            min_value=0.0,
            max_value=5000.0,
            value=1000.0,
            step=50.0,
            key="candidate_eng_min",
        )
    with c2:
        accel_max = st.number_input(
            "AccelPedalPos <= ",
            min_value=0.0,
            max_value=100.0,
            value=7.0,
            step=1.0,
            key="candidate_accel_max",
        )
    with c3:
        st.write("")

    eng = df["EngSpeed"] if "EngSpeed" in df.columns else pd.Series(np.nan, index=df.index)
    ac = df["AccelPedalPos"] if "AccelPedalPos" in df.columns else (
        df["AccelPedalPos_1587"] if "AccelPedalPos_1587" in df.columns else pd.Series(np.nan, index=df.index)
    )

    if require_lockup and "TransTorqConvLockupEngaged" in df.columns:
        lock = df["TransTorqConvLockupEngaged"]
        mask = (eng >= eng_min) & (ac <= accel_max) & (lock == 1)
    else:
        mask = (eng >= eng_min) & (ac <= accel_max)
    mask = mask.fillna(False)

    segments = find_segments(mask, min_len=int(min_seg_len))
    st.write(f"Found **{len(segments)}** candidate segments.")

    if len(segments) == 0:
        st.info("Try lowering EngSpeed threshold or increasing Accel threshold.")
        st.stop()

    seg_idx = st.slider("Select segment index", min_value=0, max_value=len(segments) - 1, value=0, step=1)
    start_i, end_i = segments[seg_idx]
    start_i2 = max(0, start_i - int(pad))
    end_i2 = min(len(df) - 1, end_i + int(pad))
    window = df.iloc[start_i2 : end_i2 + 1].copy()

    plot_signals = st.multiselect(
        "Signals to plot",
        options=[c for c in df.columns if c != time_col],
        default=[s for s in available_defaults if s != time_col],
    )

    fig = stacked_plot(window, time_col, plot_signals, f"{selected_vin} | Segment {seg_idx}")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2: Health Trend
# ============================================================
with tab2:
    ht = load_health_trends()
    if ht is None:
        st.warning("Missing vin_health_trends.csv. Run your health trend builder first.")
    else:
        g = ht[ht["vin"] == selected_vin].copy()
        if g.empty:
            st.warning(f"No health trends for {selected_vin}.")
        else:
            g = g.sort_values("year_month")
            lab = g["label"].dropna().unique()
            lab_txt = lab[0] if len(lab) else "Unknown"

            c1, c2, c3 = st.columns(3)
            c1.metric("VIN", selected_vin)
            c2.metric("Inspection Label", lab_txt)
            c3.metric("Months of data", len(g))

            st.caption(
                "Definitions: effective_rate = fraction of monthly events with min Act_RetardPctTorqExh < -60. "
                "deep_rate = fraction of monthly events with min Act_RetardPctTorqExh < -90."
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=g["year_month"],
                    y=g["effective_rate"],
                    mode="lines+markers",
                    name="effective_rate (< -60)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=g["year_month"],
                    y=g["deep_rate"],
                    mode="lines+markers",
                    name="deep_rate (< -90)",
                )
            )
            fig.add_trace(go.Bar(x=g["year_month"], y=g["n_events"], name="n_events", opacity=0.25, yaxis="y2"))

            fig.update_layout(
                height=420,
                xaxis_title="year_month",
                yaxis_title="rate",
                yaxis=dict(range=[0, 1.05]),
                yaxis2=dict(title="n_events", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=40, r=40, t=60, b=60),
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Monthly table"):
                st.dataframe(g, use_container_width=True)


# ============================================================
# TAB 3: Correlations
# ============================================================
with tab3:
    st.subheader("Correlation visualizations")

    # ---- Load precomputed correlation matrices ----
    c_left, c_right = st.columns([2, 2])
    with c_left:
        method = st.radio("Population correlation method", ["Pearson", "Spearman"], horizontal=True)
    with c_right:
        st.caption("Uses your precomputed CSVs from build_correlations_v2.py")

    pop_path = CORR_POP_PEARSON if method == "Pearson" else CORR_POP_SPEARMAN
    pop_mat = load_corr_matrix(pop_path)

    if pop_mat is None:
        st.error(f"Missing population correlation file: {pop_path}")
    else:
        st.plotly_chart(
            corr_heatmap_from_matrix(pop_mat, f"Population {method} correlation (precomputed)"),
            use_container_width=True,
        )
        

    # ---- VIN-specific heatmap from correlations_by_vin.csv ----
    st.subheader(f"{selected_vin} correlation heatmap (precomputed)")
    by_vin = load_corr_by_vin_long(CORR_BY_VIN)

    if by_vin is None:
        st.warning(f"Missing VIN correlation file: {CORR_BY_VIN}")
    else:
        # Try to infer columns
        cols_lower = {c.lower(): c for c in by_vin.columns}
        need = ["vin", "x", "y"]
        if not all(n in cols_lower for n in need):
            st.warning("correlations_by_vin.csv format not recognized. Expected columns like vin,x,y,pearson_r/spearman_r.")
        else:
            vcol = cols_lower["vin"]
            xcol = cols_lower["x"]
            ycol = cols_lower["y"]

            # pick r column based on method
            rcol = None
            if method.lower() == "pearson":
                for cand in ["pearson_r", "r", "corr"]:
                    if cand in cols_lower:
                        rcol = cols_lower[cand]
                        break
            else:
                for cand in ["spearman_r", "r", "corr"]:
                    if cand in cols_lower:
                        rcol = cols_lower[cand]
                        break

            if rcol is None:
                # fallback: find any numeric column besides x/y/vin
                num_cols = [c for c in by_vin.columns if c not in [vcol, xcol, ycol] and pd.api.types.is_numeric_dtype(by_vin[c])]
                rcol = num_cols[0] if num_cols else None

            if rcol is None:
                st.warning("No correlation value column found in correlations_by_vin.csv.")
            else:
                sub = by_vin[by_vin[vcol] == selected_vin].copy()
                if sub.empty:
                    st.warning(f"No rows for {selected_vin} in correlations_by_vin.csv")
                else:
                    mat = sub.pivot(index=ycol, columns=xcol, values=rcol)
                    st.plotly_chart(
                        corr_heatmap_from_matrix(mat, f"{selected_vin} {method} correlation (precomputed)"),
                        use_container_width=True,
                    )

    st.divider()

    # ---- Scatter plots using the SLIM dataset (dynamic, filterable) ----
    st.subheader("Scatter plots (from slim event samples)")
    if not SLIM_DS_ROOT.exists():
        st.error(f"Missing slim dataset directory: {SLIM_DS_ROOT}")
        st.stop()

    st.caption("These scatters read from vin_event_samples_slim.parquet partitions (not the dataset root).")

    f1, f2, f3, f4, f5, f6, f7 = st.columns(7)
    with f1:
        min_engspeed = st.number_input("EngSpeed >= ", 0.0, 5000.0, 1000.0, 50.0, key="scatter_min_engspeed")
    with f2:
        max_accel = st.number_input("Accel <= ", 0.0, 100.0, 7.0, 1.0, key="scatter_max_accel")
    with f3:
        min_speed = st.number_input("Speed >= ", 0.0, 120.0, 5.0, 1.0, key="scatter_min_speed")
    with f4:
        max_speed = st.number_input("Speed <= ", 0.0, 120.0, 80.0, 1.0, key="scatter_max_speed")
    with f5:
        require_lockup = st.checkbox("Require lockup", value=True)
    with f6:
        require_ret_on = st.checkbox("Require retarder ON", value=False)
    with f7:
        use_torque_gate = st.checkbox("Gate torque near 0", value=False)

    max_engpcttorq = 5.0 if use_torque_gate else None

    scope = st.radio("Scatter scope", ["Population sample", "Selected VIN"], horizontal=True)

    # choose X
    scatter_candidates = [
        c for c in PRIMARY_SIGNALS
        if c != TARGET_Y
    ]
    x = st.selectbox("X axis", options=scatter_candidates, index=0)

    # load data
    if scope == "Selected VIN":
        df_sc = load_slim_vin_samples(selected_vin, SLIM_READ_COLS)
        if df_sc is None or df_sc.empty:
            st.warning(f"No slim samples found for {selected_vin} under {vin_partition_path(selected_vin)}")
            st.stop()
    else:
        # Sample across partitions (fast)
        cA, cB = st.columns([1, 1])
        with cA:
            max_vins = st.number_input(
                "Population VINs to sample",
                min_value=5,
                max_value=200,
                value=50,
                step=5,
                key="scatter_population_max_vins",
            )
        with cB:
            max_rows = st.number_input(
                "Population rows to sample",
                min_value=10_000,
                max_value=1_000_000,
                value=250_000,
                step=10_000,
                key="scatter_population_max_rows",
            )

        with st.spinner("Sampling population partitions..."):
            df_sc = load_population_sample(SLIM_READ_COLS, max_vins=int(max_vins), max_rows=int(max_rows))

        if df_sc.empty:
            st.warning("Population sample is empty (no readable partitions).")
            st.stop()

    # apply filters
    df_f = filter_for_correlation(
        df_sc,
        min_engspeed=min_engspeed,
        max_accel=max_accel,
        require_lockup=require_lockup,
        require_retarder_on=require_ret_on,
        min_speed=min_speed,
        max_speed=max_speed,
        max_engpcttorq=max_engpcttorq,
    )

    st.caption(f"Scatter rows after filters: **{len(df_f):,}**")

    if TARGET_Y not in df_f.columns or x not in df_f.columns:
        st.warning(f"Missing columns in data for scatter: need {TARGET_Y} and {x}")
    elif len(df_f) < 200:
        st.warning("Not enough points after filters. Relax the filters.")
    else:
        st.plotly_chart(
            scatter_xy(df_f, x=x, y=TARGET_Y, title=f"{scope}: {TARGET_Y} vs {x}"),
            use_container_width=True,
        )

    # Optional: correlation table (live)
    st.subheader("Live correlation (on filtered scatter dataset)")
    corr_candidates = [TARGET_Y, x, "EngSpeed", "VehSpeedEng", "FuelRate", "EngPctTorq", "TrRgAttai", "BoostPres"]
    numeric_cols = list(dict.fromkeys(c for c in corr_candidates if c in df_f.columns))
    if len(numeric_cols) >= 2:
        corr_live = df_f[numeric_cols].corr(numeric_only=True, method="pearson")
        st.dataframe(corr_live.round(3), use_container_width=True)
    else:
        st.info("Not enough numeric columns available for live correlation.")
