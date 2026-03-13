# src/app_vin_exhbrake_dashboard.py
"""
VIN Exhaust Brake Dashboard (Local Streamlit)

Includes:
1) VIN Visual Scanner:
   - Find candidate segments (EngSpeed high + accel released + optional lockup)
   - Plot stacked signals for a selected segment
2) Health Trend:
   - Monthly effective_rate / deep_rate / n_events from vin_health_trends.csv (if present)
3) Correlations:
   - Population correlation heatmap (from filtered event samples)
   - VIN-specific correlation heatmap
   - Scatter plots
   - Rolling monthly correlation trend (if vin_rolling_corr.csv exists)

Run:
  pip install streamlit plotly pandas numpy pyarrow
  streamlit run .\src\app_vin_exhbrake_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------
# Streamlit config (must be first Streamlit call)
# -------------------------
st.set_page_config(page_title="VIN Exhaust Brake Dashboard", layout="wide")
st.title("VIN Exhaust Brake Dashboard (Local)")


# -------------------------
# Paths
# -------------------------
DATA_GLOB = "data/analysis/VIN*/timeseries.parquet"
EVENT_SAMPLES = Path("data/analysis/vin_event_samples.parquet")
HEALTH_TRENDS = Path("data/analysis/vin_health_trends.csv")
LABELS_CSV = Path("data/analysis/vin_exhaust_valve_labels.csv")
ROLLING_CORR = Path("data/analysis/vin_rolling_corr.csv")


# -------------------------
# Signals
# -------------------------
TARGET_Y = "Act_RetardPctTorqExh"

DEFAULT_SIGNALS = [
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


# -------------------------
# Helpers
# -------------------------
def list_vin_parquets() -> list[Path]:
    return sorted(Path(".").glob(DATA_GLOB))


def guess_time_col(cols: list[str]) -> str | None:
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
    fig.update_layout(height=min(250 * n, 1800), title=title, showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text=time_col, row=n, col=1)
    return fig


@st.cache_data
def load_labels() -> dict[str, str]:
    if not LABELS_CSV.exists():
        return {}
    lab = pd.read_csv(LABELS_CSV)
    return dict(zip(lab["vin"], lab["valve_label"]))


def vin_display(v: str, labels: dict[str, str]) -> str:
    lab = labels.get(v, "Unknown")
    if lab == "GOOD":
        return f"✅ {v} (GOOD)"
    if lab == "BAD":
        return f"❌ {v} (BAD)"
    return f"⚪ {v} (Unknown)"


@st.cache_data
def load_health_trends() -> pd.DataFrame | None:
    if not HEALTH_TRENDS.exists():
        return None
    df = pd.read_csv(HEALTH_TRENDS)
    return df


@st.cache_data
def load_event_samples() -> pd.DataFrame | None:
    if not EVENT_SAMPLES.exists():
        return None
    return pd.read_parquet(EVENT_SAMPLES)


def accel_col(df: pd.DataFrame) -> str | None:
    if "AccelPedalPos" in df.columns:
        return "AccelPedalPos"
    if "AccelPedalPos_1587" in df.columns:
        return "AccelPedalPos_1587"
    return None


def filter_for_correlation(df: pd.DataFrame,
                           min_engspeed: float,
                           max_accel: float,
                           require_lockup: bool,
                           require_retarder_on: bool,
                           min_speed: float,
                           max_speed: float) -> pd.DataFrame:
    d = df.copy()

    # Timestamp normalization
    if "timestamp" in d.columns:
        d["timestamp"] = to_datetime_safe(d["timestamp"])
    else:
        # If your samples are called UTC_1Hz instead:
        if "UTC_1Hz" in d.columns:
            d["timestamp"] = to_datetime_safe(d["UTC_1Hz"])
        else:
            st.error("Event samples missing 'timestamp' (or UTC_1Hz).")
            return d.iloc[0:0]

    # Filters
    if "EngSpeed" in d.columns:
        d = d[d["EngSpeed"] >= min_engspeed]

    ac = accel_col(d)
    if ac:
        d = d[d[ac] <= max_accel]

    if "VehSpeedEng" in d.columns:
        d = d[(d["VehSpeedEng"] >= min_speed) & (d["VehSpeedEng"] <= max_speed)]

    if require_lockup and "TransTorqConvLockupEngaged" in d.columns:
        d = d[d["TransTorqConvLockupEngaged"] == 1]

    if require_retarder_on and "EngRetarderStat_1587" in d.columns:
        d = d[d["EngRetarderStat_1587"] == 1]

    return d


def corr_heatmap(df: pd.DataFrame, cols: list[str], title: str) -> go.Figure:
    cols2 = [c for c in cols if c in df.columns]
    if len(cols2) < 2:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    c = df[cols2].corr(method="pearson")
    fig = go.Figure(data=go.Heatmap(
        z=c.values,
        x=c.columns,
        y=c.index,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Pearson r")
    ))
    fig.update_layout(title=title, height=520, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def scatter_xy(df: pd.DataFrame, x: str, y: str, title: str, max_points: int = 200_000) -> go.Figure:
    d = df[[x, y]].dropna()
    if len(d) > max_points:
        d = d.sample(max_points, random_state=7)
    fig = go.Figure(data=go.Scattergl(x=d[x], y=d[y], mode="markers"))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=520, margin=dict(l=40, r=20, t=60, b=40))
    return fig


@st.cache_data
def load_rolling_corr() -> pd.DataFrame | None:
    if not ROLLING_CORR.exists():
        return None
    df = pd.read_csv(ROLLING_CORR)
    return df


def rolling_corr_plot(df_roll: pd.DataFrame, vin: str, x: str) -> go.Figure:
    sub = df_roll[(df_roll["vin"] == vin) & (df_roll["x"] == x)].sort_values("year_month")
    fig = go.Figure()
    if sub.empty:
        fig.update_layout(title=f"No rolling corr found for {vin} vs {x}")
        return fig
    fig.add_trace(go.Scatter(x=sub["year_month"], y=sub["pearson_r"], mode="lines+markers"))
    fig.update_layout(
        title=f"{vin} monthly correlation: {TARGET_Y} vs {x}",
        xaxis_title="year_month",
        yaxis_title="pearson_r",
        yaxis=dict(range=[-1.0, 1.0]),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
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
    format_func=lambda v: vin_display(v, labels)
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

    available_defaults = [s for s in DEFAULT_SIGNALS if s in schema_cols]
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
        min_seg_len = st.number_input("Min segment length (seconds/samples)", min_value=1, max_value=600, value=3, step=1)
    with col2:
        pad = st.number_input("Padding around segment (seconds/samples)", min_value=0, max_value=600, value=20, step=5)
    with col3:
        require_lockup = st.checkbox("Require TransTorqConvLockupEngaged == 1", value=True)

    st.subheader("Candidate Window Filter (adjustable)")
    c1, c2, c3 = st.columns(3)
    with c1:
        eng_min = st.number_input("EngSpeed >= ", min_value=0.0, max_value=5000.0, value=1000.0, step=50.0)
    with c2:
        accel_max = st.number_input("AccelPedalPos <= ", min_value=0.0, max_value=100.0, value=7.0, step=1.0)
    with c3:
        st.write("")

    eng = df["EngSpeed"] if "EngSpeed" in df.columns else pd.Series(np.nan, index=df.index)
    ac = df["AccelPedalPos"] if "AccelPedalPos" in df.columns else (df["AccelPedalPos_1587"] if "AccelPedalPos_1587" in df.columns else pd.Series(np.nan, index=df.index))

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

    st.subheader("Export window")
    out_dir = Path("data/analysis") / selected_vin / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_name = st.text_input("Export filename (CSV)", value=f"{selected_vin}_segment{seg_idx}_window.csv")
    if st.button("Export Window CSV"):
        out_path = out_dir / export_name
        window.to_csv(out_path, index=False)
        st.success(f"Wrote: {out_path}")


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

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g["year_month"], y=g["effective_rate"], mode="lines+markers", name="effective_rate (< -60)"))
            fig.add_trace(go.Scatter(x=g["year_month"], y=g["deep_rate"], mode="lines+markers", name="deep_rate (< -90)"))
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
    samples = load_event_samples()
    if samples is None:
        st.error("Missing vin_event_samples.parquet. Build it first.")
        st.stop()

    st.subheader("Correlation filters (applied before computing correlation)")
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    with f1:
        min_engspeed = st.number_input("EngSpeed >= ", 0.0, 5000.0, 1000.0, 50.0)
    with f2:
        max_accel = st.number_input("Accel <= ", 0.0, 100.0, 7.0, 1.0)
    with f3:
        min_speed = st.number_input("Speed >= ", 0.0, 120.0, 5.0, 1.0)
    with f4:
        max_speed = st.number_input("Speed <= ", 0.0, 120.0, 80.0, 1.0)
    with f5:
        require_lockup = st.checkbox("Require lockup", value=True)
    with f6:
        require_retarder_on = st.checkbox("Require retarder ON", value=False)

    filtered = filter_for_correlation(
        samples,
        min_engspeed=min_engspeed,
        max_accel=max_accel,
        require_lockup=require_lockup,
        require_retarder_on=require_retarder_on,
        min_speed=min_speed,
        max_speed=max_speed,
    )

    st.caption(f"Filtered sample rows: **{len(filtered):,}**")

    numeric_cols = [c for c in DEFAULT_SIGNALS if c in filtered.columns]
    numeric_cols = [c for c in numeric_cols if c != "AccelPedalPos_1587"]  # avoid duplicates unless you want both
    if "AccelPedalPos" not in filtered.columns and "AccelPedalPos_1587" in filtered.columns:
        numeric_cols.append("AccelPedalPos_1587")

    # Population heatmap
    st.subheader("Population correlation (filtered)")
    fig_pop = corr_heatmap(filtered.select_dtypes(include=[np.number]), numeric_cols, "Population Pearson correlation")
    st.plotly_chart(fig_pop, use_container_width=True)

    # VIN heatmap
    st.subheader(f"{selected_vin} correlation (filtered)")
    fvin = filtered[filtered["vin"] == selected_vin]
    if len(fvin) < 200:
        st.warning("Not enough filtered rows for this VIN under the current filter.")
    else:
        fig_vin = corr_heatmap(fvin.select_dtypes(include=[np.number]), numeric_cols, f"{selected_vin} Pearson correlation")
        st.plotly_chart(fig_vin, use_container_width=True)

    # Scatter plot controls
    st.subheader("Scatter plot")
    x_choices = [c for c in numeric_cols if c != TARGET_Y and c in filtered.columns]
    x = st.selectbox("X axis", options=x_choices, index=0 if x_choices else None)
    scope = st.radio("Scope", options=["Population", "Selected VIN"], horizontal=True)

    if x:
        plot_df = filtered if scope == "Population" else fvin
        if len(plot_df) < 200:
            st.warning("Not enough points for scatter under current filters.")
        else:
            fig_sc = scatter_xy(plot_df, x=x, y=TARGET_Y, title=f"{scope}: {TARGET_Y} vs {x}")
            st.plotly_chart(fig_sc, use_container_width=True)

    # Rolling correlation plot (if built)
    st.subheader("Rolling monthly correlation (if available)")
    rc = load_rolling_corr()
    if rc is None:
        st.info("No vin_rolling_corr.csv found. Run: python src/build_correlation_plots.py")
    else:
        x_roll = st.selectbox("Rolling correlation X", options=["EngSpeed", "VehSpeedEng", "FuelRate", "EngPctTorq"])
        fig_roll = rolling_corr_plot(rc, selected_vin, x_roll)
        st.plotly_chart(fig_roll, use_container_width=True)
