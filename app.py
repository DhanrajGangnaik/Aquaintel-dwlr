# AquaIntel DWLR Dashboard — with ntfy push notifications (no MQTT)
# Drop-in replacement file

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests  # for ntfy push
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AquaIntel • DWLR Dashboard", layout="wide")

# ===============================
# Theme colors (blue scheme)
# ===============================
PRIMARY_DARK = "#0d47a1"  # deep blue
PRIMARY = "#1565c0"
PRIMARY_LIGHT = "#1976d2"
ACCENT = "#42a5f5"
GRAD_1 = "#0d47a1"
GRAD_2 = "#1976d2"


# ===============================
# Theming / UI bits
# ===============================
def local_css():
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 5.0rem; }}
        .stMetric {{ font-size: 1.3rem; font-weight: bold; }}
        .stTabs [data-baseweb="tab"] {{ font-size: 1.1rem; }}

        .stButton>button {{
            background-color: {PRIMARY};
            color: white;
            border-radius: 8px;
            border: 0;
        }}
        .stButton>button:hover {{ background-color: {PRIMARY_LIGHT}; }}

        .ai-topbar {{
            position: fixed; top: 0; left: 0; right: 0; height: 56px;
            display: flex; align-items: center; gap: 12px; padding: 0 16px;
            background: {PRIMARY_DARK}; box-shadow: 0 2px 8px rgba(0,0,0,.2); z-index: 9999;
        }}
        .ai-logo {{ display: flex; align-items: center; gap: 10px; color: #fff; font-weight: 700; font-size: 1.15rem; letter-spacing: .3px; }}

        .ai-hero {{
            background: linear-gradient(90deg, {GRAD_1} 0%, {GRAD_2} 100%);
            padding: 2rem 1rem; border-radius: 12px; margin-bottom: 2rem;
        }}
        .ai-hero h1, .ai-hero p {{ color: white; }}

        /* Sticky AquaIntel branding in sidebar (acts like Home link) */
        [data-testid="stSidebar"] .ai-sb-brand {{
          position: sticky; top: 0; z-index: 10; background: {PRIMARY_DARK};
          border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.15);
        }}
        [data-testid="stSidebar"] .ai-sb-brand a {{
          color: #ffffff; text-decoration: none; font-weight: 800; font-size: 1.05rem; letter-spacing: .2px; display: inline-block;
        }}
        [data-testid="stSidebar"] .ai-sb-brand a:hover {{ opacity: .95; }}
        </style>
    """,
        unsafe_allow_html=True,
    )


def top_navbar():
    st.markdown(
        f"""
        <div class="ai-topbar">
            <div class="ai-logo">💧 AquaIntel <span style="opacity:.85;font-size:.95rem;margin-left:6px;">DWLR</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_banner():
    st.markdown(
        """
        <div class="ai-hero">
            <h1 style="margin-bottom: 0.5rem;">💧 AquaIntel — DWLR Water Logger Dashboard</h1>
            <p style="font-size: 1.1rem;">
                Explore water-level and quality metrics interactively.<br>
                <span style="font-size: 0.95rem; opacity: 0.95;">Powered by Streamlit</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        f"""
        <hr style="margin-top:2rem;">
        <div style="text-align:center; color:gray; font-size:0.9rem;">
            Made with ❤️ using Streamlit · <span style="color:{PRIMARY}">AquaIntel</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===============================
# Data loading & mapping
# ===============================
@st.cache_data(show_spinner=False)
def load_data(file, dayfirst=False):
    with st.spinner("Loading data..."):
        try:
            df = pd.read_csv(
                file, low_memory=False, parse_dates=["Date"], dayfirst=dayfirst
            )
        except Exception:
            df = pd.read_csv(file, low_memory=False)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(
                    df["Date"], errors="coerce", dayfirst=dayfirst
                )
        if "Date" in df.columns:
            df = df.sort_values("Date")
    return df


def column_mapper(df):
    st.sidebar.subheader("🗂️ Column Mapper")
    expected = [
        "Date",
        "Water_Level_m",
        "Temperature_C",
        "Rainfall_mm",
        "Dissolved_Oxygen_mg_L",
    ]
    mapping = {}
    for col in expected:
        options = [None] + list(df.columns)
        default_idx = options.index(col) if col in df.columns else 0
        selected = st.sidebar.selectbox(
            f"Map '{col}' to:", options, index=default_idx, key=f"map_{col}"
        )
        if selected:
            mapping[col] = selected
    df = df.rename(columns={v: k for k, v in mapping.items() if v})
    return df


# ===============================
# Sidebar filters & thresholds
# ===============================


def _ensure_datetime_datecol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df['Date'] is pandas datetime (naive). Tries both MDY and DMY,
    picks the parse with fewer NaT. Returns a copy.
    """
    if "Date" not in df.columns:
        return df.copy()

    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        # strip timezone if present
        out = df.copy()
        try:
            out["Date"] = out["Date"].dt.tz_localize(None)
        except Exception:
            pass
        return out

    out = df.copy()
    s = out["Date"].astype(str)
    a = pd.to_datetime(s, errors="coerce")  # default (MDY first)
    b = pd.to_datetime(s, errors="coerce", dayfirst=True)  # DMY
    out["Date"] = a if a.notna().sum() >= b.notna().sum() else b

    # strip timezone if any slipped in
    try:
        out["Date"] = out["Date"].dt.tz_localize(None)
    except Exception:
        pass
    return out


def add_sidebar_filters(df):
    # Force datetime and drop NaT
    df = _ensure_datetime_datecol(df)
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        st.error(
            "No valid dates after parsing the 'Date' column. Please check your dataset or Column Mapper."
        )
        st.stop()

    min_date = pd.to_datetime(df["Date"].min())
    max_date = pd.to_datetime(df["Date"].max())

    if "date_range" not in st.session_state:
        st.session_state.date_range = (min_date, max_date)
    if "resample_freq" not in st.session_state:
        st.session_state.resample_freq = "Daily (no resample)"

    # Streamlit returns date objects; convert to Timestamp
    start_date, end_date = st.sidebar.date_input(
        "📅 Date range",
        value=st.session_state.date_range,
        min_value=min_date.to_pydatetime().date(),
        max_value=max_date.to_pydatetime().date(),
        key="date_input_key",
    )

    # Normalize to inclusive end-of-day timestamp
    start_ts = pd.Timestamp(start_date)
    end_ts = (
        pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    )

    st.session_state.date_range = (start_ts, end_ts)

    mask = (df["Date"] >= start_ts) & (df["Date"] <= end_ts)
    df = df.loc[mask].copy()

    freq = st.sidebar.selectbox(
        "⏱️ Resample frequency",
        ["Daily (no resample)", "Weekly", "Monthly"],
        key="resample_freq",
    )

    num_cols = [
        c for c in df.columns if c != "Date" and pd.api.types.is_numeric_dtype(df[c])
    ]
    other_cols = [c for c in df.columns if c not in ["Date"] + num_cols]
    agg_map = {**{c: "mean" for c in num_cols}, **{c: "first" for c in other_cols}}

    if freq == "Weekly":
        df = df.set_index("Date").resample("W").agg(agg_map).reset_index()
    elif freq == "Monthly":
        df = df.set_index("Date").resample("MS").agg(agg_map).reset_index()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Select variables to show")
    available_cols = [c for c in df.columns if c != "Date"]
    numeric_cols = [c for c in available_cols if pd.api.types.is_numeric_dtype(df[c])]
    default_select = numeric_cols[:3]
    y_cols = st.sidebar.multiselect(
        "Numeric columns", numeric_cols, default=default_select, key="y_cols"
    )

    def _reset():
        st.session_state.date_range = (min_date, max_date)
        st.session_state.resample_freq = "Daily (no resample)"
        st.session_state.y_cols = default_select
        st.experimental_rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset filters"):
        _reset()

    st.sidebar.info("ℹ️ Use these controls to filter and explore your data.")
    return df, y_cols


def sidebar_thresholds():
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Thresholds")
    safe_thr = st.sidebar.number_input(
        "Safe water level threshold (m)", value=1.5, step=0.1
    )
    critical_thr = st.sidebar.number_input(
        "Critical water level threshold (m)", value=1.0, step=0.1
    )
    drop_thr = st.sidebar.number_input("Sudden drop detection (m)", value=2.0, step=0.1)
    return float(safe_thr), float(critical_thr), float(drop_thr)


# ===============================
# ntfy Push Helpers (simple HTTP push)
# ===============================
def ntfy_push(
    topic: str,
    title: str,
    message: str,
    priority: str = "default",
    server: str = "https://ntfy.sh",
    auth_token: str | None = None,
    timeout_sec: float = 10.0,
):
    """
    Send a push via ntfy.
    Returns a dict: {"ok": bool, "status": int|None, "url": str, "text": str|None, "error": str|None}
    NOTE: HTTP headers must be latin-1; we sanitize the Title (and other headers) accordingly.
    """

    def _latin1_safe(s: str) -> str:
        replacements = {
            "•": "-",
            "–": "-",
            "—": "-",
            "“": '"',
            "”": '"',
            "’": "'",
            "…": "...",
            "✓": "OK",
            "✅": "OK",
            "Δ": "Delta",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        return s.encode("latin-1", "ignore").decode("latin-1")

    url = f"{server.rstrip('/')}/{topic.strip()}"
    headers = {
        "Title": _latin1_safe(title),
        "Priority": _latin1_safe(priority),
        "Content-Type": "text/plain; charset=utf-8",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        r = requests.post(
            url, data=message.encode("utf-8"), headers=headers, timeout=timeout_sec
        )
        ok = r.status_code in (200, 201)
        return {
            "ok": ok,
            "status": r.status_code,
            "url": url,
            "text": r.text,
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "status": None, "url": url, "text": None, "error": str(e)}


def manual_alarm_button():
    st.sidebar.markdown("## 🧪 Manual Test")
    if st.sidebar.button("Manually trigger alarm", use_container_width=True):
        cfg = st.session_state.get("ntfy_cfg", {})
        if not cfg.get("enabled"):
            st.sidebar.error("Enable ntfy push first.")
            return

        st.session_state["manual_push_active"] = True

        res = ntfy_push(
            topic=cfg["topic"],
            title="AquaIntel - Test Alarm",
            message="Manual trigger from dashboard. If you see this, push works ✅",
            priority="high",
            server=cfg["server"],
            auth_token=cfg["token"] or None,
        )

        if res["ok"]:
            st.sidebar.success(f"Sent to {res['url']}")
        else:
            if res["status"] is not None:
                st.sidebar.error(f"Failed to send (HTTP {res['status']}).")
                if res["text"]:
                    st.sidebar.caption(res["text"][:500])
            else:
                st.sidebar.error("Failed to send (no HTTP response).")
                st.sidebar.caption(res["error"] or "Unknown error")
            st.sidebar.info(
                "Tips: 1) Confirm the exact topic in the ntfy app. "
                "2) If HTTPS is blocked, try http://ntfy.sh. "
                "3) Ensure topic has no spaces or special characters."
            )

        st.session_state["manual_push_active"] = False


# ===============================
# Small helpers / KPIs / Panels
# ===============================
def dataset_info(df):
    st.write(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")


def kpi_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    wl = df["Water_Level_m"].mean() if "Water_Level_m" in df else np.nan
    temp = df["Temperature_C"].mean() if "Temperature_C" in df else np.nan
    rain = df["Rainfall_mm"].sum() if "Rainfall_mm" in df else np.nan
    do = df["Dissolved_Oxygen_mg_L"].mean() if "Dissolved_Oxygen_mg_L" in df else np.nan
    col1.metric("🌊 Avg Water Level (m)", f"{wl:.3f}" if pd.notna(wl) else "—")
    col2.metric("🌡️ Avg Temperature (°C)", f"{temp:.2f}" if pd.notna(temp) else "—")
    col3.metric("🌧️ Total Rainfall (mm)", f"{rain:.1f}" if pd.notna(rain) else "—")
    col4.metric("🫧 Avg Dissolved O₂ (mg/L)", f"{do:.2f}" if pd.notna(do) else "—")


def transparency_panel(df):
    if "Water_Level_m" in df and not df["Water_Level_m"].dropna().empty:
        low = float(df["Water_Level_m"].min())
        missing = int(df["Water_Level_m"].isna().sum())
    else:
        low = 0.0
        missing = 0
    total_rain = float(df["Rainfall_mm"].sum()) if "Rainfall_mm" in df else 0.0
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, {GRAD_1} 0%, {GRAD_2} 100%);
            padding: 1.5rem 1rem; border-radius: 12px; margin: 1.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        ">
            <h2 style="color: white; margin-bottom: 0.5rem;">🔎 Transparency Panel</h2>
            <ul style="color: white; font-size: 1.05rem; margin-left: 1rem;">
                <li><b>Lowest water level:</b> {low:.2f} m</li>
                <li><b>Total rainfall:</b> {total_rain:.1f} mm</li>
                <li><b>Anomalies detected:</b> {missing} missing values</li>
            </ul>
            <p style="color: #f8f8f8; font-size: 0.95rem; margin-top: 0.5rem;">
                📢 <b>Stay informed!</b> This panel summarizes key alerts and water usage for your farm.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def low_water_alert(df, safe_threshold: float):
    if "Water_Level_m" in df and not df["Water_Level_m"].dropna().empty:
        if df["Water_Level_m"].min() < safe_threshold:
            st.warning(
                f"⚠️ Water level is below safe threshold ({safe_threshold:.2f} m)! Consider reducing extraction.",
                icon="💧",
            )


def government_notifications(df, critical_threshold: float):
    st.subheader("Government Notifications")
    if "Water_Level_m" in df and not df["Water_Level_m"].dropna().empty:
        if df["Water_Level_m"].min() < critical_threshold:
            st.error("🚨 Water level critically low! Immediate action required.")


# ===============================
# Analysis components
# ===============================
def time_series_section(df, y_cols):
    st.subheader("Time Series")
    if not y_cols:
        st.info("Select at least one numeric column from the sidebar.")
        return
    for col in y_cols:
        fig = px.line(df, x="Date", y=col, title=col)
        st.plotly_chart(fig, use_container_width=True)


def correlation_heatmap(df):
    st.subheader("Correlation (Pearson)")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        st.info("No numeric columns to compute correlations.")
        return
    corr = num_df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Correlation Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)


def scatter_matrix(df):
    st.subheader("Scatter Matrix")
    cols = st.multiselect(
        "Select variables for scatter matrix:",
        [
            c
            for c in df.columns
            if c not in ["Date"] and pd.api.types.is_numeric_dtype(df[c])
        ],
        default=(
            ["Rainfall_mm", "Water_Level_m"]
            if "Rainfall_mm" in df and "Water_Level_m" in df
            else []
        ),
    )
    if len(cols) >= 2:
        fig = px.scatter_matrix(df, dimensions=cols)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Select at least two variables to plot scatter matrix.")


def anomaly_detection(df):
    st.subheader("Simple Anomaly Detection (Z-score of rolling residuals)")
    numeric_candidates = [
        c
        for c in df.columns
        if c not in ["Date"] and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_candidates:
        st.info("No numeric columns found for anomaly detection.")
        return
    target = st.selectbox("Select series for anomaly check", numeric_candidates)
    if target:
        work = df[["Date", target]].dropna().copy()
        work[target + "_roll"] = work[target].rolling(window=14, min_periods=7).mean()
        work["resid"] = work[target] - work[target + "_roll"]
        std = work["resid"].rolling(window=14, min_periods=7).std().replace(0, np.nan)
        work["z"] = (work["resid"] / std).fillna(0)
        thresh = st.slider("Z-score threshold", 1.5, 4.0, 3.0, 0.1)
        anomalies = work[np.abs(work["z"]) >= thresh]

        fig = px.line(work, x="Date", y=target, title=f"Anomalies in {target}")
        fig.add_scatter(
            x=work["Date"],
            y=work[target + "_roll"],
            mode="lines",
            name="Rolling mean (14d)",
        )
        fig.add_scatter(
            x=anomalies["Date"],
            y=anomalies[target],
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Anomaly table"):
            st.dataframe(
                anomalies[["Date", target, "z"]].reset_index(drop=True),
                use_container_width=True,
            )

        cfg = st.session_state.get("ntfy_cfg", {})
        if cfg.get("enabled") and not anomalies.empty:
            msg = f"{len(anomalies)} anomalies in '{target}' (z ≥ {thresh}). Range: {work['Date'].min()} → {work['Date'].max()}"
            ntfy_push(
                cfg["topic"],
                "AquaIntel • Anomaly Alert",
                msg,
                priority="high",
                server=cfg["server"],
                auth_token=cfg["token"] or None,
            )


def data_download(df):
    st.subheader("Download filtered data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", csv, file_name="DWLR_filtered.csv", mime="text/csv"
    )


def missingness(df):
    st.subheader("Missing Data Overview")
    miss = df.isna().sum().reset_index()
    miss.columns = ["Column", "Missing_Count"]
    st.dataframe(miss, use_container_width=True)


def data_preview(df):
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)


def data_quality_metrics(df):
    st.subheader("Data Quality Metrics")
    total = len(df)
    if total == 0:
        st.info("No rows to compute data quality metrics.")
        return
    miss = df.isna().sum()
    st.write(f"**Missing values:**")
    for col, val in miss.items():
        st.write(f"- {col}: {val} ({(val/total):.1%})")


def historical_comparison(df):
    st.subheader("Historical Water Level Comparison")
    if "Date" in df and "Water_Level_m" in df:
        df_year = df.dropna(subset=["Date", "Water_Level_m"]).copy()
        if df_year.empty:
            st.info("No data for historical comparison.")
            return
        if df_year["Date"].dt.year.nunique() > 1:
            df_year["Year"] = df_year["Date"].dt.year
            yearly = df_year.groupby("Year")["Water_Level_m"].mean().reset_index()
            fig = px.bar(
                yearly,
                x="Year",
                y="Water_Level_m",
                title="Average Water Level by Year",
                labels={"Water_Level_m": "Avg Water Level (m)"},
            )
        else:
            df_year["MonthNum"] = df_year["Date"].dt.month
            monthly = df_year.groupby("MonthNum")["Water_Level_m"].mean().reset_index()
            monthly["Month"] = monthly["MonthNum"].map(
                lambda m: pd.Timestamp(2000, m, 1).strftime("%b")
            )
            fig = px.bar(
                monthly,
                x="Month",
                y="Water_Level_m",
                title="Average Water Level by Month",
                labels={"Water_Level_m": "Avg Water Level (m)"},
            )
        st.plotly_chart(fig, use_container_width=True)


def illegal_extraction_detection(df, sudden_drop_threshold: float):
    st.subheader("Illegal Extraction Detection")
    if "Water_Level_m" in df:
        w = df[["Date", "Water_Level_m"]].copy()
        w["diff"] = w["Water_Level_m"].diff()
        suspicious = w[w["diff"] < -abs(sudden_drop_threshold)]
        if not suspicious.empty:
            st.error(
                f"🚨 {len(suspicious)} possible illegal extraction events detected!"
            )
            st.dataframe(
                suspicious[["Date", "Water_Level_m", "diff"]], use_container_width=True
            )
        else:
            st.success("No suspicious extraction events detected.")

        cfg = st.session_state.get("ntfy_cfg", {})
        if cfg.get("enabled") and not suspicious.empty:
            first = suspicious.iloc[0]
            diff_val = float(first["diff"]) if pd.notna(first["diff"]) else 0.0
            msg = f"{len(suspicious)} sudden drops detected (>{abs(sudden_drop_threshold)} m). First at {first['Date']} (Δ={diff_val:.2f} m)"
            ntfy_push(
                cfg["topic"],
                "AquaIntel • Illegal Extraction",
                msg,
                priority="max",
                server=cfg["server"],
                auth_token=cfg["token"] or None,
            )


def future_prediction(df):
    st.subheader("Future Water Level Prediction (Simple Linear Regression)")
    if "Date" in df and "Water_Level_m" in df:
        df_pred = df.dropna(subset=["Date", "Water_Level_m"]).copy()
        if len(df_pred) > 10:
            df_pred["ordinal_date"] = df_pred["Date"].map(datetime.toordinal)
            X = df_pred[["ordinal_date"]]
            y = df_pred["Water_Level_m"]
            try:
                model = LinearRegression().fit(X, y)
            except Exception:
                st.info("Prediction model could not be fit (insufficient variance).")
                return
            future_days = 30
            last_date = df_pred["Date"].max()
            future_dates = [
                last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)
            ]
            future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            preds = model.predict(future_ord)
            pred_df = pd.DataFrame(
                {"Date": future_dates, "Predicted_Water_Level_m": preds}
            )
            fig = px.line(
                df_pred, x="Date", y="Water_Level_m", title="Water Level & Prediction"
            )
            fig.add_scatter(
                x=pred_df["Date"],
                y=pred_df["Predicted_Water_Level_m"],
                mode="lines",
                name="Prediction",
                line=dict(dash="dot"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for prediction.")


# ===============================
# Future-looking crop suggestions
# ===============================
def _estimate_future_conditions(df_full, months_ahead: int):
    wl_future_mean = None
    rain_future_per_month = None
    temp_future_mean = None

    if "Date" in df_full and "Water_Level_m" in df_full:
        df_wl = df_full.dropna(subset=["Date", "Water_Level_m"]).copy()
        if len(df_wl) > 10:
            df_wl["ordinal_date"] = df_wl["Date"].map(datetime.toordinal)
            X = df_wl[["ordinal_date"]]
            y = df_wl["Water_Level_m"]
            try:
                model = LinearRegression().fit(X, y)
                last_date = pd.to_datetime(df_wl["Date"].max())
                horizon_days = int(round(months_ahead * 30))
                future_dates = [
                    last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)
                ]
                future_ord = np.array([d.toordinal() for d in future_dates]).reshape(
                    -1, 1
                )
                wl_preds = model.predict(future_ord)
                wl_future_mean = float(np.mean(wl_preds))
            except Exception:
                wl_future_mean = float(df_wl["Water_Level_m"].mean())
        elif not df_wl.empty:
            wl_future_mean = float(df_wl["Water_Level_m"].mean())

    if "Date" in df_full and "Rainfall_mm" in df_full:
        df_r = df_full.dropna(subset=["Date"]).copy()
        end_dt = pd.to_datetime(df_r["Date"].max())
        start_dt = end_dt - pd.Timedelta(days=90)
        recent = df_r[(df_r["Date"] >= start_dt) & (df_r["Date"] <= end_dt)].copy()
        if not recent.empty and "Rainfall_mm" in recent:
            total = float(recent["Rainfall_mm"].sum())
            days = (recent["Date"].max() - recent["Date"].min()).days + 1
            if days > 0:
                rain_future_per_month = (total / days) * 30.0
        if rain_future_per_month is None and "Rainfall_mm" in df_r:
            by_month = df_r.set_index("Date")["Rainfall_mm"].resample("MS").sum()
            if not by_month.empty:
                rain_future_per_month = float(by_month.mean())

    if "Date" in df_full and "Temperature_C" in df_full:
        df_t = df_full.dropna(subset=["Date"]).copy()
        end_dt = pd.to_datetime(df_t["Date"].max())
        start_dt = end_dt - pd.Timedelta(days=90)
        recent = df_t[(df_t["Date"] >= start_dt) & (df_t["Date"] <= end_dt)]
        if "Temperature_C" in recent and not recent["Temperature_C"].dropna().empty:
            temp_future_mean = float(recent["Temperature_C"].mean())
        elif "Temperature_C" in df_full and not df_full["Temperature_C"].dropna().empty:
            temp_future_mean = float(df_full["Temperature_C"].mean())

    return wl_future_mean, rain_future_per_month, temp_future_mean


def _suggest_crops_from_estimates(wl, rpm, t):
    if wl is not None and rpm is not None:
        if wl > 3 and rpm > 80:
            crops = ["Paddy (Rice)", "Sugarcane", "Banana"]
        elif wl > 2 and rpm > 40:
            crops = ["Maize", "Groundnut", "Cotton", "Pulses"]
        else:
            crops = ["Millets (Sorghum, Pearl Millet)", "Chickpea", "Sunflower"]
    else:
        crops = ["Millets (Sorghum, Pearl Millet)", "Chickpea", "Sunflower"]
    if t is not None and t > 32:
        for x in ["Chili", "Okra", "Sweet Potato"]:
            if x not in crops:
                crops.append(x)
    return crops


def crop_suggestions(df):
    st.subheader("🌱 Crop Suggestions (Current Filter)")
    wl = df["Water_Level_m"].mean() if "Water_Level_m" in df else None
    rain = df["Rainfall_mm"].mean() if "Rainfall_mm" in df else None
    temp = df["Temperature_C"].mean() if "Temperature_C" in df else None
    crops = _suggest_crops_from_estimates(wl, rain, temp)
    if crops:
        st.success(f"Based on current conditions, consider: **{', '.join(crops)}**")
    else:
        st.info("Not enough data to suggest crops.")


def crop_suggestions_period(df_full):
    st.subheader("🌱 Farmer Suggestions for the Next Period")
    period = st.radio(
        "Select period",
        ["Next 3 months", "Next 6 months", "Next 1 year"],
        horizontal=True,
    )
    months_map = {"Next 3 months": 3, "Next 6 months": 6, "Next 1 year": 12}
    months = months_map[period]

    wl_future, rain_pm_future, temp_future = _estimate_future_conditions(
        df_full, months_ahead=months
    )

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.metric(
            "Expected Avg Water Level (m)",
            f"{wl_future:.2f}" if wl_future is not None else "—",
        )
    with c2:
        st.metric(
            "Expected Avg Rainfall / month (mm)",
            f"{rain_pm_future:.1f}" if rain_pm_future is not None else "—",
        )
    with c3:
        st.metric(
            "Expected Avg Temperature (°C)",
            f"{temp_future:.1f}" if temp_future is not None else "—",
        )

    crops = _suggest_crops_from_estimates(wl_future, rain_pm_future, temp_future)
    st.success(
        f"For **{period.lower()}**, based on projected conditions, consider: **{', '.join(crops)}**"
    )

    cfg = st.session_state.get("ntfy_cfg", {})
    if cfg.get("enabled") and not st.session_state.get("manual_push_active", False):
        ntfy_push(
            cfg["topic"],
            "AquaIntel • Crop Recommendation",
            f"{period}: {', '.join(crops)}",
            priority="low",
            server=cfg["server"],
            auth_token=cfg["token"] or None,
        )


# ===============================
# Views
# ===============================
def farmer_view(df_full, df_filt, thresholds):
    safe_thr, _, drop_thr = thresholds
    st.header("👩‍🌾 Farmer Dashboard")
    transparency_panel(df_filt)
    st.info(
        "Get alerts on low water, cost-saving tips, and pump usage insights.", icon="🚜"
    )
    kpi_cards(df_filt)
    low_water_alert(df_filt, safe_thr)

    st.subheader("Water Level Over Time")
    if "Water_Level_m" in df_filt.columns:
        st.plotly_chart(
            px.line(df_filt, x="Date", y="Water_Level_m", title="Water Level (m)"),
            use_container_width=True,
        )

    st.subheader("Rainfall Over Time")
    if "Rainfall_mm" in df_filt.columns:
        st.plotly_chart(
            px.bar(df_filt, x="Date", y="Rainfall_mm", title="Rainfall Over Time"),
            use_container_width=True,
        )

    crop_suggestions(df_filt)
    crop_suggestions_period(df_full)

    historical_comparison(df_filt)

    st.subheader("Pump Usage Tips")
    st.markdown(
        "- **Save costs:** Run pumps only when water level is above safe threshold."
    )
    st.markdown(
        f"- **Alert:** If water level drops below {safe_thr:.2f} m, consider reducing extraction."
    )

    illegal_extraction_detection(df_filt, drop_thr)
    future_prediction(df_filt)


def researcher_view(df_filt, y_cols, thresholds):
    _, _, drop_thr = thresholds
    st.header("🔬 Researcher Dashboard")
    st.info("Analyze anomalies, correlations, and download full datasets.", icon="📊")
    kpi_cards(df_filt)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Time Series", "Correlation", "Anomalies", "Quality & Extraction"]
    )
    with tab1:
        time_series_section(df_filt, y_cols)
        future_prediction(df_filt)
    with tab2:
        correlation_heatmap(df_filt)
        scatter_matrix(df_filt)
    with tab3:
        anomaly_detection(df_filt)
        missingness(df_filt)
    with tab4:
        data_quality_metrics(df_filt)
        illegal_extraction_detection(df_filt, drop_thr)
    data_download(df_filt)


def govt_view(df_filt, thresholds):
    _, critical_thr, drop_thr = thresholds
    st.header("🏛️ Government Dashboard")
    government_notifications(df_filt, critical_thr)
    illegal_extraction_detection(df_filt, drop_thr)
    st.subheader("Regional Water Statistics")
    if "Water_Level_m" in df_filt and "Rainfall_mm" in df_filt:
        stats = {
            "Lowest Water Level (m)": f"{df_filt['Water_Level_m'].min():.2f}",
            "Avg Water Level (m)": f"{df_filt['Water_Level_m'].mean():.2f}",
            "Total Rainfall (mm)": f"{df_filt['Rainfall_mm'].sum():.1f}",
            "Avg Rainfall (mm)": f"{df_filt['Rainfall_mm'].mean():.1f}",
        }
        for k, v in stats.items():
            st.metric(k, v)

    st.subheader("Water Level Trend")
    if "Water_Level_m" in df_filt:
        fig = px.line(
            df_filt, x="Date", y="Water_Level_m", title="Water Level Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rainfall Trend")
    if "Rainfall_mm" in df_filt:
        fig = px.bar(df_filt, x="Date", y="Rainfall_mm", title="Rainfall Over Time")
        st.plotly_chart(fig, use_container_width=True)


def citizen_view(df_filt):
    st.header("👥 Citizen Dashboard")
    transparency_panel(df_filt)
    st.subheader("Water Level Trend")
    if "Water_Level_m" in df_filt:
        fig = px.line(
            df_filt, x="Date", y="Water_Level_m", title="Water Level Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Rainfall Trend")
    if "Rainfall_mm" in df_filt:
        fig = px.bar(df_filt, x="Date", y="Rainfall_mm", title="Rainfall Over Time")
        st.plotly_chart(fig, use_container_width=True)
    st.info(
        "This dashboard provides a transparent summary of water and rainfall data for your area."
    )


# ===============================
# Main
# ===============================
def main():
    local_css()
    top_navbar()
    hero_banner()
    st.caption("Interactive exploration of water-level and quality metrics.")

    # --- Sidebar Branding (Home) ---
    with st.sidebar:
        st.markdown(
            '<div class="ai-sb-brand">'
            '  <a href="#" onclick="window.location.reload(); return false;">💧 AquaIntel</a>'
            "</div>",
            unsafe_allow_html=True,
        )

    # Sidebar: User type selection
    st.sidebar.markdown("## 👤 Dashboard View")
    user_type = st.sidebar.selectbox(
        "Choose your dashboard view",
        ["Farmer", "Government", "Researcher", "Citizen"],
        key="user_type",
    )

    # Sidebar: Data source selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📂 Data Source")
    src = st.sidebar.radio(
        "Select data source", ["Use sample (DWLR_Dataset_2023.csv)", "Upload CSV"]
    )
    dayfirst = st.sidebar.checkbox("Treat dates as Day-First (DD/MM/YYYY)", value=False)

    if src == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is None:
            st.warning(
                "Please upload a CSV to continue, or switch to the sample dataset."
            )
            st.stop()
        df = load_data(uploaded, dayfirst=dayfirst)
    else:
        df = load_data("DWLR_Dataset_2023.csv", dayfirst=dayfirst)

    # Column mapping (run before checks)
    st.sidebar.markdown("---")
    df = column_mapper(df)
    # Make sure 'Date' is proper datetime before anything else
    df = _ensure_datetime_datecol(df)

    # Basic checks
    if "Date" not in df.columns:
        st.error("The dataset must contain a 'Date' column (map it via Column Mapper).")
        st.stop()

    # Keep a full copy for future suggestions
    df_full = df.copy()

    # Data preview & info (main area)
    data_preview(df)
    dataset_info(df)

    # Filters & thresholds (sidebar + main effects)
    df_filt, y_cols = add_sidebar_filters(df)
    thresholds = sidebar_thresholds()

    # Views (main area)
    if user_type == "Farmer":
        farmer_view(df_full, df_filt, thresholds)
    elif user_type == "Government":
        govt_view(df_filt, thresholds)
    elif user_type == "Researcher":
        researcher_view(df_filt, y_cols, thresholds)
    elif user_type == "Citizen":
        citizen_view(df_filt)

    # --- Push Notifications (BOTTOM of the sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔔 Push Notifications")
    use_ntfy = st.sidebar.checkbox(
        "Enable ntfy push", value=True, help="Send pushes to your phone via ntfy"
    )
    ntfy_topic = st.sidebar.text_input(
        "ntfy Topic", value="aquaintel-3fa7b2c9", help="Use a hard-to-guess string"
    )
    ntfy_server = st.sidebar.text_input("ntfy Server", value="https://ntfy.sh")
    ntfy_token = st.sidebar.text_input(
        "ntfy Auth Token (optional)", value="", type="password"
    )
    st.session_state.ntfy_cfg = {
        "enabled": use_ntfy,
        "topic": ntfy_topic.strip(),
        "server": ntfy_server.strip(),
        "token": ntfy_token.strip(),
    }
    manual_alarm_button()

    st.divider()
    with st.expander("About / Tips"):
        st.markdown(
            "- Use **Resample frequency** to aggregate to weekly or monthly means.\n"
            "- **Anomaly Detection** uses a 14-day rolling mean and residual z-scores.\n"
            "- If your schema differs, rename columns via the **Column Mapper**.\n"
            "- Configure alert thresholds in the sidebar under **Thresholds**.\n"
            "- **Farmer Suggestions for the Next Period** projects 3/6/12 months ahead using simple trends.\n"
            "- **Push notifications** use ntfy: subscribe to your topic in the ntfy app on your phone."
        )
    footer()


if __name__ == "__main__":
    main()
