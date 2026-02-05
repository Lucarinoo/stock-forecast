from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = ROOT / "outputs" / "forecasts"
PLOTS_DIR = ROOT / "outputs" / "plots"
EVAL_DIR = ROOT / "outputs" / "evaluations"
BACKTEST_DIR = ROOT / "outputs" / "backtests"


def latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


@st.cache_data
def load_csv(path: Path, parse_dates=None):
    if path and path.exists():
        return pd.read_csv(path, parse_dates=parse_dates)
    return pd.DataFrame()


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    layout="wide",
)

# =========================
# Sidebar – Theme toggle
# =========================
st.sidebar.header("Settings")
dark_mode = st.sidebar.toggle("Dark mode", value=True)

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #e6edf3; }
        [data-testid="stSidebar"] { background-color: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    PLOTLY_TEMPLATE = "plotly_dark"
else:
    st.markdown(
        """
        <style>
        .stApp { background-color: #ffffff; color: #111111; }
        [data-testid="stSidebar"] { background-color: #f6f7fb; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    PLOTLY_TEMPLATE = "plotly_white"


# =========================
# Title
# =========================
st.title("📈 Stock Forecast Dashboard")

# =========================
# Load files
# =========================
forecast_file = latest_file(FORECAST_DIR, "*_forecast.csv")
eval_log_file = EVAL_DIR / "direction_log.csv"
eval_summary_file = EVAL_DIR / "summary.csv"
bt_daily_file = latest_file(BACKTEST_DIR, "*_daily.csv")
bt_trades_file = latest_file(BACKTEST_DIR, "*_trades.csv")

st.sidebar.subheader("Data status")
st.sidebar.write("Forecast:", forecast_file.name if forecast_file else "None")
st.sidebar.write("Evaluation:", "OK" if eval_log_file.exists() else "None")
st.sidebar.write("Backtest:", bt_daily_file.name if bt_daily_file else "None")

forecast = load_csv(forecast_file, parse_dates=["date"]) if forecast_file else pd.DataFrame()
eval_log = load_csv(eval_log_file, parse_dates=["asof_date", "forecast_date"])
eval_summary = load_csv(eval_summary_file)
bt_daily = load_csv(bt_daily_file, parse_dates=["asof_date"]) if bt_daily_file else pd.DataFrame()
bt_trades = load_csv(bt_trades_file, parse_dates=["asof_date"]) if bt_trades_file else pd.DataFrame()

tabs = st.tabs(["Today Forecast", "Evaluation", "Backtest", "Plots"])


# =========================================================
# TAB 1 – Forecast
# =========================================================
with tabs[0]:
    st.subheader("Latest Forecast")

    if forecast.empty:
        st.info("No forecast available yet.")
    else:
        tickers = sorted(forecast["ticker"].unique())
        sel = st.multiselect("Tickers", tickers, default=tickers)

        df = forecast[forecast["ticker"].isin(sel)].copy()

        if "pred_pct_p50" in df.columns:
            df = df.sort_values("pred_pct_p50", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Assets", len(df))
        c2.metric("Avg P50", f"{df['pred_pct_p50'].mean():.2f}%")
        c3.metric(
            "Avg band width",
            f"{(df['pred_pct_p90'] - df['pred_pct_p10']).mean():.2f}%",
        )

        st.dataframe(df, use_container_width=True)

        fig = px.bar(
            df,
            x="ticker",
            y="pred_pct_p50",
            title="Predicted move (P50, %)",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 2 – Evaluation
# =========================================================
with tabs[1]:
    st.subheader("Directional Accuracy")

    if eval_log.empty:
        st.info("No evaluation data yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(eval_log))
        c2.metric("Overall accuracy", f"{eval_log['correct'].mean()*100:.2f}%")
        c3.metric(
            "Date range",
            f"{eval_log['asof_date'].min().date()} → {eval_log['asof_date'].max().date()}",
        )

        by_ticker = (
            eval_log.groupby("ticker")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .reset_index()
        )
        by_ticker["accuracy"] *= 100

        st.dataframe(by_ticker, use_container_width=True)

        fig = px.bar(
            by_ticker,
            x="ticker",
            y="accuracy",
            title="Accuracy by ticker (%)",
            hover_data=["n"],
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

        daily = (
            eval_log.groupby("asof_date")["correct"]
            .mean()
            .reset_index(name="accuracy")
        )
        daily["accuracy"] *= 100

        fig2 = px.line(
            daily,
            x="asof_date",
            y="accuracy",
            title="Daily accuracy (%)",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig2, use_container_width=True)


# =========================================================
# TAB 3 – Backtest
# =========================================================
with tabs[2]:
    st.subheader("Backtest")

    if bt_daily.empty:
        st.info("No backtest results yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Days", len(bt_daily))
        c2.metric("Total trades", int(bt_daily["n_trades"].sum()))
        c3.metric(
            "Net total return",
            f"{(bt_daily['net_equity'].iloc[-1]-1)*100:.2f}%",
        )

        fig = px.line(
            bt_daily,
            x="asof_date",
            y=["gross_equity", "net_equity"],
            title="Equity curve",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(bt_daily, use_container_width=True)

        if not bt_trades.empty:
            st.subheader("Trades")
            st.dataframe(bt_trades, use_container_width=True)


# =========================================================
# TAB 4 – Plots
# =========================================================
with tabs[3]:
    st.subheader("Latest Forecast Plots")

    if not PLOTS_DIR.exists():
        st.info("No plots directory.")
    else:
        runs = sorted([p for p in PLOTS_DIR.iterdir() if p.is_dir()])
        if not runs:
            st.info("No plot runs found.")
        else:
            latest = runs[-1]
            st.write("Run:", latest.name)

            imgs = list(latest.glob("*.png"))
            cols = st.columns(3)
            for i, img in enumerate(imgs[:24]):
                with cols[i % 3]:
                    st.image(img, caption=img.name, use_container_width=True)
