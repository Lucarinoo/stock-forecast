import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class RunCfg:
    lookback_years: int = 8
    market_tickers: Tuple[str, ...] = ("SPY", "QQQ", "^VIX")
    n_bootstrap_models: int = 30
    lower_q: float = 0.10
    upper_q: float = 0.90
    plot_lookback_days: int = 180


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def download_ohlcv(ticker: str, lookback_years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{lookback_years}y", auto_adjust=False, progress=False, group_by="column")
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        if ticker == "^VIX" and {"Open", "High", "Low", "Close"}.issubset(set(df.columns)):
            df["Volume"] = np.nan
        else:
            return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def make_features_one_ticker(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    close = d["Close"].astype(float)
    volume = d["Volume"].replace(0, np.nan).astype(float)

    d["ret_1"] = np.log(close / close.shift(1))
    d["ret_2"] = np.log(close / close.shift(2))
    d["ret_5"] = np.log(close / close.shift(5))

    d["vol_5"] = d["ret_1"].rolling(5).std()
    d["vol_10"] = d["ret_1"].rolling(10).std()
    d["vol_20"] = d["ret_1"].rolling(20).std()

    for w in (5, 10, 20, 50, 100, 200):
        ma = close.rolling(w).mean()
        d[f"ma_ratio_{w}"] = ma / close

    d["rsi_14"] = rsi(close, 14)

    d["vol_chg_1"] = np.log(volume / volume.shift(1))
    d["vol_ma_ratio_20"] = volume.rolling(20).mean() / volume

    d["hl_range"] = (d["High"].astype(float) - d["Low"].astype(float)) / close
    d["oc_change"] = (d["Close"].astype(float) - d["Open"].astype(float)) / close

    for lag in (1, 2, 3, 5, 10):
        d[f"ret_1_lag_{lag}"] = d["ret_1"].shift(lag)

    y = d["ret_1"].shift(-1)

    drop_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    feature_cols = [c for c in d.columns if c not in drop_cols]
    X = d[feature_cols]
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask].copy(), y.loc[mask].copy()


def make_market_features(market_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for t, df in market_dfs.items():
        if df.empty:
            continue
        close = df["Close"].astype(float)
        ret1 = np.log(close / close.shift(1))
        vol10 = ret1.rolling(10).std()
        vol20 = ret1.rolling(20).std()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        rsi14 = rsi(close, 14)

        if t == "^VIX":
            level = close
            feats = pd.DataFrame({
                "m_vix_level": level,
                "m_vix_level_chg": level.diff(),
                "m_vix_ret1": ret1,
                "m_vix_vol10": vol10,
                "m_vix_vol20": vol20,
                "m_vix_rsi14": rsi14,
                "m_vix_ma20_ratio": ma20 / (level + 1e-12),
                "m_vix_ma50_ratio": ma50 / (level + 1e-12),
            }, index=df.index)
        else:
            prefix = "m_spy" if t == "SPY" else ("m_qqq" if t == "QQQ" else f"m_{t.lower()}")
            feats = pd.DataFrame({
                f"{prefix}_ret1": ret1,
                f"{prefix}_vol10": vol10,
                f"{prefix}_vol20": vol20,
                f"{prefix}_rsi14": rsi14,
                f"{prefix}_ma20_ratio": ma20 / (close + 1e-12),
                f"{prefix}_ma50_ratio": ma50 / (close + 1e-12),
            }, index=df.index)

        out.append(feats)

    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=1).sort_index().ffill().dropna()


def load_tickers_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def build_dataset_for_run(tickers: List[str], run_cfg: RunCfg, feature_cols: List[str], ticker_to_id: Dict[str, int]):
    market_dfs = {mt: download_ohlcv(mt, run_cfg.lookback_years) for mt in run_cfg.market_tickers}
    market_X = make_market_features(market_dfs)

    frames_X, frames_y, frames_meta = [], [], []
    price_series: Dict[str, pd.Series] = {}

    for t in tickers:
        df = download_ohlcv(t, run_cfg.lookback_years)
        if df.empty:
            print(f"[{t}] Download leer/fehlgeschlagen.")
            continue

        price_series[t] = df["Close"].astype(float).copy()

        X_t, y_t = make_features_one_ticker(df)
        if not market_X.empty:
            X_t = X_t.join(market_X, how="inner")
            y_t = y_t.loc[X_t.index]

        # ticker_id (unseen -> -1)
        tid = ticker_to_id.get(t, -1)
        if tid == -1:
            print(f"[WARN] {t} war nicht im Training. ticker_id = -1 (kann schlechter sein).")

        meta_t = pd.DataFrame({
            "date": X_t.index,
            "ticker": t,
            "last_close": df.loc[X_t.index, "Close"].astype(float).values,
            "ticker_id": tid
        }, index=X_t.index)

        X_t = X_t.copy()
        X_t["ticker_id"] = tid

        # enforce exact feature columns order (missing -> NaN -> drop)
        X_t = X_t.reindex(columns=feature_cols)
        mask = X_t.notna().all(axis=1) & y_t.notna()
        X_t = X_t.loc[mask]
        y_t = y_t.loc[mask]
        meta_t = meta_t.loc[X_t.index]

        if len(X_t) < 50:
            print(f"[{t}] Zu wenig verwertbare Rows nach Alignment: {len(X_t)}")
            continue

        frames_X.append(X_t)
        frames_y.append(y_t)
        frames_meta.append(meta_t)

    if not frames_X:
        raise RuntimeError("Keine verwertbaren Daten für Run. Prüfe Ticker oder Feature-Alignment.")

    X = pd.concat(frames_X, axis=0).sort_index()
    y = pd.concat(frames_y, axis=0).sort_index()
    meta = pd.concat(frames_meta, axis=0).sort_index()
    return X, y, meta, price_series


def latest_rows(X: pd.DataFrame, meta: pd.DataFrame):
    df = meta.copy()
    df["row_idx"] = np.arange(len(df))
    latest = df.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    X_latest = X.iloc[latest["row_idx"].values]
    return latest.reset_index(drop=True), X_latest


def bootstrap_bands(
    X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
    X_latest: pd.DataFrame, latest_meta: pd.DataFrame,
    run_cfg: RunCfg, base_random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(base_random_state)
    unique_dates = np.array(sorted(pd.to_datetime(meta["date"].unique())))
    if len(unique_dates) < 50:
        raise RuntimeError("Zu wenig unique Dates für Bootstrap-Bands.")

    # precompute indices per date
    meta_dates = meta["date"].values
    date_to_idx = {d: np.where(meta_dates == d)[0] for d in unique_dates}

    n_models = run_cfg.n_bootstrap_models
    preds = np.zeros((n_models, len(latest_meta)), dtype=float)

    for m in range(n_models):
        sampled_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        idx_list = [date_to_idx[d] for d in sampled_dates]
        boot_idx = np.concatenate(idx_list)

        model_b = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=500,
            random_state=base_random_state + 100 + m
        )
        model_b.fit(X.iloc[boot_idx], y.iloc[boot_idx])
        preds[m, :] = model_b.predict(X_latest)

    q_low = np.quantile(preds, run_cfg.lower_q, axis=0)
    q_med = np.quantile(preds, 0.50, axis=0)
    q_high = np.quantile(preds, run_cfg.upper_q, axis=0)

    out = latest_meta[["ticker", "date", "last_close"]].copy()
    out["pred_logret_p50"] = q_med
    out["pred_logret_p10"] = q_low
    out["pred_logret_p90"] = q_high

    out["pred_price_p50"] = out["last_close"] * np.exp(out["pred_logret_p50"])
    out["pred_price_p10"] = out["last_close"] * np.exp(out["pred_logret_p10"])
    out["pred_price_p90"] = out["last_close"] * np.exp(out["pred_logret_p90"])

    out["pred_pct_p50"] = (np.exp(out["pred_logret_p50"]) - 1.0) * 100.0
    out["pred_pct_p10"] = (np.exp(out["pred_logret_p10"]) - 1.0) * 100.0
    out["pred_pct_p90"] = (np.exp(out["pred_logret_p90"]) - 1.0) * 100.0

    return out.sort_values("pred_pct_p50", ascending=False).reset_index(drop=True)


def plot_one(ticker: str, closes: pd.Series, row: pd.Series, run_cfg: RunCfg, out_dir: str):
    closes = closes.dropna().astype(float)
    if closes.empty:
        return

    last_date = pd.to_datetime(row["date"])
    closes = closes.loc[closes.index <= last_date]
    if closes.empty:
        return

    closes_tail = closes.tail(run_cfg.plot_lookback_days)

    next_date = last_date + pd.Timedelta(days=1)
    p50 = float(row["pred_price_p50"])
    p10 = float(row["pred_price_p10"])
    p90 = float(row["pred_price_p90"])

    plt.figure()
    plt.plot(closes_tail.index, closes_tail.values, linewidth=1)
    plt.scatter([next_date], [p50], s=30)
    plt.vlines(x=next_date, ymin=p10, ymax=p90, linewidth=2)
    plt.title(f"{ticker} | 1-Day Forecast Band (P10-P50-P90)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=30)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{ticker.replace('^','').replace('/','_')}_forecast.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/latest.joblib")
    parser.add_argument("--tickers-file", default="config/tickers.txt")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--lookback-years", type=int, default=8)
    parser.add_argument("--bands", action="store_true", help="Bootstrap-Bands rechnen (langsamer, aber besser).")
    parser.add_argument("--no-bands", action="store_true", help="Keine Bands (schneller).")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    ticker_to_id = bundle["ticker_to_id"]
    base_rs = bundle.get("cfg", {}).get("random_state", 42)

    run_cfg = RunCfg(lookback_years=args.lookback_years)
    tickers = load_tickers_from_file(args.tickers_file)

    if not tickers:
        raise RuntimeError("config/tickers.txt ist leer.")

    X, y, meta, price_series = build_dataset_for_run(tickers, run_cfg, feature_cols, ticker_to_id)
    latest_meta, X_latest = latest_rows(X, meta)

    # Forecast distribution (bands) OR just p50 via model
    use_bands = True
    if args.no_bands:
        use_bands = False
    if args.bands:
        use_bands = True

    if use_bands:
        preds = bootstrap_bands(X, y, meta, X_latest, latest_meta, run_cfg, base_rs)
    else:
        p50 = model.predict(X_latest).astype(float)
        preds = latest_meta[["ticker", "date", "last_close"]].copy()
        preds["pred_logret_p50"] = p50
        preds["pred_logret_p10"] = np.nan
        preds["pred_logret_p90"] = np.nan
        preds["pred_price_p50"] = preds["last_close"] * np.exp(preds["pred_logret_p50"])
        preds["pred_price_p10"] = np.nan
        preds["pred_price_p90"] = np.nan
        preds["pred_pct_p50"] = (np.exp(preds["pred_logret_p50"]) - 1.0) * 100.0
        preds["pred_pct_p10"] = np.nan
        preds["pred_pct_p90"] = np.nan
        preds = preds.sort_values("pred_pct_p50", ascending=False).reset_index(drop=True)

    # Save outputs
    date_tag = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    forecasts_dir = os.path.join(args.out_dir, "forecasts")
    plots_dir = os.path.join(args.out_dir, "plots", date_tag)

    os.makedirs(forecasts_dir, exist_ok=True)
    out_csv = os.path.join(forecasts_dir, f"{date_tag}_forecast.csv")
    preds.to_csv(out_csv, index=False)

    # Plots (only if bands available)
    if use_bands:
        for _, row in preds.iterrows():
            t = row["ticker"]
            closes = price_series.get(t, pd.Series(dtype=float))
            plot_one(t, closes, row, run_cfg, plots_dir)

    # Print summary
    cols = ["ticker", "date", "last_close", "pred_pct_p10", "pred_pct_p50", "pred_pct_p90", "pred_price_p10", "pred_price_p50", "pred_price_p90"]
    print("\n== Forecast ==")
    print(preds[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\n✅ CSV: {out_csv}")
    if use_bands:
        print(f"✅ Plots: {plots_dir}")


if __name__ == "__main__":
    main()
