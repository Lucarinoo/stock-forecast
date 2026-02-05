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

from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


@dataclass
class RunCfg:
    lookback_years: int = 8
    market_tickers: Tuple[str, ...] = ("SPY", "QQQ", "^VIX")
    holdout_frac_dates: float = 0.20
    perm_repeats: int = 7
    top_k: int = 30


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def download_ohlcv(ticker: str, lookback_years: int) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{lookback_years}y",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
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

    # target: next-day log return
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


def build_dataset_for_run(
    tickers: List[str],
    run_cfg: RunCfg,
    feature_cols: List[str],
    ticker_to_id: Dict[str, int],
):
    # market context
    market_dfs = {mt: download_ohlcv(mt, run_cfg.lookback_years) for mt in run_cfg.market_tickers}
    market_X = make_market_features(market_dfs)

    frames_X, frames_y, frames_meta = [], [], []

    for t in tickers:
        df = download_ohlcv(t, run_cfg.lookback_years)
        if df.empty:
            print(f"[{t}] Download leer/fehlgeschlagen.")
            continue

        X_t, y_t = make_features_one_ticker(df)

        if not market_X.empty:
            X_t = X_t.join(market_X, how="inner")
            y_t = y_t.loc[X_t.index]

        tid = ticker_to_id.get(t, -1)
        if tid == -1:
            print(f"[WARN] {t} war nicht im Training. ticker_id = -1 (kann schlechter sein).")

        X_t = X_t.copy()
        X_t["ticker_id"] = tid

        # align exact feature set
        X_t = X_t.reindex(columns=feature_cols)
        mask = X_t.notna().all(axis=1) & y_t.notna()
        X_t = X_t.loc[mask]
        y_t = y_t.loc[mask]

        if len(X_t) < 80:
            print(f"[{t}] Zu wenig Rows nach Alignment: {len(X_t)}")
            continue

        meta_t = pd.DataFrame({
            "date": X_t.index,
            "ticker": t,
        }, index=X_t.index)

        frames_X.append(X_t)
        frames_y.append(y_t)
        frames_meta.append(meta_t)

    if not frames_X:
        raise RuntimeError("Keine verwertbaren Daten. Prüfe tickers.txt / Feature-Alignment / Lookback.")

    X = pd.concat(frames_X, axis=0).sort_index()
    y = pd.concat(frames_y, axis=0).sort_index()
    meta = pd.concat(frames_meta, axis=0).sort_index()
    return X, y, meta


def split_by_last_dates(meta: pd.DataFrame, holdout_frac: float):
    unique_dates = np.array(sorted(pd.to_datetime(meta["date"].unique())))
    n = len(unique_dates)
    cut = int(np.floor(n * (1.0 - holdout_frac)))
    cut = max(1, min(cut, n - 1))

    train_dates = set(unique_dates[:cut])
    test_dates = set(unique_dates[cut:])
    train_mask = meta["date"].isin(train_dates)
    test_mask = meta["date"].isin(test_dates)
    return train_mask, test_mask, unique_dates[cut]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/latest.joblib")
    parser.add_argument("--tickers-file", default="config/tickers.txt")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--lookback-years", type=int, default=8)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    ticker_to_id = bundle["ticker_to_id"]
    base_rs = bundle.get("cfg", {}).get("random_state", 42)

    run_cfg = RunCfg(
        lookback_years=args.lookback_years,
        holdout_frac_dates=args.holdout_frac,
        perm_repeats=args.repeats,
        top_k=args.topk,
    )

    tickers = load_tickers_from_file(args.tickers_file)
    if not tickers:
        raise RuntimeError("config/tickers.txt ist leer.")

    print("== Baue Dataset für Importance ==")
    X, y, meta = build_dataset_for_run(tickers, run_cfg, feature_cols, ticker_to_id)
    print(f"Global rows: {len(X)} | features: {X.shape[1]}")

    train_mask, test_mask, split_date = split_by_last_dates(meta, run_cfg.holdout_frac_dates)
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    print(f"== Holdout (zeitlich) ==")
    print(f"Test rows: {len(X_test)} | split start approx: {pd.to_datetime(split_date).date()}")

    # baseline holdout RMSE
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    print(f"Holdout RMSE (return): {rmse:.6f}")

    def neg_rmse(estimator, Xv, yv):
        p = estimator.predict(Xv)
        return -np.sqrt(mean_squared_error(yv, p))

    print("== Permutation Importance ==")
    pi = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=neg_rmse,
        n_repeats=run_cfg.perm_repeats,
        random_state=base_rs,
        n_jobs=-1,
    )

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(args.out_dir, f"{tag}_importance.csv")
    png_path = os.path.join(args.out_dir, f"{tag}_importance.png")

    imp.to_csv(csv_path, index=False)

    top = imp.head(run_cfg.top_k).copy()
    print(f"\n== Top {run_cfg.top_k} Features ==")
    print(top.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    # plot
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top))[::-1]
    plt.barh(y_pos, top["importance_mean"].values[::-1])
    plt.yticks(y_pos, top["feature"].values[::-1])
    plt.xlabel("Importance (Δ -RMSE) via permutation")
    plt.title(f"Top {run_cfg.top_k} Feature Importances (Holdout)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=170)
    plt.close()

    print(f"\n✅ CSV:  {csv_path}")
    print(f"✅ PNG:  {png_path}")
    print("Fertig.")


if __name__ == "__main__":
    main()
