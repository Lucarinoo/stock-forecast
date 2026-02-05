import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, List, Tuple

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class Config:
    lookback_years: int = 8
    min_rows_per_ticker: int = 250
    random_state: int = 42
    market_tickers: Tuple[str, ...] = ("SPY", "QQQ", "^VIX")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def download_ohlcv(ticker: str, lookback_years: int) -> pd.DataFrame:
    period = f"{lookback_years}y"
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False,
        group_by="column"
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        # ^VIX sometimes has no volume -> accept OHLC
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

    # Target: next-day log return
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

    m = pd.concat(out, axis=1).sort_index().ffill().dropna()
    return m


def load_tickers_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    tickers = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        tickers.append(ln)
    return tickers


def build_global_dataset(tickers: List[str], cfg: Config) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, int]]:
    print("== Market-Kontext laden ==")
    market_dfs = {mt: download_ohlcv(mt, cfg.lookback_years) for mt in cfg.market_tickers}
    market_X = make_market_features(market_dfs)
    if market_X.empty:
        print("WARNUNG: Market-Features fehlen (läuft trotzdem).")

    frames_X, frames_y, frames_meta = [], [], []

    print("\n== Ticker laden & Features bauen ==")
    for t in tickers:
        df = download_ohlcv(t, cfg.lookback_years)
        if df.empty:
            print(f"[{t}] Download leer/fehlgeschlagen.")
            continue

        X_t, y_t = make_features_one_ticker(df)
        if len(X_t) < cfg.min_rows_per_ticker:
            print(f"[{t}] Zu wenig Daten nach Features: {len(X_t)}")
            continue

        if not market_X.empty:
            X_t = X_t.join(market_X, how="inner")
            y_t = y_t.loc[X_t.index]
            if len(X_t) < cfg.min_rows_per_ticker:
                print(f"[{t}] Zu wenig nach Market-Join: {len(X_t)}")
                continue

        meta_t = pd.DataFrame({
            "date": X_t.index,
            "ticker": t,
            "last_close": df.loc[X_t.index, "Close"].astype(float).values
        }, index=X_t.index)

        frames_X.append(X_t)
        frames_y.append(y_t)
        frames_meta.append(meta_t)
        print(f"[{t}] OK: {len(X_t)} rows")

    if not frames_X:
        raise RuntimeError("Keine verwertbaren Daten. Prüfe Ticker/Internet/Lookback.")

    X = pd.concat(frames_X, axis=0).sort_index()
    y = pd.concat(frames_y, axis=0).sort_index()
    meta = pd.concat(frames_meta, axis=0).sort_index()

    # ticker_id mapping (stable for inference)
    uniq = sorted(meta["ticker"].unique().tolist())
    ticker_to_id = {t: i for i, t in enumerate(uniq)}
    meta["ticker_id"] = meta["ticker"].map(ticker_to_id).astype(int)
    X = X.copy()
    X["ticker_id"] = meta["ticker_id"].values

    return X, y, meta, ticker_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers-file", default="config/tickers.txt")
    parser.add_argument("--lookback-years", type=int, default=8)
    parser.add_argument("--min-rows", type=int, default=250)
    parser.add_argument("--out", default="models/latest.joblib")
    args = parser.parse_args()

    cfg = Config(
        lookback_years=args.lookback_years,
        min_rows_per_ticker=args.min_rows,
    )

    tickers = load_tickers_from_file(args.tickers_file)
    if not tickers:
        raise RuntimeError("config/tickers.txt ist leer.")

    X, y, meta, ticker_to_id = build_global_dataset(tickers, cfg)

    print(f"\nGlobal rows: {len(X)} | features: {X.shape[1]}")
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=900,
        random_state=cfg.random_state
    )
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_cols": X.columns.tolist(),
        "ticker_to_id": ticker_to_id,
        "cfg": cfg.__dict__,
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"\n✅ Modell gespeichert: {args.out}")
    print(f"Tickers im Modell: {len(ticker_to_id)}")


if __name__ == "__main__":
    import os
    main()
