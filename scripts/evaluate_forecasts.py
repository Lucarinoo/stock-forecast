import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import os
os.environ["PYTHONIOENCODING"] = "utf-8"


FORECAST_DIR = Path("outputs/forecasts")
EVAL_DIR = Path("outputs/evaluations")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = EVAL_DIR / "direction_log.csv"
SUMMARY_PATH = EVAL_DIR / "summary.csv"
CACHE_PATH = EVAL_DIR / "price_cache.csv"


def load_price_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        c = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        # expected columns: ticker, date, next_close
        return c
    return pd.DataFrame(columns=["ticker", "date", "next_close"])


def save_price_cache(cache: pd.DataFrame) -> None:
    cache = cache.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    cache.to_csv(CACHE_PATH, index=False)


def get_next_close_yf(ticker: str, date: pd.Timestamp) -> float:
    """
    Get first available close AFTER 'date' (typically next trading day).
    Robust against yfinance returning MultiIndex columns.
    """
    start = date + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=7)  # buffer for weekends/holidays

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if df is None or df.empty:
        return np.nan

    # If MultiIndex columns: take first level (e.g., 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return np.nan

    val = df["Close"].iloc[0]

    # Sometimes val is a Series (edge case) -> take first element
    if isinstance(val, pd.Series):
        val = val.iloc[0]

    try:
        return float(val)
    except Exception:
        return np.nan



def ensure_next_close(rows: pd.DataFrame, cache: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each row (ticker, date), ensure we have next_close. Use cache first, download missing.
    Returns (rows_with_next_close, updated_cache)
    """
    rows = rows.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.tz_localize(None)

    cache = cache.copy()
    cache["date"] = pd.to_datetime(cache["date"]).dt.tz_localize(None)

    # merge cached values
    merged = rows.merge(cache, on=["ticker", "date"], how="left")

    missing = merged["next_close"].isna()
    if missing.any():
        missing_rows = merged.loc[missing, ["ticker", "date"]].drop_duplicates()
        print(f"Fetching next_close for {len(missing_rows)} missing ticker-date pairs...")

        new_cache_rows = []
        for t, d in zip(missing_rows["ticker"], missing_rows["date"]):
            nc = get_next_close_yf(t, pd.to_datetime(d))
            new_cache_rows.append({"ticker": t, "date": pd.to_datetime(d), "next_close": nc})

        new_cache = pd.DataFrame(new_cache_rows)
        cache = pd.concat([cache, new_cache], ignore_index=True)
        cache = cache.drop_duplicates(subset=["ticker", "date"])

        # re-merge after updating cache
        merged = rows.merge(cache, on=["ticker", "date"], how="left")

    return merged, cache


def load_existing_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        # If file exists but is empty, treat as no log yet
        if LOG_PATH.stat().st_size == 0:
            return pd.DataFrame(columns=[
                "forecast_file","ticker","asof_date","last_close","pred_pct_p50",
                "next_close","real_move_pct","pred_direction","real_direction",
                "correct","forecast_date"
            ])
        return pd.read_csv(LOG_PATH, parse_dates=["forecast_date", "asof_date"])
    return pd.DataFrame(columns=[
        "forecast_file","ticker","asof_date","last_close","pred_pct_p50",
        "next_close","real_move_pct","pred_direction","real_direction",
        "correct","forecast_date"
    ])



def main():
    forecast_files = sorted(FORECAST_DIR.glob("*_forecast.csv"))
    if not forecast_files:
        raise RuntimeError("No forecast CSV files found in outputs/forecasts.")

    existing_log = load_existing_log()

    # We'll evaluate each file only once; key = forecast_file + ticker + asof_date
    existing_keys = set(
        (existing_log["forecast_file"].astype(str) + "|" +
         existing_log["ticker"].astype(str) + "|" +
         existing_log["asof_date"].astype(str)).tolist()
    ) if len(existing_log) else set()

    cache = load_price_cache()

    all_new_rows = []

    for f in forecast_files:
        df = pd.read_csv(f)
        if df.empty:
            continue

        # Required columns from your forecast outputs
        required = {"ticker", "date", "last_close", "pred_pct_p50"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            print(f"[SKIP] {f.name} missing columns: {missing_cols}")
            continue

        df = df[["ticker", "date", "last_close", "pred_pct_p50"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["forecast_file"] = f.name
        df["forecast_date"] = pd.to_datetime(f.stem.split("_forecast")[0], errors="coerce")

        # filter already evaluated rows
        keys = (df["forecast_file"].astype(str) + "|" + df["ticker"].astype(str) + "|" + df["date"].astype(str))
        keep_mask = ~keys.isin(existing_keys)
        df = df.loc[keep_mask].copy()
        if df.empty:
            continue

        # Ensure next_close (cached)
        df2, cache = ensure_next_close(df.rename(columns={"date": "date"}), cache)

        # Compute realized move based on last_close -> next_close
        df2["next_close"] = pd.to_numeric(df2["next_close"], errors="coerce")
        df2["last_close"] = pd.to_numeric(df2["last_close"], errors="coerce")
        df2["pred_pct_p50"] = pd.to_numeric(df2["pred_pct_p50"], errors="coerce")

        df2 = df2.dropna(subset=["next_close", "last_close", "pred_pct_p50"]).copy()

        df2["real_move_pct"] = (df2["next_close"] / df2["last_close"] - 1.0) * 100.0

        # Direction: +1 up, -1 down, 0 flat
        df2["pred_direction"] = np.sign(df2["pred_pct_p50"]).astype(int)
        df2["real_direction"] = np.sign(df2["real_move_pct"]).astype(int)

        # Define correctness: exact direction match (0 counts only if real is also 0)
        df2["correct"] = (df2["pred_direction"] == df2["real_direction"]).astype(int)

        out = df2.rename(columns={"date": "asof_date"})[
            [
                "forecast_file",
                "ticker",
                "asof_date",
                "last_close",
                "pred_pct_p50",
                "next_close",
                "real_move_pct",
                "pred_direction",
                "real_direction",
                "correct",
                "forecast_date",
            ]
        ].copy()

        all_new_rows.append(out)

    # persist cache
    save_price_cache(cache)

    if not all_new_rows:
        print("No new forecast rows to evaluate (everything up to date).")
        # still regenerate summary from existing log if present
        if LOG_PATH.exists():
            log = pd.read_csv(LOG_PATH, parse_dates=["forecast_date", "asof_date"])
            write_summary(log)
        return

    new_log = pd.concat(all_new_rows, ignore_index=True)

    # Append to log
    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH, parse_dates=["forecast_date", "asof_date"])
        log = pd.concat([old, new_log], ignore_index=True)
    else:
        log = new_log

    # de-dup
    log = log.drop_duplicates(subset=["forecast_file", "ticker", "asof_date"]).sort_values(["asof_date", "ticker"])
    log.to_csv(LOG_PATH, index=False)

    print(f"✅ Appended {len(new_log)} rows to {LOG_PATH}")

    # Write summary
    write_summary(log)


def write_summary(log: pd.DataFrame) -> None:
    log = log.copy()
    if log.empty:
        print("No evaluation data to summarize.")
        return

    overall = {
        "scope": "overall",
        "accuracy": float(log["correct"].mean()),
        "n": int(len(log)),
        "start": str(pd.to_datetime(log["asof_date"]).min().date()),
        "end": str(pd.to_datetime(log["asof_date"]).max().date()),
    }

    by_ticker = (
        log.groupby("ticker")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .reset_index()
    )
    by_ticker["scope"] = "ticker"
    by_ticker["start"] = overall["start"]
    by_ticker["end"] = overall["end"]

    summary = pd.concat(
        [
            pd.DataFrame([overall])[["scope", "accuracy", "n", "start", "end"]],
            by_ticker[["scope", "ticker", "accuracy", "n", "start", "end"]],
        ],
        ignore_index=True,
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    # Print quick view
    print("\n=== Directional Accuracy Summary ===")
    print(f"Overall accuracy: {overall['accuracy']*100:.2f}%  (n={overall['n']}, {overall['start']} → {overall['end']})")

    top = by_ticker.sort_values(["accuracy", "n"], ascending=[False, False]).head(10)
    print("\nTop tickers:")
    for _, r in top.iterrows():
        print(f"  {r['ticker']}: {r['accuracy']*100:.2f}% (n={int(r['n'])})")

    print(f"\nSummary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
