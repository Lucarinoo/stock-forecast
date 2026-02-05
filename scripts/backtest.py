import argparse
from pathlib import Path
import numpy as np
import pandas as pd


LOG_PATH = Path("outputs/evaluations/direction_log.csv")


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-abs-pred", type=float, default=0.0,
                        help="Only trade if abs(pred_pct_p50) >= this threshold (in percent).")
    parser.add_argument("--cost-bps", type=float, default=0.0,
                        help="Transaction cost per trade in basis points (e.g., 5 = 0.05%). Applied per asset per day when traded.")
    parser.add_argument("--per-ticker", action="store_true",
                        help="Also print per-ticker stats.")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        raise RuntimeError("Missing direction_log.csv. Run evaluate_forecasts.py first.")

    df = pd.read_csv(LOG_PATH, parse_dates=["asof_date", "forecast_date"])
    if df.empty:
        raise RuntimeError("direction_log.csv is empty.")
# Keep only the latest forecast per (ticker, asof_date)
    df = df.sort_values(["asof_date", "ticker", "forecast_file"])
    df = df.groupby(["asof_date", "ticker"], as_index=False).tail(1)

    # Basic cleaning
    df = df.dropna(subset=["pred_pct_p50", "real_move_pct"]).copy()
    df["pred_pct_p50"] = pd.to_numeric(df["pred_pct_p50"], errors="coerce")
    df["real_move_pct"] = pd.to_numeric(df["real_move_pct"], errors="coerce")
    df = df.dropna(subset=["pred_pct_p50", "real_move_pct"]).copy()

    # Threshold filter
    if args.min_abs_pred > 0:
        df = df[df["pred_pct_p50"].abs() >= args.min_abs_pred].copy()

    if df.empty:
        raise RuntimeError("No rows left after filtering. Lower --min-abs-pred.")

    # Signal: long/short from prediction sign
    df["signal"] = np.sign(df["pred_pct_p50"]).astype(int)
    df = df[df["signal"] != 0].copy()  # ignore exactly 0

    if df.empty:
        raise RuntimeError("No non-zero signals available.")

    # Convert pct -> decimal
    df["real_ret"] = df["real_move_pct"] / 100.0

    # Strategy return per trade (1-day)
    df["gross_trade_ret"] = df["signal"] * df["real_ret"]

    # Transaction cost per traded asset per day
    cost = (args.cost_bps / 10000.0)
    df["net_trade_ret"] = df["gross_trade_ret"] - cost

    # Portfolio daily return: equal-weight across all trades that day
    daily = (
        df.groupby("asof_date")
        .agg(
            n_trades=("ticker", "count"),
            gross_ret=("gross_trade_ret", "mean"),
            net_ret=("net_trade_ret", "mean"),
            hit_rate=("correct", "mean"),
        )
        .sort_index()
    )

    # Equity curves
    daily["gross_equity"] = (1.0 + daily["gross_ret"]).cumprod()
    daily["net_equity"] = (1.0 + daily["net_ret"]).cumprod()

    # Summary stats
    n_days = len(daily)
    n_trades = int(daily["n_trades"].sum())

    gross_total = float(daily["gross_equity"].iloc[-1] - 1.0)
    net_total = float(daily["net_equity"].iloc[-1] - 1.0)

    # Annualized (rough, assumes 252 trading days)
    def ann_return(equity_end: float, days: int) -> float:
        if days <= 1:
            return float("nan")
        return float((equity_end ** (252.0 / days)) - 1.0)

    gross_ann = ann_return(float(daily["gross_equity"].iloc[-1]), n_days)
    net_ann = ann_return(float(daily["net_equity"].iloc[-1]), n_days)

    # Sharpe (daily, annualized)
    def sharpe(returns: pd.Series) -> float:
        r = returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return float("nan")
        return float((r.mean() / r.std()) * np.sqrt(252.0))

    gross_sharpe = sharpe(daily["gross_ret"])
    net_sharpe = sharpe(daily["net_ret"])

    gross_mdd = max_drawdown(daily["gross_equity"])
    net_mdd = max_drawdown(daily["net_equity"])

    # Print report
    print("\n=== BACKTEST (1-day, equal-weight portfolio) ===")
    print(f"Rows used (trades): {len(df)}")
    print(f"Days: {n_days} | Total trades: {n_trades} | Avg trades/day: {n_trades / n_days:.2f}")
    print(f"Filter: min_abs_pred={args.min_abs_pred:.3f}% | cost_bps={args.cost_bps:.2f}")

    print("\n--- Performance ---")
    print(f"Gross total return: {gross_total*100:.2f}%")
    print(f"Net total return:   {net_total*100:.2f}%")
    print(f"Gross ann. return:  {gross_ann*100:.2f}%")
    print(f"Net ann. return:    {net_ann*100:.2f}%")
    print(f"Gross Sharpe:       {gross_sharpe:.2f}")
    print(f"Net Sharpe:         {net_sharpe:.2f}")
    print(f"Gross max drawdown: {gross_mdd*100:.2f}%")
    print(f"Net max drawdown:   {net_mdd*100:.2f}%")

    print("\n--- Forecast quality (direction) ---")
    print(f"Daily hit-rate avg: {daily['hit_rate'].mean()*100:.2f}%")

    # Save outputs
    out_dir = Path("outputs/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    daily_path = out_dir / f"{tag}_daily.csv"
    trades_path = out_dir / f"{tag}_trades.csv"
    daily.to_csv(daily_path, index=True)
    df.sort_values(["asof_date", "ticker"]).to_csv(trades_path, index=False)
    print(f"\nSaved: {daily_path}")
    print(f"Saved: {trades_path}")

    # Optional per-ticker
    if args.per_ticker:
        per = (
            df.groupby("ticker")
            .agg(
                trades=("ticker", "count"),
                hit_rate=("correct", "mean"),
                gross_ret=("gross_trade_ret", "mean"),
                net_ret=("net_trade_ret", "mean"),
            )
            .sort_values("net_ret", ascending=False)
        )
        per["hit_rate"] = per["hit_rate"] * 100.0
        per["gross_ret"] = per["gross_ret"] * 100.0
        per["net_ret"] = per["net_ret"] * 100.0

        print("\n--- Per ticker (avg daily trade return %) ---")
        print(per.to_string(float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
