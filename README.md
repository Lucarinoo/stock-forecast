# 📈 Stock Forecast – Global ML System

A **global machine-learning forecasting system for stocks** with a 1-day horizon.
The project covers the **full ML lifecycle**: training, forecasting, evaluation,
backtesting, automation, and a clean dashboard.

> Research & portfolio project  
> Not a trading bot. No financial advice.

---

## ✨ Key Features

### 🔮 Forecasting
- Global ML model (single model across all stocks)
- 1-day horizon (next trading day)
- Predicts:
  - Return distribution (P10 / P50 / P90)
  - Price projections
- Market context features (e.g. market & volatility proxies)

### 📊 Evaluation
- Directional accuracy (up/down)
- Persistent evaluation log (grows over time)
- Accuracy by:
  - overall
  - ticker
  - day

### 📉 Backtesting
- Long **and** short positions
- Equal-weight portfolio
- 1-day holding period
- Optional:
  - prediction threshold
  - transaction costs
- Metrics:
  - total & annualized return
  - Sharpe ratio
  - max drawdown

### 🔁 Automation
- Daily pipeline via Windows Task Scheduler
- One command runs:
  1. Forecast
  2. Evaluation
- Robust logging (no silent failures)

### 🖥️ Dashboard (Streamlit)
- Dark / Light mode toggle
- Tabs:
  - Latest forecast
  - Evaluation & accuracy
  - Backtest results
  - Forecast plots
- Auto-start on login (Task Scheduler)

---

## 🗂️ Project Structure

```text
stock-forecast/
├─ scripts/
│  ├─ train_model.py
│  ├─ run_forecast.py
│  ├─ evaluate_forecasts.py
│  ├─ backtest.py
│  ├─ run_daily.py
│  ├─ start_dashboard.py
│  └─ dashboard.py
├─ outputs/
│  ├─ forecasts/
│  ├─ evaluations/
│  ├─ backtests/
│  ├─ plots/
│  └─ logs/
├─ models/
│  └─ latest.joblib
├─ config/
│  └─ tickers.txt
├─ README.md
├─ requirements.txt
└─ .gitignore



⚙️ Setup
Python

Python 3.11 recommended

Install dependencies
py -3.11 -m pip install -r requirements.txt

▶️ Usage
Run daily pipeline (forecast + evaluation)
py -3.11 scripts/run_daily.py

Run evaluation manually
py -3.11 scripts/evaluate_forecasts.py

Run backtest
py -3.11 scripts/backtest.py --min-abs-pred 0.3 --cost-bps 5

Start dashboard manually
py -3.11 -m streamlit run scripts/dashboard.py


Dashboard:

http://localhost:8501

🔁 Automation
Daily forecast & evaluation

Managed via Windows Task Scheduler

Uses:

py.exe -3.11 scripts/run_daily.py

Dashboard auto-start

Streamlit dashboard starts automatically on login

Wrapper:

scripts/start_dashboard.py

📊 Interpretation Notes

Directional accuracy > 52% indicates signal beyond randomness

Backtest metrics are only meaningful with sufficient history

Early results (few days) are exploratory, not statistically conclusive

🚫 Disclaimer

This project is for research and educational purposes only.
It does not provide investment advice.

👤 Author

Built by Luca
Machine Learning · Data Science · Quant-oriented projects