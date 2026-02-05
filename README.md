# 📈 Stock Forecast – Global 1-Day ML Model

Ein **globales Machine-Learning-Forecast-Tool** für Aktienmärkte.  
Das Modell sagt **die nächste Tagesbewegung (1-Day Forecast)** für beliebige Aktien voraus – inklusive **Unsicherheitsband** und **Markt-Kontext**.

> ⚠️ Kein Trading-Bot.  
> 🎯 Fokus: Forecasting, Analyse, Research & saubere ML-Architektur.

---

## ✨ Features

- 🔁 **Globales Modell** (ein Modell für alle Aktien)
- 📊 **1-Day Forecast** (log-return → Preisprojektion)
- 🌍 **Market-Features**:
  - SPY (Gesamtmarkt)
  - QQQ (Nasdaq / Tech)
  - VIX (Volatilität)
- 📉 **Confidence Bands (P10 / P50 / P90)** via Bootstrap-Ensemble
- 🖼️ **Plots pro Aktie** (Preisverlauf + Forecast-Band)
- 🧠 **Explainability**:
  - Permutation Feature Importance
- 🧪 **Zeitlich sauberes Backtesting** (kein Lookahead)

---

## 🗂️ Projektstruktur

```text
stock-forecast/
├─ config/
│  └─ tickers.txt          # Liste der Aktien
├─ models/
│  └─ latest.joblib        # Aktives Modell
├─ outputs/
│  ├─ forecasts/           # CSV Forecasts
│  ├─ plots/               # PNG Plots pro Run
│  └─ reports/             # Feature Importance
├─ scripts/
│  ├─ train_model.py       # Modell trainieren
│  ├─ run_forecast.py      # Forecast + Bands + Plots
│  └─ run_importance.py    # Feature Importance
├─ data/                   # (ignoriert, optionales Caching)
├─ archive/                # alte Versionen / Experimente
├─ README.md
└─ .gitignore



⚙️ Installation
Voraussetzungen

Python 3.9+ empfohlen

Dependencies installieren
pip install yfinance pandas numpy scikit-learn joblib matplotlib

📌 Ticker festlegen

In config/tickers.txt (eine Aktie pro Zeile):

AAPL
MSFT
NVDA
AMD
TSLA
SAP.DE
ASML

🏋️ Modell trainieren

Trainiert ein globales Modell und speichert es als:

models/latest.joblib

python scripts/train_model.py


Optional:

python scripts/train_model.py --lookback-years 10

🔮 Forecast ausführen
Mit Confidence Bands (empfohlen)
python scripts/run_forecast.py --bands

Schnell (ohne Bootstrap-Bands)
python scripts/run_forecast.py --no-bands

Outputs

outputs/forecasts/<timestamp>_forecast.csv

outputs/plots/<timestamp>/*.png

🧠 Feature Importance (Explainability)

Berechnet Permutation Importance auf zeitlichem Holdout.

python scripts/run_importance.py


Outputs:

outputs/reports/<timestamp>_importance.csv

outputs/reports/<timestamp>_importance.png

🧪 Modell-Interpretation

P50 → erwartete Bewegung

P10 / P90 → Unsicherheitsband

VIX-Features zeigen Marktstress

ticker_id kodiert aktienspezifische Muster

🚫 Was dieses Projekt NICHT ist

❌ Kein Trading-System

❌ Keine Kauf-/Verkaufsempfehlung

❌ Kein Intraday-Forecast

❌ Keine News/Sentiment-Analyse

🚀 Mögliche Erweiterungen

📈 Trading-Signals (Thresholds, Risk-Management)

📰 News- & Sentiment-Features

🧠 Regime-Detection (Bullen/Bärenmarkt)

🌐 API oder Dashboard (FastAPI / Streamlit)

🔄 Auto-Retraining (daily/weekly)

📜 Disclaimer

Dieses Projekt dient Forschung, Lernen und Analyse.
Keine Anlageberatung. Nutzung auf eigene Verantwortung.

👤 Autor

Built by Luca
AI / ML · Quant-interessiert · Forecasting & Data Science