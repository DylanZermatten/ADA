# Predicting Financial Market Volatility and Safe-Haven Behavior  
## Is Bitcoin the New Digital Gold?

This project examines whether machine learning models can predict financial market volatility more accurately than traditional econometric approaches, and whether Bitcoin exhibits safe-haven characteristics comparable to gold.

The analysis focuses on three major asset classes:
- **S&P 500** – traditional equity market
- **Bitcoin** – digital asset
- **Gold** – historical safe-haven asset

The study covers the period **2018–2025** and combines volatility forecasting with supervised classification techniques to identify safe-haven regimes.

---

## Research Questions

1. Can machine learning models outperform classical econometric models (e.g. GARCH) in forecasting market volatility?
2. Does Bitcoin behave like a safe-haven asset similar to gold, particularly during periods of market stress?

---

## Methodology Overview

This project implements a complete end-to-end quantitative pipeline in Python.

### Data
- Daily financial data retrieved from **Yahoo Finance** using the `yfinance` API
- Assets analyzed: Bitcoin (BTC-USD), S&P 500 (^GSPC), Gold Futures (GC=F), and VIX (^VIX)
- Business-day alignment to ensure consistency between cryptocurrency markets (7-day trading) and traditional markets (5-day trading)

### Feature Engineering
- Log-returns
- 30-day rolling volatility (annualized)
- Rolling correlations (Bitcoin vs S&P 500)
- Rolling betas (Bitcoin and Gold vs S&P 500)
- Lagged features to avoid look-ahead bias

### Models

**Volatility Forecasting**
- GARCH(1,1) as econometric benchmark
- Random Forest
- Gradient Boosting
- XGBoost

**Safe-Haven Classification**
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

### Evaluation Metrics
- RMSE and MAE for volatility forecasting
- Accuracy, Precision, Recall, F1-score, and AUC-ROC for classification

---

## Repository Structure

ADA/
│
├── Final.ipynb # Complete analysis notebook (main entry point)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore
│
└── outputs/
├── data/ # Processed datasets and model outputs
├── plots/ # Generated figures
└── reports/ # Final tables and reports



##  Setup Instructions

Follow the steps below to install and run the project locally.

### 1. Clone the repository
```bash
git clone https://github.com/DylanZermatten/ADA.git
cd ADA

python -m venv venv

venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

### 2 Options

python -m jupyter lab notebooks/Final.ipynb

python main.py





