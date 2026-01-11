# Raw Data Directory

This directory is reserved for raw input data.

In this project, no raw datasets are stored locally in the repository.
All financial data is downloaded dynamically at runtime using the
`yfinance` Python library.

## Data Sources

The following assets are retrieved from Yahoo Finance:
- Bitcoin (BTC-USD)
- S&P 500 Index (^GSPC)
- Gold Futures (GC=F)
- VIX Index (^VIX)

## Reproducibility

Data is fetched automatically when running:
- `python main.py`
- or the notebook in `notebooks/Final.ipynb`

This ensures that the analysis is fully reproducible without requiring
manual data downloads or large files in the repository.
