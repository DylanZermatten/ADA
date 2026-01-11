# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# =============================================================================
# SPRINT 1: DATA PREPARATION - COMPLETE CODE
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

CONFIG = {
    'dates': {
        'start': '2017-11-20',
        'end': '2025-01-01',
        'train_start': '2018-01-01',
        'train_end': '2021-12-31',
        'test_start': '2022-01-01',
        'test_end': '2025-01-01'
    },
    'windows': {
        'volatility': 30,
        'correlation': 30,
        'beta': 60,
        'horizon': 5
    },
    'tickers': {
        'SP500': '^GSPC',
        'BTC': 'BTC-USD',
        'Gold': 'GC=F',
        'VIX': '^VIX'
    },
    'assets': ['SP500', 'BTC', 'Gold', 'VIX'],
    'random_state': 42,
    'paths': {
        'data': 'outputs/data/',
        'plots': 'outputs/plots/',
        'reports': 'outputs/reports/'
    }
}

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import time
import warnings
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def setup_directories(config):
    """Create output directories if they don't exist"""
    for path in config['paths'].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    print("Output directories created")

def save_plot(name, config, dpi=150):
    """Save plot to outputs/plots/ directory"""
    filepath = f"{config['paths']['plots']}{name}.png"
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {filepath}")

class Timer:
    """Context manager to measure execution time"""
    def __init__(self, name="Operation"):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        print(f"Starting: {self.name}")
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f" Completed: {self.name} ({elapsed:.2f}s)\n")

def print_section(title):
    """Print formatted section title"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)

# -----------------------------------------------------------------------------
# DATA DOWNLOAD AND PROCESSING CLASSES
# -----------------------------------------------------------------------------

class DataDownloader:
    """Download financial data via yfinance"""
    
    def __init__(self, config):
        self.config = config
        self.tickers = config['tickers']
        self.start = config['dates']['start']
        self.end = config['dates']['end']
    
    def fetch_data(self):
        """Download data for all tickers"""
        print(f" Downloading data ({self.start} â†’ {self.end})...")
        
        raw = yf.download(
            list(self.tickers.values()),
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False
        )['Close']
        
        # Rename columns
        name_map = {v: k for k, v in self.tickers.items()}
        raw = raw.rename(columns=name_map)
        
        print(f" Data downloaded: {raw.shape}")
        return raw


class DataProcessor:
    """Clean and align data"""
    
    def __init__(self, config):
        self.config = config
        self.assets = config['assets']
    
    def clean(self, raw_data):
        """Clean and align data"""
        print(" Cleaning data...")
        
        # Convert to business day frequency
        prices = raw_data.asfreq('B')
        
        # Remove NaN values
        prices = prices.dropna(subset=self.assets)
        
        print(f" Data cleaned: {prices.shape}")
        print(f"   Period: {prices.index.min()} â†’ {prices.index.max()}")
        return prices


class DataValidator:
    """Validate data integrity"""
    
    @staticmethod
    def check_missing_values(df):
        """Check for missing values"""
        missing = df.isnull().sum()
        if missing.any():
            print("  Missing values detected:")
            print(missing[missing > 0])
        else:
            print(" No missing values")
    
    @staticmethod
    def check_date_range(df, expected_start, expected_end):
        """Check date range"""
        actual_start = df.index.min()
        actual_end = df.index.max()
        
        print(f" Date range:")
        print(f"   Expected: {expected_start} â†’ {expected_end}")
        print(f"   Actual:   {actual_start} â†’ {actual_end}")
    
    @staticmethod
    def test_stationarity(series, name):
        """ADF stationarity test"""
        result = adfuller(series.dropna())
        p_value = result[1]
        status = "STATIONARY" if p_value < 0.05 else "NON-STATIONARY"
        print(f"   {name:20s}: p-value={p_value:.4f} â†’ {status}")
        return p_value < 0.05

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING CLASSES
# -----------------------------------------------------------------------------

class ReturnsCalculator:
    """Calculate log returns"""
    
    def compute(self, prices, assets):
        """Calculate returns for all assets"""
        print(" Calculating returns...")
        
        df = prices.copy()
        log_prices = np.log(prices[assets])
        
        for asset in assets:
            df[f'ret_{asset}'] = log_prices[asset].diff()
        
        print(f" Returns calculated for {len(assets)} assets")
        return df


class VolatilityCalculator:
    """Calculate rolling volatility"""
    
    def __init__(self, window=30):
        self.window = window
    
    def compute(self, df, assets):
        """Calculate volatility for all assets"""
        print(f" Calculating volatility (window={self.window})...")
        
        for asset in assets:
            ret_col = f'ret_{asset}'
            df[f'vol_{asset}'] = df[ret_col].rolling(self.window).std()
            df[f'vol_ann_{asset}'] = df[f'vol_{asset}'] * np.sqrt(252)
        
        print(f" Volatility calculated for {len(assets)} assets")
        return df


class CorrelationCalculator:
    """Calculate rolling correlations"""
    
    def __init__(self, window=30):
        self.window = window
    
    def compute(self, df):
        """Calculate BTC-SP500 correlation"""
        print(f" Calculating correlation (window={self.window})...")
        
        df['corr_BTC_SP500'] = (
            df['ret_BTC']
            .rolling(self.window)
            .corr(df['ret_SP500'])
        )
        
        print(" Correlation calculated")
        return df


class BetaCalculator:
    """Calculate rolling betas"""
    
    def __init__(self, window=60):
        self.window = window
    
    def compute(self, df):
        """Calculate BTC and Gold betas vs SP500"""
        print(f" Calculating betas (window={self.window})...")
        
        # BTC Beta
        cov_btc = df['ret_BTC'].rolling(self.window).cov(df['ret_SP500'])
        var_sp = df['ret_SP500'].rolling(self.window).var()
        df['beta_BTC'] = cov_btc / var_sp
        
        # Gold Beta
        cov_gold = df['ret_Gold'].rolling(self.window).cov(df['ret_SP500'])
        df['beta_Gold'] = cov_gold / var_sp
        
        print(" Betas calculated")
        return df


class FeatureEngineer:
    """Orchestrator for all features"""
    
    def __init__(self, config):
        self.config = config
        self.assets = config['assets']
        self.returns_calc = ReturnsCalculator()
        self.vol_calc = VolatilityCalculator(config['windows']['volatility'])
        self.corr_calc = CorrelationCalculator(config['windows']['correlation'])
        self.beta_calc = BetaCalculator(config['windows']['beta'])
    
    def create_all_features(self, prices):
        """Create all features at once"""
        print("\n FEATURE ENGINEERING")
        print("-" * 80)
        
        df = prices.copy()
        
        # Returns
        df = self.returns_calc.compute(df, self.assets)
        
        # Volatility
        df = self.vol_calc.compute(df, self.assets)
        
        # Correlation
        df = self.corr_calc.compute(df)
        
        # Beta
        df = self.beta_calc.compute(df)
        
        # Final cleaning
        df_clean = df.dropna(subset=[f'ret_{a}' for a in self.assets])
        
        print("-" * 80)
        print(f" Features created: {df_clean.shape}")
        print(f" Final dataset ready")
        
        return df_clean

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

print_section("A --- DATA PREPARATION")

# Setup
setup_directories(CONFIG)

# Download
with Timer("Data download"):
    downloader = DataDownloader(CONFIG)
    raw_data = downloader.fetch_data()

# Clean
with Timer("Data cleaning"):
    processor = DataProcessor(CONFIG)
    clean_prices = processor.clean(raw_data)

# Validation
print("\n DATA VALIDATION")
print("-" * 80)
validator = DataValidator()
validator.check_missing_values(clean_prices)
validator.check_date_range(
    clean_prices, 
    CONFIG['dates']['start'], 
    CONFIG['dates']['end']
)

# Feature Engineering
with Timer("Feature engineering"):
    engineer = FeatureEngineer(CONFIG)
    df_final = engineer.create_all_features(clean_prices)

# Stationarity tests
print("\n STATIONARITY TESTS (ADF)")
print("-" * 80)
validator.test_stationarity(df_final['ret_BTC'], 'ret_BTC')
validator.test_stationarity(df_final['vol_BTC'], 'vol_BTC')
validator.test_stationarity(df_final['corr_BTC_SP500'], 'corr_BTC_SP500')

# Data preview
print("\n FINAL DATA PREVIEW")
print("-" * 80)
print(df_final.head())
print("\n")
print(df_final.describe())

# Save
output_path = f"{CONFIG['paths']['data']}processed_data.pkl"
df_final.to_pickle(output_path)
print(f"\n Data saved: {output_path}")

print("\n" + "="*80)
print(" SPRINT 1 COMPLETED - Data Preparation Complete!")
print("="*80)


