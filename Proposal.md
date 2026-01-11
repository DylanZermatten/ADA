

Project Proposal — Predicting Financial Market Volatility and Safe-Haven
## Behavior
Financial markets constantly fluctuate, and their volatility reflects how uncertain investors
are about future risks. Comparing volatility and market sensitivity across different types of
assets can help understand how they react under stress and whether some assets can act as
safe havens.
In this project, I will analyze and compare the daily volatility of three major assets — the
S&P 500, Bitcoin, and Gold — over the period 2018–2025. These three represent distinct
categories of investments: a traditional equity index, a digital asset, and a historical store of
value. The project aims to explore whether Bitcoin is evolving toward the behavior of a
“digital gold”, and to assess whether machine learning models can help predict future
volatility or safe-haven behavior more accurately than traditional statistical approaches.
The project will be developed entirely in Python, using libraries such as pandas, numpy,
matplotlib, yfinance, statsmodels, and scikit-learn. Daily closing prices for the three assets
will be downloaded via the yfinance package. From these, I will compute log returns, rolling
30-day volatility, and beta coefficients relative to the S&P 500.
After an exploratory phase (volatility comparison, beta estimation), I will introduce a
predictive machine learning component:
 Volatility forecasting task: train models (ARIMA, Random Forest, Gradient Boosting, and
optionally LSTM) to predict next-week or next-month volatility for each asset.
 Features: lagged returns, lagged volatility, and potentially the VIX index or trading
volume.
 Goal: evaluate whether ML models outperform traditional volatility models
(ARIMA/EWMA) in predictive accuracy.
Additionally, I may test a classification model to predict “high-volatility regimes” (binary
classification between calm and stressed market periods), to further assess how these
assets behave during market shocks.
## Expected Challenges
While the computations are straightforward, a few challenges may arise:
 Different trading calendars (crypto trades 7 days a week, markets only 5).
 Data alignment and normalization between assets.
 Feature engineering for volatility prediction.
 Interpreting ML results economically (e.g., understanding model feature importance).

These will be handled by aligning time indices, using consistent data frequencies, and
validating predictive models with proper train/test splits and performance metrics (RMSE,
MAE, classification accuracy).
## Success Criteria
 Clean and reproducible code with clear exploratory and predictive analysis.
 Visual and quantitative comparison of volatility across S&P 500, Bitcoin, and Gold.
 Successfully trained ML models predicting next-week or next-month volatility.
 Discussion of whether Bitcoin is evolving toward “safe-haven” behavior based on both
beta and model predictions.
 Results summarized in a 10-page report and a 10-minute presentation video.
## Stretch Goals
 Compute rolling betas to visualize changes in Bitcoin’s sensitivity over time.
 Extend ML models to predict safe-haven behavior (positive vs. negative correlation with
the S&P 500).
 Compare model performance across pre- and post-COVID market periods.
 Implement an LSTM for volatility forecasting if time allows.

## Mail 

Hi Anna,
Thank you for the detailed feedback! Here are the clarifications you asked for:
Volatility models: I’ll include GARCH(1,1) as the main econometric baseline, alongside ARIMA, to compare whether machine learning models (Random Forest, Gradient Boosting, and optionally LSTM) can outperform GARCH in volatility forecasting.
Temporal validation: The training period will cover 2018–2021, and testing will be done on 2022–2025, with an additional walk-forward validation to avoid look-ahead bias and ensure robustness.
Crypto trading calendar: Since Bitcoin trades 24/7, I’ll resample Bitcoin data to weekdays only to align it with equity and gold market calendars before computing volatility and training models.
Safe-haven classification: The binary target will indicate whether Bitcoin behaves as a safe haven (1) or not (0), based on its rolling 30-day correlation with the S&P 500 —
1 → correlation ≤ 0 (acts as safe haven)
0 → positive correlation (behaves like a risk asset)
The model will use lagged correlations, volatility levels, market regime indicators, and trading volume ratios as features.
Please let me know if this level of detail is sufficient or if you’d like me to refine any part further.
Best,
Dylan