# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# =============================================================================
# SPRINT 3: VOLATILITY FORECASTING
# =============================================================================

# Additional imports for this sprint
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------

print_section("C --- VOLATILITY FORECASTING")

print("\n[C1-C2] Building regression dataset...")

def build_regression_dataset(df, horizon=5):
    """Build dataset for volatility forecasting"""
    target = df['vol_BTC'].shift(-horizon)
    
    features = pd.DataFrame(index=df.index)
    features['vol_BTC'] = df['vol_BTC']
    features['vol_SP500'] = df['vol_SP500']
    features['vol_Gold'] = df['vol_Gold']
    features['corr_BTC_SP500'] = df['corr_BTC_SP500']
    features['VIX'] = df['VIX']
    features['ret_BTC'] = df['ret_BTC']
    features['ret_SP500'] = df['ret_SP500']
    features['ret_Gold'] = df['ret_Gold']
    features['ret_VIX'] = df['ret_VIX']
    
    # Lags
    for lag in [1, 5, 10]:
        features[f'vol_BTC_lag{lag}'] = df['vol_BTC'].shift(lag)
        features[f'corr_lag{lag}'] = df['corr_BTC_SP500'].shift(lag)
        features[f'VIX_lag{lag}'] = df['VIX'].shift(lag)
        features[f'ret_BTC_lag{lag}'] = df['ret_BTC'].shift(lag)
    
    full = pd.concat([features, target.rename('target')], axis=1)
    full = full.dropna()
    
    return full.drop(columns=['target']), full['target']

# Build dataset
X_reg, y_reg = build_regression_dataset(df_base, horizon=CONFIG['windows']['horizon'])

# Train/test split
train_mask = (X_reg.index < CONFIG['dates']['test_start'])
test_mask = (X_reg.index >= CONFIG['dates']['test_start'])

X_train_reg = X_reg[train_mask]
y_train_reg = y_reg[train_mask]
X_test_reg = X_reg[test_mask]
y_test_reg = y_reg[test_mask]

print(f" Dataset built")
print(f"   Train: {X_train_reg.shape}")
print(f"   Test:  {X_test_reg.shape}")

# -----------------------------------------------------------------------------
# GARCH MODEL
# -----------------------------------------------------------------------------

print("\n[C3] Training GARCH(1,1)...")

with Timer("GARCH"):
    # Convert to percentage for GARCH
    btc_ret_pct = df_base['ret_BTC'].dropna() * 100
    
    # GARCH model
    am = arch_model(btc_ret_pct, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
    res_garch = am.fit(
        last_obs=CONFIG['dates']['train_end'], 
        disp='off', 
        update_freq=0
    )
    
    # Forecasts
    fcst = res_garch.forecast(
        horizon=1, 
        start=CONFIG['dates']['test_start'], 
        reindex=True
    )
    
    garch_vol = np.sqrt(
        fcst.variance[CONFIG['dates']['test_start']:CONFIG['dates']['test_end']]['h.1']
    ) / 100
    
    print(f" GARCH trained, {len(garch_vol)} forecasts generated")

# -----------------------------------------------------------------------------
# ML MODELS
# -----------------------------------------------------------------------------

print("\n[C5] Training ML models...")

models_reg = {
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=CONFIG['random_state'],
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=CONFIG['random_state']
    ),
    'XGBoost': XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )
}

predictions = {'target': y_test_reg}

for name, model in models_reg.items():
    with Timer(name):
        model.fit(X_train_reg, y_train_reg)
        predictions[name] = pd.Series(
            model.predict(X_test_reg), 
            index=X_test_reg.index
        )

# Align predictions
common_idx = predictions['target'].index.intersection(garch_vol.index)
predictions = {
    k: v.loc[common_idx] if isinstance(v, pd.Series) else v 
    for k, v in predictions.items()
}
predictions['GARCH'] = garch_vol.loc[common_idx]

all_results = pd.DataFrame(predictions)

# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------

print("\n[C6] Model evaluation...")

def compute_metrics(y_true, y_pred):
    """Calculate RMSE and MAE"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

metrics_dict = {}
for col in all_results.columns:
    if col != 'target':
        metrics_dict[col] = compute_metrics(
            all_results['target'], 
            all_results[col]
        )

summary_table = pd.DataFrame(metrics_dict).T.sort_values('RMSE')

print("\n" + "="*70)
print("VOLATILITY FORECASTING PERFORMANCE")
print("="*70)
print(summary_table.to_string())
print("="*70)

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

print("\n[C7] Creating visualizations...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Time series
axes[0].plot(all_results.index, all_results['target'],
             label='True volatility', linewidth=2, color='black')

colors = {
    'GARCH': 'red',
    'Random Forest': 'blue',
    'Gradient Boosting': 'green',
    'XGBoost': 'purple'
}

for col in all_results.columns:
    if col != 'target':
        axes[0].plot(all_results.index, all_results[col],
                     label=col, alpha=0.8, color=colors.get(col, 'gray'))

axes[0].set_title('BTC Volatility: True vs Predictions', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Volatility')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Metrics comparison
summary_table[['RMSE', 'MAE']].plot(
    kind='bar', 
    ax=axes[1], 
    color=['steelblue', 'coral']
)
axes[1].set_title('Model Comparison (RMSE / MAE)', 
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Error')
axes[1].set_xlabel('Model')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
save_plot('04_volatility_models', CONFIG)

# -----------------------------------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------------------------------

# Save predictions
all_results.to_csv(f"{CONFIG['paths']['data']}volatility_predictions.csv")
summary_table.to_csv(f"{CONFIG['paths']['data']}volatility_performance.csv")

print("\n" + "="*80)
print(" SPRINT 3 COMPLETED - Volatility Forecasting Complete!")
print("="*80)
print("\n Files created:")
print("   - outputs/plots/04_volatility_models.png")
print("   - outputs/data/volatility_predictions.csv")
print("   - outputs/data/volatility_performance.csv")


