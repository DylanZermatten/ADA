# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# =============================================================================
# SPRINT 2: EXPLORATORY ANALYSIS - VISUALIZATIONS
# =============================================================================

# Set matplotlib style
import seaborn as sns
sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

print_section("B --- EXPLORATORY ANALYSIS")

# Load data from Sprint 1
df_base = pd.read_pickle(f"{CONFIG['paths']['data']}processed_data.pkl")
print(f" Data loaded: {df_base.shape}")
print(f"   Period: {df_base.index.min()} → {df_base.index.max()}\n")

# Required columns
prices = df_base[CONFIG['assets']]

# -----------------------------------------------------------------------------
# VISUALIZATION 1: NORMALIZED PRICES + VIX
# -----------------------------------------------------------------------------

print("\n[B1] Normalized prices and VIX...")

assets_viz = ['SP500', 'BTC', 'Gold']
norm_prices = prices[assets_viz].divide(prices[assets_viz].iloc[0]) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Normalized prices
for col in assets_viz:
    ax1.plot(norm_prices.index, norm_prices[col], label=col, linewidth=1.5)

ax1.set_title('Normalized Price Evolution (Base 100)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Index (Base 100)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# VIX
ax2.plot(prices.index, prices['VIX'], color='red', linewidth=1.5)
ax2.set_title('VIX Index Level', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('VIX')
ax2.grid(True, alpha=0.3)
ax2.axhline(20, linestyle='--', color='orange', label='Normal threshold')
ax2.axhline(30, linestyle='--', color='red', label='High volatility')
ax2.legend()

plt.tight_layout()
save_plot('01_prices', CONFIG)

# -----------------------------------------------------------------------------
# VISUALIZATION 2: RETURNS AND VOLATILITY
# -----------------------------------------------------------------------------

print("\n[B2] Returns and volatility...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Returns in percentage
returns_pct = df_base[[f'ret_{a}' for a in assets_viz]] * 100
for col in returns_pct.columns:
    axes[0].plot(returns_pct.index, returns_pct[col], 
                 label=col.replace('ret_', ''), alpha=0.7)

axes[0].set_title('Daily Log-Returns (%)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Return (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Annualized volatility
ann_vol = df_base[[f'vol_{a}' for a in assets_viz]] * np.sqrt(252) * 100
for col in ann_vol.columns:
    axes[1].plot(ann_vol.index, ann_vol[col], 
                 label=col.replace('vol_', ''), linewidth=1.5)

axes[1].set_title(f'{CONFIG["windows"]["volatility"]}-day Annualized Volatility',
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Volatility (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_plot('02_returns_volatility', CONFIG)

# -----------------------------------------------------------------------------
# VISUALIZATION 3: CORRELATION AND BETA
# -----------------------------------------------------------------------------

print("\n[B3] Correlation and beta...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# BTC-SP500 Correlation
axes[0].plot(df_base.index, df_base['corr_BTC_SP500'], 
             color='purple', linewidth=1.5)
axes[0].axhline(0, linestyle='--', color='black', linewidth=1)
axes[0].fill_between(df_base.index, df_base['corr_BTC_SP500'], 0,
                      where=(df_base['corr_BTC_SP500'] <= 0),
                      color='green', alpha=0.3, label='Safe-haven periods')
axes[0].set_title(f'{CONFIG["windows"]["correlation"]}-day Rolling Correlation: BTC vs SP500',
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Correlation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# BTC Beta
axes[1].plot(df_base.index, df_base['beta_BTC'], 
             label='Beta BTC', color='orange', linewidth=1.5)
axes[1].axhline(0, linestyle='--', color='black', linewidth=1)
axes[1].axhline(1, linestyle='--', color='gray', linewidth=1, label='Market beta')
axes[1].set_title(f'BTC Beta vs SP500 (rolling {CONFIG["windows"]["beta"]} days)',
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Beta')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# BTC vs Gold Beta Comparison
axes[2].plot(df_base.index, df_base['beta_BTC'], label='Beta BTC', linewidth=1.5)
axes[2].plot(df_base.index, df_base['beta_Gold'], label='Beta Gold', linewidth=1.5)
axes[2].axhline(0, linestyle='--', color='black', linewidth=1)
axes[2].set_title('BTC vs Gold Beta Comparison', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Beta')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_plot('03_correlation_beta', CONFIG)

# -----------------------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print(" DESCRIPTIVE STATISTICS")
print("="*80)

# Average volatility by asset
print("\n1. AVERAGE VOLATILITY (annualized)")
print("-" * 60)
for asset in assets_viz:
    vol_mean = df_base[f'vol_ann_{asset}'].mean() * 100
    vol_std = df_base[f'vol_ann_{asset}'].std() * 100
    print(f"{asset:8s}: {vol_mean:6.2f}% ± {vol_std:5.2f}%")

# Average correlation
print("\n2. BTC-SP500 CORRELATION")
print("-" * 60)
corr_mean = df_base['corr_BTC_SP500'].mean()
corr_std = df_base['corr_BTC_SP500'].std()
safe_haven_pct = (df_base['corr_BTC_SP500'] <= 0).mean() * 100
print(f"Average correlation: {corr_mean:+.3f} ± {corr_std:.3f}")
print(f"% safe-haven periods (corr <= 0): {safe_haven_pct:.1f}%")

# Average beta
print("\n3. AVERAGE BETA")
print("-" * 60)
beta_btc_mean = df_base['beta_BTC'].mean()
beta_gold_mean = df_base['beta_Gold'].mean()
print(f"Beta BTC:  {beta_btc_mean:+.3f}")
print(f"Beta Gold: {beta_gold_mean:+.3f}")

# Average returns
print("\n4. AVERAGE RETURNS (annualized)")
print("-" * 60)
for asset in assets_viz:
    ret_mean = df_base[f'ret_{asset}'].mean() * 252 * 100
    ret_std = df_base[f'ret_{asset}'].std() * np.sqrt(252) * 100
    sharpe = ret_mean / ret_std if ret_std > 0 else 0
    print(f"{asset:8s}: {ret_mean:+7.2f}% ± {ret_std:6.2f}% | Sharpe: {sharpe:+.3f}")

print("\n" + "="*80)
print(" SPRINT 2 COMPLETED - Exploratory Analysis Complete!")
print("="*80)
print("\n 3 plots created in outputs/plots/:")
print("   - 01_prices.png")
print("   - 02_returns_volatility.png")
print("   - 03_correlation_beta.png")


