# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# =============================================================================
# SPRINT 5: CONCLUSIONS & FINAL REPORT
# =============================================================================

# Additional imports for this sprint
import json
from io import StringIO

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

print_section("E --- CONCLUSIONS")

# Load results from previous sprints
all_results = pd.read_pickle(f"{CONFIG['paths']['data']}processed_data.pkl")
vol_predictions = pd.read_csv(
    f"{CONFIG['paths']['data']}volatility_predictions.csv", 
    index_col=0, 
    parse_dates=True
)
vol_performance = pd.read_csv(
    f"{CONFIG['paths']['data']}volatility_performance.csv", 
    index_col=0
)
cls_performance = pd.read_csv(
    f"{CONFIG['paths']['data']}classification_performance.csv", 
    index_col=0
)

print(" Results loaded")

# -----------------------------------------------------------------------------
# QUESTION 1: BTC VS GOLD VOLATILITY
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("1. VOLATILITY COMPARISON (BTC vs GOLD)")
print("="*80)

btc_vol_mean = vol_predictions['target'].mean()
gold_vol_mean = all_results['vol_Gold'].loc[
    CONFIG['dates']['test_start']:CONFIG['dates']['test_end']
].mean()
vol_ratio = btc_vol_mean / gold_vol_mean

print(f"   BTC average volatility (2022-2025):  {btc_vol_mean:.4f}")
print(f"   Gold average volatility (2022-2025): {gold_vol_mean:.4f}")
print(f"   Ratio BTC/Gold: {vol_ratio:.2f}x")
print(f"   → BTC is {vol_ratio:.1f} times MORE VOLATILE than Gold")

# -----------------------------------------------------------------------------
# QUESTION 2: GARCH VS ML
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("2. FORECASTING PERFORMANCE (GARCH vs ML)")
print("="*80)

best_ml_idx = vol_performance.drop('GARCH', errors='ignore')['RMSE'].idxmin()
best_ml_rmse = vol_performance.loc[best_ml_idx, 'RMSE']

if 'GARCH' in vol_performance.index:
    garch_rmse = vol_performance.loc['GARCH', 'RMSE']
    improvement = ((garch_rmse - best_ml_rmse) / garch_rmse * 100)
    
    print(f"   GARCH RMSE:           {garch_rmse:.6f}")
    print(f"   Best ML ({best_ml_idx}): {best_ml_rmse:.6f}")
    print(f"   Improvement:          {improvement:.1f}%")
    print(f"   → ML models OUTPERFORM classical GARCH")
else:
    print(f"   Best ML model: {best_ml_idx}")
    print(f"   RMSE: {best_ml_rmse:.6f}")

# -----------------------------------------------------------------------------
# QUESTION 3: CORRELATION EVOLUTION
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("3. BTC CORRELATION EVOLUTION")
print("="*80)

pre_2020 = all_results['corr_BTC_SP500'].loc[:'2019-12-31'].mean()
post_2020 = all_results['corr_BTC_SP500'].loc['2020-01-01':].mean()

print(f"   Average correlation BTC-SP500 before 2020: {pre_2020:+.3f}")
print(f"   Average correlation BTC-SP500 after 2020:  {post_2020:+.3f}")
print(f"   Change: {post_2020 - pre_2020:+.3f}")

if abs(post_2020) < abs(pre_2020):
    print(f"   → BTC becomes LESS correlated (moving toward safe-haven)")
else:
    print(f"   → BTC becomes MORE correlated (risky asset behavior)")

# -----------------------------------------------------------------------------
# QUESTION 4: BEHAVIOR DURING STRESS
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("4. SAFE-HAVEN BEHAVIOR DURING MARKET STRESS")
print("="*80)

# Recalculate labels
df_stress = all_results.copy()
df_stress['label_sh_now'] = (df_stress['corr_BTC_SP500'] <= 0).astype(int)

vix_high = df_stress['VIX'] > df_stress['VIX'].quantile(0.75)
sh_stress = df_stress.loc[vix_high, 'label_sh_now'].mean()
sh_calm = df_stress.loc[~vix_high, 'label_sh_now'].mean()

print(f"   % Safe-haven days (VIX high):   {sh_stress:.1%}")
print(f"   % Safe-haven days (VIX normal): {sh_calm:.1%}")
print(f"   Difference: {sh_stress - sh_calm:+.1%}")

if sh_stress > sh_calm:
    print(f"   → BTC acts as SAFE-HAVEN during stress periods ✓")
else:
    print(f"   → BTC does NOT act as safe-haven during stress ✗")

# -----------------------------------------------------------------------------
# QUESTION 5: FINAL VERDICT - DIGITAL GOLD?
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("5. FINAL VERDICT: IS BITCOIN A DIGITAL GOLD?")
print("="*80)

# Calculate BTC and Gold safe-haven frequency
btc_sh_pct = df_stress['label_sh_now'].mean()
df_stress['corr_Gold_SP500'] = (
    df_stress['ret_Gold'].rolling(30).corr(df_stress['ret_SP500'])
)
gold_sh_pct = (df_stress['corr_Gold_SP500'] <= 0).mean()

# Best AUC
best_auc = cls_performance['AUC'].max()

# Scoring system
score = 0
criteria = []

# Criterion 1: Acceptable volatility
if vol_ratio < 5:
    score += 1
    criteria.append("✓ Acceptable volatility level")
else:
    criteria.append("✗ Too volatile compared to gold")

# Criterion 2: Safe-haven frequency
if btc_sh_pct > 0.25:
    score += 1
    criteria.append("✓ Regular safe-haven behavior")
else:
    criteria.append("✗ Insufficient safe-haven periods")

# Criterion 3: Stress behavior
if sh_stress > sh_calm * 1.2:
    score += 1
    criteria.append("✓ Safe-haven during market stress")
else:
    criteria.append("✗ No clear safe-haven during stress")

# Criterion 4: ML predictability
if best_auc > 0.65:
    score += 1
    criteria.append(f"✓ Predictable safe-haven behavior (AUC={best_auc:.3f})")
else:
    criteria.append(f"✗ Unpredictable behavior (AUC={best_auc:.3f})")

# Display criteria
for c in criteria:
    print(f"   {c}")

print(f"\n   Score: {score}/4")
print("   " + "="*76)

# Final verdict
if score >= 3:
    verdict = "YES - Bitcoin exhibits digital gold characteristics"
elif score == 2:
    verdict = "PARTIAL - Bitcoin shows some gold-like properties"
else:
    verdict = "NO - Bitcoin remains a high-risk speculative asset"

print(f"\n   CONCLUSION: {verdict}")
print("   " + "="*76)

# -----------------------------------------------------------------------------
# BTC VS GOLD COMPARISON TABLE
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("BTC VS GOLD COMPARISON")
print("="*80)

comparison_data = {
    'Metric': ['Volatility ratio', 'Safe-haven %', 'Stress behavior', 'Best AUC'],
    'BTC': [
        f'{vol_ratio:.2f}x',
        f'{btc_sh_pct:.1%}',
        f'{sh_stress:.1%}',
        f'{best_auc:.3f}'
    ],
    'Gold': [
        '1.00x',
        f'{gold_sh_pct:.1%}',
        '-',
        '-'
    ]
}
comparison_table = pd.DataFrame(comparison_data)
print(comparison_table.to_string(index=False))

# -----------------------------------------------------------------------------
# EXPORT RESULTS
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Export CSV
comparison_table.to_csv(
    f"{CONFIG['paths']['reports']}btc_vs_gold_comparison.csv",
    index=False
)

# Export JSON
export_data = {
    'config': CONFIG,
    'volatility_metrics': vol_performance.to_dict(),
    'classification_metrics': cls_performance.to_dict(),
    'conclusions': {
        'vol_ratio': float(vol_ratio),
        'btc_safe_haven_pct': float(btc_sh_pct),
        'gold_safe_haven_pct': float(gold_sh_pct),
        'stress_sh_pct': float(sh_stress),
        'calm_sh_pct': float(sh_calm),
        'score': int(score),
        'verdict': verdict
    }
}

with open(f"{CONFIG['paths']['reports']}analysis_results.json", 'w') as f:
    json.dump(export_data, f, indent=4)

# Export text report
report = StringIO()
report.write("="*80 + "\n")
report.write("BITCOIN AS DIGITAL GOLD - ANALYSIS REPORT\n")
report.write("="*80 + "\n\n")
report.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.write(f"Data period: {CONFIG['dates']['start']} to {CONFIG['dates']['end']}\n")
report.write(f"Training: {CONFIG['dates']['train_start']} to {CONFIG['dates']['train_end']}\n")
report.write(f"Testing: {CONFIG['dates']['test_start']} to {CONFIG['dates']['test_end']}\n\n")

report.write("VOLATILITY FORECASTING RESULTS\n")
report.write("-"*80 + "\n")
report.write(vol_performance.to_string() + "\n\n")

report.write("CLASSIFICATION RESULTS\n")
report.write("-"*80 + "\n")
report.write(cls_performance.to_string() + "\n\n")

report.write("CONCLUSIONS\n")
report.write("-"*80 + "\n")
for c in criteria:
    report.write(f"{c}\n")
report.write(f"\nFinal Score: {score}/4\n")
report.write(f"Verdict: {verdict}\n\n")

report.write("BTC vs GOLD COMPARISON\n")
report.write("-"*80 + "\n")
report.write(comparison_table.to_string(index=False) + "\n")

with open(f"{CONFIG['paths']['reports']}analysis_report.txt", 'w') as f:
    f.write(report.getvalue())

print("   ✓ btc_vs_gold_comparison.csv")
print("   ✓ analysis_results.json")
print("   ✓ analysis_report.txt")

print("\n" + "="*80)
print(" SPRINT 5 COMPLETED - Analysis Complete!")
print("="*80)
print("\n ALL FILES CREATED:")
print("\nPlots:")
print("   - outputs/plots/01_prices.png")
print("   - outputs/plots/02_returns_volatility.png")
print("   - outputs/plots/03_correlation_beta.png")
print("   - outputs/plots/04_volatility_models.png")
print("   - outputs/plots/05_classification.png")
print("   - outputs/plots/06_feature_importance.png")
print("\nData:")
print("   - outputs/data/processed_data.pkl")
print("   - outputs/data/volatility_predictions.csv")
print("   - outputs/data/volatility_performance.csv")
print("   - outputs/data/classification_performance.csv")
print("\nReports:")
print("   - outputs/reports/btc_vs_gold_comparison.csv")
print("   - outputs/reports/analysis_results.json")
print("   - outputs/reports/analysis_report.txt")
print("\n" + "="*80)
print(" COMPLETE ANALYSIS FINISHED!")
print("="*80)
