# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# =============================================================================
# SPRINT 4: SAFE-HAVEN CLASSIFICATION
# =============================================================================

# Additional imports for this sprint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# -----------------------------------------------------------------------------
# LABEL CREATION
# -----------------------------------------------------------------------------

print_section("D --- SAFE-HAVEN CLASSIFICATION")

print("\n[D1] Defining safe-haven labels...")

# Create copy for classification
df_classif = df_base.copy()

# Label based on negative correlation
df_classif['label_sh_now'] = (df_classif['corr_BTC_SP500'] <= 0).astype(int)
df_classif['label_sh_future'] = df_classif['label_sh_now'].shift(-CONFIG['windows']['horizon'])

# Calculate for Gold (comparison)
df_classif['corr_Gold_SP500'] = (
    df_classif['ret_Gold'].rolling(30).corr(df_classif['ret_SP500'])
)
df_classif['label_gold_sh'] = (df_classif['corr_Gold_SP500'] <= 0).astype(int)

# Cleaning
df_classif = df_classif.dropna(subset=['label_sh_future'])
df_classif['label_sh_future'] = df_classif['label_sh_future'].astype(int)

# Statistics
btc_sh_pct = df_classif['label_sh_future'].mean()
gold_sh_pct = df_classif['label_gold_sh'].mean()

print(f" Labels created")
print(f"   BTC safe-haven frequency:  {btc_sh_pct:.1%}")
print(f"   Gold safe-haven frequency: {gold_sh_pct:.1%}")

# -----------------------------------------------------------------------------
# FEATURE CONSTRUCTION
# -----------------------------------------------------------------------------

print("\n[D2] Building classification features...")

# Features to use with lag
features_to_lag = [
    'ret_BTC', 'ret_SP500', 'ret_Gold', 'ret_VIX',
    'vol_BTC', 'vol_SP500', 'vol_Gold', 
    'corr_BTC_SP500', 'VIX'
]

# Create lags
for feat in features_to_lag:
    df_classif[f'{feat}_lag1'] = df_classif[feat].shift(1)

# Final cleaning
df_classif_ml = df_classif.dropna().copy()

# Feature selection
feature_cols = [f'{f}_lag1' for f in features_to_lag]
feature_cols += ['vol_BTC', 'vol_SP500', 'vol_Gold']

X_cls = df_classif_ml[feature_cols]
y_cls = df_classif_ml['label_sh_future']

print(f" Features built: {X_cls.shape}")

# -----------------------------------------------------------------------------
# TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------

print("\n[D3] Train/test split...")

X_train_cls = X_cls.loc[CONFIG['dates']['train_start']:CONFIG['dates']['train_end']]
y_train_cls = y_cls.loc[CONFIG['dates']['train_start']:CONFIG['dates']['train_end']]
X_test_cls = X_cls.loc[CONFIG['dates']['test_start']:CONFIG['dates']['test_end']]
y_test_cls = y_cls.loc[CONFIG['dates']['test_start']:CONFIG['dates']['test_end']]

print(f"   Train: {X_train_cls.shape}")
print(f"   Test:  {X_test_cls.shape}")

# Class imbalance calculation
neg = (y_train_cls == 0).sum()
pos = (y_train_cls == 1).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0

print(f"   Class balance: {neg} negatives, {pos} positives")
print(f"   Scale pos weight: {scale_pos_weight:.2f}")

# -----------------------------------------------------------------------------
# CLASSIFIER TRAINING
# -----------------------------------------------------------------------------

print("\n[D4] Training classifiers...")

models_cls = {
    'Logistic Regression': LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        solver='liblinear',
        random_state=CONFIG['random_state']
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=CONFIG['random_state'],
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=CONFIG['random_state']
    ),
    'XGBoost': XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )
}

predictions_cls = {}
probas_cls = {}

for name, model in models_cls.items():
    with Timer(name):
        model.fit(X_train_cls, y_train_cls)
        predictions_cls[name] = model.predict(X_test_cls)
        probas_cls[name] = model.predict_proba(X_test_cls)[:, 1]

# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------

print("\n[D5] Classifier evaluation...")

metrics_cls = {}
for name in models_cls.keys():
    metrics_cls[name] = {
        'Accuracy': accuracy_score(y_test_cls, predictions_cls[name]),
        'Precision': precision_score(y_test_cls, predictions_cls[name], zero_division=0),
        'Recall': recall_score(y_test_cls, predictions_cls[name], zero_division=0),
        'F1-Score': f1_score(y_test_cls, predictions_cls[name], zero_division=0),
        'AUC': roc_auc_score(y_test_cls, probas_cls[name])
    }

summary_cls = pd.DataFrame(metrics_cls).T.sort_values('AUC', ascending=False)

print("\n" + "="*70)
print("CLASSIFICATION PERFORMANCE")
print("="*70)
print(summary_cls.to_string())
print("="*70)

# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------

print("\n[D6] Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curves
for name in models_cls.keys():
    fpr, tpr, _ = roc_curve(y_test_cls, probas_cls[name])
    auc = metrics_cls[name]['AUC']
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Metrics bar chart
summary_cls[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar',
    ax=axes[1]
)
axes[1].set_title('Classification Metrics', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score')
axes[1].set_xlabel('Model')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)
axes[1].legend(loc='lower right')

plt.tight_layout()
save_plot('05_classification', CONFIG)

# -----------------------------------------------------------------------------
# FEATURE IMPORTANCE (best model)
# -----------------------------------------------------------------------------

print("\n[D7] Feature importance of best model...")

best_model_name = summary_cls.index[0]
best_model = models_cls[best_model_name]

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(
        best_model.feature_importances_,
        index=X_train_cls.columns
    ).sort_values(ascending=False)
    
    fig = plt.figure(figsize=(12, 6))
    importances[:15].sort_values().plot(kind='barh', color='steelblue')
    plt.title(f'Top 15 Features - {best_model_name}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_plot('06_feature_importance', CONFIG)
    
    print(f" Feature importance calculated for {best_model_name}")
else:
    print(f"  {best_model_name} doesn't have feature_importances_ attribute")

# -----------------------------------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------------------------------

summary_cls.to_csv(f"{CONFIG['paths']['data']}classification_performance.csv")
print(f"\n Results saved: classification_performance.csv")

print("\n" + "="*80)
print(" SPRINT 4 COMPLETED - Safe-Haven Classification Complete!")
print("="*80)
print("\n Files created:")
print("   - outputs/plots/05_classification.png")
print("   - outputs/plots/06_feature_importance.png")
print("   - outputs/data/classification_performance.csv")


