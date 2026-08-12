"""
EMG Teleoperation Project
=========================
Pipeline:
  1. Load dataset
  2. Explore — understand what we have
  3. Handle missing values (imputation)
  4. Train Random Forest classifier
  5. Evaluate results

Dataset columns:
  timestamp        — when the reading was taken
  bicep_raw        — raw ADC value from AD8232 (0–1023)
  tricep_raw       — raw ADC value from ExG Pill (0–1023)
  bicep_envelope   — smoothed signal after abs(raw-512), 10-pt moving avg
  tricep_envelope  — same for tricep
  ratio_b_to_t     — bicep_envelope / tricep_envelope
  co_activation    — lower of the two envelope values (both muscles firing)
  gesture          — label: what movement was happening
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────

print("=" * 55)
print("  EMG GESTURE CLASSIFICATION — FULL PIPELINE")
print("=" * 55)

df = pd.read_csv("EMG_Teleoperation_Dataset.csv", on_bad_lines='skip')

print(f"\n[1] Dataset loaded")
print(f"    Rows    : {len(df):,}")
print(f"    Columns : {list(df.columns)}")
print(f"\n    First 5 rows:")
print(df.head().to_string())

# ─────────────────────────────────────────────────────────────
# STEP 2 — EXPLORE
# ─────────────────────────────────────────────────────────────

print(f"\n[2] Exploring the data")

print(f"\n    Label counts:")
print(df["label"].value_counts().to_string())

print(f"\n    Basic statistics (sensor columns):")
sensor_cols = ["bicep_raw","tricep_raw","bicep_envelope",
               "tricep_envelope","ratio_b_to_t","co_activation"]
print(df[sensor_cols].describe().round(2).to_string())

print(f"\n    Missing values per column:")
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df = pd.DataFrame({"Missing Count": miss, "Missing %": miss_pct})
print(miss_df[miss_df["Missing Count"] > 0].to_string())

# ─────────────────────────────────────────────────────────────
# STEP 3 — IMPUTATION
# ─────────────────────────────────────────────────────────────
# Strategy:
#   bicep_raw, tricep_raw         → median  (raw ADC, skewed by spikes)
#   bicep_envelope, tricep_envelope → median (same reason)
#   ratio_b_to_t                  → median  (ratio can be skewed by near-zero tricep)
#   co_activation                 → mean    (bounded, symmetric distribution)
#
# We use median for most because our raw columns have spike outliers
# from electrode slippage. Mean would be pulled by those spikes.
# co_activation is the minimum of two envelopes — less affected by spikes,
# so mean is acceptable there.

print(f"\n[3] Imputation")
print(f"    Before — total missing: {df[sensor_cols].isnull().sum().sum():,}")

df_clean = df.copy()

median_cols = ["bicep_raw", "tricep_raw",
               "bicep_envelope", "tricep_envelope", "ratio_b_to_t"]
mean_cols   = ["co_activation"]

for col in median_cols:
    fill_val = df_clean[col].median()
    df_clean[col].fillna(fill_val, inplace=True)
    print(f"    {col:<25} filled with median = {fill_val:.4f}")

for col in mean_cols:
    fill_val = df_clean[col].mean()
    df_clean[col].fillna(fill_val, inplace=True)
    print(f"    {col:<25} filled with mean   = {fill_val:.4f}")

print(f"    After  — total missing: {df_clean[sensor_cols].isnull().sum().sum()}")

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# ... existing code ...

# ─────────────────────────────────────────────────────────────
# STEP 4 — PREPARE FOR ML
# ─────────────────────────────────────────────────────────────

print(f"\n[4] Preparing features")

# Enhanced feature engineering
df_clean['bicep_tricep_diff'] = df_clean['bicep_envelope'] - df_clean['tricep_envelope']
df_clean['bicep_tricep_sum'] = df_clean['bicep_envelope'] + df_clean['tricep_envelope']
df_clean['envelope_product'] = df_clean['bicep_envelope'] * df_clean['tricep_envelope']
df_clean['raw_diff'] = df_clean['bicep_raw'] - df_clean['tricep_raw']
df_clean['envelope_ratio_log'] = np.log1p(df_clean['ratio_b_to_t'])  # log transform for skewed ratios

feature_cols = ["bicep_envelope", "tricep_envelope", "ratio_b_to_t", "co_activation",
                "bicep_raw", "tricep_raw", "bicep_tricep_diff", "bicep_tricep_sum",
                "envelope_product", "raw_diff", "envelope_ratio_log"]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(df_clean[feature_cols].values)
le = LabelEncoder()
y = le.fit_transform(df_clean["label"].values)

print(f"    Enhanced features: {feature_cols}")
print(f"    Classes          : {list(le.classes_)}")
print(f"    X shape          : {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"    Train size       : {len(X_train):,}")
print(f"    Test size        : {len(X_test):,}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — MODEL TRAINING WITH TUNING
# ─────────────────────────────────────────────────────────────

print(f"\n[5] Training Models")

# Random Forest with hyperparameter tuning
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [8, 12, 16, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print(f"    Random Forest best params: {rf_grid.best_params_}")
print(f"    Random Forest CV score   : {rf_grid.best_score_:.4f}")

# Train best RF model
rf_model = rf_grid.best_estimator_
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"    Random Forest test acc   : {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# Try XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"    XGBoost test acc         : {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

# Try SVM
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"    SVM test acc             : {svm_acc:.4f} ({svm_acc*100:.2f}%)")

# Calculate metrics for each algorithm
rf_precision = precision_score(y_test, rf_pred, average='weighted')
rf_recall = recall_score(y_test, rf_pred, average='weighted')
rf_f1 = f1_score(y_test, rf_pred, average='weighted')

xgb_precision = precision_score(y_test, xgb_pred, average='weighted')
xgb_recall = recall_score(y_test, xgb_pred, average='weighted')
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')

svm_precision = precision_score(y_test, svm_pred, average='weighted')
svm_recall = recall_score(y_test, svm_pred, average='weighted')
svm_f1 = f1_score(y_test, svm_pred, average='weighted')

# Use the best model
models = {'Random Forest': rf_acc, 'XGBoost': xgb_acc, 'SVM': svm_acc}
best_model_name = max(models, key=models.get)
best_acc = models[best_model_name]

if best_model_name == 'Random Forest':
    model = rf_model
    y_pred = rf_pred
elif best_model_name == 'XGBoost':
    model = xgb_model
    y_pred = xgb_pred
else:
    model = svm_model
    y_pred = svm_pred

print(f"    Random Forest CV score   : {rf_grid.best_score_:.4f}")
print(f"    XGBoost test acc         : {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
print(f"    SVM test acc             : {svm_acc:.4f} ({svm_acc*100:.2f}%)")
print(f"    Best model: {best_model_name} with {best_acc*100:.2f}% accuracy")

# ─────────────────────────────────────────────────────────────
# STEP 6 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────

print(f"\n[6] Model Evaluation")

# Cross-validation with best model
cv_scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
acc = best_acc

print(f"\n    ═══════════════════════════════════════════════════")
print(f"    ALGORITHM PERFORMANCE COMPARISON")
print(f"    ═══════════════════════════════════════════════════")
print(f"\n    {'Algorithm':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"    {'-'*68}")
print(f"    {'Random Forest':<20} {rf_acc:.4f}      {rf_precision:.4f}      {rf_recall:.4f}      {rf_f1:.4f}")
print(f"    {'XGBoost':<20} {xgb_acc:.4f}      {xgb_precision:.4f}      {xgb_recall:.4f}      {xgb_f1:.4f}")
print(f"    {'SVM':<20} {svm_acc:.4f}      {svm_precision:.4f}      {svm_recall:.4f}      {svm_f1:.4f}")
print(f"    {'-'*68}")
print(f"    {'BEST: ' + best_model_name:<20} {best_acc:.4f}      ✓ SELECTED")

print(f"\n    ═══════════════════════════════════════════════════")
print(f"    CROSS-VALIDATION SCORE")
print(f"    ═══════════════════════════════════════════════════")
print(f"    CV Accuracy (5-fold)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print(f"\n    ═══════════════════════════════════════════════════")
print(f"    PER-LABEL BREAKDOWN ({best_model_name})")
print(f"    ═══════════════════════════════════════════════════")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print(f"\n    ═══════════════════════════════════════════════════")
print(f"    FEATURE IMPORTANCES")
print(f"    ═══════════════════════════════════════════════════")
if hasattr(model, 'feature_importances_'):
    for feat, imp in sorted(zip(feature_cols, model.feature_importances_),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 60)
        print(f"      {feat:<25} {bar} {imp:.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 7 — PLOTS
# ─────────────────────────────────────────────────────────────

print(f"\n[7] Generating plots...")

colors = {
    "up":                   "#378ADD",
    "down":                 "#D85A30",
    "yaw towards left":     "#BA7517",
    "yaw towards right":    "#1D9E75",
    "roll":                 "#888780",
    "bicep extend":         "#6A0DAD",
}
label_order = sorted(list(colors.keys()))

def get_joint_positions(q):
    # q = [q1, q2, q3, q4] in radians
    p0 = np.array([0, 0, 0])  # base
    p1 = np.array([0, 0, 0.1])  # shoulder joint
    # From shoulder to elbow
    R1 = np.array([[np.cos(q[0]), -np.sin(q[0]), 0],
                   [np.sin(q[0]), np.cos(q[0]), 0],
                   [0, 0, 1]])
    p2 = p1 + R1 @ np.array([0.15, 0, 0])
    # From elbow to wrist
    R2 = np.array([[np.cos(q[0] + q[1]), -np.sin(q[0] + q[1]), 0],
                   [np.sin(q[0] + q[1]), np.cos(q[0] + q[1]), 0],
                   [0, 0, 1]])
    p3 = p2 + R2 @ np.array([0.10, 0, 0])
    # From wrist to end effector
    R3 = np.array([[np.cos(q[0] + q[1] + q[2]), -np.sin(q[0] + q[1] + q[2]), 0],
                   [np.sin(q[0] + q[1] + q[2]), np.cos(q[0] + q[1] + q[2]), 0],
                   [0, 0, 1]])
    p4 = p3 + R3 @ np.array([0.06, 0, 0])
    return [p0, p1, p2, p3, p4]

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.patch.set_facecolor('#FAFAFA')
fig.suptitle("EMG Teleoperation — Analysis & Classification Results",
             fontsize=13, fontweight='bold', y=1.01)

# ── Plot 1: Data completeness ──
ax = axes[0, 0]
completeness = [(df[col].notnull().sum() / len(df)) * 100 for col in sensor_cols]
bars = ax.bar(sensor_cols, completeness, color='#378ADD', width=0.6, edgecolor='white')
for b, v in zip(bars, completeness):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
            f"{v:.1f}%", ha='center', fontsize=8)
ax.set_title("1. Data Completeness (%)\nAll sensor columns are fully complete",
             fontsize=10, fontweight='bold')
ax.set_ylabel("Completeness (%)")
ax.set_xticklabels(sensor_cols, rotation=35, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#F5F5F5')
ax.set_ylim(0, 105)  # Ensure bars are visible

# ── Plot 2: Gesture distribution ──
ax = axes[0, 1]
cnts = df_clean["label"].value_counts().reindex(label_order)
bars = ax.bar(label_order, cnts.values,
              color=[colors[g] for g in label_order],
              width=0.6, edgecolor='white')
for b, v in zip(bars, cnts.values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+20,
            str(v), ha='center', fontsize=8)
ax.set_title("2. Samples per label\n(balanced dataset)",
             fontsize=10, fontweight='bold')
ax.set_ylabel("Count")
ax.set_xticklabels(label_order, rotation=30, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#F5F5F5')

# ── Plot 3: Bicep vs Tricep scatter ──
ax = axes[1, 0]
samp_size = min(1200, len(df_clean))
samp = df_clean.sample(samp_size, random_state=1)
for g in label_order:
    sub = samp[samp["label"] == g]
    ax.scatter(sub["bicep_envelope"], sub["tricep_envelope"],
               c=colors[g], label=g, alpha=0.5, s=14, edgecolors='none')
ax.set_title("3. Bicep vs Tricep envelope\n(separated clusters = classifiable)",
             fontsize=10, fontweight='bold')
ax.set_xlabel("Bicep envelope"); ax.set_ylabel("Tricep envelope")
ax.legend(fontsize=7, framealpha=0.85)
ax.grid(alpha=0.25); ax.set_facecolor('#F5F5F5')

# ── Plot 4: Boxplot of ratio per gesture ──
ax = axes[1, 1]
data_box = [df_clean[df_clean["label"] == g]["ratio_b_to_t"].values
            for g in label_order]
bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='.', markersize=2, alpha=0.3))
for patch, g in zip(bp['boxes'], label_order):
    patch.set_facecolor(colors[g]); patch.set_alpha(0.8)
ax.set_title("4. Bicep/Tricep ratio per label\n(key feature for flex vs extend)",
             fontsize=10, fontweight='bold')
ax.set_ylabel("ratio_b_to_t")
ax.set_xticklabels(label_order, rotation=30, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3); ax.set_facecolor('#F5F5F5')

# ── Plot 5: Confusion matrix ──
ax = axes[2, 0]
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=ax, linewidths=0.5, annot_kws={"size": 8},
            cbar_kws={"shrink": 0.8})
ax.set_title(f"5. Confusion Matrix (% correct per gesture)\nTest accuracy: {acc*100:.2f}%",
             fontsize=10, fontweight='bold')
ax.set_xlabel("Predicted →", fontsize=9)
ax.set_ylabel("Actual →", fontsize=9)
ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.tick_params(axis='y', rotation=0, labelsize=8)

# ── Plot 6: Feature importance ──
ax = axes[2, 1]
imp_sorted = sorted(zip(feature_cols, model.feature_importances_),
                    key=lambda x: x[1])
fs, vs = zip(*imp_sorted)
bar_c = ['#378ADD' if i >= len(fs)-3 else '#B4B2A9'
         for i in range(len(fs))]
ax.barh(fs, vs, color=bar_c, height=0.5, edgecolor='white')
ax.set_title("6. Feature importance\n(blue = top 3 most useful features)",
             fontsize=10, fontweight='bold')
ax.set_xlabel("Importance score")
ax.tick_params(labelsize=9)
ax.grid(axis='x', alpha=0.3); ax.set_facecolor('#F5F5F5')

plt.tight_layout()
plt.savefig("emg_analysis_results.png", dpi=150,
            bbox_inches='tight', facecolor='#FAFAFA')
print("    Saved: emg_analysis_results.png")

print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  Dataset rows        : {len(df_clean):,}")
print(f"  Missing values fixed: {df[sensor_cols].isnull().sum().sum():,}")
print(f"  Imputation method   : median (sensor cols), mean (co_activation)")
print(f"  Algorithm           : Random Forest (100 trees, max_depth=8)")
print(f"  Test accuracy       : {acc*100:.2f}%")
print(f"  CV accuracy (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print("=" * 55)