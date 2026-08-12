"""
EMG Teleoperation — Classifier + Prediction Exporter
=====================================================
Pipeline:
  1. Load  EMG_Teleoperation_Dataset.csv
  2. Impute missing values
  3. Engineer features
  4. Train best model (RF / XGBoost / SVM — auto-selected)
  5. Predict on the FULL dataset
  6. Export  predictions.csv  →  MATLAB reads this for simulation

Output columns in predictions.csv:
  row_index        — original row number
  timestamp        — original timestamp (if present)
  bicep_envelope   — sensor value
  tricep_envelope  — sensor value
  true_label       — ground-truth gesture (from 'label' column)
  predicted_label  — what the classifier says
  confidence       — max class probability (RF/XGB) or 1.0 (SVM)
  correct          — 1 if prediction matches true label, else 0
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  EMG CLASSIFIER → PREDICTION EXPORTER")
print("=" * 55)

df = pd.read_csv("EMG_Teleoperation_Dataset.csv", on_bad_lines='skip')
print(f"\n[1] Loaded  →  {len(df):,} rows,  columns: {list(df.columns)}")

sensor_cols = ["bicep_raw", "tricep_raw", "bicep_envelope",
               "tricep_envelope", "ratio_b_to_t", "co_activation"]

# ─────────────────────────────────────────────────────────────
# STEP 2 — IMPUTATION
# ─────────────────────────────────────────────────────────────
print(f"\n[2] Imputing missing values ...")
df_clean = df.copy()

median_cols = ["bicep_raw", "tricep_raw", "bicep_envelope",
               "tricep_envelope", "ratio_b_to_t"]
mean_cols   = ["co_activation"]

for col in median_cols:
    if col in df_clean.columns:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
for col in mean_cols:
    if col in df_clean.columns:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)

remaining = df_clean[sensor_cols].isnull().sum().sum()
print(f"    Missing after imputation: {remaining}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print(f"\n[3] Engineering features ...")

df_clean['bicep_tricep_diff']  = df_clean['bicep_envelope'] - df_clean['tricep_envelope']
df_clean['bicep_tricep_sum']   = df_clean['bicep_envelope'] + df_clean['tricep_envelope']
df_clean['envelope_product']   = df_clean['bicep_envelope'] * df_clean['tricep_envelope']
df_clean['raw_diff']           = df_clean['bicep_raw']      - df_clean['tricep_raw']
df_clean['envelope_ratio_log'] = np.log1p(df_clean['ratio_b_to_t'])

feature_cols = [
    "bicep_envelope", "tricep_envelope", "ratio_b_to_t", "co_activation",
    "bicep_raw", "tricep_raw", "bicep_tricep_diff", "bicep_tricep_sum",
    "envelope_product", "raw_diff", "envelope_ratio_log"
]

scaler = StandardScaler()
X_all  = scaler.fit_transform(df_clean[feature_cols].values)

le = LabelEncoder()
y_all = le.fit_transform(df_clean["label"].values)
print(f"    Classes : {list(le.classes_)}")
print(f"    X shape : {X_all.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAIN MODELS & SELECT BEST
# ─────────────────────────────────────────────────────────────
print(f"\n[4] Training models ...")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
)

# --- Random Forest (with grid search) ---
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [8, 12, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid, cv=5, scoring='f1_macro', n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf_model  = rf_grid.best_estimator_
rf_pred   = rf_model.predict(X_test)
rf_f1     = f1_score(y_test, rf_pred, average='macro')
print(f"    Random Forest  →  test F1 = {rf_f1*100:.2f}%")

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax', num_class=len(le.classes_),
    random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1
)
xgb_model.fit(X_train, y_train)
xgb_pred  = xgb_model.predict(X_test)
xgb_f1    = f1_score(y_test, xgb_pred, average='macro')
print(f"    XGBoost        →  test F1 = {xgb_f1*100:.2f}%")

# --- SVM ---
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale',
                random_state=42, probability=True)
svm_model.fit(X_train, y_train)
svm_pred  = svm_model.predict(X_test)
svm_f1    = f1_score(y_test, svm_pred, average='macro')
print(f"    SVM            →  test F1 = {svm_f1*100:.2f}%")

# --- Pick best ---
model_map = {
    'Random Forest': (rf_model,  rf_f1),
    'XGBoost':       (xgb_model, xgb_f1),
    'SVM':           (svm_model, svm_f1),
}
best_name  = max(model_map, key=lambda k: model_map[k][1])
best_model, best_f1 = model_map[best_name]
print(f"\n    ✓ Best model: {best_name}  ({best_f1*100:.2f}%)")
print(f"\n{classification_report(y_test, best_model.predict(X_test), target_names=le.classes_)}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — PREDICT ON FULL DATASET
# ─────────────────────────────────────────────────────────────
print(f"\n[5] Predicting on full dataset ({len(df_clean):,} rows) ...")

y_pred_all   = best_model.predict(X_all)
pred_labels  = le.inverse_transform(y_pred_all)
true_labels  = le.inverse_transform(y_all)

# Confidence = max class probability
if hasattr(best_model, 'predict_proba'):
    proba       = best_model.predict_proba(X_all)
    confidence  = proba.max(axis=1)
else:
    confidence  = np.ones(len(df_clean))   # SVM without probability

overall_acc = (pred_labels == true_labels).mean()
print(f"    Full-dataset accuracy: {overall_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# STEP 6 — EXPORT predictions.csv
# ─────────────────────────────────────────────────────────────
print(f"\n[6] Exporting predictions.csv ...")

out = pd.DataFrame({
    'row_index':       df_clean.index,
    'bicep_envelope':  df_clean['bicep_envelope'].values,
    'tricep_envelope': df_clean['tricep_envelope'].values,
    'ratio_b_to_t':    df_clean['ratio_b_to_t'].values,
    'co_activation':   df_clean['co_activation'].values,
    'true_label':      true_labels,
    'predicted_label': pred_labels,
    'confidence':      confidence.round(4),
    'correct':         (pred_labels == true_labels).astype(int),
})

# Preserve timestamp if it exists in original data
if 'timestamp' in df.columns:
    out.insert(1, 'timestamp', df['timestamp'].values)

out.to_csv("predictions.csv", index=False)

print(f"    Saved: predictions.csv")
print(f"    Rows : {len(out):,}")
print(f"    Cols : {list(out.columns)}")
print(f"\n    Label distribution in predictions:")
print(out['predicted_label'].value_counts().to_string())
print(f"\n    Per-gesture accuracy:")
for gesture in le.classes_:
    mask = out['true_label'] == gesture
    g_acc = out.loc[mask, 'correct'].mean() * 100
    print(f"      {gesture:<25} {g_acc:.1f}%")

print("\n" + "=" * 55)
print(f"  Done! Run MATLAB simulation next.")
print(f"  Best model : {best_name}")
print(f"  F1 Score   : {best_f1*100:.2f}%")
print("=" * 55)