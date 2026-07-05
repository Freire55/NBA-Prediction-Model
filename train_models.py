import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score, confusion_matrix
from scipy.optimize import nnls
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.base import clone
import joblib

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "ml_ready_matchups_players.csv")

# Dynamic Feature Selection
features = [col for col in df.columns if col.startswith('DELTA_')] + ['REST_ADVANTAGE', 'HOME_B2B', 'AWAY_B2B', 'SEASON_YEAR']
target = 'HOME_WIN'

# 1. STRICT CHRONOLOGICAL SPLIT (Train / Val / Test)
df['HOME_SEASON_ID'] = df['HOME_SEASON_ID'].astype(str)

train_df = df[df['HOME_SEASON_ID'] <= '22018']
val_df = df[(df['HOME_SEASON_ID'] > '22018') & (df['HOME_SEASON_ID'] <= '22020')]
test_df = df[df['HOME_SEASON_ID'] > '22020']

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]

# Scale for the Tuning/Validation Phase
scaler_val = StandardScaler()
X_train_scaled = scaler_val.fit_transform(X_train)
X_val_scaled = scaler_val.transform(X_val)

print(f"Phase 1: Tuning on {len(train_df)} games (2000-2018)...")
print(f"Phase 2: Validating on {len(val_df)} games (2019-2020)...")
print(f"Phase 3: Testing on {len(test_df)} games (2021+)...")
print("="*50)

# 2. HYPERPARAMETER GRIDS & TIME SERIES SPLIT
tscv = TimeSeriesSplit(n_splits=5)

rf_param_grid = {
    'max_depth': [6, 10, 15, 20, None],
    'min_samples_leaf': [5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'n_estimators': [100, 300, 500],
    'max_features': ['sqrt', 'log2', None]
}

xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [5, 7, 9],
    'reg_lambda': [1, 5, 10, 20], 
    'reg_alpha': [0, 0.1, 1, 5],  
    'subsample': [0.7, 0.85, 1.0], 
    'colsample_bytree': [0.7, 0.85, 1.0], 
    'n_estimators': [100, 300, 500]
}

rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, n_iter=20, cv=tscv, scoring='neg_log_loss', random_state=42, n_jobs=-1)
xgb_search = RandomizedSearchCV(XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist'), xgb_param_grid, n_iter=20, cv=tscv, scoring='neg_log_loss', random_state=42, n_jobs=-1)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# 3. TRAIN ENSEMBLE COMPONENTS (Phase 1)
print("Tuning Base Models...")
rf_search.fit(X_train, y_train)
rf_model = rf_search.best_estimator_

xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_

lr_model.fit(X_train_scaled, y_train)

# --- NEW: SAVE TUNED HYPERPARAMETERS FOR REPRODUCIBILITY ---
with open(MODELS_DIR / "rf_best_params.json", "w") as f:
    json.dump(rf_search.best_params_, f, indent=4)
with open(MODELS_DIR / "xgb_best_params.json", "w") as f:
    json.dump(xgb_search.best_params_, f, indent=4)

# 4. VALIDATION & META-LEARNER (Phase 2)
rf_val_probs = rf_model.predict_proba(X_val)[:, 1]
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
lr_val_probs = lr_model.predict_proba(X_val_scaled)[:, 1]

stacked_val_probs = np.column_stack((rf_val_probs, xgb_val_probs, lr_val_probs))
weights, _ = nnls(stacked_val_probs, y_val)
normalized_weights = weights / np.sum(weights)

print("\n--- META-LEARNER TRUST ALLOCATION (LEARNED ON VAL SET) ---")
print(f"Random Forest Trust:       {normalized_weights[0]*100:.1f}%")
print(f"XGBoost Trust:             {normalized_weights[1]*100:.1f}%")
print(f"Logistic Reg Trust:        {normalized_weights[2]*100:.1f}%")
print("=" * 50)

# 5. THE CRITICAL RETRAINING STEP
print("\nRetraining optimal models on Full Historical Data (2000-2020)...")
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_scaled_full = scaler_full.transform(X_test)

rf_final = clone(rf_model)
xgb_final = clone(xgb_model)
lr_final = clone(lr_model)

rf_final.fit(X_train_full, y_train_full)
xgb_final.fit(X_train_full, y_train_full)
lr_final.fit(X_train_full_scaled, y_train_full)

# --- FEATURE IMPORTANCE EXTRACTION ---
print("\n--- TOP 3 FEATURE DRIVERS ---")
# Random Forest
rf_importances = pd.DataFrame({'Feature': features, 'Importance': rf_final.feature_importances_}).sort_values(by='Importance', ascending=False)
print(f"Random Forest: 1) {rf_importances.iloc[0]['Feature']}, 2) {rf_importances.iloc[1]['Feature']}, 3) {rf_importances.iloc[2]['Feature']}")

# XGBoost
xgb_importances = pd.DataFrame({'Feature': features, 'Importance': xgb_final.feature_importances_}).sort_values(by='Importance', ascending=False)
print(f"XGBoost:       1) {xgb_importances.iloc[0]['Feature']}, 2) {xgb_importances.iloc[1]['Feature']}, 3) {xgb_importances.iloc[2]['Feature']}")

# Logistic Regression (Absolute Coefficients)
lr_coefs = pd.DataFrame({'Feature': features, 'Weight': lr_final.coef_[0]})
lr_coefs['Abs_Weight'] = lr_coefs['Weight'].abs()
lr_coefs = lr_coefs.sort_values(by='Abs_Weight', ascending=False)
print(f"Logistic Reg:  1) {lr_coefs.iloc[0]['Feature']}, 2) {lr_coefs.iloc[1]['Feature']}, 3) {lr_coefs.iloc[2]['Feature']}")
print("=" * 50)

# 6. FINAL UNTOUCHED TEST SET EVALUATION (Phase 3)
rf_test_probs = rf_final.predict_proba(X_test)[:, 1]
xgb_test_probs = xgb_final.predict_proba(X_test)[:, 1]
lr_test_probs = lr_final.predict_proba(X_test_scaled_full)[:, 1]

def evaluate_model(name, y_true, probs):
    predictions = (probs > 0.5).astype(int) 
    acc = accuracy_score(y_true, predictions)
    ll = log_loss(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    cm = confusion_matrix(y_true, predictions)
    
    print(f"\n{name.upper()}:")
    print(f"  Accuracy:  {acc * 100:.1f}%")
    print(f"  Log Loss:  {ll:.3f}")
    print(f"  Brier:     {brier:.3f}")
    print(f"  ROC-AUC:   {auc:.3f}")
    print(f"  Confusion Matrix: \n    TN: {cm[0][0]:<4} | FP: {cm[0][1]:<4}\n    FN: {cm[1][0]:<4} | TP: {cm[1][1]:<4}")

print("\n--- FINAL SCORES (UNTOUCHED 2021+ TEST SET) ---")
evaluate_model("Random Forest", y_test, rf_test_probs)
evaluate_model("XGBoost", y_test, xgb_test_probs)
evaluate_model("Logistic Regression", y_test, lr_test_probs)

ensemble_test_probs = (rf_test_probs * normalized_weights[0]) + (xgb_test_probs * normalized_weights[1]) + (lr_test_probs * normalized_weights[2])

print("\n--- FINAL ENSEMBLE SCORE ---")
evaluate_model("Weighted Ensemble", y_test, ensemble_test_probs)

# --- NEW: SAVE DEPLOYMENT ARTIFACTS AND FEATURE ORDER ---
print("\nSaving deployment artifacts to '/models'...")
joblib.dump(rf_final, MODELS_DIR / "rf_model.pkl")
joblib.dump(xgb_final, MODELS_DIR / "xgb_model.pkl")
joblib.dump(lr_final, MODELS_DIR / "lr_model.pkl")
joblib.dump(scaler_full, MODELS_DIR / "scaler.pkl")
joblib.dump(normalized_weights, MODELS_DIR / "ensemble_weights.pkl")
joblib.dump(features, MODELS_DIR / "feature_order.pkl")
print("Success! Models saved. Pipeline execution complete.")