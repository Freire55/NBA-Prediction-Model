import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from scipy.optimize import nnls
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import shap
import joblib

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "ml_ready_matchups_players.csv")

# Pick the model features
features = [col for col in df.columns if col.startswith('DELTA_')] + ['REST_ADVANTAGE', 'HOME_B2B', 'AWAY_B2B', 'SEASON_YEAR']
target = 'HOME_WIN'

# Split by time
df['HOME_SEASON_ID'] = df['HOME_SEASON_ID'].astype(str)

train_df = df[df['HOME_SEASON_ID'] <= '22018']
val_df = df[(df['HOME_SEASON_ID'] > '22018') & (df['HOME_SEASON_ID'] <= '22020')]
test_df = df[df['HOME_SEASON_ID'] > '22020']

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]

# Scale for tuning
scaler_val = StandardScaler()
X_train_scaled = scaler_val.fit_transform(X_train)
X_val_scaled = scaler_val.transform(X_val)

print(f"Phase 1: Tuning on {len(train_df)} games (2000-2018)...")
print(f"Phase 2: Validating on {len(val_df)} games (2019-2020)...")
print(f"Phase 3: Testing on {len(test_df)} games (2021+)...")
print("="*50)

# Set up the searches
tscv = TimeSeriesSplit(n_splits=5)

mlp_param_grid = {
    "hidden_layer_sizes": [
        (64,), (128,), (64,32), (128,64), (128,64,32), (256,128,64)
    ],
    "activation": ["relu", "tanh"],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "learning_rate_init": [0.0005, 0.001, 0.005],
    "batch_size": [32, 64, 128],
    "max_iter": [1000]
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

mlp_search = RandomizedSearchCV(
    MLPClassifier(random_state=42, early_stopping=True, learning_rate="adaptive"), 
    mlp_param_grid, n_iter=30, cv=tscv, scoring='neg_log_loss', random_state=42, n_jobs=-1
)
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist'), 
    xgb_param_grid, n_iter=20, cv=tscv, scoring='neg_log_loss', random_state=42, n_jobs=-1
)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Fit the base models
print("Tuning Base Models...")
mlp_search.fit(X_train_scaled, y_train)
mlp_model = mlp_search.best_estimator_

xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_

lr_model.fit(X_train_scaled, y_train)

# Show best params
print("\n--- BEST HYPERPARAMETERS ---")
print("Neural net:")
print(json.dumps(mlp_search.best_params_, indent=2))
print("\nXGBoost:")
print(json.dumps(xgb_search.best_params_, indent=2))
print("=" * 50)

# Save the best params
with open(MODELS_DIR / "mlp_best_params.json", "w") as f:
    json.dump(mlp_search.best_params_, f, indent=4)
with open(MODELS_DIR / "xgb_best_params.json", "w") as f:
    json.dump(xgb_search.best_params_, f, indent=4)

# Learn the blend weights
mlp_val_probs = mlp_model.predict_proba(X_val_scaled)[:, 1]
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
lr_val_probs = lr_model.predict_proba(X_val_scaled)[:, 1]

stacked_val_probs = np.column_stack((mlp_val_probs, xgb_val_probs, lr_val_probs))
weights, _ = nnls(stacked_val_probs, y_val)
normalized_weights = weights / np.sum(weights)

print("\n--- BLEND WEIGHTS ---")
print(f"Neural Network Trust:      {normalized_weights[0]*100:.1f}%")
print(f"XGBoost Trust:             {normalized_weights[1]*100:.1f}%")
print(f"Logistic Reg Trust:        {normalized_weights[2]*100:.1f}%")
print("\nBlend formula:")
print(f"  ({normalized_weights[0]:.3f} * NN) + ({normalized_weights[1]:.3f} * XGB) + ({normalized_weights[2]:.3f} * LR)")
print("=" * 50)

# Retrain on train plus val
print("\nRetraining optimal models on Full Historical Data (2000-2020)...")
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_scaled_full = scaler_full.transform(X_test)

mlp_final = clone(mlp_model)
xgb_final = clone(xgb_model)
lr_final = clone(lr_model)

mlp_final.fit(X_train_full_scaled, y_train_full)
xgb_final.fit(X_train_full, y_train_full)
lr_final.fit(X_train_full_scaled, y_train_full)

# --- Feature importance and SHAP ---
print("\n--- FEATURE IMPORTANCE & SHAP ---")

def plot_horizontal_bar(df_imp, title, filename, top_n=15):
    plt.figure(figsize=(10, 8))
    top_df = df_imp.head(top_n).sort_values(by=df_imp.columns[1], ascending=True)
    plt.barh(top_df['Feature'], top_df[df_imp.columns[1]], color='royalblue')
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / filename)
    plt.close()

# Neural net permutation importance on the validation set
r = permutation_importance(mlp_final, X_val_scaled, y_val, n_repeats=20, random_state=42, n_jobs=-1)
mlp_importances = pd.DataFrame({'Feature': features, 'Importance': r.importances_mean}).sort_values(by='Importance', ascending=False)
mlp_importances.to_csv(MODELS_DIR / "mlp_feature_importance.csv", index=False)
plot_horizontal_bar(mlp_importances, "Neural Network Permutation Importance (Top 15)", "mlp_feature_importance.png")

# XGBoost feature importance
xgb_importances = pd.DataFrame({'Feature': features, 'Importance': xgb_final.feature_importances_}).sort_values(by='Importance', ascending=False)
xgb_importances.to_csv(MODELS_DIR / "xgb_feature_importance.csv", index=False)
plot_horizontal_bar(xgb_importances, "XGBoost Feature Importance (Top 15)", "xgb_feature_importance.png")

# Logistic regression coefficients
lr_coefs = pd.DataFrame({'Feature': features, 'Weight': lr_final.coef_[0]})
lr_coefs['Abs_Weight'] = lr_coefs['Weight'].abs()
lr_coefs = lr_coefs.sort_values(by='Abs_Weight', ascending=False)
lr_coefs.to_csv(MODELS_DIR / "lr_coefficients.csv", index=False)
plot_horizontal_bar(lr_coefs, "Logistic Regression Coefficients (Absolute Top 15)", "lr_coefficients.png")

# SHAP on a training sample
print("Building SHAP plot for XGBoost...")
X_explain = X_train_full.sample(n=min(3000, len(X_train_full)), random_state=42)
explainer = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_explain)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_explain, show=False)
plt.title("XGBoost SHAP Feature Importance")
plt.tight_layout()
plt.savefig(MODELS_DIR / "xgb_shap_summary.png")
plt.close()

# --- Scoring and calibration ---
mlp_test_probs = mlp_final.predict_proba(X_test_scaled_full)[:, 1]
xgb_test_probs = xgb_final.predict_proba(X_test)[:, 1]
lr_test_probs = lr_final.predict_proba(X_test_scaled_full)[:, 1]

metrics_dict = {}

def evaluate_model(name, y_true, probs):
    predictions = (probs > 0.5).astype(int) 
    acc = accuracy_score(y_true, predictions)
    ll = log_loss(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    auc_val = roc_auc_score(y_true, probs)
    
    metrics_dict[name] = {
        "Accuracy": acc,
        "Log_Loss": ll,
        "Brier_Score": brier,
        "ROC_AUC": auc_val
    }
    
    print(f"{name:<20} -> Accuracy: {acc * 100:.1f}% | Log Loss: {ll:.3f} | Brier: {brier:.3f} | ROC-AUC: {auc_val:.3f}")
    return predictions

print("\n--- FINAL SCORES ---")
evaluate_model("Neural Network", y_test, mlp_test_probs)
evaluate_model("XGBoost", y_test, xgb_test_probs)
evaluate_model("Logistic Regression", y_test, lr_test_probs)

ensemble_test_probs = (mlp_test_probs * normalized_weights[0]) + (xgb_test_probs * normalized_weights[1]) + (lr_test_probs * normalized_weights[2])

print("\n--- FINAL ENSEMBLE SCORE ---")
ensemble_preds = evaluate_model("Weighted Ensemble", y_test, ensemble_test_probs)

# Save metrics
with open(MODELS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)

# Build the evaluation plots
print("\nBuilding evaluation plots...")
fpr, tpr, _ = roc_curve(y_test, ensemble_test_probs)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble ROC (area = {roc_auc_val:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(MODELS_DIR / "roc_curve.png")
plt.close()

# Build the calibration curve
prob_true, prob_pred = calibration_curve(y_test, ensemble_test_probs, n_bins=10)

plt.figure(figsize=(8, 8))
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Standard Ensemble')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.title("Probability Calibration Curve")
plt.xlabel("Predicted Probability")
plt.ylabel("True Win Rate")
plt.legend()
plt.tight_layout()
plt.savefig(MODELS_DIR / "calibration_curve.png")
plt.close()

# Build the confusion matrix
cm = confusion_matrix(y_test, ensemble_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Away Win", "Home Win"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Weighted Ensemble Confusion Matrix (2021+)")
plt.tight_layout()
plt.savefig(MODELS_DIR / "ensemble_confusion_matrix.png")
plt.close()

# Save the deployment files
print("\nSaving model files to /models...")
joblib.dump(mlp_final, MODELS_DIR / "mlp_model.pkl")
joblib.dump(xgb_final, MODELS_DIR / "xgb_model.pkl")
joblib.dump(lr_final, MODELS_DIR / "lr_model.pkl")
joblib.dump(scaler_full, MODELS_DIR / "scaler.pkl")
joblib.dump(normalized_weights, MODELS_DIR / "ensemble_weights.pkl")
joblib.dump(features, MODELS_DIR / "feature_order.pkl")
print("Done. Models and plots saved.")