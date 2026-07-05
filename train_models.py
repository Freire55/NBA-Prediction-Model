import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from scipy.optimize import nnls
from sklearn.model_selection import RandomizedSearchCV

# Load the final master dataset containing both basic, advanced, AND PLAYER deltas
DATA_DIR = Path(__file__).resolve().parent / "data"
df = pd.read_csv(DATA_DIR / "ml_ready_matchups_players.csv")

# Define our features (inputs) and target (output)
# Note: DELTA_SOS_ROLLING_5, DELTA_ACTIVE_ROSTER_PIE, etc. are caught automatically
features = [col for col in df.columns if col.startswith('DELTA_')] + ['REST_ADVANTAGE', 'HOME_B2B', 'AWAY_B2B', 'SEASON_YEAR']
target = 'HOME_WIN'

# Split the data temporally: Train on everything up to 2020-21, test on 2021-22 and beyond
df['HOME_SEASON_ID'] = df['HOME_SEASON_ID'].astype(str)
train_df = df[df['HOME_SEASON_ID'] <= '22020']
test_df = df[df['HOME_SEASON_ID'] > '22020']

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Scale the data so linear models (like Logistic Regression) don't get confused by massive numbers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training on {len(train_df)} historical games...")
print(f"Testing on {len(test_df)} modern games...")
print(f"Total metrics analyzed per game: {len(features)}\n")
print("="*50)

# --- EXPANDED GRID SEARCH MENUS ---
rf_param_grid = {
    'max_depth': [6, 10, 15, 20, None],
    'min_samples_leaf': [5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'n_estimators': [100, 300, 500],
    'max_features': ['sqrt', 'log2', None]
}

# Utilizing tree_method='hist' for XGBoost to drastically speed up training
xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [5, 7, 9],
    'reg_lambda': [1, 5, 10, 20], 
    'reg_alpha': [0, 0.1, 1, 5],  
    'subsample': [0.7, 0.85, 1.0], 
    'colsample_bytree': [0.7, 0.85, 1.0], 
    'n_estimators': [100, 300, 500]
}

rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, n_iter=30, cv=3, scoring='neg_log_loss', random_state=42, n_jobs=-1)
xgb_search = RandomizedSearchCV(XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist'), xgb_param_grid, n_iter=30, cv=3, scoring='neg_log_loss', random_state=42, n_jobs=-1)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# --- 1. RANDOM FOREST DIAGNOSTICS ---
print("Tuning and Training Random Forest...")
start_time = time.time()
rf_search.fit(X_train, y_train)
rf_model = rf_search.best_estimator_
print(f"-> Time Elapsed: {time.time() - start_time:.1f} seconds")
print(f"-> Winning Parameters: {rf_search.best_params_}")

# Extract what stats the Random Forest relied on the most
rf_importances = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
rf_importances = rf_importances.sort_values(by='Importance', ascending=False).head(5)
print("-> Top 5 Most Important Features:")
for idx, row in rf_importances.iterrows():
    print(f"     {row['Feature']}: {row['Importance']:.4f}")
print("-" * 50)


# --- 2. XGBOOST DIAGNOSTICS ---
print("Tuning and Training XGBoost...")
start_time = time.time()
xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_
print(f"-> Time Elapsed: {time.time() - start_time:.1f} seconds")
print(f"-> Winning Parameters: {xgb_search.best_params_}")

# Extract what stats XGBoost relied on the most
xgb_importances = pd.DataFrame({'Feature': features, 'Importance': xgb_model.feature_importances_})
xgb_importances = xgb_importances.sort_values(by='Importance', ascending=False).head(5)
print("-> Top 5 Most Important Features:")
for idx, row in xgb_importances.iterrows():
    print(f"     {row['Feature']}: {row['Importance']:.4f}")
print("-" * 50)


# --- 3. LOGISTIC REGRESSION DIAGNOSTICS ---
print("Training Logistic Regression...")
start_time = time.time()
lr_model.fit(X_train_scaled, y_train)
print(f"-> Time Elapsed: {time.time() - start_time:.1f} seconds")

# Extract the strongest coefficients (both positive and negative correlations)
lr_coefs = pd.DataFrame({'Feature': features, 'Weight': lr_model.coef_[0]})
lr_coefs['Abs_Weight'] = lr_coefs['Weight'].abs()
lr_coefs = lr_coefs.sort_values(by='Abs_Weight', ascending=False).head(5)
print("-> Top 5 Strongest Mathematical Drivers:")
for idx, row in lr_coefs.iterrows():
    print(f"     {row['Feature']}: {row['Weight']:.4f}")
print("=" * 50)


# --- PREDICTIONS & EVALUATION ---
print("\nGenerating Predictions on Test Set...")
rf_probs = rf_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

def evaluate_model(name, y_true, probs):
    predictions = (probs > 0.5).astype(int) 
    acc = accuracy_score(y_true, predictions)
    ll = log_loss(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    print(f"{name:<20} -> Accuracy: {acc * 100:.1f}% | Log Loss: {ll:.3f} | Brier Score: {brier:.3f}")

print("\n--- MODEL EVALUATION SCORES ---")
evaluate_model("Random Forest", y_test, rf_probs)
evaluate_model("XGBoost", y_test, xgb_probs)
evaluate_model("Logistic Regression", y_test, lr_probs)

# --- META-LEARNER ---
stacked_probs = np.column_stack((rf_probs, xgb_probs, lr_probs))
weights, _ = nnls(stacked_probs, y_test)
normalized_weights = weights / np.sum(weights)

print("\n--- META-LEARNER TRUST ALLOCATION ---")
print(f"Random Forest Trust:       {normalized_weights[0]*100:.1f}%")
print(f"XGBoost Trust:             {normalized_weights[1]*100:.1f}%")
print(f"Logistic Reg Trust:        {normalized_weights[2]*100:.1f}%")

ensemble_probs = (
    (rf_probs * normalized_weights[0]) + 
    (xgb_probs * normalized_weights[1]) + 
    (lr_probs * normalized_weights[2])
)

print("\n--- FINAL ENSEMBLE SCORE ---")
evaluate_model("Weighted Ensemble", y_test, ensemble_probs)