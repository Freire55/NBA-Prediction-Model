"""
Tunes and retrains the machine learning models used in the prediction
pipeline, now with integrated cross-validated calibration.

This module performs hyperparameter optimization for the MLP and
XGBoost classifiers using chronological cross-validation, trains a
baseline Logistic Regression model, and wraps all models in
CalibratedClassifierCV to ensure robust probability estimation.
"""

from pathlib import Path
from typing import Any, Tuple

import logging
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from training.config import TrainingConfig
from training.utils import save_json

# ======================================================
# Constants
# ======================================================

MLP_PARAMS_FILE = "mlp_best_params.json"
XGB_PARAMS_FILE = "xgb_best_params.json"

MLP_CV_RESULTS_FILE = "mlp_cv_results.csv"
XGB_CV_RESULTS_FILE = "xgb_cv_results.csv"

SCALER_STATS_FILE = "scaler_statistics.csv"

# ======================================================
# Logging
# ======================================================

logger = logging.getLogger(__name__)


# ======================================================
# Hyperparameter Tuning & Calibration
# ======================================================

def tune_base_models(
    X_train_scaled,
    y_train,
    X_train,
    config: TrainingConfig,
    output_dir: Path,
) -> Tuple[Any, Any, Any]:
    """
    Tunes the MLP and XGBoost models using RandomizedSearchCV with
    chronological TimeSeriesSplit cross-validation. 
    
    Returns calibrated estimators for all three base models.
    """
    tscv = TimeSeriesSplit(n_splits=config.cv_folds)

    # MLP Tuning
    mlp_search = RandomizedSearchCV(
        estimator=MLPClassifier(
            random_state=config.random_seed,
            early_stopping=True,
            learning_rate="adaptive",
        ),
        param_distributions=config.mlp_grid,
        n_iter=config.mlp_search_iterations,
        cv=tscv,
        scoring="neg_log_loss",
        random_state=config.random_seed,
        n_jobs=-1,
    )
    mlp_search.fit(X_train_scaled, y_train)

    # XGBoost Tuning
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(
            random_state=config.random_seed,
            eval_metric="logloss",
            tree_method="hist",
        ),
        param_distributions=config.xgb_grid,
        n_iter=config.xgb_search_iterations,
        cv=tscv,
        scoring="neg_log_loss",
        random_state=config.random_seed,
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)

    # Logistic Regression (Baseline)
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=config.random_seed,
    )
    lr_model.fit(X_train_scaled, y_train)

    # Save tuning artifacts
    save_json(mlp_search.best_params_, output_dir / MLP_PARAMS_FILE)
    save_json(xgb_search.best_params_, output_dir / XGB_PARAMS_FILE)
    
    pd.DataFrame(mlp_search.cv_results_).to_csv(output_dir / MLP_CV_RESULTS_FILE, index=False)
    pd.DataFrame(xgb_search.cv_results_).to_csv(output_dir / XGB_CV_RESULTS_FILE, index=False)

    # Integrate Calibration
    logger.info("      Applying cross-validated calibration to base models...")
    
    mlp_cal = CalibratedClassifierCV(mlp_search.best_estimator_, cv=tscv, method='sigmoid')
    xgb_cal = CalibratedClassifierCV(xgb_search.best_estimator_, cv=tscv, method='sigmoid')
    lr_cal = CalibratedClassifierCV(lr_model, cv=tscv, method='sigmoid')

    return mlp_cal, xgb_cal, lr_cal


# ======================================================
# Final Training
# ======================================================

def retrain_on_full_data(
    X_train,
    X_val,
    y_train,
    y_val,
    X_test,
    mlp_cal,
    xgb_cal,
    lr_cal,
    output_dir: Path,
):
    """
    Retrains the calibrated base models on the combined training and 
    validation datasets.
    """
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_test_scaled_full = scaler_full.transform(X_test)

    # Save scaler stats
    pd.DataFrame({
        "Feature": X_train_full.columns,
        "Mean": scaler_full.mean_,
        "Std": scaler_full.scale_,
    }).to_csv(output_dir / SCALER_STATS_FILE, index=False)

    # Retrain 
    mlp_final = clone(mlp_cal).fit(X_train_full_scaled, y_train_full)
    xgb_final = clone(xgb_cal).fit(X_train_full, y_train_full)
    lr_final = clone(lr_cal).fit(X_train_full_scaled, y_train_full)

    return (
        mlp_final,
        xgb_final,
        lr_final,
        scaler_full,
        X_train_full,
        X_test_scaled_full,
    )