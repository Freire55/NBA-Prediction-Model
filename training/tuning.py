"""
Hyperparameter tuning and cross-validated probability calibration for NBA models.

This module optimizes hyperparameters for the base classification architectures:
- Multi-Layer Perceptron (MLP) via RandomizedSearchCV with adaptive learning rate
- Gradient Boosted Trees (XGBoost) via RandomizedSearchCV
- L2-Regularized Logistic Regression via GridSearchCV

All hyperparameter selections use chronological TimeSeriesSplit cross-validation
evaluated on negative log-loss to optimize calibrated probability distributions
rather than raw thresholded accuracy. Winning estimators undergo cross-validated
sigmoid (Platt) calibration to mitigate overconfident probabilities.
"""

import logging
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from training.config import ModelArtifacts, TrainingConfig, TrainingData
from training.utils import save_json

logger = logging.getLogger(__name__)

# ======================================================
# Output Artifact Filenames
# ======================================================

MLP_PARAMS_FILE = "mlp_best_params.json"
XGB_PARAMS_FILE = "xgb_best_params.json"
LR_PARAMS_FILE = "lr_best_params.json"

MLP_RESULTS_FILE = "mlp_cv_results.csv"
XGB_RESULTS_FILE = "xgb_cv_results.csv"
LR_RESULTS_FILE = "lr_cv_results.csv"


# ======================================================
# Architecture-Specific Tuning Functions
# ======================================================

def tune_mlp(
    X_train: Any,
    y_train: Any,
    config: TrainingConfig,
    tscv: TimeSeriesSplit,
) -> RandomizedSearchCV:
    """
    Tunes Multi-Layer Perceptron hyperparameters over the configured parameter distributions.
    Early stopping on training validation slices prevents overfitting.
    """
    search = RandomizedSearchCV(
        estimator=MLPClassifier(
            random_state=config.random_seed,
            early_stopping=True,
            n_iter_no_change=15,
            tol=1e-4,
            learning_rate="adaptive",
        ),
        param_distributions=config.mlp_grid,
        n_iter=config.mlp_search_iterations,
        cv=tscv,
        scoring="neg_log_loss",
        random_state=config.random_seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("      Best MLP CV Log Loss: %.4f", -search.best_score_)
    return search


def tune_xgboost(
    X_train: Any,
    y_train: Any,
    config: TrainingConfig,
    tscv: TimeSeriesSplit,
) -> RandomizedSearchCV:
    """
    Tunes XGBoost gradient boosted trees with histogram-based binning.
    Uses log-loss objective to penalize inaccurate probability distributions.
    """
    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            random_state=config.random_seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        ),
        param_distributions=config.xgb_grid,
        n_iter=config.xgb_search_iterations,
        cv=tscv,
        scoring="neg_log_loss",
        random_state=config.random_seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("      Best XGBoost CV Log Loss: %.4f", -search.best_score_)
    return search


def tune_logistic_regression(
    X_train: Any,
    y_train: Any,
    config: TrainingConfig,
    tscv: TimeSeriesSplit,
) -> GridSearchCV:
    """
    Exhaustively searches regularization penalty strengths (C) for Logistic Regression.
    """
    search = GridSearchCV(
        estimator=LogisticRegression(
            max_iter=1000,
            random_state=config.random_seed,
        ),
        param_grid=config.lr_grid,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("      Best Logistic Regression CV Log Loss: %.4f", -search.best_score_)
    return search


# ======================================================
# Probability Calibration
# ======================================================

def calibrate_classifier(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    tscv: TimeSeriesSplit,
) -> CalibratedClassifierCV:
    """
    Applies cross-validated Platt scaling (sigmoid calibration) to an estimator.

    Platt scaling maps raw uncalibrated margins/logits into true posterior
    win probabilities: P(Y=1|f) = 1 / (1 + exp(A*f + B)).
    """
    calibrated = CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=tscv,
        n_jobs=-1,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def calibrate_best_models(
    mlp_estimator: Any,
    xgb_estimator: Any,
    lr_estimator: Any,
    data: TrainingData,
    tscv: TimeSeriesSplit,
) -> Tuple[CalibratedClassifierCV, CalibratedClassifierCV, CalibratedClassifierCV]:
    """Fits cross-validated sigmoid probability calibration across all base models."""
    mlp_cal = calibrate_classifier(mlp_estimator, data.mlp.X_train_processed, data.y_train, tscv)
    xgb_cal = calibrate_classifier(xgb_estimator, data.xgb.X_train, data.y_train, tscv)
    lr_cal = calibrate_classifier(lr_estimator, data.lr.X_train_processed, data.y_train, tscv)
    return mlp_cal, xgb_cal, lr_cal


# ======================================================
# Serialization & Orchestration
# ======================================================

def save_tuning_results(
    mlp_search: RandomizedSearchCV,
    xgb_search: RandomizedSearchCV,
    lr_search: GridSearchCV,
    output_dir: Path,
) -> None:
    """Serializes best hyperparameters and cross-validation search dataframes."""
    save_json(mlp_search.best_params_, output_dir / MLP_PARAMS_FILE)
    save_json(xgb_search.best_params_, output_dir / XGB_PARAMS_FILE)
    save_json(lr_search.best_params_, output_dir / LR_PARAMS_FILE)

    pd.DataFrame(mlp_search.cv_results_).to_csv(output_dir / MLP_RESULTS_FILE, index=False)
    pd.DataFrame(xgb_search.cv_results_).to_csv(output_dir / XGB_RESULTS_FILE, index=False)
    pd.DataFrame(lr_search.cv_results_).to_csv(output_dir / LR_RESULTS_FILE, index=False)


def tune_base_models(
    data: TrainingData,
    config: TrainingConfig,
    output_dir: Path,
) -> Tuple[ModelArtifacts, ModelArtifacts, ModelArtifacts]:
    """
    Orchestrates hyperparameter tuning and calibration for all three base classifiers.

    Returns:
        Tuple of (mlp_artifacts, xgb_artifacts, lr_artifacts) containing calibrated models.
    """
    tscv = TimeSeriesSplit(n_splits=config.cv_folds)

    mlp_search = tune_mlp(data.mlp.X_train_processed, data.y_train, config, tscv)
    xgb_search = tune_xgboost(data.xgb.X_train, data.y_train, config, tscv)
    lr_search = tune_logistic_regression(data.lr.X_train_processed, data.y_train, config, tscv)

    logger.info("      Applying cross-validated calibration to base models...")
    mlp_calibrated, xgb_calibrated, lr_calibrated = calibrate_best_models(
        mlp_search.best_estimator_,
        xgb_search.best_estimator_,
        lr_search.best_estimator_,
        data,
        tscv,
    )

    save_tuning_results(mlp_search, xgb_search, lr_search, output_dir)

    mlp_artifacts = ModelArtifacts(feature_set=data.mlp, model=mlp_calibrated)
    xgb_artifacts = ModelArtifacts(feature_set=data.xgb, model=xgb_calibrated)
    lr_artifacts = ModelArtifacts(feature_set=data.lr, model=lr_calibrated)

    return mlp_artifacts, xgb_artifacts, lr_artifacts