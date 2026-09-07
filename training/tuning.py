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
from typing import Any, Dict, Tuple

import pandas as pd
from catboost import CatBoostClassifier
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

from training.calibration import calibrate_estimator_with_model_selection
from training.config import ModelArtifacts, TrainingConfig, TrainingData
from training.utils import save_json

logger = logging.getLogger(__name__)

# ======================================================
# Output Artifact Filenames
# ======================================================

MLP_PARAMS_FILE = "mlp_best_params.json"
XGB_PARAMS_FILE = "xgb_best_params.json"
CATBOOST_PARAMS_FILE = "catboost_best_params.json"
LR_PARAMS_FILE = "lr_best_params.json"

MLP_RESULTS_FILE = "mlp_cv_results.csv"
XGB_RESULTS_FILE = "xgb_cv_results.csv"
CATBOOST_RESULTS_FILE = "catboost_cv_results.csv"
LR_RESULTS_FILE = "lr_cv_results.csv"
CALIBRATION_REPORT_FILE = "calibration_model_selection.json"


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


def tune_catboost(
    X_train: Any,
    y_train: Any,
    config: TrainingConfig,
    tscv: TimeSeriesSplit,
) -> RandomizedSearchCV:
    """
    Tunes CatBoost gradient boosted trees with symmetric oblivious splits.
    Uses log-loss objective to optimize predictive probability distributions.
    """
    search = RandomizedSearchCV(
        estimator=CatBoostClassifier(
            random_seed=config.random_seed,
            eval_metric="Logloss",
            thread_count=-1,
            verbose=False,
        ),
        param_distributions=config.catboost_grid,
        n_iter=config.catboost_search_iterations,
        cv=tscv,
        scoring="neg_log_loss",
        random_state=config.random_seed,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    logger.info("      Best CatBoost CV Log Loss: %.4f", -search.best_score_)
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

def calibrate_best_models(
    mlp_estimator: Any,
    xgb_estimator: Any,
    catboost_estimator: Any,
    lr_estimator: Any,
    data: TrainingData,
    tscv: TimeSeriesSplit,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """
    Fits cross-validated probability calibration across all base models.
    Performs model selection between Platt Scaling and Beta Calibration for each model.
    """
    logger.info("      [Calibration] Evaluating Platt vs Beta calibration for MLP...")
    mlp_cal, mlp_meta = calibrate_estimator_with_model_selection(
        mlp_estimator, data.mlp.X_train_processed, data.y_train, tscv
    )

    logger.info("      [Calibration] Evaluating Platt vs Beta calibration for XGBoost...")
    xgb_cal, xgb_meta = calibrate_estimator_with_model_selection(
        xgb_estimator, data.xgb.X_train, data.y_train, tscv
    )

    logger.info("      [Calibration] Evaluating Platt vs Beta calibration for CatBoost...")
    cb_cal, cb_meta = calibrate_estimator_with_model_selection(
        catboost_estimator, data.catboost.X_train, data.y_train, tscv
    )

    logger.info("      [Calibration] Evaluating Platt vs Beta calibration for Logistic Regression...")
    lr_cal, lr_meta = calibrate_estimator_with_model_selection(
        lr_estimator, data.lr.X_train_processed, data.y_train, tscv
    )

    calibration_report = {
        "MLP": mlp_meta,
        "XGBoost": xgb_meta,
        "CatBoost": cb_meta,
        "Logistic_Regression": lr_meta,
    }
    return mlp_cal, xgb_cal, cb_cal, lr_cal, calibration_report


# ======================================================
# Serialization & Orchestration
# ======================================================

def save_tuning_results(
    mlp_search: RandomizedSearchCV,
    xgb_search: RandomizedSearchCV,
    catboost_search: RandomizedSearchCV,
    lr_search: GridSearchCV,
    calibration_report: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Serializes best hyperparameters, cross-validation dataframes, and calibration selection."""
    save_json(mlp_search.best_params_, output_dir / MLP_PARAMS_FILE)
    save_json(xgb_search.best_params_, output_dir / XGB_PARAMS_FILE)
    save_json(catboost_search.best_params_, output_dir / CATBOOST_PARAMS_FILE)
    save_json(lr_search.best_params_, output_dir / LR_PARAMS_FILE)
    save_json(calibration_report, output_dir / CALIBRATION_REPORT_FILE)

    pd.DataFrame(mlp_search.cv_results_).to_csv(output_dir / MLP_RESULTS_FILE, index=False)
    pd.DataFrame(xgb_search.cv_results_).to_csv(output_dir / XGB_RESULTS_FILE, index=False)
    pd.DataFrame(catboost_search.cv_results_).to_csv(output_dir / CATBOOST_RESULTS_FILE, index=False)
    pd.DataFrame(lr_search.cv_results_).to_csv(output_dir / LR_RESULTS_FILE, index=False)


def tune_base_models(
    data: TrainingData,
    config: TrainingConfig,
    output_dir: Path,
) -> Tuple[ModelArtifacts, ModelArtifacts, ModelArtifacts, ModelArtifacts]:
    """
    Orchestrates hyperparameter tuning and calibration for all four base classifiers:
    MLP, XGBoost, CatBoost, and Logistic Regression.

    Returns:
        Tuple of (mlp_artifacts, xgb_artifacts, catboost_artifacts, lr_artifacts)
        containing tuned and calibrated models.
    """
    tscv = TimeSeriesSplit(n_splits=config.cv_folds)

    mlp_search = tune_mlp(data.mlp.X_train_processed, data.y_train, config, tscv)
    xgb_search = tune_xgboost(data.xgb.X_train, data.y_train, config, tscv)
    catboost_search = tune_catboost(data.catboost.X_train, data.y_train, config, tscv)
    lr_search = tune_logistic_regression(data.lr.X_train_processed, data.y_train, config, tscv)

    logger.info("      Applying leak-free calibration model selection (Platt vs. Beta)...")
    mlp_cal, xgb_cal, cb_cal, lr_cal, cal_report = calibrate_best_models(
        mlp_search.best_estimator_,
        xgb_search.best_estimator_,
        catboost_search.best_estimator_,
        lr_search.best_estimator_,
        data,
        tscv,
    )

    save_tuning_results(mlp_search, xgb_search, catboost_search, lr_search, cal_report, output_dir)

    mlp_artifacts = ModelArtifacts(feature_set=data.mlp, model=mlp_cal)
    xgb_artifacts = ModelArtifacts(feature_set=data.xgb, model=xgb_cal)
    catboost_artifacts = ModelArtifacts(feature_set=data.catboost, model=cb_cal)
    lr_artifacts = ModelArtifacts(feature_set=data.lr, model=lr_cal)

    return mlp_artifacts, xgb_artifacts, catboost_artifacts, lr_artifacts