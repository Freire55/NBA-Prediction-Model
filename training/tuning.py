"""
Hyperparameter tuning utilities for the NBA prediction pipeline.

This module performs hyperparameter optimization for the MLP and
XGBoost classifiers using RandomizedSearchCV with chronological
cross-validation. Logistic Regression is trained using GridSearchCV.

The best models and search results are saved for reproducibility,
and all resulting estimators are probability-calibrated.

Functions:
    tune_base_models()
"""

import logging
from pathlib import Path
from typing import Tuple

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
# Output Files
# ======================================================

MLP_PARAMS_FILE = "mlp_best_params.json"
XGB_PARAMS_FILE = "xgb_best_params.json"
LR_PARAMS_FILE = "lr_best_params.json"

MLP_RESULTS_FILE = "mlp_cv_results.csv"
XGB_RESULTS_FILE = "xgb_cv_results.csv"
LR_RESULTS_FILE = "lr_cv_results.csv"


# ======================================================
# Hyperparameter Tuning
# ======================================================

def tune_base_models(
    data: TrainingData,
    config: TrainingConfig,
    output_dir: Path,
) -> Tuple[ModelArtifacts, ModelArtifacts, ModelArtifacts]:
    """
    Tunes the MLP, XGBoost, and Logistic Regression classifiers using
    chronological cross-validation.

    The resulting best estimators are probability-calibrated using
    CalibratedClassifierCV before being bundled into ModelArtifacts.

    Args:
        data:
            The prepared training data containing model-specific feature sets.
        config:
            Training configuration containing search spaces.
        output_dir:
            Directory where tuning artifacts are saved.

    Returns:
        Tuple containing:
            - MLP ModelArtifacts
            - XGBoost ModelArtifacts
            - Logistic Regression ModelArtifacts
    """
    # --------------------------------------------------
    # Time-series cross-validation
    # --------------------------------------------------

    tscv = TimeSeriesSplit(
        n_splits=config.cv_folds
    )

    # --------------------------------------------------
    # Hyperparameter searches
    # --------------------------------------------------

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

    lr_search = GridSearchCV(
        estimator=LogisticRegression(
            max_iter=1000,
            random_state=config.random_seed,
        ),
        param_grid=config.lr_grid,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=-1,
    )

    # --------------------------------------------------
    # Model training
    # --------------------------------------------------

    mlp_search.fit(
        data.mlp.X_train_processed,
        data.y_train,
    )
    
    xgb_search.fit(
        data.xgb.X_train,
        data.y_train,
    )
    
    lr_search.fit(
        data.lr.X_train_processed,
        data.y_train,
    )

    logger.info(
        f"      Best MLP CV Log Loss: {-mlp_search.best_score_:.4f}"
    )
    logger.info(
        f"      Best XGBoost CV Log Loss: {-xgb_search.best_score_:.4f}"
    )
    logger.info(
        f"      Best Logistic Regression CV Log Loss: {-lr_search.best_score_:.4f}"
    )

    # --------------------------------------------------
    # Probability calibration (Platt / sigmoid scaling)
    # --------------------------------------------------
    
    logger.info("      Applying cross-validated calibration to base models...")

    mlp_calibrated = CalibratedClassifierCV(
        clone(mlp_search.best_estimator_),
        method="sigmoid",
        cv=tscv,
        n_jobs=-1
    )
    mlp_calibrated.fit(
        data.mlp.X_train_processed,
        data.y_train,
    )

    xgb_calibrated = CalibratedClassifierCV(
        clone(xgb_search.best_estimator_),
        method="sigmoid",
        cv=tscv,
        n_jobs=-1
    )
    xgb_calibrated.fit(
        data.xgb.X_train,
        data.y_train,
    )

    lr_calibrated = CalibratedClassifierCV(
        clone(lr_search.best_estimator_),
        method="sigmoid",
        cv=tscv,
        n_jobs=-1
    )
    lr_calibrated.fit(
        data.lr.X_train_processed,
        data.y_train,
    )

    # --------------------------------------------------
    # Save tuning artifacts
    # --------------------------------------------------

    save_json(
        mlp_search.best_params_,
        output_dir / MLP_PARAMS_FILE,
    )

    save_json(
        xgb_search.best_params_,
        output_dir / XGB_PARAMS_FILE,
    )

    save_json(
        lr_search.best_params_,
        output_dir / LR_PARAMS_FILE,
    )

    pd.DataFrame(
        mlp_search.cv_results_
    ).to_csv(
        output_dir / MLP_RESULTS_FILE,
        index=False,
    )

    pd.DataFrame(
        xgb_search.cv_results_
    ).to_csv(
        output_dir / XGB_RESULTS_FILE,
        index=False,
    )

    pd.DataFrame(
        lr_search.cv_results_
    ).to_csv(
        output_dir / LR_RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Package into ModelArtifacts
    # --------------------------------------------------

    mlp_artifacts = ModelArtifacts(
        feature_set=data.mlp,
        model=mlp_calibrated,
    )
    
    xgb_artifacts = ModelArtifacts(
        feature_set=data.xgb,
        model=xgb_calibrated,
    )
    
    lr_artifacts = ModelArtifacts(
        feature_set=data.lr,
        model=lr_calibrated,
    )

    return mlp_artifacts, xgb_artifacts, lr_artifacts