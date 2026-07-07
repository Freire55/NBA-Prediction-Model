"""
Hyperparameter tuning utilities for the NBA prediction pipeline.

This module performs hyperparameter optimization for the MLP and
XGBoost classifiers using RandomizedSearchCV with chronological
cross-validation. Logistic Regression is trained directly using
its default configuration since it has relatively few hyperparameters.

The best models and search results are saved for reproducibility.

Functions:
    tune_base_models()
"""

from pathlib import Path
from typing import Any, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from training.config import TrainingConfig
from training.utils import save_json

logger = logging.getLogger(__name__)

# ======================================================
# Output Files
# ======================================================

MLP_PARAMS_FILE = "mlp_best_params.json"
XGB_PARAMS_FILE = "xgb_best_params.json"

MLP_RESULTS_FILE = "mlp_cv_results.csv"
XGB_RESULTS_FILE = "xgb_cv_results.csv"


# ======================================================
# Hyperparameter Tuning
# ======================================================

def tune_base_models(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_train: pd.DataFrame,
    config: TrainingConfig,
    output_dir: Path,
) -> Tuple[Any, Any, Any]:
    """
    Tunes the MLP and XGBoost classifiers using randomized
    hyperparameter search with chronological cross-validation.

    Logistic Regression is trained using its default configuration
    as a strong linear baseline.

    Args:
        X_train_scaled:
            Standardized training features for MLP and Logistic Regression.
        y_train:
            Training labels.
        X_train:
            Original (unscaled) training features for XGBoost.
        config:
            Training configuration containing search spaces and
            optimization settings.
        output_dir:
            Directory where tuning artifacts are saved.

    Returns:
        Tuple containing:
            - Best MLP model
            - Best XGBoost model
            - Trained Logistic Regression model
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

    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=config.random_seed,
    )

    # --------------------------------------------------
    # Model training
    # --------------------------------------------------

    mlp_search.fit(X_train_scaled, y_train)
    xgb_search.fit(X_train, y_train)
    lr_model.fit(X_train_scaled, y_train)

    logger.info(
        f"      Best MLP CV Log Loss: {-mlp_search.best_score_:.4f}"
    )
    logger.info(
        f"      Best XGBoost CV Log Loss: {-xgb_search.best_score_:.4f}"
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

    return (
        mlp_search.best_estimator_,
        xgb_search.best_estimator_,
        lr_model,
    )