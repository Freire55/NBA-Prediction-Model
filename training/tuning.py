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
from typing import Any, Dict, Optional, Tuple

import numpy as np
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
# Early-Stopping Estimator Wrappers
# ======================================================

class EarlyStoppingXGBClassifier(XGBClassifier):
    """
    XGBClassifier with built-in chronological early stopping.
    Automatically splits off the trailing fraction of training data as an evaluation set.
    """

    def __init__(
        self,
        early_stopping_rounds: Optional[int] = 25,
        val_fraction: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.val_fraction = val_fraction
        super().__init__(early_stopping_rounds=early_stopping_rounds, **kwargs)

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params.pop("val_fraction", None)
        return params

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any:
        n = len(X)
        split = max(int(n * (1.0 - self.val_fraction)), 1)
        if split < n and self.early_stopping_rounds:
            X_tr = X.iloc[:split] if hasattr(X, "iloc") else X[:split]
            X_val = X.iloc[split:] if hasattr(X, "iloc") else X[split:]
            y_tr = y.iloc[:split] if hasattr(y, "iloc") else y[:split]
            y_val = y.iloc[split:] if hasattr(y, "iloc") else y[split:]
            fit_kwargs = dict(kwargs)
            if "sample_weight" in fit_kwargs and fit_kwargs["sample_weight"] is not None:
                sw = fit_kwargs["sample_weight"]
                fit_kwargs["sample_weight"] = sw.iloc[:split] if hasattr(sw, "iloc") else sw[:split]
            return super().fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                **fit_kwargs,
            )
        return super().fit(X, y, verbose=False, **kwargs)


class EarlyStoppingCatBoostClassifier(CatBoostClassifier):
    """
    CatBoostClassifier with built-in chronological early stopping.
    Automatically splits off the trailing fraction of training data as an evaluation set.
    """

    def __init__(
        self,
        early_stopping_rounds: int = 25,
        val_fraction: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        super().__init__(**kwargs)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any:
        n = len(X)
        split = max(int(n * (1.0 - self.val_fraction)), 1)
        if split < n and self.early_stopping_rounds:
            X_tr = X.iloc[:split] if hasattr(X, "iloc") else X[:split]
            X_val = X.iloc[split:] if hasattr(X, "iloc") else X[split:]
            y_tr = y.iloc[:split] if hasattr(y, "iloc") else y[:split]
            y_val = y.iloc[split:] if hasattr(y, "iloc") else y[split:]
            fit_kwargs = dict(kwargs)
            if "sample_weight" in fit_kwargs and fit_kwargs["sample_weight"] is not None:
                sw = fit_kwargs["sample_weight"]
                fit_kwargs["sample_weight"] = sw.iloc[:split] if hasattr(sw, "iloc") else sw[:split]
            return super().fit(
                X_tr,
                y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
                **fit_kwargs,
            )
        return super().fit(X, y, verbose=False, **kwargs)


# ======================================================
# Search Factory Helper (Successive Halving vs Random)
# ======================================================

def _create_search_cv(
    estimator: Any,
    param_distributions: Dict[str, Any],
    n_iter: int,
    cv: TimeSeriesSplit,
    config: TrainingConfig,
    n_jobs: int = -1,
) -> Any:
    """Instantiates HalvingRandomSearchCV or standard RandomizedSearchCV based on config."""
    if getattr(config, "use_halving_search", True):
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV

        return HalvingRandomSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            cv=cv,
            scoring="neg_log_loss",
            random_state=config.random_seed,
            n_candidates=n_iter,
            min_resources="exhaust",
            factor=getattr(config, "halving_factor", 2),
            n_jobs=n_jobs,
        )

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_log_loss",
        random_state=config.random_seed,
        n_jobs=n_jobs,
    )


# ======================================================
# Architecture-Specific Tuning Functions
# ======================================================

def tune_mlp(
    X_train: Any,
    y_train: Any,
    config: TrainingConfig,
    tscv: TimeSeriesSplit,
    sample_weight: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> Any:
    """
    Tunes Multi-Layer Perceptron hyperparameters over the configured parameter distributions.
    Evaluates architectures on the full sample split with validation-based early stopping
    to preserve deep multi-layer representation learning.
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
        n_jobs=n_jobs,
    )
    if sample_weight is not None:
        search.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        search.fit(X_train, y_train)
    logger.info("      Best MLP CV Log Loss: %.4f", -search.best_score_)
    return search


def tune_xgboost(
    X_train: Any,
    y_train: Any,
    config: TrainingConfig,
    tscv: TimeSeriesSplit,
    sample_weight: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> Any:
    """
    Tunes XGBoost gradient boosted trees with early stopping and histogram binning.
    Uses log-loss objective to penalize inaccurate probability distributions.
    """
    search = _create_search_cv(
        estimator=EarlyStoppingXGBClassifier(
            early_stopping_rounds=getattr(config, "early_stopping_rounds", 25),
            val_fraction=getattr(config, "early_stopping_val_fraction", 0.1),
            random_state=config.random_seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        ),
        param_distributions=config.xgb_grid,
        n_iter=config.xgb_search_iterations,
        cv=tscv,
        config=config,
        n_jobs=n_jobs,
    )
    if sample_weight is not None:
        search.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        search.fit(X_train, y_train)
    logger.info("      Best XGBoost CV Log Loss: %.4f", -search.best_score_)
    return search


def tune_catboost(
    X_train: Any,
    y_train: Any,
    config: TrainingConfig,
    tscv: TimeSeriesSplit,
    sample_weight: Optional[np.ndarray] = None,
    thread_count: int = -1,
) -> Any:
    """
    Tunes CatBoost gradient boosted trees with early stopping and symmetric oblivious splits.
    Uses log-loss objective to optimize predictive probability distributions.
    """
    search = _create_search_cv(
        estimator=EarlyStoppingCatBoostClassifier(
            early_stopping_rounds=getattr(config, "early_stopping_rounds", 25),
            val_fraction=getattr(config, "early_stopping_val_fraction", 0.1),
            random_seed=config.random_seed,
            eval_metric="Logloss",
            thread_count=thread_count,
            verbose=False,
        ),
        param_distributions=config.catboost_grid,
        n_iter=config.catboost_search_iterations,
        cv=tscv,
        config=config,
        n_jobs=1,
    )
    if sample_weight is not None:
        search.fit(X_train, y_train, sample_weight=sample_weight)
    else:
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
    lr_estimator: Optional[Any],
    data: TrainingData,
    tscv: TimeSeriesSplit,
    config: Optional[TrainingConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """
    Fits cross-validated probability calibration across available base models.
    Performs model selection between Platt Scaling and Beta Calibration for each model.
    Applies selective recency weighting: MLP receives sample weights, while tree models remain unweighted.
    """
    sw = sample_weight if sample_weight is not None else (
        getattr(data, "sample_weights_train", None)
        if (config is None or getattr(config, "use_recency_weights", True))
        else None
    )

    logger.info("      [Calibration] Evaluating Platt vs Beta vs Spline calibration for MLP...")
    mlp_cal, mlp_meta = calibrate_estimator_with_model_selection(
        mlp_estimator, data.mlp.X_train_processed, data.y_train, tscv, sample_weight=sw
    )

    logger.info("      [Calibration] Evaluating Platt vs Beta vs Spline calibration for XGBoost...")
    xgb_cal, xgb_meta = calibrate_estimator_with_model_selection(
        xgb_estimator, data.xgb.X_train, data.y_train, tscv, sample_weight=None
    )

    logger.info("      [Calibration] Evaluating Platt vs Beta vs Spline calibration for CatBoost...")
    cb_cal, cb_meta = calibrate_estimator_with_model_selection(
        catboost_estimator, data.catboost.X_train, data.y_train, tscv, sample_weight=None
    )

    lr_cal = None
    lr_meta = {"method": "none", "selected_log_loss": None}
    if lr_estimator is not None and getattr(config, "include_logistic_regression", False):
        logger.info("      [Calibration] Evaluating Platt vs Beta vs Spline calibration for Logistic Regression...")
        lr_cal, lr_meta = calibrate_estimator_with_model_selection(
            lr_estimator, data.lr.X_train_processed, data.y_train, tscv, sample_weight=None
        )

    calibration_report = {
        "MLP": mlp_meta,
        "XGBoost": xgb_meta,
        "CatBoost": cb_meta,
    }
    if lr_cal is not None:
        calibration_report["Logistic_Regression"] = lr_meta

    return mlp_cal, xgb_cal, cb_cal, lr_cal, calibration_report


# ======================================================
# Serialization & Orchestration
# ======================================================

def save_tuning_results(
    mlp_search: Any,
    xgb_search: Any,
    catboost_search: Any,
    lr_search: Optional[Any],
    calibration_report: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Serializes best hyperparameters, cross-validation dataframes, and calibration selection."""
    save_json(mlp_search.best_params_, output_dir / MLP_PARAMS_FILE)
    save_json(xgb_search.best_params_, output_dir / XGB_PARAMS_FILE)
    save_json(catboost_search.best_params_, output_dir / CATBOOST_PARAMS_FILE)
    if lr_search is not None:
        save_json(lr_search.best_params_, output_dir / LR_PARAMS_FILE)
    save_json(calibration_report, output_dir / CALIBRATION_REPORT_FILE)

    pd.DataFrame(mlp_search.cv_results_).to_csv(output_dir / MLP_RESULTS_FILE, index=False)
    pd.DataFrame(xgb_search.cv_results_).to_csv(output_dir / XGB_RESULTS_FILE, index=False)
    pd.DataFrame(catboost_search.cv_results_).to_csv(output_dir / CATBOOST_RESULTS_FILE, index=False)
    if lr_search is not None:
        pd.DataFrame(lr_search.cv_results_).to_csv(output_dir / LR_RESULTS_FILE, index=False)


def tune_base_models(
    data: TrainingData,
    config: TrainingConfig,
    output_dir: Path,
) -> Tuple[ModelArtifacts, ModelArtifacts, ModelArtifacts, ModelArtifacts]:
    """
    Orchestrates hyperparameter tuning and calibration across base classifiers.
    Applies architecture-selective recency weighting exclusively to MLP while
    keeping decision trees and linear models unweighted.
    If include_logistic_regression is False, skips LR to accelerate training.

    Returns:
        Tuple of (mlp_artifacts, xgb_artifacts, catboost_artifacts, lr_artifacts)
        containing tuned and calibrated models.
    """
    tscv = TimeSeriesSplit(n_splits=config.cv_folds)
    sw = (
        getattr(data, "sample_weights_train", None)
        if getattr(config, "use_recency_weights", True)
        else None
    )

    mlp_search = tune_mlp(data.mlp.X_train_processed, data.y_train, config, tscv, sample_weight=sw)
    xgb_search = tune_xgboost(data.xgb.X_train, data.y_train, config, tscv, sample_weight=None)
    catboost_search = tune_catboost(data.catboost.X_train, data.y_train, config, tscv, sample_weight=None)

    lr_search = None
    if getattr(config, "include_logistic_regression", False):
        lr_search = tune_logistic_regression(data.lr.X_train_processed, data.y_train, config, tscv)
    else:
        logger.info("      [Pruning] Binary Logistic Regression is disabled (0.00% ensemble contribution).")

    logger.info("      Applying leak-free calibration model selection (Platt vs. Beta vs. Spline)...")
    mlp_cal, xgb_cal, cb_cal, lr_cal, cal_report = calibrate_best_models(
        mlp_estimator=mlp_search.best_estimator_,
        xgb_estimator=xgb_search.best_estimator_,
        catboost_estimator=catboost_search.best_estimator_,
        lr_estimator=lr_search.best_estimator_ if lr_search is not None else None,
        data=data,
        tscv=tscv,
        config=config,
        sample_weight=sw,
    )

    save_tuning_results(mlp_search, xgb_search, catboost_search, lr_search, cal_report, output_dir)

    mlp_artifacts = ModelArtifacts(feature_set=data.mlp, model=mlp_cal)
    xgb_artifacts = ModelArtifacts(feature_set=data.xgb, model=xgb_cal)
    catboost_artifacts = ModelArtifacts(feature_set=data.catboost, model=cb_cal)
    lr_artifacts = ModelArtifacts(feature_set=data.lr, model=lr_cal)

    return mlp_artifacts, xgb_artifacts, catboost_artifacts, lr_artifacts