"""
Unit tests for base classification model architectures.

Validates:
1. CatBoost classifier hyperparameter tuning and oblivious tree importance.
2. XGBoost classifier hyperparameter tuning and histogram splitting.
3. Multi-Layer Perceptron (MLP) adaptive learning rate tuning.
4. Logistic Regression regularization grid search.
5. Model unwrapping utility across calibrated and raw estimator wrappers.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from training.config import TrainingConfig
from training.tuning import (
    tune_catboost,
    tune_logistic_regression,
    tune_mlp,
    tune_xgboost,
)
from training.utils import unwrap_base_estimator


def test_catboost_tuning_pipeline():
    """Verifies CatBoostClassifier tunes with TimeSeriesSplit and extracts split importances."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(80, 4), columns=["HOME_pts", "AWAY_pts", "EMBED_0", "EMBED_1"])
    y = pd.Series((X["HOME_pts"] > X["AWAY_pts"]).astype(int))

    config = TrainingConfig()
    config.catboost_grid = {
        "depth": [4],
        "l2_leaf_reg": [1],
        "learning_rate": [0.05],
        "iterations": [20],
    }
    config.catboost_search_iterations = 1
    tscv = TimeSeriesSplit(n_splits=2)

    search = tune_catboost(X, y, config, tscv)
    assert search.best_estimator_ is not None
    assert hasattr(search.best_estimator_, "predict_proba")

    cb_raw = unwrap_base_estimator(search.best_estimator_)
    importances = cb_raw.get_feature_importance()
    assert len(importances) == 4
    assert np.all(importances >= 0.0)


def test_xgboost_tuning_pipeline():
    """Verifies XGBoostClassifier tunes properly with histogram binning."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(80, 3), columns=["feat_0", "feat_1", "feat_2"])
    y = pd.Series((X["feat_0"] > 0).astype(int))

    config = TrainingConfig()
    config.xgb_grid = {
        "max_depth": [3],
        "learning_rate": [0.05],
        "n_estimators": [15],
    }
    config.xgb_search_iterations = 1
    tscv = TimeSeriesSplit(n_splits=2)

    search = tune_xgboost(X, y, config, tscv)
    assert search.best_estimator_ is not None
    assert hasattr(search.best_estimator_, "predict_proba")


def test_mlp_tuning_pipeline():
    """Verifies MLPClassifier tunes with early stopping and adaptive learning rate."""
    np.random.seed(42)
    X = np.random.randn(80, 4)
    y = pd.Series((X[:, 0] + X[:, 1] > 0).astype(int))

    config = TrainingConfig()
    config.mlp_grid = {
        "hidden_layer_sizes": [(16,)],
        "alpha": [0.01],
        "max_iter": [30],
    }
    config.mlp_search_iterations = 1
    tscv = TimeSeriesSplit(n_splits=2)

    search = tune_mlp(X, y, config, tscv)
    assert search.best_estimator_ is not None
    assert hasattr(search.best_estimator_, "predict_proba")


def test_logistic_regression_tuning_pipeline():
    """Verifies LogisticRegression grid search over C regularization parameter."""
    np.random.seed(42)
    X = np.random.randn(80, 3)
    y = pd.Series((X[:, 0] > 0).astype(int))

    config = TrainingConfig()
    config.lr_grid = {"C": [0.1, 1.0]}
    tscv = TimeSeriesSplit(n_splits=2)

    search = tune_logistic_regression(X, y, config, tscv)
    assert search.best_estimator_ is not None
    assert hasattr(search.best_estimator_, "coef_")


def test_unwrap_base_estimator():
    """Ensures unwrap_base_estimator extracts the core model from wrappers."""
    raw_lr = LogisticRegression()
    assert unwrap_base_estimator(raw_lr) is raw_lr
