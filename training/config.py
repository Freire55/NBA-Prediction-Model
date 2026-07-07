"""
Configuration objects for the NBA prediction training pipeline.

This module centralizes:

- Training configuration
- Hyperparameter search spaces
- Dataset split settings
- Runtime artifact storage
- Experiment metadata generation
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainingConfig:
    """
    Stores all configurable parameters used throughout the training pipeline.
    """

    # ======================================================
    # Dataset Splits
    # ======================================================

    train_end: str = "22018"
    validation_end: str = "22020"

    # ======================================================
    # Feature Configuration
    # ======================================================

    feature_prefix: str = "DELTA_"

    extra_features: list[str] = field(
        default_factory=lambda: [
            "REST_ADVANTAGE",
            "HOME_B2B",
            "AWAY_B2B",
            "SEASON_YEAR",
        ]
    )

    # ======================================================
    # General Machine Learning
    # ======================================================

    random_seed: int = 42
    cv_folds: int = 5

    # ======================================================
    # Explainability & Evaluation
    # ======================================================

    n_shap_samples: int = 3000
    permutation_repeats: int = 20
    calibration_bins: int = 10

    # ======================================================
    # Hyperparameter Search
    # ======================================================

    mlp_search_iterations: int = 30
    xgb_search_iterations: int = 20

    # ======================================================
    # Hyperparameter Search Spaces
    # ======================================================

    mlp_grid: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_layer_sizes": [
                (64,),
                (128,),
                (64, 32),
                (128, 64),
                (128, 64, 32),
                (256, 128, 64),
            ],
            "activation": ["relu", "tanh"],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [0.0005, 0.001, 0.005],
            "batch_size": [32, 64, 128],
            "max_iter": [1000],
        }
    )

    xgb_grid: dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [5, 7, 9],
            "reg_lambda": [1, 5, 10, 20],
            "reg_alpha": [0, 0.1, 1, 5],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "n_estimators": [100, 300, 500],
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Returns the configuration as a serializable dictionary."""
        return asdict(self)


@dataclass
class TrainingArtifacts:
    """
    Stores datasets, trained models, scalers, and intermediate
    artifacts produced throughout the training pipeline.

    Passing a single TrainingArtifacts instance greatly reduces
    function argument lists while keeping the pipeline state
    centrally organized.
    """

    config: TrainingConfig
    output_dir: Path

    # ======================================================
    # Dataset
    # ======================================================

    X_train: pd.DataFrame | None = None
    y_train: pd.Series | None = None

    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None

    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None

    X_train_scaled: np.ndarray | None = None
    X_val_scaled: np.ndarray | None = None
    X_test_scaled_full: np.ndarray | None = None

    X_train_full: pd.DataFrame | None = None

    features: list[str] | None = None

    # ======================================================
    # Tuned Models
    # ======================================================

    mlp_model: BaseEstimator | None = None
    xgb_model: BaseEstimator | None = None
    lr_model: BaseEstimator | None = None

    # ======================================================
    # Final Retrained Models
    # ======================================================

    mlp_final: BaseEstimator | None = None
    xgb_final: BaseEstimator | None = None
    lr_final: BaseEstimator | None = None

    # ======================================================
    # Scalers & Ensemble
    # ======================================================

    scaler_val: StandardScaler | None = None
    scaler_full: StandardScaler | None = None

    ensemble_weights: np.ndarray | None = None


def get_experiment_metadata() -> dict[str, str]:
    """
    Returns metadata describing the current training run.
    """

    return {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dataset_version": "v1.0",
    }