"""
Configuration objects for the NBA prediction training pipeline.

This module centralizes:

- Training configuration
- Hyperparameter search spaces
- Dataset split settings
- Runtime artifact storage
- Experiment metadata generation
"""

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


# ======================================================
# Universal JSON Serialization
# ======================================================

class NumpyEncoder(json.JSONEncoder):
    """Safely converts NumPy arrays and scalars to native Python types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


# ======================================================
# Dataset Containers
# ======================================================

@dataclass
class DatasetSummary:
    """Strongly typed summary metrics for the prepared dataset."""
    train_games: int
    validation_games: int
    test_games: int
    
    lr_feature_names: list[str]
    xgb_feature_names: list[str]
    mlp_feature_names: list[str]


@dataclass
class FeatureSet:
    """Container for a model's specific dataset representation."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame

    X_train_full: pd.DataFrame | None = None

    X_train_processed: np.ndarray | None = None
    X_val_processed: np.ndarray | None = None
    X_test_processed: np.ndarray | None = None

    X_train_full_processed: np.ndarray | None = None

    scaler: StandardScaler | None = None

    feature_names: list[str] | None = None


@dataclass
class TrainingData:
    """Return payload from the data preparation stage."""
    mlp: FeatureSet
    xgb: FeatureSet
    lr: FeatureSet

    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    summary: DatasetSummary


# ======================================================
# Configurations
# ======================================================

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
    # Feature Configuration (Heterogeneous)
    # ======================================================

    lr_prefixes: list[str] = field(
        default_factory=lambda: ["DELTA_"]
    )
    
    xgb_prefixes: list[str] = field(
        default_factory=lambda: ["HOME_", "AWAY_"]
    )
    
    mlp_prefixes: list[str] = field(
        default_factory=lambda: ["DELTA_", "EMBED_DELTA_"]
    )

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
    lr_search_iterations: int = 12

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

    lr_grid: dict[str, Any] = field(
        default_factory=lambda: {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "liblinear"],
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Returns the configuration as a serializable dictionary."""
        return asdict(self)


# ======================================================
# Artifact Containers
# ======================================================

@dataclass
class ModelArtifacts:
    """Groups all artifacts specific to a single model architecture."""
    feature_set: FeatureSet | None = None

    model: BaseEstimator | None = None
    final_model: BaseEstimator | None = None


@dataclass
class TrainingArtifacts:
    """
    Central state container for the training pipeline.
    """

    config: TrainingConfig
    output_dir: Path

    data: TrainingData | None = None

    lr: ModelArtifacts = field(default_factory=ModelArtifacts)
    xgb: ModelArtifacts = field(default_factory=ModelArtifacts)
    mlp: ModelArtifacts = field(default_factory=ModelArtifacts)

    ensemble_weights: np.ndarray | None = None


def get_experiment_metadata() -> dict[str, str]:
    """Returns metadata describing the current training run."""
    return {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dataset_version": "v2.0-heterogeneous",
    }