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
import os
import platform
import subprocess
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
    """
    JSON encoder capable of serializing NumPy objects.

    Training artifacts frequently contain NumPy arrays and scalar types,
    which are not directly serializable by Python's default JSON encoder.
    This custom encoder converts them into native Python objects before
    writing experiment metadata to disk.
    """
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
    """
    Stores high-level information about the prepared datasets.

    Used primarily for logging and experiment tracking so every training
    run records exactly how many games and features were used.
    """

    train_games: int
    validation_games: int
    test_games: int
    
    lr_feature_names: list[str]
    xgb_feature_names: list[str]
    mlp_feature_names: list[str]


@dataclass
class FeatureSet:
    """
    Container holding every representation of a feature matrix for one model.

    A model may require:

    • Raw pandas DataFrames
    • Standardized NumPy arrays
    • A fitted scaler
    • Feature names

    Keeping everything together avoids passing dozens of variables
    throughout the pipeline.
    """

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
    """
    Master container returned by the data preparation stage.

    After feature engineering, every model receives its own FeatureSet,
    while all models share the same target variables and dataset summary.
    """

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
    Central configuration object for the entire training pipeline.

    Every configurable aspect of the project is defined here, including:

    • Chronological dataset splits
    • Feature selection rules
    • Hyperparameter search spaces
    • Cross-validation settings
    • Explainability configuration

    Centralizing these parameters guarantees reproducible experiments and
    makes changing the training pipeline possible without modifying the
    implementation code.
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
        default_factory=lambda: ["DELTA_", "EMBED_"]
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
    # Feature Pruning 
    # ======================================================

    features_to_remove: dict[str, list[str]] = field(
        default_factory=lambda: {
            "mlp": [],
            "xgb": [],
            "lr": []
        }
    )

    # ======================================================
    # General Machine Learning
    # ======================================================

    random_seed: int = 42
    cv_folds: int = 5

    # ======================================================
    # Explainability & Evaluation
    # ======================================================

    n_shap_samples: int = 1000
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
            "max_iter": [500],
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
    """
    Groups every artifact associated with one model architecture.

    Separating artifacts by model keeps the pipeline modular and greatly
    simplifies saving, evaluation, explainability, and ensemble creation.
    """
    feature_set: FeatureSet | None = None

    model: BaseEstimator | None = None
    final_model: BaseEstimator | None = None


@dataclass
class TrainingArtifacts:
    """
    Central state object passed throughout the training pipeline.

    Rather than returning dozens of independent objects between stages,
    every component reads from and writes to this shared container.

    It effectively acts as the project's in-memory workspace.
    """

    config: TrainingConfig
    output_dir: Path

    data: TrainingData | None = None

    lr: ModelArtifacts = field(default_factory=ModelArtifacts)
    xgb: ModelArtifacts = field(default_factory=ModelArtifacts)
    mlp: ModelArtifacts = field(default_factory=ModelArtifacts)

    ensemble_weights: np.ndarray | None = None


def get_experiment_metadata(data_dir: Path | None = None) -> dict[str, Any]:
    """
    Collects runtime information describing the current experiment.

    This metadata is saved alongside every trained model to improve
    reproducibility and simplify future comparisons between runs.
    """
    git_info: dict[str, Any] = {
        "git_commit": "unknown",
        "git_branch": "unknown",
        "git_dirty": False,
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        git_info = {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": dirty,
        }
    except Exception:
        pass

    metadata: dict[str, Any] = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "dataset_version": "v2.0-heterogeneous",
        **git_info,
    }

    if data_dir is not None:
        dataset_path = Path(data_dir) / "ml_ready_matchups_players.csv"
        if dataset_path.exists():
            import hashlib
            hasher = hashlib.sha256()
            with open(dataset_path, "rb") as f:
                hasher.update(f.read(1024 * 1024))
            metadata["dataset_fingerprint_sha256_1mb"] = hasher.hexdigest()

    return metadata