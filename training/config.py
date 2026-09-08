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
    margin_feature_names: list[str] = field(default_factory=list)
    catboost_feature_names: list[str] = field(default_factory=list)


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

    margin: FeatureSet | None = None
    catboost: FeatureSet | None = None
    y_margin_train: pd.Series | None = None
    y_margin_val: pd.Series | None = None
    y_margin_test: pd.Series | None = None
    sample_weights_train: np.ndarray | None = None
    dates_train: pd.Series | None = None
    dates_val: pd.Series | None = None
    dates_test: pd.Series | None = None


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

    train_end: str = "22020"
    validation_end: str = "22021"

    # Exponential recency sample weighting
    use_recency_weights: bool = True
    recency_half_life_years: float = 7.0

    # ======================================================
    # Feature Configuration (Heterogeneous)
    # ======================================================


    lr_prefixes: list[str] = field(
        default_factory=lambda: ["DELTA_"]
    )
    
    xgb_prefixes: list[str] = field(
        default_factory=lambda: ["HOME_", "AWAY_", "EMBED_"]
    )
    
    mlp_prefixes: list[str] = field(
        default_factory=lambda: ["DELTA_", "EMBED_"]
    )

    catboost_prefixes: list[str] = field(
        default_factory=lambda: ["HOME_", "AWAY_", "EMBED_"]
    )

    margin_prefixes: list[str] = field(
        default_factory=lambda: ["DELTA_"]
    )

    extra_features: list[str] = field(
        default_factory=lambda: [
            "REST_ADVANTAGE",
            "HOME_B2B",
            "AWAY_B2B",
            "SEASON_YEAR",
            "ALTITUDE_ADVANTAGE",
            "ALTITUDE_B2B_PENALTY",
            "MATCHUP_EXPECTED_PACE",
        ]
    )

    # ======================================================
    # Feature Pruning 
    # ======================================================

    # Set to True to prune the SBS-optimized redundant/noisy features (recommended).
    # Set to False to retain the full baseline feature set.
    prune_optimized_features: bool = True

    # SBS-identified redundant/noisy features to prune when prune_optimized_features is True
    optimized_features_to_remove: dict[str, list[str]] = field(
        default_factory=lambda: {
            "mlp": [],
            "xgb": [
                "EMBED_DELTA_1_MAX",
            ],
            "lr": [
                "DELTA_Z_FG3A_EWMA_5",
                "DELTA_SOS_ROLLING_8",
                "DELTA_ACTIVE_ROSTER_EXPECTED_FTR",
                "HOME_B2B",
            ],
        }
    )

    # Base structural and empirically redundant features pruned across architectures
    features_to_remove: dict[str, list[str]] = field(
        default_factory=lambda: {
            "mlp": [
                # Base structural & rolling duplicates
                "DELTA_ROLLING_PACE",
                "DELTA_Z_FTA_ROLLING_8",
                "DELTA_Z_FGM_ROLLING_8",
                # Exact affine & mathematical duplicates (r = 1.0000)
                "DELTA_D_3PT_ACTUAL",
                "DELTA_FATIGUE_IMPORTANCE_5_STD",
                "DELTA_FATIGUE_IMPORTANCE_5_SUM",
                "DELTA_FATIGUE_IMPORTANCE_5_MAX",
                "DELTA_FOUR_FACTORS_NET_EFG",
                "DELTA_ACTIVE_ROSTER_TOP_3_SHARE",
                "DELTA_ROAD_TRIP_LENGTH",
                # Redundant embedding sums (r > 0.98 with mean)
                "EMBED_DELTA_1_SUM",
                "EMBED_DELTA_2_SUM",
                "EMBED_DELTA_3_SUM",
                "EMBED_DELTA_4_SUM",
                "EMBED_DELTA_5_SUM",
                "EMBED_DELTA_6_SUM",
                "EMBED_DELTA_7_SUM",
                "EMBED_DELTA_8_SUM",
                # Features with negative / noise permutation importance for MLP dense layers
                "DELTA_Z_TOV_ROLLING_8",
                "AWAY_B2B",
                "EMBED_RAW_DELTA_4_MAX",
                "DELTA_Z_FGM_EWMA_10",
                "DELTA_Z_FG3M_ROLLING_8",
                "DELTA_NET_RATING_EWMA_5",
                "EMBED_RAW_DELTA_3_STD",
                "DELTA_REBOUND_PACE_CLASH",
                "DELTA_OPP_3PA_RATE",
                "DELTA_Z_FT_PCT_EWMA_5",
                "DELTA_Z_FG_PCT_EWMA_10",
                "DELTA_TURNOVER_PRESSURE_CLASH",
                "DELTA_Z_FG3A_ROLLING_8",
                "EMBED_DELTA_6_MEAN",
                "DELTA_Z_FGA_ROLLING_8",
                "DELTA_TRAVEL_7D",
                "DELTA_D_3PT_TRUE",
                "DELTA_Z_FG_PCT_EWMA_5",
                "DELTA_Z_FTA_EWMA_10",
                "DELTA_FOUR_FACTOR_TOV_ROLLING_8",
                "EMBED_DELTA_8_MEAN",
                "SEASON_YEAR",
            ],
            "xgb": [
                # Base metadata / structural removals
                "HOME_IS_AWAY",
                "AWAY_IS_AWAY",
                "HOME_VIDEO_AVAILABLE",
                "AWAY_VIDEO_AVAILABLE",
                "HOME_AWAY_GROUP",
                "AWAY_AWAY_GROUP",
                "AWAY_SEASON_ID",
                "HOME_SEASON_YEAR",
                "AWAY_SEASON_YEAR",
                "HOME_ROLLING_PACE",
                "AWAY_ROLLING_PACE",
                "HOME_ROAD_TRIP_LENGTH",
                "HOME_4_IN_5",
                # Zero-variance constant
                "HOME_ALTITUDE_FATIGUE_IMPACT",
                # Exact affine duplicates (r = 1.0000)
                "HOME_D_3PT_ACTUAL",
                "AWAY_D_3PT_ACTUAL",
                "HOME_FATIGUE_IMPORTANCE_5_STD",
                "HOME_FATIGUE_IMPORTANCE_5_SUM",
                "HOME_FATIGUE_IMPORTANCE_5_MAX",
                "AWAY_FATIGUE_IMPORTANCE_5_STD",
                "AWAY_FATIGUE_IMPORTANCE_5_SUM",
                "AWAY_FATIGUE_IMPORTANCE_5_MAX",
                # Near-exact collinear duplicates (r >= 0.98)
                "HOME_ACTIVE_ROSTER_TOP_3_SHARE",
                "AWAY_ACTIVE_ROSTER_TOP_3_SHARE",
                "AWAY_TZ_EASTWARD_LOSS",
                "HOME_4_IN_6",
                "HOME_OFF_RATING_EWMA_10",
                "AWAY_OFF_RATING_EWMA_10",
                "AWAY_OPP_3PA_RATE",
                "AWAY_OPP_3PA_RATE_EWMA_10",
                "HOME_OPP_PRE_GAME_STRENGTH",
                "AWAY_OPP_PRE_GAME_STRENGTH",
                "EMBED_DELTA_1_SUM",
                "EMBED_DELTA_2_SUM",
                "EMBED_DELTA_3_SUM",
                "EMBED_DELTA_4_SUM",
                "EMBED_DELTA_5_SUM",
                "EMBED_DELTA_6_SUM",
                "EMBED_DELTA_7_SUM",
                "EMBED_DELTA_8_SUM",
            ],
            "catboost": [
                # Base metadata / structural removals & zero-variance
                "HOME_IS_AWAY",
                "AWAY_IS_AWAY",
                "HOME_VIDEO_AVAILABLE",
                "AWAY_VIDEO_AVAILABLE",
                "HOME_AWAY_GROUP",
                "AWAY_AWAY_GROUP",
                "AWAY_SEASON_ID",
                "HOME_SEASON_YEAR",
                "AWAY_SEASON_YEAR",
                "HOME_ROLLING_PACE",
                "AWAY_ROLLING_PACE",
                "HOME_ROAD_TRIP_LENGTH",
                "HOME_4_IN_5",
                "HOME_ALTITUDE_FATIGUE_IMPACT",
                # Exact affine duplicates (r = 1.0000)
                "HOME_D_3PT_ACTUAL",
                "AWAY_D_3PT_ACTUAL",
                "HOME_FATIGUE_IMPORTANCE_5_STD",
                "HOME_FATIGUE_IMPORTANCE_5_SUM",
                "HOME_FATIGUE_IMPORTANCE_5_MAX",
                "AWAY_FATIGUE_IMPORTANCE_5_STD",
                "AWAY_FATIGUE_IMPORTANCE_5_SUM",
                "AWAY_FATIGUE_IMPORTANCE_5_MAX",
                # Near-exact collinear duplicates (r >= 0.98)
                "HOME_ACTIVE_ROSTER_TOP_3_SHARE",
                "AWAY_ACTIVE_ROSTER_TOP_3_SHARE",
                "AWAY_TZ_EASTWARD_LOSS",
                "HOME_4_IN_6",
                "HOME_OFF_RATING_EWMA_10",
                "AWAY_OFF_RATING_EWMA_10",
                "AWAY_OPP_3PA_RATE",
                "AWAY_OPP_3PA_RATE_EWMA_10",
                "HOME_OPP_PRE_GAME_STRENGTH",
                "AWAY_OPP_PRE_GAME_STRENGTH",
                "EMBED_DELTA_1_SUM",
                "EMBED_DELTA_2_SUM",
                "EMBED_DELTA_3_SUM",
                "EMBED_DELTA_4_SUM",
                "EMBED_DELTA_5_SUM",
                "EMBED_DELTA_6_SUM",
                "EMBED_DELTA_7_SUM",
                "EMBED_DELTA_8_SUM",
                # Zero-importance features specific to CatBoost oblivious trees
                "HOME_DEF_FTR",
                "HOME_DEF_REB_RATE",
                "AWAY_DEF_FTR",
                "AWAY_DEF_TOV_RATE",
                "AWAY_D_3PT_TRUE",
                "AWAY_D_3PT_TRUE_EWMA_10",
                "AWAY_D_3PT_TRUE_EWMA_5",
                "AWAY_FOUR_FACTOR_EFG_EWMA_10",
                "HOME_FOUR_FACTOR_EFG_ROLLING_8",
                "AWAY_FOUR_FACTOR_OREB_EWMA_10",
                "AWAY_FOUR_FACTOR_OREB_EWMA_5",
                "AWAY_FOUR_FACTOR_FTR_EWMA_10",
                "AWAY_FOUR_FACTOR_FTR_EWMA_5",
                "AWAY_FOUR_FACTOR_TOV_ROLLING_8",
                "HOME_FOUR_FACTOR_TOV_EWMA_10",
                "HOME_ROLLING_OFF_RATING",
                "AWAY_ROLLING_PTS_8",
                "HOME_ROLLING_PTS_8",
                "HOME_3_IN_4",
                "AWAY_3_IN_4",
                "AWAY_4_IN_6",
                "AWAY_ALTITUDE",
                "HOME_B2B",
                "HOME_CIRCADIAN_FATIGUE_INDEX",
                "HOME_FATIGUE_EWMA_MINUTES_10_MAX",
                "HOME_FATIGUE_EWMA_MINUTES_5_MAX",
                "HOME_FATIGUE_IMPORTANCE_10_MAX",
                "HOME_FATIGUE_IMPORTANCE_10_STD",
                "HOME_FATIGUE_SURGE_MAX",
                "HOME_TRAVEL_7D",
                "AWAY_ACTIVE_ROSTER_EXPECTED_DBPM",
                "AWAY_ACTIVE_ROSTER_EXPECTED_OREB",
                "AWAY_ACTIVE_ROSTER_EXPECTED_TOV",
                "AWAY_ACTIVE_ROSTER_ROBUST_FORM_MAX",
                "AWAY_ACTIVE_ROSTER_ROBUST_FORM_SUM",
                "AWAY_PLAYMAKER_CONCENTRATION_RATIO",
                "AWAY_ROLLING_TOP_2_SHARE_10",
                "AWAY_SPACING_GRAVITY_INDEX",
                "HOME_ACTIVE_ROSTER_EXPECTED_EFG",
                "HOME_ACTIVE_ROSTER_EXPECTED_NET_BPM",
                "HOME_ACTIVE_ROSTER_STAR_SHARE",
                "HOME_ACTIVE_ROSTER_TOP_2_SHARE",
                "HOME_PLAYMAKER_CONCENTRATION_RATIO",
                "HOME_SPACING_GRAVITY_INDEX",
                "AWAY_GLASS_DOMINANCE",
                "AWAY_TURNOVER_PRESSURE_CLASH",
                "AWAY_REBOUND_PACE_CLASH",
                "AWAY_SOS_EWMA_5",
                "AWAY_SOS_ROLLING_8",
                "HOME_SOS_EWMA_5",
                "HOME_SOS_ROLLING_8",
                "EMBED_DELTA_2_STD",
                "EMBED_DELTA_3_STD",
                "EMBED_DELTA_5_MEAN",
                "EMBED_DELTA_6_MAX",
                "EMBED_DELTA_6_MEAN",
                "EMBED_DELTA_6_STD",
                "EMBED_DELTA_7_MAX",
                "EMBED_DELTA_7_MEAN",
                "EMBED_DELTA_7_STD",
                "EMBED_DELTA_8_MAX",
                "EMBED_RAW_DELTA_2_MAX",
                "EMBED_RAW_DELTA_2_STD",
                "EMBED_RAW_DELTA_4_STD",
                "EMBED_RAW_DELTA_5_STD",
                "EMBED_RAW_DELTA_6_MAX",
                "EMBED_RAW_DELTA_6_STD",
                "EMBED_RAW_DELTA_7_MAX",
                "EMBED_RAW_DELTA_7_STD",
                "EMBED_RAW_DELTA_8_STD",
            ],
            "margin": [
                "DELTA_ROLLING_PACE",
                "DELTA_D_3PT_ACTUAL",
                "DELTA_FATIGUE_IMPORTANCE_5_STD",
                "DELTA_FATIGUE_IMPORTANCE_5_SUM",
                "DELTA_FATIGUE_IMPORTANCE_5_MAX",
                "DELTA_FOUR_FACTORS_NET_EFG",
                "DELTA_ACTIVE_ROSTER_TOP_3_SHARE",
                "DELTA_ROAD_TRIP_LENGTH",
            ],
            "lr": [
                "DELTA_ROLLING_PACE",
                "DELTA_D_3PT_ACTUAL",
                "DELTA_FATIGUE_IMPORTANCE_5_STD",
                "DELTA_FATIGUE_IMPORTANCE_5_SUM",
                "DELTA_FATIGUE_IMPORTANCE_5_MAX",
                "DELTA_FOUR_FACTORS_NET_EFG",
                "DELTA_ACTIVE_ROSTER_TOP_3_SHARE",
                "DELTA_ROAD_TRIP_LENGTH",
            ],
        }
    )


    # ======================================================
    # General Machine Learning
    # ======================================================

    random_seed: int = 42
    cv_folds: int = 5

    # ======================================================
    # Speed & Architecture Optimizations
    # ======================================================

    include_logistic_regression: bool = False
    ensemble_shrinkage: float = 0.05
    use_halving_search: bool = True
    halving_factor: int = 2
    early_stopping_rounds: int = 25
    early_stopping_val_fraction: float = 0.1

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
    margin_search_iterations: int = 15

    # ======================================================
    # Hyperparameter Search Spaces
    # ======================================================

    mlp_grid: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_layer_sizes": [
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

    margin_model_type: str = "ridge"
    margin_bivariate_efficiency: bool = True

    margin_ridge_grid: dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0],
        }
    )

    margin_xgb_grid: dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "reg_lambda": [1, 5, 10, 20],
            "n_estimators": [100, 200, 300],
            "subsample": [0.8, 1.0],
        }
    )

    catboost_grid: dict[str, Any] = field(
        default_factory=lambda: {
            "depth": [4, 6],
            "l2_leaf_reg": [1, 5, 10],
            "learning_rate": [0.03, 0.07],
            "iterations": [350],
        }
    )
    catboost_search_iterations: int = 4

    def __post_init__(self) -> None:
        """Applies feature pruning if prune_optimized_features is True or via environment override."""
        env_override = os.environ.get("USE_OPTIMIZED_FEATURES")
        if env_override is not None:
            self.prune_optimized_features = (env_override == "1")

        if os.environ.get("USE_LOGISTIC_REGRESSION") is not None:
            self.include_logistic_regression = (os.environ.get("USE_LOGISTIC_REGRESSION") == "1")

        if os.environ.get("USE_HALVING_SEARCH") is not None:
            self.use_halving_search = (os.environ.get("USE_HALVING_SEARCH") == "1")

        # CatBoost inherits tree feature removals by default
        if "catboost" not in self.features_to_remove:
            self.features_to_remove["catboost"] = list(self.features_to_remove.get("xgb", []))

        if self.prune_optimized_features:
            for m in ["mlp", "xgb", "lr", "catboost", "margin"]:
                existing = set(self.features_to_remove.get(m, []))
                for feat in self.optimized_features_to_remove.get(m, []):
                    if feat not in existing:
                        self.features_to_remove.setdefault(m, []).append(feat)

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
    catboost: ModelArtifacts = field(default_factory=ModelArtifacts)
    margin: ModelArtifacts = field(default_factory=ModelArtifacts)

    ensemble_weights: np.ndarray | None = None
    ensemble_formula: dict[str, float] | None = None
    calibration_report: dict[str, Any] | None = None


def get_git_metadata() -> dict[str, Any]:
    """Retrieves current git commit, branch, and working tree status."""
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
        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": dirty,
        }
    except Exception:
        return {
            "git_commit": "unknown",
            "git_branch": "unknown",
            "git_dirty": False,
        }


def get_dataset_fingerprint(dataset_path: Path) -> str | None:
    """Computes a SHA-256 fingerprint on the first 1MB of the dataset."""
    if not dataset_path.exists():
        return None
    import hashlib
    hasher = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        hasher.update(f.read(1024 * 1024))
    return hasher.hexdigest()


def get_experiment_metadata(data_dir: Path | None = None) -> dict[str, Any]:
    """
    Collects runtime information describing the current experiment.

    This metadata is saved alongside every trained model to improve
    reproducibility and simplify future comparisons between runs.
    """
    metadata: dict[str, Any] = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "dataset_version": "v2.0-heterogeneous",
        **get_git_metadata(),
    }

    if data_dir is not None:
        fingerprint = get_dataset_fingerprint(Path(data_dir) / "ml_ready_matchups_players.csv")
        if fingerprint is not None:
            metadata["dataset_fingerprint_sha256_1mb"] = fingerprint

    return metadata