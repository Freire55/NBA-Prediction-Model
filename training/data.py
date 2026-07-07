"""
Loads and prepares the dataset for model training.

This module reads the engineered matchup dataset, validates the required
schema, selects the configured feature set, performs chronological
train/validation/test splits, and standardizes numerical features for
models that require feature scaling.

Input:
    data/ml_ready_matchups_players.csv

Output:
    Train, validation and test datasets ready for the training pipeline.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.config import TrainingConfig

logger = logging.getLogger(__name__)

# ======================================================
# Constants
# ======================================================

DATASET_FILE = "ml_ready_matchups_players.csv"

TARGET_COLUMN = "HOME_WIN"
SEASON_COLUMN = "HOME_SEASON_ID"

DatasetSplit = Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Dict,
]


# ======================================================
# Helper Functions
# ======================================================

def select_features(df: pd.DataFrame, config: TrainingConfig) -> List[str]:
    """
    Returns the configured feature list after validating that every
    requested feature exists in the dataset.
    """
    features = [
        col for col in df.columns
        if col.startswith(config.feature_prefix)
    ] + config.extra_features

    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        raise ValueError(
            f"Configured features are missing from the dataset: {missing_features}"
        )

    return features


# ======================================================
# Data Preparation
# ======================================================

def load_and_prep_data(
    data_dir: Path,
    config: TrainingConfig,
) -> DatasetSplit:
    """
    Loads the engineered matchup dataset, validates its schema,
    selects the configured feature set, and performs chronological
    train/validation/test splits.

    Returns
    -------
    DatasetSplit
        Tuple containing train, validation and test features,
        labels, feature names, and dataset summary statistics.
    """
    df = pd.read_csv(data_dir / DATASET_FILE)

    # Validate required columns
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column '{TARGET_COLUMN}'.")

    if SEASON_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required season column '{SEASON_COLUMN}'."
        )

    features = select_features(df, config)

    # Ensure chronological comparisons use string values
    df[SEASON_COLUMN] = df[SEASON_COLUMN].astype(str)

    # Chronological train / validation / test split
    train_df = df[df[SEASON_COLUMN] <= config.train_end]

    val_df = df[
        (df[SEASON_COLUMN] > config.train_end)
        & (df[SEASON_COLUMN] <= config.validation_end)
    ]

    test_df = df[df[SEASON_COLUMN] > config.validation_end]

    dataset_summary = {
        "train_games": len(train_df),
        "validation_games": len(val_df),
        "test_games": len(test_df),
        "feature_count": len(features),
    }

    logger.info(
        "      Train: %d | Val: %d | Test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return (
        train_df[features],
        train_df[TARGET_COLUMN],
        val_df[features],
        val_df[TARGET_COLUMN],
        test_df[features],
        test_df[TARGET_COLUMN],
        features,
        dataset_summary,
    )


# ======================================================
# Feature Scaling
# ======================================================

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits a StandardScaler on the training data and applies it to both
    the training and validation feature matrices.

    Returns
    -------
    tuple
        (scaled_train_features, scaled_validation_features, fitted_scaler)
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, scaler