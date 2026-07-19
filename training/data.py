"""
Loads and prepares the dataset for model training.

This module reads the engineered matchup dataset, validates the required
schema, selects the configured feature set, performs chronological
train/validation/test splits, and standardizes numerical features for
models that require feature scaling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.config import (
    DatasetSummary,
    FeatureSet,
    TrainingData,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

# ======================================================
# Constants
# ======================================================

DATASET_FILE = "ml_ready_matchups_players.csv"

TARGET_COLUMN = "HOME_WIN"
SEASON_COLUMN = "HOME_SEASON_ID"


# ======================================================
# Helper Functions
# ======================================================

def get_model_features(
    df: pd.DataFrame, 
    config: TrainingConfig
) -> Dict[str, List[str]]:
    """Builds distinct feature lists for each model architecture."""
    
    numeric_cols = set(df.select_dtypes(include=["number"]).columns)
    safe_extra = [col for col in config.extra_features if col in numeric_cols]
    
    exclude_cols = {TARGET_COLUMN}
    
    lr_raw = [
        col for col in df.columns 
        if any(col.startswith(p) for p in config.lr_prefixes)
        and col in numeric_cols
        and col not in exclude_cols
    ] + safe_extra
    
    xgb_raw = [
        col for col in df.columns 
        if any(col.startswith(p) for p in config.xgb_prefixes) 
        and "_Z_" not in col
        and col in numeric_cols
        and col not in exclude_cols
    ] + safe_extra
    
    mlp_raw = [
        col for col in df.columns 
        if any(col.startswith(p) for p in config.mlp_prefixes)
        and col in numeric_cols
        and col not in exclude_cols
    ] + safe_extra

    lr_removals = set(config.features_to_remove.get("lr", []))
    xgb_removals = set(config.features_to_remove.get("xgb", []))
    mlp_removals = set(config.features_to_remove.get("mlp", []))

    return {
        "mlp": [f for f in dict.fromkeys(mlp_raw) if f not in mlp_removals],
        "xgb": [f for f in dict.fromkeys(xgb_raw) if f not in xgb_removals],
        "lr":  [f for f in dict.fromkeys(lr_raw) if f not in lr_removals],
    }


def load_and_prep_data(
    data_dir: Path, 
    config: TrainingConfig
) -> TrainingData:
    """Loads dataset and performs chronological splits for all feature sets."""
    df = pd.read_csv(data_dir / DATASET_FILE)

    df[SEASON_COLUMN] = df[SEASON_COLUMN].astype(str)
    feature_dict = get_model_features(df, config)

    # Chronological Split
    train_df = df[df[SEASON_COLUMN] <= config.train_end]
    val_df = df[(df[SEASON_COLUMN] > config.train_end) & (df[SEASON_COLUMN] <= config.validation_end)]
    test_df = df[df[SEASON_COLUMN] > config.validation_end]

    summary = DatasetSummary(
        train_games=len(train_df),
        validation_games=len(val_df),
        test_games=len(test_df),
        lr_feature_names=feature_dict["lr"],
        xgb_feature_names=feature_dict["xgb"],
        mlp_feature_names=feature_dict["mlp"],
    )

    logger.info(
        f"      Train: {summary.train_games} | "
        f"Val: {summary.validation_games} | "
        f"Test: {summary.test_games}"
    )
    logger.info(
        f"      Features -> LR: {len(summary.lr_feature_names)} | "
        f"XGB: {len(summary.xgb_feature_names)} | "
        f"MLP: {len(summary.mlp_feature_names)}"
    )

    return TrainingData(
        lr=FeatureSet(
            X_train=train_df[feature_dict["lr"]].copy(),
            X_val=val_df[feature_dict["lr"]].copy(),
            X_test=test_df[feature_dict["lr"]].copy(),
            feature_names=feature_dict["lr"]
        ),
        xgb=FeatureSet(
            X_train=train_df[feature_dict["xgb"]].copy(),
            X_val=val_df[feature_dict["xgb"]].copy(),
            X_test=test_df[feature_dict["xgb"]].copy(),
            feature_names=feature_dict["xgb"]
        ),
        mlp=FeatureSet(
            X_train=train_df[feature_dict["mlp"]].copy(),
            X_val=val_df[feature_dict["mlp"]].copy(),
            X_test=test_df[feature_dict["mlp"]].copy(),
            feature_names=feature_dict["mlp"]
        ),
        y_train=train_df[TARGET_COLUMN].copy(),
        y_val=val_df[TARGET_COLUMN].copy(),
        y_test=test_df[TARGET_COLUMN].copy(),
        summary=summary
    )


# ======================================================
# Feature Scaling
# ======================================================

def scale_features(
    feature_set: FeatureSet,
) -> None:

    scaler = StandardScaler()

    feature_set.X_train_processed = scaler.fit_transform(
        feature_set.X_train
    )

    feature_set.X_val_processed = scaler.transform(
        feature_set.X_val
    )

    feature_set.X_test_processed = scaler.transform(
        feature_set.X_test
    )

    feature_set.scaler = scaler