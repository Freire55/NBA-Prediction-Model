"""
Loads and prepares matchup datasets for NBA model training.

This module enforces strict chronological data partitioning and guarantees zero
lookahead or post-game data leakage. It loads raw/engineered game logs,
constructs architecture-tailored feature matrices for each model (MLP, XGBoost,
Logistic Regression, Margin Regressor), extracts continuous and binary targets,
and standardizes numerical inputs where required.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.config import (
    DatasetSummary,
    FeatureSet,
    TrainingConfig,
    TrainingData,
)

logger = logging.getLogger(__name__)

# ======================================================
# Constants & Column Definitions
# ======================================================

DATASET_FILE = "ml_ready_matchups_players.csv"
ERA_ADJUSTED_FILE = "era_adjusted_nba.csv"

TARGET_COLUMN = "HOME_WIN"
SEASON_COLUMN = "HOME_SEASON_ID"


# ======================================================
# Data Ingestion & Leakage Prevention
# ======================================================

def _load_raw_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Loads matchup records from Parquet (preferred for speed) or CSV.

    Converts 64-bit floating point columns to 32-bit floats to reduce memory
    footprint by 50% during matrix transformations.
    """
    parquet_path = (data_dir / DATASET_FILE).with_suffix(".parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(data_dir / DATASET_FILE)

    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")

    return df


def _identify_post_game_leakage_columns() -> Set[str]:
    """
    Returns an explicit set of post-game box score and in-game statistics that
    must NEVER appear in predictive feature spaces prior to tip-off.
    """
    base_box_stats = [
        "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL",
        "BLK", "TOV", "PF", "PLUS_MINUS", "POSSESSIONS", "PACE", "MIN",
        "WL", "WIN", "FOUR_FACTOR_EFG", "FOUR_FACTOR_TOV",
        "FOUR_FACTOR_OREB", "FOUR_FACTOR_FTR",
    ]
    post_game_cols = {
        f"{prefix}{stat}"
        for prefix in ["HOME_", "AWAY_", "DELTA_"]
        for stat in base_box_stats
    }
    return {TARGET_COLUMN, "TARGET_MARGIN", "MARGIN"} | post_game_cols


# ======================================================
# Feature Matrix Construction
# ======================================================

def get_model_features(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> Dict[str, List[str]]:
    """
    Constructs model-specific feature column lists based on architectural biases.

    - Logistic Regression & Margin: Differential features ("DELTA_") and game context
      to enforce linear interpretability and avoid multicollinearity.
    - XGBoost: Separate home/away levels and embeddings to exploit non-linear interactions.
    - MLP: Standardized differential features and dense roster embeddings.

    Pruning lists specified in TrainingConfig (e.g., from Sequential Backward Selection)
    are applied to eliminate redundant or noisy features.
    """
    numeric_cols = set(df.select_dtypes(include=["number"]).columns)
    safe_extra = [col for col in config.extra_features if col in numeric_cols]
    exclude_cols = _identify_post_game_leakage_columns()

    def _filter_candidates(prefixes: List[str], exclude_substr: Optional[str] = None) -> List[str]:
        return [
            col for col in df.columns
            if any(col.startswith(p) for p in prefixes)
            and (exclude_substr is None or exclude_substr not in col)
            and col in numeric_cols
            and col not in exclude_cols
        ] + safe_extra

    lr_raw = _filter_candidates(config.lr_prefixes)
    xgb_raw = _filter_candidates(config.xgb_prefixes, exclude_substr="_Z_")
    mlp_raw = _filter_candidates(config.mlp_prefixes)

    margin_prefixes = getattr(config, "margin_prefixes", ["DELTA_"])
    margin_raw = _filter_candidates(margin_prefixes)

    # Apply configuration-defined feature removals / pruning
    lr_removals = set(config.features_to_remove.get("lr", []))
    xgb_removals = set(config.features_to_remove.get("xgb", []))
    mlp_removals = set(config.features_to_remove.get("mlp", []))
    margin_removals = set(
        config.features_to_remove.get("margin", config.features_to_remove.get("lr", []))
    )

    return {
        "mlp": [f for f in dict.fromkeys(mlp_raw) if f not in mlp_removals],
        "xgb": [f for f in dict.fromkeys(xgb_raw) if f not in xgb_removals],
        "lr": [f for f in dict.fromkeys(lr_raw) if f not in lr_removals],
        "margin": [f for f in dict.fromkeys(margin_raw) if f not in margin_removals],
    }


# ======================================================
# Target Extraction & Chronological Splitting
# ======================================================

def _extract_margin_target(
    df: pd.DataFrame,
    data_dir: Path,
) -> Optional[pd.Series]:
    """
    Extracts the continuous point differential target (HOME_PTS - AWAY_PTS).

    If 'TARGET_MARGIN' already exists in the dataset, it is used directly.
    Otherwise, it is safely joined from era_adjusted_nba.csv via GAME_ID.
    Target series is kept strictly isolated from feature matrices.
    """
    if "TARGET_MARGIN" in df.columns:
        return df["TARGET_MARGIN"].astype("float32")

    if "HOME_GAME_ID" not in df.columns:
        return None

    era_path = data_dir / ERA_ADJUSTED_FILE
    if not era_path.exists():
        return None

    era_df = pd.read_csv(era_path, usecols=["GAME_ID", "MATCHUP", "PTS"])
    home_pts = (
        era_df[era_df["MATCHUP"].str.contains(" vs. ")][["GAME_ID", "PTS"]]
        .rename(columns={"PTS": "HOME_PTS"})
    )
    away_pts = (
        era_df[era_df["MATCHUP"].str.contains(" @ ")][["GAME_ID", "PTS"]]
        .rename(columns={"PTS": "AWAY_PTS"})
    )
    game_margins = home_pts.merge(away_pts, on="GAME_ID")
    game_margins["MARGIN"] = (game_margins["HOME_PTS"] - game_margins["AWAY_PTS"]).astype("float32")
    game_margins["GAME_ID"] = game_margins["GAME_ID"].astype(str)

    margin_map = dict(zip(game_margins["GAME_ID"], game_margins["MARGIN"]))
    return df["HOME_GAME_ID"].astype(str).map(margin_map).astype("float32")


def _split_chronologically(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits records chronologically using the NBA Season ID code (e.g. '22018').

    Chronological splitting prevents temporal data leakage: models are trained
    on past seasons and validated/tested strictly on subsequent chronological seasons.
    """
    train_df = df[df[SEASON_COLUMN] <= config.train_end]
    val_df = df[
        (df[SEASON_COLUMN] > config.train_end)
        & (df[SEASON_COLUMN] <= config.validation_end)
    ]
    test_df = df[df[SEASON_COLUMN] > config.validation_end]
    return train_df, val_df, test_df


def _build_feature_set(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> FeatureSet:
    """Instantiates a strongly typed FeatureSet container for one model family."""
    return FeatureSet(
        X_train=train_df[features].copy(),
        X_val=val_df[features].copy(),
        X_test=test_df[features].copy(),
        feature_names=features,
    )


# ======================================================
# Main Pipeline API
# ======================================================

def load_and_prep_data(
    data_dir: Path,
    config: TrainingConfig,
) -> TrainingData:
    """
    Loads dataset, builds model-specific feature subsets, and partitions splits.

    Returns a TrainingData container ready for training and evaluation.
    """
    df = _load_raw_dataframe(data_dir)
    df[SEASON_COLUMN] = df[SEASON_COLUMN].astype(str)

    feature_dict = get_model_features(df, config)
    train_df, val_df, test_df = _split_chronologically(df, config)

    # Extract continuous target series (for margin regressors)
    margin_series = _extract_margin_target(df, data_dir)
    y_margin_train = margin_series.loc[train_df.index].copy() if margin_series is not None else None
    y_margin_val = margin_series.loc[val_df.index].copy() if margin_series is not None else None
    y_margin_test = margin_series.loc[test_df.index].copy() if margin_series is not None else None

    summary = DatasetSummary(
        train_games=len(train_df),
        validation_games=len(val_df),
        test_games=len(test_df),
        lr_feature_names=feature_dict["lr"],
        xgb_feature_names=feature_dict["xgb"],
        mlp_feature_names=feature_dict["mlp"],
        margin_feature_names=feature_dict.get("margin", []),
    )

    logger.info(
        f"      Train: {summary.train_games} | "
        f"Val: {summary.validation_games} | "
        f"Test: {summary.test_games}"
    )
    logger.info(
        f"      Features -> LR: {len(summary.lr_feature_names)} | "
        f"XGB: {len(summary.xgb_feature_names)} | "
        f"MLP: {len(summary.mlp_feature_names)} | "
        f"Margin: {len(summary.margin_feature_names)}"
    )

    return TrainingData(
        lr=_build_feature_set(train_df, val_df, test_df, feature_dict["lr"]),
        xgb=_build_feature_set(train_df, val_df, test_df, feature_dict["xgb"]),
        mlp=_build_feature_set(train_df, val_df, test_df, feature_dict["mlp"]),
        margin=_build_feature_set(train_df, val_df, test_df, feature_dict["margin"]),
        y_train=train_df[TARGET_COLUMN].copy(),
        y_val=val_df[TARGET_COLUMN].copy(),
        y_test=test_df[TARGET_COLUMN].copy(),
        y_margin_train=y_margin_train,
        y_margin_val=y_margin_val,
        y_margin_test=y_margin_test,
        summary=summary,
    )


def scale_features(feature_set: FeatureSet) -> None:
    """
    Fits a StandardScaler on training data and transforms validation and test sets.

    Crucially, the scaler fits EXCLUSIVELY on X_train to ensure no test data
    distribution information leaks into the training pipeline.
    """
    scaler = StandardScaler()
    feature_set.X_train_processed = scaler.fit_transform(feature_set.X_train)
    feature_set.X_val_processed = scaler.transform(feature_set.X_val)
    feature_set.X_test_processed = scaler.transform(feature_set.X_test)
    feature_set.scaler = scaler