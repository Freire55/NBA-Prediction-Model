"""
Model retraining utilities for the NBA prediction pipeline.

This module retrains the best-performing models using the combined
training and validation datasets after hyperparameter tuning has
completed. It also fits the final feature scaler, exports scaler
statistics for reproducibility, and returns the trained models
ready for evaluation on the held-out test set.

Functions:
    retrain_on_full_data()
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from training.config import FeatureSet, TrainingArtifacts

# ======================================================
# Output Files
# ======================================================

SCALER_STATS_FILE = "scaler_statistics.csv"


# ======================================================
# Training Functions
# ======================================================

def retrain_on_full_data(
    artifacts: TrainingArtifacts,
) -> None:
    """
    Retrains the selected models on the full historical dataset.

    The training and validation datasets are combined before fitting
    the final models. A new feature scaler is trained on the complete
    dataset to avoid wasting available information, and scaler
    statistics are exported for reproducibility.

    This function mutates `artifacts` in place: it populates
    `final_model` on each of `artifacts.mlp`, `artifacts.xgb`, and
    `artifacts.lr`, and refreshes each model's `feature_set` with the
    scaler and combined train+val data fit on the full dataset, so
    that downstream stages (explainability, evaluation) see final,
    fully-trained models rather than the dataclass defaults.

    Args:
        artifacts:
            The central pipeline state, including tuned (but not yet
            finally-retrained) models under `.model` for each of
            `mlp`, `xgb`, and `lr`.
    """
    # --------------------------------------------------
    # Combine training and validation datasets
    # --------------------------------------------------

    data = artifacts.data

    mlp_train_full = pd.concat([
        data.mlp.X_train,
        data.mlp.X_val,
    ])

    xgb_train_full = pd.concat([
        data.xgb.X_train,
        data.xgb.X_val,
    ])

    lr_train_full = pd.concat([
        data.lr.X_train,
        data.lr.X_val,
    ])

    y_train_full = pd.concat([
        data.y_train,
        data.y_val,
    ])


    # --------------------------------------------------
    # Fit scalers on the complete datasets
    # --------------------------------------------------

    mlp_scaler = StandardScaler()
    mlp_train_full_scaled = mlp_scaler.fit_transform(mlp_train_full)
    mlp_test_scaled = mlp_scaler.transform(data.mlp.X_test)

    lr_scaler = StandardScaler()
    lr_train_full_scaled = lr_scaler.fit_transform(lr_train_full)
    lr_test_scaled = lr_scaler.transform(data.lr.X_test)

    # --------------------------------------------------
    # Export scaler statistics
    # --------------------------------------------------

    scaler_stats = pd.DataFrame(
        {
            "Feature": lr_train_full.columns,
            "Mean": lr_scaler.mean_,
            "Std": lr_scaler.scale_,
        }
    )

    scaler_stats.to_csv(
        artifacts.output_dir / SCALER_STATS_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Clone tuned models and retrain
    # --------------------------------------------------

    mlp_final = clone(artifacts.mlp.model)
    xgb_final = clone(artifacts.xgb.model)
    lr_final = clone(artifacts.lr.model)

    mlp_final.fit(mlp_train_full_scaled, y_train_full)
    xgb_final.fit(xgb_train_full, y_train_full)
    lr_final.fit(lr_train_full_scaled, y_train_full)

    # --------------------------------------------------
    # Write results back onto artifacts
    # --------------------------------------------------

    artifacts.mlp.final_model = mlp_final
    artifacts.xgb.final_model = xgb_final
    artifacts.lr.final_model = lr_final

    artifacts.mlp.feature_set.scaler = mlp_scaler
    artifacts.lr.feature_set.scaler = lr_scaler

    artifacts.mlp.feature_set.X_test_processed = mlp_test_scaled
    artifacts.lr.feature_set.X_test_processed = lr_test_scaled

    artifacts.mlp.feature_set.X_train_full = mlp_train_full
    artifacts.mlp.feature_set.X_train_full_processed = mlp_train_full_scaled

    artifacts.xgb.feature_set.X_train_full = xgb_train_full

    artifacts.lr.feature_set.X_train_full = lr_train_full
    artifacts.lr.feature_set.X_train_full_processed = lr_train_full_scaled