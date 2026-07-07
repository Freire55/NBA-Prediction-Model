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

# ======================================================
# Output Files
# ======================================================

SCALER_STATS_FILE = "scaler_statistics.csv"


# ======================================================
# Training Functions
# ======================================================

def retrain_on_full_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    mlp_model: Any,
    xgb_model: Any,
    lr_model: Any,
    output_dir: Path,
) -> Tuple[
    Any,
    Any,
    Any,
    StandardScaler,
    pd.DataFrame,
    np.ndarray,
]:
    """
    Retrains the selected models on the full historical dataset.

    The training and validation datasets are combined before fitting
    the final models. A new feature scaler is trained on the complete
    dataset to avoid wasting available information, and scaler
    statistics are exported for reproducibility.

    Args:
        X_train:
            Training feature matrix.
        X_val:
            Validation feature matrix.
        y_train:
            Training labels.
        y_val:
            Validation labels.
        X_test:
            Test feature matrix.
        mlp_model:
            Tuned MLP model.
        xgb_model:
            Tuned XGBoost model.
        lr_model:
            Tuned Logistic Regression model.
        output_dir:
            Directory where training artifacts are saved.

    Returns:
        Tuple containing:
            - Final MLP model
            - Final XGBoost model
            - Final Logistic Regression model
            - StandardScaler fitted on the combined dataset
            - Combined training feature matrix
            - Scaled test feature matrix
    """
    # --------------------------------------------------
    # Combine training and validation datasets
    # --------------------------------------------------

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    # --------------------------------------------------
    # Fit scaler on the complete training dataset
    # --------------------------------------------------

    scaler_full = StandardScaler()

    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_test_scaled_full = scaler_full.transform(X_test)

    # --------------------------------------------------
    # Export scaler statistics
    # --------------------------------------------------

    scaler_stats = pd.DataFrame(
        {
            "Feature": X_train_full.columns,
            "Mean": scaler_full.mean_,
            "Std": scaler_full.scale_,
        }
    )

    scaler_stats.to_csv(
        output_dir / SCALER_STATS_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Clone tuned models and retrain
    # --------------------------------------------------

    mlp_final = clone(mlp_model)
    xgb_final = clone(xgb_model)
    lr_final = clone(lr_model)

    mlp_final.fit(X_train_full_scaled, y_train_full)
    xgb_final.fit(X_train_full, y_train_full)
    lr_final.fit(X_train_full_scaled, y_train_full)

    return (
        mlp_final,
        xgb_final,
        lr_final,
        scaler_full,
        X_train_full,
        X_test_scaled_full,
    )