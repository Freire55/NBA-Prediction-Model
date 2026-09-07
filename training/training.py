"""
Retrains tuned NBA prediction models on the combined train and validation sets.

After hyperparameter search spaces and ensemble weights are resolved, this module
retrains estimators on the complete historical window (train + val) to maximize
statistical efficiency before prospective testing. Feature scalers are refit on
the combined historical dataset, and their parameters (means, scales) are exported
to disk for auditability and production inference.
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from training.config import FeatureSet, TrainingArtifacts
from training.margin import MarginRegressor, PaceModulatedMarginClassifier

# ======================================================
# Output Artifact Filenames
# ======================================================

SCALER_STATS_FILE = "scaler_statistics.csv"


# ======================================================
# Transformation & Scaling Helpers
# ======================================================

def _combine_splits(
    train_part: pd.DataFrame | pd.Series,
    val_part: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Concatenates chronological training and validation partitions."""
    return pd.concat([train_part, val_part], axis=0)


def _fit_and_apply_scaler(
    train_full: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
    """
    Fits a new StandardScaler on the combined historical dataset and scales
    both full training and held-out test matrices.
    """
    scaler = StandardScaler()
    train_full_scaled = scaler.fit_transform(train_full)
    test_scaled = scaler.transform(test)
    return scaler, train_full_scaled, test_scaled


def _export_scaler_statistics(
    scaler: StandardScaler,
    features: pd.Index | list[str],
    output_path: Path,
) -> None:
    """Exports scaler means and standard deviations for reproducibility."""
    stats_df = pd.DataFrame(
        {
            "Feature": list(features),
            "Mean": scaler.mean_,
            "Std": scaler.scale_,
        }
    )
    stats_df.to_csv(output_path, index=False)


# ======================================================
# Estimator Retraining
# ======================================================

def _retrain_classifier(
    estimator: Any,
    X_train: Any,
    y_train: Any,
) -> Any:
    """Clones a tuned estimator configuration and refits it on full historical data."""
    final_estimator = clone(estimator)
    final_estimator.fit(X_train, y_train)
    return final_estimator


def _retrain_margin_pipeline(artifacts: TrainingArtifacts) -> None:
    """
    Retrains the continuous margin model and pace converter on full historical data.

    Handles both raw MarginRegressor instances and PaceModulatedMarginClassifier wrappers.
    """
    if (
        artifacts.margin.model is None
        or artifacts.data is None
        or artifacts.data.margin is None
        or artifacts.data.y_margin_train is None
    ):
        return

    margin_fs = artifacts.data.margin
    margin_train_full = _combine_splits(margin_fs.X_train, margin_fs.X_val)
    y_margin_full = _combine_splits(artifacts.data.y_margin_train, artifacts.data.y_margin_val)

    scaler, train_scaled, test_scaled = _fit_and_apply_scaler(margin_train_full, margin_fs.X_test)
    margin_fs.scaler = scaler
    margin_fs.X_train_full = margin_train_full
    margin_fs.X_train_full_processed = train_scaled
    margin_fs.X_test_processed = test_scaled

    model = artifacts.margin.model
    if isinstance(model, PaceModulatedMarginClassifier):
        reg = model.regressor
        if reg.model_type == "ridge":
            reg.fit(train_scaled, y_margin_full, feature_names=margin_fs.feature_names)
        else:
            reg.fit(margin_train_full, y_margin_full)
        artifacts.margin.final_model = model
    elif isinstance(model, MarginRegressor):
        margin_final = clone(model)
        if model.model_type == "ridge":
            margin_final.fit(train_scaled, y_margin_full, feature_names=margin_fs.feature_names)
        else:
            margin_final.fit(margin_train_full, y_margin_full)
        artifacts.margin.final_model = margin_final
    else:
        # Generic fallback
        margin_final = clone(model)
        margin_final.fit(train_scaled, y_margin_full)
        artifacts.margin.final_model = margin_final


# ======================================================
# Main Pipeline API
# ======================================================

def retrain_on_full_data(artifacts: TrainingArtifacts) -> None:
    """
    Retrains all selected models on the combined training and validation dataset.

    Mutates `artifacts` in-place by setting `final_model` and updated `feature_set`
    containers with fresh scalers and scaled test data.
    """
    data = artifacts.data
    mlp_train_full = _combine_splits(data.mlp.X_train, data.mlp.X_val)
    xgb_train_full = _combine_splits(data.xgb.X_train, data.xgb.X_val)
    lr_train_full = _combine_splits(data.lr.X_train, data.lr.X_val)
    y_train_full = _combine_splits(data.y_train, data.y_val)

    # Fit and apply scalers
    mlp_scaler, mlp_train_scaled, mlp_test_scaled = _fit_and_apply_scaler(
        mlp_train_full, data.mlp.X_test
    )
    lr_scaler, lr_train_scaled, lr_test_scaled = _fit_and_apply_scaler(
        lr_train_full, data.lr.X_test
    )

    _export_scaler_statistics(
        lr_scaler, lr_train_full.columns, artifacts.output_dir / SCALER_STATS_FILE
    )

    # Retrain binary base classifiers
    artifacts.mlp.final_model = _retrain_classifier(
        artifacts.mlp.model, mlp_train_scaled, y_train_full
    )
    artifacts.xgb.final_model = _retrain_classifier(
        artifacts.xgb.model, xgb_train_full, y_train_full
    )
    artifacts.lr.final_model = _retrain_classifier(
        artifacts.lr.model, lr_train_scaled, y_train_full
    )

    if artifacts.catboost.model is not None and data.catboost is not None:
        catboost_train_full = _combine_splits(data.catboost.X_train, data.catboost.X_val)
        artifacts.catboost.final_model = _retrain_classifier(
            artifacts.catboost.model, catboost_train_full, y_train_full
        )
        artifacts.catboost.feature_set.X_train_full = catboost_train_full

    # Update feature set state
    artifacts.mlp.feature_set.scaler = mlp_scaler
    artifacts.mlp.feature_set.X_train_full = mlp_train_full
    artifacts.mlp.feature_set.X_train_full_processed = mlp_train_scaled
    artifacts.mlp.feature_set.X_test_processed = mlp_test_scaled

    artifacts.xgb.feature_set.X_train_full = xgb_train_full

    artifacts.lr.feature_set.scaler = lr_scaler
    artifacts.lr.feature_set.X_train_full = lr_train_full
    artifacts.lr.feature_set.X_train_full_processed = lr_train_scaled
    artifacts.lr.feature_set.X_test_processed = lr_test_scaled

    # Retrain margin regressor and pace converter if present
    _retrain_margin_pipeline(artifacts)