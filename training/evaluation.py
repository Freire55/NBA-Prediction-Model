"""
Evaluates trained models and computes classification metrics.

This module generates probability predictions for each trained model,
computes standard evaluation metrics, evaluates the weighted ensemble,
prints a formatted performance summary, and exports the results to CSV.

Outputs:
    01_model_comparison.csv
"""

from pathlib import Path
from typing import Dict, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from training.config import TrainingArtifacts

# ======================================================
# Constants
# ======================================================

CLASSIFICATION_THRESHOLD = 0.5

# ======================================================
# Logging
# ======================================================

logger = logging.getLogger(__name__)


# ======================================================
# Metric Computation
# ======================================================

def compute_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Computes classification predictions and evaluation metrics.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth labels.

    probabilities : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    tuple
        Binary predictions and a dictionary containing Accuracy,
        Log Loss, Brier Score, and ROC-AUC.
    """
    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    metrics = {
        "Accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "Log_Loss": float(
            log_loss(y_true, probabilities)
        ),
        "Brier_Score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "ROC_AUC": float(
            roc_auc_score(y_true, probabilities)
        ),
    }

    return predictions, metrics


# ======================================================
# Model Evaluation
# ======================================================

def evaluate_all_models(
    artifacts: TrainingArtifacts,
) -> Tuple[
    Dict[str, Dict[str, float]],
    np.ndarray,
    np.ndarray,
]:
    """
    Evaluates every trained model together with the weighted ensemble.

    Returns
    -------
    tuple
        - Dictionary of evaluation metrics.
        - Ensemble binary predictions.
        - Ensemble probability predictions.
    """
    
    mlp_probs = artifacts.mlp.final_model.predict_proba(
        artifacts.mlp.feature_set.X_test_processed
    )[:, 1]

    xgb_probs = artifacts.xgb.final_model.predict_proba(
        artifacts.xgb.feature_set.X_test
    )[:, 1]

    lr_probs = artifacts.lr.final_model.predict_proba(
        artifacts.lr.feature_set.X_test_processed
    )[:, 1]

    weights = artifacts.ensemble_weights

    ensemble_probs = (
        weights[0] * mlp_probs
        + weights[1] * xgb_probs
        + weights[2] * lr_probs
    )

    model_probabilities = {
        "MLP": mlp_probs,
        "XGBoost": xgb_probs,
        "Logistic Regression": lr_probs,
        "Ensemble": ensemble_probs,
    }

    metrics = {}
    predictions = {}

    for model_name, probs in model_probabilities.items():
        preds, model_metrics = compute_metrics(
            artifacts.data.y_test,
            probs,
        )

        predictions[model_name] = preds
        metrics[model_name] = model_metrics

    return (
        metrics,
        predictions["Ensemble"],
        ensemble_probs,
    )


# ======================================================
# Reporting
# ======================================================

def print_metrics(
    metrics: Dict[str, Dict[str, float]],
) -> None:
    """
    Logs a formatted comparison table containing the evaluation
    metrics for every model.
    """
    logger.info("\n      Final Evaluation Metrics:")
    logger.info(
        "      %-20s %-8s %-10s %-8s %-8s",
        "Model",
        "Acc",
        "LogLoss",
        "Brier",
        "ROC-AUC",
    )

    logger.info("      %s", "─" * 60)

    for model_name, values in metrics.items():
        logger.info(
            "      %-20s %-8.1f %-10.3f %-8.3f %-8.3f",
            model_name,
            values["Accuracy"] * 100,
            values["Log_Loss"],
            values["Brier_Score"],
            values["ROC_AUC"],
        )

    logger.info("")


def save_metrics(
    metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Saves the evaluation metrics as a CSV comparison table.
    """
    comparison_df = pd.DataFrame(metrics).T
    comparison_df.index.name = "Model"

    comparison_df.to_csv(
        output_dir / "01_model_comparison.csv"
    )