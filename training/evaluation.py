"""
Evaluates trained models and computes prospective testing classification metrics.

This module generates out-of-sample probability forecasts across all fitted
architectures (MLP, XGBoost, Logistic Regression, Pace-Modulated Margin Regressor,
and the convex Ensemble). It assesses performance across complementary metrics:
- Accuracy: Proportion of games where argmax(P(Home Win)) matched the winner
- Log Loss: Cross-entropy penalizing overconfident miscalibrated errors
- Brier Score: Mean squared error of probabilities: (1/N) sum (p_i - y_i)^2
- ROC-AUC: Probability that a random home win was scored higher than a random home loss

Outputs:
    01_model_comparison.csv
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from training.config import TrainingArtifacts
from training.margin import MarginRegressor, PaceModulatedMarginClassifier

logger = logging.getLogger(__name__)

CLASSIFICATION_THRESHOLD = 0.5


# ======================================================
# Metric Calculation
# ======================================================

def compute_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Computes binary thresholded predictions and probabilistic classification metrics.

    Args:
        y_true: Ground truth binary target vector (1 for home win, 0 for away win).
        probabilities: Predicted win probabilities for the positive class (Home Win).

    Returns:
        Tuple of (binary_predictions_vector, metrics_dictionary)
    """
    safe_probs = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    predictions = (safe_probs >= CLASSIFICATION_THRESHOLD).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Log_Loss": float(log_loss(y_true, safe_probs)),
        "Brier_Score": float(brier_score_loss(y_true, safe_probs)),
        "ROC_AUC": float(roc_auc_score(y_true, safe_probs)),
    }
    return predictions, metrics


# ======================================================
# Prediction Extraction Helpers
# ======================================================

def _extract_margin_test_probabilities(
    artifacts: TrainingArtifacts,
) -> np.ndarray | None:
    """Extracts positive-class win probabilities from the fitted margin model if available."""
    margin_model = artifacts.margin.final_model
    if margin_model is None or artifacts.margin.feature_set is None:
        return None

    if isinstance(margin_model, PaceModulatedMarginClassifier):
        reg_type = getattr(margin_model.regressor, "model_type", "ridge")
        X_test = (
            artifacts.margin.feature_set.X_test_processed
            if reg_type == "ridge"
            else artifacts.margin.feature_set.X_test
        )
        return margin_model.predict_proba(X_test)[:, 1]

    if isinstance(margin_model, MarginRegressor):
        reg_type = margin_model.model_type
        X_test = (
            artifacts.margin.feature_set.X_test_processed
            if reg_type == "ridge"
            else artifacts.margin.feature_set.X_test
        )
        margins = margin_model.predict(X_test)
        sigma = getattr(margin_model, "residual_std_", 12.5)
        from scipy.special import ndtr
        return np.clip(ndtr(margins / sigma), 1e-7, 1.0 - 1e-7)

    return None


def _collect_test_probabilities(
    artifacts: TrainingArtifacts,
) -> Dict[str, np.ndarray]:
    """Generates out-of-sample probability forecasts for each individual model and ensemble."""
    mlp_probs = artifacts.mlp.final_model.predict_proba(
        artifacts.mlp.feature_set.X_test_processed
    )[:, 1]

    xgb_probs = artifacts.xgb.final_model.predict_proba(
        artifacts.xgb.feature_set.X_test
    )[:, 1]

    lr_probs = artifacts.lr.final_model.predict_proba(
        artifacts.lr.feature_set.X_test_processed
    )[:, 1]

    probs_dict = {
        "MLP": mlp_probs,
        "XGBoost": xgb_probs,
        "Logistic Regression": lr_probs,
    }

    # Margin Model
    margin_probs = _extract_margin_test_probabilities(artifacts)
    if margin_probs is not None:
        probs_dict["Pace Margin (CDF)"] = margin_probs

    # Weighted Ensemble
    weights = artifacts.ensemble_weights
    if weights is not None:
        if len(weights) == 3:
            ensemble_probs = (
                weights[0] * mlp_probs
                + weights[1] * xgb_probs
                + weights[2] * lr_probs
            )
        elif len(weights) == 4 and margin_probs is not None:
            ensemble_probs = (
                weights[0] * mlp_probs
                + weights[1] * xgb_probs
                + weights[2] * lr_probs
                + weights[3] * margin_probs
            )
        else:
            ensemble_probs = (
                weights[0] * mlp_probs
                + weights[1] * xgb_probs
                + weights[2] * lr_probs
            )
        probs_dict["Ensemble"] = ensemble_probs

    return probs_dict


# ======================================================
# Main Pipeline API
# ======================================================

def evaluate_all_models(
    artifacts: TrainingArtifacts,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray]:
    """
    Evaluates every trained model alongside the weighted ensemble on the test set.

    Returns:
        Tuple containing:
            - Dictionary mapping each model name to its performance metrics dict
            - Ensemble binary predictions
            - Ensemble probability predictions
    """
    prob_dict = _collect_test_probabilities(artifacts)
    y_test = artifacts.data.y_test

    metrics = {}
    predictions = {}

    for model_name, probs in prob_dict.items():
        preds, model_metrics = compute_metrics(y_test, probs)
        predictions[model_name] = preds
        metrics[model_name] = model_metrics

    ensemble_key = "Ensemble" if "Ensemble" in prob_dict else "XGBoost"
    return metrics, predictions[ensemble_key], prob_dict[ensemble_key]


# ======================================================
# Reporting & Serialization
# ======================================================

def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Logs a formatted comparison table containing evaluation metrics for each model."""
    logger.info("\n      Final Evaluation Metrics:")
    logger.info(
        "      %-22s %-8s %-10s %-8s %-8s",
        "Model", "Acc", "LogLoss", "Brier", "ROC-AUC"
    )
    logger.info("      %s", "─" * 62)

    for model_name, values in metrics.items():
        logger.info(
            "      %-22s %-8.1f %-10.3f %-8.3f %-8.3f",
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
    """Saves the evaluation metrics as a standardized CSV comparison table."""
    comparison_df = pd.DataFrame(metrics).T
    comparison_df.index.name = "Model"
    comparison_df.to_csv(output_dir / "01_model_comparison.csv")