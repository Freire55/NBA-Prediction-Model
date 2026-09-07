"""
Learns optimal convex ensemble weights across NBA prediction models.

This module combines validation-set probability distributions from calibrated
models (MLP, XGBoost, Logistic Regression, and optionally Pace-Modulated Margin)
by minimizing logarithmic loss on the probability simplex:

    minimize_{w} - 1/N sum_{i=1}^N [ y_i ln(p_i(w)) + (1 - y_i) ln(1 - p_i(w)) ]
    subject to:
        sum_{j=1}^M w_j = 1.0
        w_j >= 0.0  for all j in {1, ..., M}

Sequential Least Squares Programming (SLSQP) solves this constrained convex
optimization, ensuring the ensemble output remains a valid probability distribution.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from training.config import ModelArtifacts

logger = logging.getLogger(__name__)

# Numerical clipping threshold to avoid log(0) divergences in log-loss calculations
LOG_LOSS_EPSILON = 1e-15


# ======================================================
# Probability Extraction
# ======================================================

def _extract_validation_probabilities(
    mlp: ModelArtifacts,
    xgb: ModelArtifacts,
    lr: ModelArtifacts,
    catboost: Optional[ModelArtifacts] = None,
    margin: Optional[ModelArtifacts] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts positive-class validation probability forecasts from all available models.

    Returns:
        Tuple of (stacked_predictions_matrix of shape [N, M], list_of_model_names)
    """
    mlp_probs = mlp.model.predict_proba(mlp.feature_set.X_val_processed)[:, 1]
    xgb_probs = xgb.model.predict_proba(xgb.feature_set.X_val)[:, 1]

    columns = [mlp_probs, xgb_probs]
    names = ["MLP", "XGBoost"]

    if catboost is not None and catboost.model is not None and catboost.feature_set is not None:
        cb_probs = catboost.model.predict_proba(catboost.feature_set.X_val)[:, 1]
        columns.append(cb_probs)
        names.append("CatBoost")

    if lr is not None and lr.model is not None and lr.feature_set is not None and lr.feature_set.X_val_processed is not None:
        lr_probs = lr.model.predict_proba(lr.feature_set.X_val_processed)[:, 1]
        columns.append(lr_probs)
        names.append("Logistic Regression")

    if margin is not None and margin.model is not None:
        reg_type = getattr(getattr(margin.model, "regressor", margin.model), "model_type", "ridge")
        X_val = (
            margin.feature_set.X_val_processed
            if reg_type == "ridge"
            else margin.feature_set.X_val
        )
        val_pace = (
            margin.feature_set.X_val["MATCHUP_EXPECTED_PACE"].to_numpy(dtype=np.float32)
            if isinstance(margin.feature_set.X_val, pd.DataFrame)
            and "MATCHUP_EXPECTED_PACE" in margin.feature_set.X_val.columns
            else None
        )
        try:
            margin_probs = margin.model.predict_proba(X_val, pace=val_pace)[:, 1]
        except TypeError:
            margin_probs = margin.model.predict_proba(X_val)[:, 1]
        columns.append(margin_probs)
        names.append("Pace Margin (CDF)")

    return np.column_stack(columns), names


# ======================================================
# Optimization Engine
# ======================================================

def _optimize_weights_slsqp(
    predictions_matrix: np.ndarray,
    y_true: pd.Series,
) -> np.ndarray:
    """
    Solves the constrained convex optimization problem over the probability simplex.

    Args:
        predictions_matrix: Array of shape (n_samples, n_models) containing probabilities.
        y_true: Ground truth binary target vector.

    Returns:
        Optimal weight vector w of shape (n_models,) summing to 1.0 with w_j >= 0.
    """
    n_models = predictions_matrix.shape[1]

    def _cross_entropy_objective(weights: np.ndarray) -> float:
        blended = np.dot(predictions_matrix, weights)
        blended_safe = np.clip(blended, LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
        return float(log_loss(y_true, blended_safe))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    initial_weights = np.full(n_models, 1.0 / n_models)

    result = minimize(
        _cross_entropy_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        logger.warning(
            "      Ensemble optimization failed to converge (%s). Falling back to equal weights.",
            result.message,
        )
        return initial_weights

    # Ensure strictly normalized weights (guard against numerical precision fuzz)
    normalized = np.maximum(result.x, 0.0)
    return normalized / np.sum(normalized)


# ======================================================
# Main Pipeline API
# ======================================================

def learn_ensemble_weights(
    mlp: ModelArtifacts,
    xgb: ModelArtifacts,
    lr: Optional[ModelArtifacts] = None,
    y_val: pd.Series = None,
    catboost: Optional[ModelArtifacts] = None,
    margin: Optional[ModelArtifacts] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Learns non-negative ensemble blending weights using validation set predictions.

    Args:
        mlp: Trained MLP model artifact.
        xgb: Trained XGBoost model artifact.
        lr: Trained Logistic Regression model artifact.
        y_val: Ground truth validation labels.
        catboost: Optional trained CatBoost model artifact.
        margin: Optional trained Pace-Modulated Margin model artifact.

    Returns:
        Tuple containing:
            - Normalized weight vector
            - Dictionary mapping model names to learned blending weights
    """
    logger.info("      Optimizing ensemble weights via constrained log-loss minimization (SLSQP)...")

    predictions_matrix, model_names = _extract_validation_probabilities(
        mlp=mlp, xgb=xgb, lr=lr, catboost=catboost, margin=margin
    )
    weights = _optimize_weights_slsqp(predictions_matrix, y_val)

    formula = {name: float(w) for name, w in zip(model_names, weights)}

    formula_str = " + ".join([f"({w:.3f} * {name})" for name, w in formula.items()])
    logger.info("      Learned Formula: %s", formula_str)

    return weights, formula