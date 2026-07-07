"""
Learns the optimal ensemble weights for the prediction models.

This module combines the validation-set probability predictions from the
calibrated MLP, XGBoost, and Logistic Regression models using Non-Negative
Least Squares (NNLS).
"""

from typing import Any, Dict, Tuple

import logging
import numpy as np
from scipy.optimize import nnls

# ======================================================
# Logging
# ======================================================

logger = logging.getLogger(__name__)


# ======================================================
# Ensemble Weight Learning
# ======================================================

def learn_ensemble_weights(
    mlp_model: Any,
    xgb_model: Any,
    lr_model: Any,
    X_val_scaled: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Learns non-negative ensemble weights using validation predictions.

    Assumes base models are calibrated (e.g., via CalibratedClassifierCV) 
    to ensure reliable probability estimation for the blending weights.

    Parameters
    ----------
    mlp_model : Any
        Calibrated MLP classifier.

    xgb_model : Any
        Calibrated XGBoost classifier.

    lr_model : Any
        Calibrated Logistic Regression classifier.

    X_val_scaled : np.ndarray
        Standardized validation features (used by MLP and LR).

    X_val : np.ndarray
        Original validation features (used by XGBoost).

    y_val : np.ndarray
        Validation labels.

    Returns
    -------
    tuple
        A tuple containing:
        - normalized ensemble weights
        - dictionary describing the learned ensemble formula
    """
    # Generate validation probabilities from calibrated models
    mlp_probs = mlp_model.predict_proba(X_val_scaled)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    lr_probs = lr_model.predict_proba(X_val_scaled)[:, 1]

    # Stack predictions for optimization
    stacked_predictions = np.column_stack(
        (mlp_probs, xgb_probs, lr_probs)
    )

    # Learn optimal non-negative weights
    weights, _ = nnls(stacked_predictions, y_val)

    total_weight = weights.sum()

    # Fallback to uniform weighting if NNLS produces all zeros
    if total_weight == 0:
        normalized_weights = np.full(3, 1 / 3)
    else:
        normalized_weights = weights / total_weight

    ensemble_formula = {
        "MLP": float(normalized_weights[0]),
        "XGBoost": float(normalized_weights[1]),
        "Logistic Regression": float(normalized_weights[2]),
    }

    logger.info(
        "      Learned Formula: "
        "(%.3f * MLP) + (%.3f * XGB) + (%.3f * LR)",
        ensemble_formula["MLP"],
        ensemble_formula["XGBoost"],
        ensemble_formula["Logistic Regression"],
    )

    return normalized_weights, ensemble_formula