"""
Learns the optimal ensemble weights for the prediction models.

This module combines the validation-set probability predictions from the
calibrated MLP, XGBoost, and Logistic Regression models using Non-Negative
Least Squares (NNLS).
"""

from typing import Any, Dict, Tuple

import logging
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


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

    # Define the Log Loss objective function
    def objective(weights: np.ndarray) -> float:
        blended_probs = np.dot(stacked_predictions, weights)
        # Clip to prevent log(0) explosion
        blended_probs = np.clip(blended_probs, 1e-15, 1 - 1e-15)
        return log_loss(y_val, blended_probs)

    # Constraint: Weights must sum to 1.0
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    # Bounds: Weights must be between 0.0 and 1.0
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    # Initial guess: Equal weighting
    initial_guess = [1 / 3, 1 / 3, 1 / 3]

    # Run the Sequential Least Squares Programming optimizer
    result = minimize(
        objective, 
        initial_guess, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=constraints,
        options={
            "ftol": 1e-9,
            "maxiter": 1000,
        },
    )

    if not result.success:
        logger.warning(
            "      Ensemble optimization failed (%s). Using equal weights.",
            result.message,
        )
        normalized_weights = np.full(3, 1 / 3)
    else:
        normalized_weights = result.x
    
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