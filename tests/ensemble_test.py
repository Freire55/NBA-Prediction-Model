"""
Tests for the ensemble weighting module.

Validates the mathematical properties of the learned blending formula,
specifically verifying the non-negativity and sum-to-one constraints
required for valid probability combinations.
"""

import numpy as np
import pytest

from training.ensemble import learn_ensemble_weights

# ======================================================
# Test Mocks
# ======================================================

class MockCalibratedModel:
    """Simulates a calibrated sklearn estimator for testing."""
    def __init__(self, proba_output: list[float]):
        self.proba_output = proba_output
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([[1 - p, p] for p in self.proba_output])

# ======================================================
# Test Cases
# ======================================================

def test_ensemble_weights_mathematical_constraints():
    """
    Ensures that the constrained Log Loss optimizer respects the 
    defined bounds and strictly returns a normalized distribution.
    """
    mlp_mock = MockCalibratedModel([0.9, 0.1, 0.8])
    xgb_mock = MockCalibratedModel([0.85, 0.15, 0.7])
    lr_mock = MockCalibratedModel([0.95, 0.05, 0.9])
    
    # Dummy validation data
    X_val_scaled = np.zeros((3, 5))
    X_val = np.zeros((3, 5))
    y_val = np.array([1, 0, 1])
    
    weights, formula = learn_ensemble_weights(
        mlp_mock, 
        xgb_mock, 
        lr_mock, 
        X_val_scaled, 
        X_val, 
        y_val,
    )
    
    # Constraint 1: Weights must sum to 1.0 (allowing for tiny float variance)
    assert np.isclose(weights.sum(), 1.0)
    
    # Constraint 2: All weights must be valid probability multipliers
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)