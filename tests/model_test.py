"""
Tests for the ensemble weighting module.

Validates the mathematical properties of the learned blending formula,
specifically verifying the non-negativity and sum-to-one constraints
required for valid probability combinations.
"""

import numpy as np
import pytest
import pandas as pd

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
        # Ensure you import these at the top of your test file!
        from training.config import ModelArtifacts, FeatureSet
        
        mlp_mock = MockCalibratedModel([0.9, 0.1, 0.8])
        xgb_mock = MockCalibratedModel([0.85, 0.15, 0.7])
        lr_mock = MockCalibratedModel([0.95, 0.05, 0.9])
    
        # Dummy validation data arrays
        dummy_processed_val = np.zeros((3, 5))
        dummy_raw_val = pd.DataFrame(np.zeros((3, 5)))
        y_val = pd.Series([1, 0, 1])
        
        # CHANGE: Package models and data into ModelArtifacts
        mlp_artifact = ModelArtifacts(
            model=mlp_mock, 
            feature_set=FeatureSet(X_train=None, X_val=None, X_test=None, X_val_processed=dummy_processed_val)
        )
        xgb_artifact = ModelArtifacts(
            model=xgb_mock, 
            feature_set=FeatureSet(X_train=None, X_val=dummy_raw_val, X_test=None)
        )
        lr_artifact = ModelArtifacts(
            model=lr_mock, 
            feature_set=FeatureSet(X_train=None, X_val=None, X_test=None, X_val_processed=dummy_processed_val)
        )
    
        # CHANGE: Pass the 4 new arguments
        weights, formula = learn_ensemble_weights(
            mlp_artifact,
            xgb_artifact,
            lr_artifact,
            y_val,
        )
        
        # (Keep your existing assertions here)
        assert np.isclose(np.sum(weights), 1.0)