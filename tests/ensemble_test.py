"""
Unit tests for the constrained meta-ensemble weighting module.

Validates the mathematical properties of the learned blending formula,
specifically verifying the non-negativity and sum-to-one simplex constraints
across any combination of models (3-model, 4-model, and 5-model blends).
"""

import numpy as np
import pandas as pd
import pytest

from training.config import FeatureSet, ModelArtifacts
from training.ensemble import learn_ensemble_weights


class MockCalibratedModel:
    """Simulates a calibrated estimator with fixed positive class probability outputs."""

    def __init__(self, proba_output: list[float]) -> None:
        self.proba_output = proba_output

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([[1.0 - p, p] for p in self.proba_output])


def test_3_model_ensemble_weights_simplex():
    """
    Ensures that the constrained Log Loss optimizer respects simplex bounds
    and strictly returns a normalized distribution across 3 base models.
    """
    mlp_mock = MockCalibratedModel([0.9, 0.1, 0.8])
    xgb_mock = MockCalibratedModel([0.85, 0.15, 0.7])
    lr_mock = MockCalibratedModel([0.95, 0.05, 0.9])

    dummy_processed_val = np.zeros((3, 5))
    dummy_raw_val = pd.DataFrame(np.zeros((3, 5)))
    y_val = pd.Series([1, 0, 1])

    mlp_artifact = ModelArtifacts(
        model=mlp_mock,
        feature_set=FeatureSet(X_train=None, X_val=None, X_test=None, X_val_processed=dummy_processed_val),
    )
    xgb_artifact = ModelArtifacts(
        model=xgb_mock,
        feature_set=FeatureSet(X_train=None, X_val=dummy_raw_val, X_test=None),
    )
    lr_artifact = ModelArtifacts(
        model=lr_mock,
        feature_set=FeatureSet(X_train=None, X_val=None, X_test=None, X_val_processed=dummy_processed_val),
    )

    weights, formula = learn_ensemble_weights(
        mlp=mlp_artifact,
        xgb=xgb_artifact,
        lr=lr_artifact,
        y_val=y_val,
    )

    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)
    assert set(formula.keys()) == {"MLP", "XGBoost", "Logistic Regression"}


def test_5_model_ensemble_weights_simplex():
    """Verifies that the SLSQP optimizer learns valid simplex weights across all 5 models."""
    mlp_mock = MockCalibratedModel([0.80, 0.20, 0.70])
    xgb_mock = MockCalibratedModel([0.75, 0.25, 0.65])
    cb_mock = MockCalibratedModel([0.85, 0.15, 0.75])
    lr_mock = MockCalibratedModel([0.70, 0.30, 0.60])
    margin_mock = MockCalibratedModel([0.90, 0.10, 0.85])

    dummy_val = pd.DataFrame(np.zeros((3, 4)))
    dummy_proc = np.zeros((3, 4))
    y_val = pd.Series([1, 0, 1])

    mlp_art = ModelArtifacts(model=mlp_mock, feature_set=FeatureSet(None, None, None, X_val_processed=dummy_proc))
    xgb_art = ModelArtifacts(model=xgb_mock, feature_set=FeatureSet(None, dummy_val, None))
    cb_art = ModelArtifacts(model=cb_mock, feature_set=FeatureSet(None, dummy_val, None))
    lr_art = ModelArtifacts(model=lr_mock, feature_set=FeatureSet(None, None, None, X_val_processed=dummy_proc))
    margin_art = ModelArtifacts(model=margin_mock, feature_set=FeatureSet(None, dummy_val, None))

    weights, formula = learn_ensemble_weights(
        mlp=mlp_art,
        xgb=xgb_art,
        lr=lr_art,
        y_val=y_val,
        catboost=cb_art,
        margin=margin_art,
    )

    assert len(weights) == 5
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)
    assert set(formula.keys()) == {"MLP", "XGBoost", "CatBoost", "Logistic Regression", "Pace Margin (CDF)"}


def test_4_model_ensemble_weights_simplex_without_lr():
    """Verifies that the SLSQP optimizer learns valid simplex weights when LR is retired."""
    mlp_mock = MockCalibratedModel([0.80, 0.20, 0.70])
    xgb_mock = MockCalibratedModel([0.75, 0.25, 0.65])
    cb_mock = MockCalibratedModel([0.85, 0.15, 0.75])
    margin_mock = MockCalibratedModel([0.90, 0.10, 0.85])

    dummy_val = pd.DataFrame(np.zeros((3, 4)))
    dummy_proc = np.zeros((3, 4))
    y_val = pd.Series([1, 0, 1])

    mlp_art = ModelArtifacts(model=mlp_mock, feature_set=FeatureSet(None, None, None, X_val_processed=dummy_proc))
    xgb_art = ModelArtifacts(model=xgb_mock, feature_set=FeatureSet(None, dummy_val, None))
    cb_art = ModelArtifacts(model=cb_mock, feature_set=FeatureSet(None, dummy_val, None))
    margin_art = ModelArtifacts(model=margin_mock, feature_set=FeatureSet(None, dummy_val, None))

    weights, formula = learn_ensemble_weights(
        mlp=mlp_art,
        xgb=xgb_art,
        lr=None,
        y_val=y_val,
        catboost=cb_art,
        margin=margin_art,
    )

    assert len(weights) == 4
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)
    assert set(formula.keys()) == {"MLP", "XGBoost", "CatBoost", "Pace Margin (CDF)"}
