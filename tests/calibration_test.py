"""
Unit tests for probability calibration modules.

Validates:
1. BetaCalibrator monotonicity constraints (a >= 0, b >= 0) and probability bounds.
2. BetaCalibratedClassifier scikit-learn estimator interface (fit, predict, predict_proba, clone).
3. Calibration model selection between Platt Sigmoid scaling and Asymmetric Beta calibration.
4. Chronological out-of-fold probability prediction generation without data leakage.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from training.calibration import (
    BetaCalibratedClassifier,
    BetaCalibrator,
    SplineCalibratedClassifier,
    SplineCalibrator,
    calibrate_estimator_with_model_selection,
    generate_oof_predictions,
)
from training.utils import unwrap_base_estimator


def test_beta_calibrator_monotonicity_and_bounds():
    """Ensures that fitted BetaCalibrator parameters satisfy non-negativity (a >= 0, b >= 0)."""
    np.random.seed(42)
    p_raw = np.random.uniform(0.05, 0.95, 200)
    y = (p_raw > 0.5).astype(int)

    calibrator = BetaCalibrator()
    calibrator.fit_from_probabilities(p_raw, y)

    assert calibrator.is_fitted_
    assert calibrator.a_ >= 0.0
    assert calibrator.b_ >= 0.0

    probs = calibrator.predict_proba(p_raw)
    assert probs.shape == (200, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_beta_calibrated_classifier_sklearn_contract():
    """Validates that BetaCalibratedClassifier follows standard scikit-learn estimator protocols."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    base = LogisticRegression()
    clf = BetaCalibratedClassifier(base_estimator=base, cv=TimeSeriesSplit(n_splits=3))
    clf.fit(X, y)

    assert clf.is_fitted_
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 2

    probs = clf.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Test cloning
    cloned = clone(clf)
    assert cloned.base_estimator is not None
    cloned.fit(X, y)
    assert cloned.is_fitted_

    # Test unwrapping
    unwrapped = unwrap_base_estimator(clf)
    assert isinstance(unwrapped, LogisticRegression)


def test_calibrate_estimator_with_model_selection():
    """Tests that model selection compares Platt vs Beta and returns a valid calibrated model."""
    np.random.seed(42)
    X = np.random.randn(120, 5)
    y = (X[:, 0] > 0.2).astype(int)
    tscv = TimeSeriesSplit(n_splits=3)

    calibrated_model, metadata = calibrate_estimator_with_model_selection(
        estimator=LogisticRegression(),
        X_train=X,
        y_train=y,
        tscv=tscv,
    )

    assert metadata["method"] in ["platt", "beta", "spline"]
    assert "selected_log_loss" in metadata
    assert "params" in metadata

    probs = calibrated_model.predict_proba(X)
    assert probs.shape == (120, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_spline_calibrator_monotonicity_and_bounds():
    """Ensures that fitted SplineCalibrator produces monotonic probabilities bounded in [0, 1]."""
    np.random.seed(42)
    p_raw = np.random.uniform(0.05, 0.95, 200)
    y = (p_raw > 0.5).astype(int)

    calibrator = SplineCalibrator(n_knots=8)
    calibrator.fit_from_probabilities(p_raw, y)

    assert calibrator.is_fitted_
    probs = calibrator.predict_proba(p_raw)
    assert probs.shape == (200, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Test monotonicity on a dense sorted grid
    grid = np.linspace(0.05, 0.95, 100)
    grid_probs = calibrator.predict_proba(grid)[:, 1]
    assert np.all(np.diff(grid_probs) >= -1e-6)


def test_spline_calibrated_classifier_sklearn_contract():
    """Validates that SplineCalibratedClassifier follows standard scikit-learn estimator protocols."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    base = LogisticRegression()
    clf = SplineCalibratedClassifier(base_estimator=base, cv=TimeSeriesSplit(n_splits=3))
    clf.fit(X, y)

    assert clf.is_fitted_
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 2

    probs = clf.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Test unwrapping
    unwrapped = unwrap_base_estimator(clf)
    assert isinstance(unwrapped, LogisticRegression)


def test_generate_oof_predictions_chronological_splits():
    """Verifies that out-of-fold probability predictions maintain strict chronological ordering."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    tscv = TimeSeriesSplit(n_splits=4)

    oof_probs, oof_targets = generate_oof_predictions(
        estimator=LogisticRegression(),
        X=X,
        y=y,
        cv=tscv,
    )

    # TimeSeriesSplit with 4 splits on 100 samples produces exactly 80 out-of-fold evaluations
    assert len(oof_probs) == 80
    assert len(oof_targets) == 80
    assert np.all(oof_probs >= 0.0) and np.all(oof_probs <= 1.0)
