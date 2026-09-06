"""
Tests for optimize_features.py.

Validates model-specific feature reduction, candidate deduplication,
and fast-track logloss computations.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from training.config import DatasetSummary, FeatureSet, TrainingData
from optimize_features import (
    apply_model_specific_reduction,
    format_time,
    partial_fast_track,
)


@pytest.fixture
def mock_training_data():
    X_train = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feat_b": [2.0, 1.0, 0.5, 3.0, 1.5, 2.5],
        "feat_c": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    X_val = pd.DataFrame({
        "feat_a": [1.5, 2.5],
        "feat_b": [1.0, 2.0],
        "feat_c": [0.2, 0.4],
    })
    X_test = X_val.copy()
    y_train = pd.Series([1, 0, 1, 0, 1, 0])
    y_val = pd.Series([1, 0])
    y_test = pd.Series([1, 0])

    fs = FeatureSet(
        X_train=X_train.copy(),
        X_val=X_val.copy(),
        X_test=X_test.copy(),
        feature_names=["feat_a", "feat_b", "feat_c"]
    )
    summary = DatasetSummary(
        train_games=6,
        validation_games=2,
        test_games=2,
        lr_feature_names=["feat_a", "feat_b", "feat_c"],
        xgb_feature_names=["feat_a", "feat_b", "feat_c"],
        mlp_feature_names=["feat_a", "feat_b", "feat_c"],
    )
    return TrainingData(
        mlp=fs,
        xgb=FeatureSet(X_train=X_train.copy(), X_val=X_val.copy(), X_test=X_test.copy(), feature_names=["feat_a", "feat_b", "feat_c"]),
        lr=FeatureSet(X_train=X_train.copy(), X_val=X_val.copy(), X_test=X_test.copy(), feature_names=["feat_a", "feat_b", "feat_c"]),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        summary=summary,
    )


def test_apply_model_specific_reduction(mock_training_data):
    removals = {"mlp": ["feat_b"], "xgb": ["feat_c"], "lr": []}
    reduced = apply_model_specific_reduction(mock_training_data, removals)

    assert "feat_b" not in reduced.mlp.feature_names
    assert "feat_b" not in reduced.mlp.X_train.columns
    assert "feat_a" in reduced.mlp.feature_names
    assert "feat_c" in reduced.mlp.feature_names

    assert "feat_c" not in reduced.xgb.feature_names
    assert "feat_c" not in reduced.xgb.X_train.columns

    assert len(reduced.lr.feature_names) == 3
    assert reduced.mlp.X_train_processed is not None
    assert reduced.lr.X_train_processed is not None


def test_format_time():
    assert format_time(45) == "45s"
    assert format_time(125) == "2m 5s"
    assert format_time(0) == "0s"


def test_partial_fast_track_baseline_matching(mock_training_data):
    removals = {"mlp": [], "xgb": [], "lr": []}
    reduced = apply_model_specific_reduction(mock_training_data, removals)

    clf = LogisticRegression(random_state=42)
    clf.fit(reduced.lr.X_train_processed, reduced.y_train)

    base_estimators = {"mlp": clf, "xgb": clf, "lr": clf}
    cached_probs = {
        "mlp": np.array([0.6, 0.4]),
        "xgb": np.array([0.6, 0.4]),
        "lr": np.array([0.6, 0.4]),
    }
    weights = np.array([0.0, 0.0, 1.0])

    base_loss = partial_fast_track("lr", reduced, base_estimators, cached_probs, weights)
    assert isinstance(base_loss, float)
    assert base_loss > 0.0

    # Removing nothing yields exact same loss (diff = 0.0)
    repeat_loss = partial_fast_track("lr", reduced, base_estimators, cached_probs, weights)
    assert pytest.approx(base_loss - repeat_loss, abs=1e-9) == 0.0


def test_get_model_importance_ranking():
    from optimize_features import get_model_importance
    
    np.random.seed(42)
    n = 100
    # Predictive feature correlates with target; noise feature has zero signal
    x_pred = np.random.randn(n)
    x_noise = np.random.randn(n)
    y = (x_pred > 0).astype(int)
    
    X = pd.DataFrame({"noise": x_noise, "signal": x_pred})
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    
    ranked = get_model_importance(clf, X, y, ["noise", "signal"], random_seed=42)
    # The noise feature must be ranked weaker (first in the queue) than the signal feature
    assert ranked[0] == "noise"
    assert ranked[1] == "signal"


def test_fit_and_calibrate_single_model(mock_training_data):
    from optimize_features import fit_and_calibrate_single_model
    
    removals = {"mlp": [], "xgb": [], "lr": []}
    reduced = apply_model_specific_reduction(mock_training_data, removals)
    base_lr = LogisticRegression(random_state=42)
    
    cal_lr, probs, standalone_loss = fit_and_calibrate_single_model(
        "lr", base_lr, reduced, cv_folds=2
    )
    
    assert cal_lr is not None
    assert len(probs) == len(mock_training_data.y_val)
    assert isinstance(standalone_loss, float)
    assert 0.0 < standalone_loss < 2.0
