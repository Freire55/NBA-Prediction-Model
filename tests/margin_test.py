"""
Unit tests for the Continuous Margin Regressor module (Step 7).

Validates:
1. MarginRegressor estimator API compatibility (fit, predict, get_residuals).
2. Support for both Ridge and XGBoost regression architectures.
3. Residual standard error (sigma) and bias calculation.
4. Evaluation metrics (RMSE, MAE, R², Directional Accuracy).
5. Monotonicity: superior ratings consistently produce higher predicted margins.
6. Zero leakage: no post-game statistics or raw targets in the margin feature set.
7. End-to-end hyperparameter tuning with chronological TimeSeriesSplit.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from training.config import DatasetSummary, FeatureSet, TrainingConfig, TrainingData
from training.margin import (
    MarginRegressor,
    PaceModulatedMarginClassifier,
    evaluate_classification_metrics,
    evaluate_margin_metrics,
    tune_margin_regressor,
    tune_pace_margin_classifier,
)


# ======================================================
# Test Fixtures
# ======================================================

@pytest.fixture
def synthetic_margin_data():
    """Generates synthetic training and validation data for margin regression."""
    np.random.seed(42)
    n_train = 60
    n_val = 20

    # Simulate features: Delta Net Rating, Delta Elo, Rest Advantage
    X_train_df = pd.DataFrame({
        "DELTA_NET_RATING": np.random.normal(0, 5, n_train),
        "DELTA_ELO": np.random.normal(0, 100, n_train),
        "REST_ADVANTAGE": np.random.choice([-1, 0, 1, 2], n_train),
    })

    # Ground truth: continuous point differential + noise
    # Margin ~ 2.5 (HCA) + 0.8 * NetRating + 0.03 * DeltaElo + 1.5 * Rest + noise
    y_train = (
        2.5
        + 0.8 * X_train_df["DELTA_NET_RATING"]
        + 0.03 * X_train_df["DELTA_ELO"]
        + 1.5 * X_train_df["REST_ADVANTAGE"]
        + np.random.normal(0, 3, n_train)
    ).astype("float32")

    X_val_df = pd.DataFrame({
        "DELTA_NET_RATING": np.random.normal(0, 5, n_val),
        "DELTA_ELO": np.random.normal(0, 100, n_val),
        "REST_ADVANTAGE": np.random.choice([-1, 0, 1, 2], n_val),
    })

    y_val = (
        2.5
        + 0.8 * X_val_df["DELTA_NET_RATING"]
        + 0.03 * X_val_df["DELTA_ELO"]
        + 1.5 * X_val_df["REST_ADVANTAGE"]
        + np.random.normal(0, 3, n_val)
    ).astype("float32")

    return X_train_df, y_train, X_val_df, y_val


@pytest.fixture
def mock_training_data(synthetic_margin_data):
    """Packages synthetic data into the canonical TrainingData dataclass."""
    X_train_df, y_train, X_val_df, y_val = synthetic_margin_data

    margin_fs = FeatureSet(
        X_train=X_train_df,
        X_val=X_val_df,
        X_test=X_val_df.copy(),
        feature_names=list(X_train_df.columns),
    )

    dummy_fs = FeatureSet(
        X_train=X_train_df.copy(),
        X_val=X_val_df.copy(),
        X_test=X_val_df.copy(),
        feature_names=list(X_train_df.columns),
    )

    summary = DatasetSummary(
        train_games=len(X_train_df),
        validation_games=len(X_val_df),
        test_games=len(X_val_df),
        lr_feature_names=list(X_train_df.columns),
        xgb_feature_names=list(X_train_df.columns),
        mlp_feature_names=list(X_train_df.columns),
        margin_feature_names=list(X_train_df.columns),
    )

    return TrainingData(
        mlp=dummy_fs,
        xgb=dummy_fs,
        lr=dummy_fs,
        margin=margin_fs,
        y_train=pd.Series(np.where(y_train > 0, 1, 0)),
        y_val=pd.Series(np.where(y_val > 0, 1, 0)),
        y_test=pd.Series(np.where(y_val > 0, 1, 0)),
        y_margin_train=y_train,
        y_margin_val=y_val,
        y_margin_test=y_val.copy(),
        summary=summary,
    )


# ======================================================
# Test Cases
# ======================================================

def test_margin_regressor_ridge_fit_predict(synthetic_margin_data):
    """Tests fitting and prediction with Ridge MarginRegressor."""
    X_train, y_train, X_val, y_val = synthetic_margin_data

    reg = MarginRegressor(model_type="ridge", alpha=10.0, random_state=42)
    reg.fit(X_train, y_train)

    assert reg.is_fitted_
    preds = reg.predict(X_val)

    assert len(preds) == len(y_val)
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any()
    assert reg.residual_std_ > 0.0


def test_margin_regressor_xgb_fit_predict(synthetic_margin_data):
    """Tests fitting and prediction with XGBoost MarginRegressor."""
    X_train, y_train, X_val, y_val = synthetic_margin_data

    reg = MarginRegressor(
        model_type="xgb",
        xgb_params={"n_estimators": 20, "max_depth": 3},
        random_state=42,
    )
    reg.fit(X_train, y_train)

    assert reg.is_fitted_
    preds = reg.predict(X_val)

    assert len(preds) == len(y_val)
    assert not np.isnan(preds).any()


def test_margin_metrics_evaluation():
    """Validates computation of regression and directional diagnostic metrics."""
    y_true = np.array([10.0, -4.0, 5.0, -8.0, 2.0])
    y_pred = np.array([8.0, -2.0, 4.0, -10.0, -1.0])  # Last one has sign mismatch

    metrics = evaluate_margin_metrics(y_true, y_pred)

    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "R2" in metrics
    assert "Residual_Mean" in metrics
    assert "Residual_Std" in metrics
    assert "Directional_Accuracy" in metrics
    assert "Within_5_Pts_Pct" in metrics

    assert metrics["RMSE"] > 0.0
    assert metrics["MAE"] > 0.0
    # 4 out of 5 signs match -> 80%
    assert metrics["Directional_Accuracy"] == 0.8
    # All errors are <= 3 pts -> 100% within 5 pts
    assert metrics["Within_5_Pts_Pct"] == 1.0


def test_monotonicity_higher_ratings_yield_higher_margins():
    """
    Verifies domain consistency: a team with strictly superior ratings
    must be predicted to win by a higher point differential.
    """
    reg = MarginRegressor(model_type="ridge", alpha=1.0)
    X = np.array([
        [-10.0],  # Heavy underdog
        [0.0],    # Even matchup
        [10.0],   # Heavy favorite
    ])
    y = np.array([-8.0, 2.0, 12.0])
    reg.fit(X, y)

    preds = reg.predict(X)
    assert preds[0] < preds[1] < preds[2], "MarginRegressor violated monotonic ordering!"


def test_tune_margin_regressor_end_to_end(mock_training_data, tmp_path):
    """Verifies that hyperparameter tuning runs cleanly with TimeSeriesSplit."""
    config = TrainingConfig(cv_folds=3, margin_model_type="ridge")
    config.margin_ridge_grid = {"alpha": [1.0, 100.0]}

    artifact, val_metrics = tune_margin_regressor(
        data=mock_training_data,
        config=config,
        model_type="ridge",
        output_dir=tmp_path,
    )

    assert artifact.model is not None
    assert artifact.final_model is not None
    assert isinstance(val_metrics, dict)
    assert "RMSE" in val_metrics
    assert (tmp_path / "margin_best_params.json").exists()
    assert (tmp_path / "margin_cv_results.csv").exists()


def test_margin_feature_set_contains_zero_leakage():
    """Guarantees that the margin feature matrix contains no postgame stats or target columns."""
    from training.data import get_model_features

    mock_df = pd.DataFrame({
        "HOME_WIN": [1, 0],
        "TARGET_MARGIN": [8.0, -5.0],
        "HOME_PTS": [110, 95],
        "AWAY_PTS": [102, 100],
        "DELTA_OFF_RATING": [4.0, -2.0],
        "DELTA_FOUR_FACTOR_EFG": [0.05, -0.03],
        "REST_ADVANTAGE": [1, 0],
    })

    config = TrainingConfig()
    features = get_model_features(mock_df, config)

    assert "margin" in features
    margin_feats = features["margin"]

    forbidden = ["HOME_WIN", "TARGET_MARGIN", "MARGIN", "HOME_PTS", "AWAY_PTS"]
    for feat in margin_feats:
        for f in forbidden:
            assert feat != f, f"Leakage detected in margin features: {feat}!"

    assert "REST_ADVANTAGE" in margin_feats
    assert "DELTA_OFF_RATING" in margin_feats


# ======================================================
# Step 8: Pace-Modulated Normal CDF Tests
# ======================================================

def test_pace_modulated_classifier_proba_properties():
    """Validates that PaceModulatedMarginClassifier outputs valid probability distributions."""
    reg = MarginRegressor(model_type="ridge", alpha=10.0)
    X = np.array([[-5.0], [0.0], [5.0], [12.0]])
    y = np.array([-4.0, 1.0, 6.0, 11.0])
    reg.fit(X, y)

    clf = PaceModulatedMarginClassifier(regressor=reg, sigma_0=12.0, fit_sigma=False)
    clf.fit(X, np.array([0, 1, 1, 1]))

    probs = clf.predict_proba(X)

    assert probs.shape == (4, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # Monotonicity: higher margin must yield strictly higher home win probability
    assert probs[0, 1] < probs[1, 1] < probs[2, 1] < probs[3, 1]

    # Even matchup (near 0 margin) should have win probability near 50%
    preds = clf.predict(X)
    assert preds[0] == 0
    assert preds[3] == 1


def test_pace_modulation_favorite_safety_and_upset_volatility():
    """
    Validates domain invariance:
    - For a heavy favorite (M_hat > 0), a slower pace compresses variance,
      making the favorite safer (higher win probability).
    - For an underdog (M_hat < 0), a faster pace expands variance,
      increasing upset probability.
    """
    reg = MarginRegressor(model_type="ridge", alpha=1.0)
    X = np.array([[10.0], [-10.0]])
    y = np.array([8.0, -8.0])
    reg.fit(X, y)

    clf = PaceModulatedMarginClassifier(regressor=reg, sigma_0=12.0, base_pace=100.0, fit_sigma=False)
    clf.fit(X, np.array([1, 0]))

    slow_pace = np.array([90.0, 90.0])
    fast_pace = np.array([110.0, 110.0])

    probs_slow = clf.predict_proba(X, pace=slow_pace)
    probs_fast = clf.predict_proba(X, pace=fast_pace)

    # Row 0 is the favorite (+8 margin):
    # Under slow pace (low variance), favorite win prob is HIGHER than under fast pace
    assert probs_slow[0, 1] > probs_fast[0, 1], (
        f"Favorite safety violated: slow {probs_slow[0, 1]:.4f} vs fast {probs_fast[0, 1]:.4f}"
    )

    # Row 1 is the underdog (-8 margin):
    # Under fast pace (high variance), underdog win prob is HIGHER than under slow pace
    assert probs_fast[1, 1] > probs_slow[1, 1], (
        f"Underdog upset volatility violated: fast {probs_fast[1, 1]:.4f} vs slow {probs_slow[1, 1]:.4f}"
    )


def test_pace_modulated_classifier_sigma_calibration():
    """Verifies that sigma_0 can be calibrated via log loss minimization."""
    reg = MarginRegressor(model_type="ridge", alpha=1.0)
    X = np.array([[-6.0], [-2.0], [2.0], [6.0]] * 10)
    y_margin = np.array([-5.0, -1.0, 3.0, 7.0] * 10)
    y_binary = np.array([0, 0, 1, 1] * 10)
    reg.fit(X, y_margin)

    clf = PaceModulatedMarginClassifier(regressor=reg, sigma_0=25.0, fit_sigma=True)
    clf.fit(X, y_binary)

    # Sigma should be tuned away from arbitrary initial value
    assert clf.sigma_0_ != 25.0
    assert 5.0 <= clf.sigma_0_ <= 30.0


def test_tune_pace_margin_classifier_end_to_end(mock_training_data, tmp_path):
    """Verifies end-to-end execution of tune_pace_margin_classifier."""
    config = TrainingConfig(cv_folds=3, margin_model_type="ridge")
    config.margin_ridge_grid = {"alpha": [1.0, 100.0]}

    artifact, val_metrics = tune_pace_margin_classifier(
        data=mock_training_data,
        config=config,
        output_dir=tmp_path,
    )

    assert artifact.model is not None
    assert artifact.final_model is not None
    assert "Log_Loss" in val_metrics
    assert "Accuracy" in val_metrics
    assert "ROC_AUC" in val_metrics
    assert "Brier_Score" in val_metrics
    assert "Calibrated_Sigma" in val_metrics
    assert (tmp_path / "pace_margin_classifier_metrics.json").exists()

