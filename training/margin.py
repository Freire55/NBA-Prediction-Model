"""
Continuous Margin-of-Victory Regressor and Pace-Modulated Normal CDF Classifier.

This module implements continuous margin regression and pace-modulated probability conversion:
- Continuous Margin Regressor predicting expected point differential
  (Delta PTS = HOME_PTS - AWAY_PTS) via regularized Ridge and tree regressors
  with TimeSeriesSplit cross-validation and residual diagnostic analysis.
- Pace-Modulated Normal CDF Converter translating continuous margin
  forecasts into well-calibrated win probabilities:
      sigma_game = sigma_0 * sqrt(expected_pace / 100.0)
      P(Home Win) = Phi(M_hat / sigma_game)
  exposing a standard scikit-learn .predict_proba() API.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from training.config import ModelArtifacts, TrainingConfig, TrainingData
from training.data import scale_features
from training.utils import save_json

logger = logging.getLogger(__name__)

# ======================================================
# Output Artifact Filenames
# ======================================================

MARGIN_PARAMS_FILE = "margin_best_params.json"
MARGIN_RESULTS_FILE = "margin_cv_results.csv"
PACE_MARGIN_METRICS_FILE = "pace_margin_classifier_metrics.json"

DEFAULT_BASE_PACE = 100.0
PACE_COLUMN_NAME = "MATCHUP_EXPECTED_PACE"


# ======================================================
# Metrics & Diagnostics
# ======================================================

def evaluate_margin_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Computes regression performance metrics and residual error diagnostics.

    Args:
        y_true: Ground truth point differential (HOME_PTS - AWAY_PTS).
        y_pred: Predicted point differential.

    Returns:
        Dictionary containing RMSE, MAE, R², Residual Mean (Bias),
        Residual Std (Sigma), Implied Directional Accuracy, and tolerance percentages.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    residuals = y_t - y_p

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    bias = float(np.mean(residuals))
    std = float(np.std(residuals))

    # Implied win/loss classification accuracy (positive predicted margin = predicted home win)
    dir_acc = float(np.mean((y_p > 0) == (y_t > 0)))

    # Error tolerances: % of predictions within N points of actual final score
    abs_err = np.abs(residuals)
    within_5 = float(np.mean(abs_err <= 5.0))
    within_10 = float(np.mean(abs_err <= 10.0))

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Residual_Mean": round(bias, 4),
        "Residual_Std": round(std, 4),
        "Directional_Accuracy": round(dir_acc, 4),
        "Within_5_Pts_Pct": round(within_5, 4),
        "Within_10_Pts_Pct": round(within_10, 4),
    }


def evaluate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Computes standard probabilistic classification metrics.

    Args:
        y_true: Binary outcome (1 for home win, 0 for away win).
        y_prob: Predicted home win probability P(y = 1).

    Returns:
        Dictionary containing Log Loss, Accuracy, ROC-AUC, and Brier Score.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(y_prob, dtype=np.float64)
    p_clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    preds = (p_clipped >= 0.5).astype(np.int32)

    return {
        "Log_Loss": round(float(log_loss(y_t, p_clipped)), 4),
        "Accuracy": round(float(accuracy_score(y_t, preds)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_t, p_clipped)), 4),
        "Brier_Score": round(float(brier_score_loss(y_t, p_clipped)), 4),
    }


# ======================================================
# Margin Regressor Estimator
# ======================================================

class MarginRegressor(BaseEstimator, RegressorMixin):
    """
    Estimator predicting continuous game point differential (HOME_PTS - AWAY_PTS).

    Supports Ridge regression (optimal for linear additive differentials),
    XGBoost regression (for non-linear blowout dynamics), and
    CatBoost regression (symmetric oblivious trees with ordered boosting).
    Computes and stores out-of-fold and training residual statistics (sigma)
    required for Normal CDF win-probability conversion.
    """

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 100.0,
        xgb_params: Optional[Dict[str, Any]] = None,
        catboost_params: Optional[Dict[str, Any]] = None,
        scaler: Optional[StandardScaler] = None,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.alpha = alpha
        self.xgb_params = xgb_params or {}
        self.catboost_params = catboost_params or {}
        self.scaler = scaler
        self.random_state = random_state

        self.estimator_: Optional[BaseEstimator] = None
        self.residual_mean_: float = 0.0
        self.residual_std_: float = 13.5
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _build_estimator(self) -> BaseEstimator:
        """Instantiates the underlying regression algorithm."""
        if self.model_type == "ridge":
            return Ridge(
                alpha=self.alpha,
                random_state=self.random_state,
            )
        elif self.model_type == "xgb":
            default_xgb = {
                "learning_rate": 0.05,
                "max_depth": 5,
                "reg_lambda": 10.0,
                "n_estimators": 200,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "tree_method": "hist",
                "random_state": self.random_state,
                "n_jobs": 1,
            }
            default_xgb.update(self.xgb_params)
            return XGBRegressor(**default_xgb)
        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor
            default_cb = {
                "iterations": 300,
                "learning_rate": 0.05,
                "depth": 5,
                "l2_leaf_reg": 5.0,
                "random_seed": self.random_state,
                "thread_count": -1,
                "verbose": False,
            }
            default_cb.update(self.catboost_params)
            return CatBoostRegressor(**default_cb)
        else:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. Choose 'ridge', 'xgb', or 'catboost'."
            )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "MarginRegressor":
        """
        Fits the continuous point differential regressor and computes residual standard error.

        Args:
            X: Training feature matrix (raw DataFrame or standardized array).
            y: Continuous margin target series (HOME_PTS - AWAY_PTS).
            feature_names: Optional column names for explainability.

        Returns:
            Fitted MarginRegressor instance.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            if self.scaler is not None and self.model_type == "ridge":
                X_arr = self.scaler.transform(X).astype(np.float32)
            else:
                X_arr = X.to_numpy(dtype=np.float32)
        else:
            self.feature_names_ = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y, dtype=np.float32)

        self.estimator_ = self._build_estimator()
        self.estimator_.fit(X_arr, y_arr)

        preds = self.estimator_.predict(X_arr)
        residuals = y_arr - preds

        self.residual_mean_ = float(np.mean(residuals))
        self.residual_std_ = float(np.std(residuals))
        self.is_fitted_ = True

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts expected game margin (HOME_PTS - AWAY_PTS).

        Args:
            X: Feature matrix.

        Returns:
            1D array of expected point differentials.
        """
        if not self.is_fitted_ or self.estimator_ is None:
            raise RuntimeError("MarginRegressor is not fitted yet. Call .fit() first.")

        if isinstance(X, pd.DataFrame):
            if self.scaler is not None and self.model_type == "ridge":
                X_arr = self.scaler.transform(X).astype(np.float32)
            else:
                X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        return self.estimator_.predict(X_arr)

    def get_residuals(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Computes residuals (y_true - y_pred)."""
        y_arr = np.asarray(y, dtype=np.float32)
        return y_arr - self.predict(X)


# ======================================================
# Pace-Modulated Normal CDF Classifier
# ======================================================

class PaceModulatedMarginClassifier(BaseEstimator, ClassifierMixin):
    """
    Translates continuous point differential predictions into well-calibrated
    win probabilities using a Pace-Modulated Normal Cumulative Distribution Function (CDF).

    Mathematical Formulation:
        Point differential variance scales directly with the number of expected
        possessions in the game (central limit theorem on possession outcomes):
            sigma(Pace) = sigma_0 * sqrt(expected_pace / base_pace)

        The probability of a home team victory (Margin > 0) is given by:
            P(Home Win) = Phi(M_hat / sigma(Pace))
            P(Away Win) = 1.0 - P(Home Win)

    Key Domain Invariance:
        - In high-possession games (faster tempo), game variance is expanded,
          increasing upset volatility.
        - In low-possession games (grind-it-out tempo), game variance is compressed,
          increasing the probability that the superior team prevails.
    """

    def __init__(
        self,
        regressor: Optional[MarginRegressor] = None,
        sigma_0: Optional[float] = None,
        base_pace: float = DEFAULT_BASE_PACE,
        pace_col: str = PACE_COLUMN_NAME,
        fit_sigma: bool = True,
    ) -> None:
        self.regressor = regressor
        self.sigma_0 = sigma_0
        self.base_pace = base_pace
        self.pace_col = pace_col
        self.fit_sigma = fit_sigma

        self.classes_: np.ndarray = np.array([0, 1], dtype=np.int32)
        self.sigma_0_: float = 12.0
        self.is_fitted_: bool = False

    def _extract_pace(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        pace: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Safely extracts game tempo vector (possessions per 48 mins).

        Prioritizes:
        1. Explicitly supplied pace vector.
        2. Unscaled DataFrame column (MATCHUP_EXPECTED_PACE).
        3. Fallback to constant base_pace (100.0).
        """
        if pace is not None:
            p = np.asarray(pace, dtype=np.float32)
        elif isinstance(X, pd.DataFrame) and self.pace_col in X.columns:
            # Verify values are unscaled raw possessions (~80 to ~115)
            vals = X[self.pace_col].to_numpy(dtype=np.float32)
            if np.nanmean(vals) > 50.0:
                p = vals
            else:
                p = np.full(len(X), self.base_pace, dtype=np.float32)
        else:
            p = np.full(len(X), self.base_pace, dtype=np.float32)

        # Sanitize missing or unphysical pace estimates (< 60 possessions)
        p_clean = np.where(np.isnan(p) | (p <= 60.0) | (p >= 140.0), self.base_pace, p)
        return p_clean

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        y_margin: Optional[Union[pd.Series, np.ndarray]] = None,
        pace: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "PaceModulatedMarginClassifier":
        """
        Fits the underlying MarginRegressor (if needed) and calibrates sigma_0.

        Args:
            X: Feature matrix (DataFrame or array).
            y: Binary outcome (1 for home win, 0 for away win).
            y_margin: Optional continuous point differential target series.
            pace: Optional vector of expected game pace.

        Returns:
            Fitted PaceModulatedMarginClassifier.
        """
        if self.regressor is None:
            self.regressor = MarginRegressor(model_type="ridge")

        if not getattr(self.regressor, "is_fitted_", False):
            target = y_margin if y_margin is not None else y
            self.regressor.fit(X, target)

        # Determine initial sigma_0 baseline from regressor residuals
        initial_sigma = (
            self.sigma_0
            if self.sigma_0 is not None
            else getattr(self.regressor, "residual_std_", 12.5)
        )

        pace_vec = self._extract_pace(X, pace)
        margins = self.regressor.predict(X)
        y_binary = np.asarray(y, dtype=np.int32)

        # Calibrate sigma_0 by minimizing cross-entropy (log-loss) if binary targets available
        if self.fit_sigma and len(np.unique(y_binary)) == 2:
            def log_loss_objective(s: float) -> float:
                sigmas = s * np.sqrt(pace_vec / self.base_pace)
                probs = np.clip(ndtr(margins / sigmas), 1e-7, 1.0 - 1e-7)
                return float(log_loss(y_binary, probs))

            res = minimize_scalar(
                log_loss_objective,
                bounds=(5.0, 30.0),
                method="bounded",
            )
            self.sigma_0_ = float(res.x)
        else:
            self.sigma_0_ = float(initial_sigma)

        self.is_fitted_ = True
        return self

    def predict_margin(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns predicted point differential M_hat."""
        if not self.is_fitted_ or self.regressor is None:
            raise RuntimeError("PaceModulatedMarginClassifier is not fitted yet.")
        return self.regressor.predict(X)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        pace: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Computes calibrated win probabilities for away and home teams.

        Returns:
            2D array of shape (n_samples, 2) where column 0 is P(Away Win)
            and column 1 is P(Home Win).
        """
        if not self.is_fitted_:
            raise RuntimeError("PaceModulatedMarginClassifier is not fitted yet.")

        margins = self.predict_margin(X)
        pace_vec = self._extract_pace(X, pace)

        # Tempo-modulated standard error
        game_sigmas = self.sigma_0_ * np.sqrt(pace_vec / self.base_pace)

        # Gaussian Normal CDF
        z_scores = margins / game_sigmas
        p_home = ndtr(z_scores)

        # Guard against zero/one clipping artifacts
        p_home_clipped = np.clip(p_home, 1e-7, 1.0 - 1e-7)
        p_away = 1.0 - p_home_clipped

        return np.column_stack([p_away, p_home_clipped])

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        pace: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Predicts binary win outcome (1 for home win, 0 for away win)."""
        probs = self.predict_proba(X, pace)
        return (probs[:, 1] >= 0.5).astype(np.int32)


# ======================================================
# Cross-Validation & Hyperparameter Tuning
# ======================================================

def tune_margin_regressor(
    data: TrainingData,
    config: TrainingConfig,
    model_type: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[ModelArtifacts, Dict[str, float]]:
    """
    Performs chronological TimeSeriesSplit hyperparameter tuning for MarginRegressor.

    Args:
        data: Prepared dataset container with feature sets and y_margin targets.
        config: Training configuration specifying search spaces and splits.
        model_type: "ridge" or "xgb" (defaults to config.margin_model_type).
        output_dir: Optional path to save tuning metadata.

    Returns:
        Tuple containing:
            - ModelArtifacts for the tuned MarginRegressor
            - Dictionary of validation evaluation metrics
    """
    m_type = model_type or getattr(config, "margin_model_type", "ridge")

    if data.margin is None or data.y_margin_train is None:
        raise ValueError(
            "TrainingData is missing margin feature set or y_margin_train. "
            "Ensure load_and_prep_data() extracted the margin target."
        )

    # Ensure scaled features exist for linear models
    if data.margin.X_train_processed is None:
        scale_features(data.margin)

    tscv = TimeSeriesSplit(n_splits=config.cv_folds)

    if m_type == "ridge":
        X_train = data.margin.X_train_processed
        X_val = data.margin.X_val_processed
        y_train = data.y_margin_train.to_numpy(dtype=np.float32)
        y_val = data.y_margin_val.to_numpy(dtype=np.float32)

        grid = config.margin_ridge_grid
        search = GridSearchCV(
            estimator=Ridge(random_state=config.random_seed),
            param_grid=grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_alpha = search.best_params_["alpha"]
        logger.info(
            f"      Best Margin Ridge CV RMSE: {-search.best_score_:.4f} (alpha={best_alpha})"
        )

        regressor = MarginRegressor(
            model_type="ridge",
            alpha=best_alpha,
            scaler=data.margin.scaler,
            random_state=config.random_seed,
        )
        regressor.fit(X_train, y_train, feature_names=data.margin.feature_names)

    elif m_type == "xgb":
        X_train = data.margin.X_train
        X_val = data.margin.X_val
        y_train = data.y_margin_train
        y_val = data.y_margin_val

        grid = config.margin_xgb_grid
        search = RandomizedSearchCV(
            estimator=XGBRegressor(
                random_state=config.random_seed,
                tree_method="hist",
                n_jobs=1,
            ),
            param_distributions=grid,
            n_iter=getattr(config, "margin_search_iterations", 15),
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=config.random_seed,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_params = search.best_params_
        logger.info(
            f"      Best Margin XGB CV RMSE: {-search.best_score_:.4f} (params={best_params})"
        )

        regressor = MarginRegressor(
            model_type="xgb",
            xgb_params=best_params,
            random_state=config.random_seed,
        )
        regressor.fit(X_train, y_train)

    elif m_type == "catboost":
        from catboost import CatBoostRegressor

        X_train = data.margin.X_train
        X_val = data.margin.X_val
        y_train = data.y_margin_train
        y_val = data.y_margin_val

        grid = getattr(
            config,
            "margin_catboost_grid",
            {
                "depth": [4, 6],
                "l2_leaf_reg": [1, 5, 10],
                "learning_rate": [0.03, 0.05, 0.08],
                "iterations": [250, 350],
            },
        )
        search = RandomizedSearchCV(
            estimator=CatBoostRegressor(
                random_seed=config.random_seed,
                thread_count=-1,
                verbose=False,
            ),
            param_distributions=grid,
            n_iter=getattr(config, "margin_catboost_search_iterations", 6),
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=config.random_seed,
            n_jobs=1,
        )
        search.fit(X_train, y_train)

        best_params = search.best_params_
        logger.info(
            f"      Best Margin CatBoost CV RMSE: {-search.best_score_:.4f} (params={best_params})"
        )

        regressor = MarginRegressor(
            model_type="catboost",
            xgb_params=best_params,
            random_state=config.random_seed,
        )
        regressor.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown margin model type '{m_type}'. Choose 'ridge', 'xgb', or 'catboost'.")

    # Evaluate on held-out validation set
    val_preds = regressor.predict(X_val)
    val_metrics = evaluate_margin_metrics(y_val, val_preds)

    logger.info(
        f"      Margin Validation -> RMSE: {val_metrics['RMSE']:.3f} | "
        f"MAE: {val_metrics['MAE']:.3f} | "
        f"R²: {val_metrics['R2']:.3f} | "
        f"Dir Acc: {val_metrics['Directional_Accuracy']:.3%}"
    )

    # Save search artifacts if output directory is provided
    if output_dir is not None:
        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_df.to_csv(output_dir / MARGIN_RESULTS_FILE, index=False)
        save_json(search.best_params_, output_dir / MARGIN_PARAMS_FILE)

    artifact = ModelArtifacts(
        feature_set=data.margin,
        model=regressor,
        final_model=regressor,
    )

    return artifact, val_metrics


def tune_pace_margin_classifier(
    data: TrainingData,
    config: TrainingConfig,
    output_dir: Optional[Path] = None,
) -> Tuple[ModelArtifacts, Dict[str, float]]:
    """
    End-to-end tuning for the Pace-Modulated Normal CDF Margin Classifier.

    1. Tunes and fits the optimal continuous MarginRegressor on training data.
    2. Calibrates tempo-scaled standard error (sigma_0) on chronological validation data.
    3. Evaluates probabilistic classification metrics (Log Loss, Accuracy, ROC-AUC, Brier).

    Args:
        data: Prepared dataset container.
        config: Training configuration.
        output_dir: Optional directory for serialization.

    Returns:
        Tuple containing ModelArtifacts and validation classification metrics.
    """
    # Train optimal margin regressor
    regressor_artifact, margin_metrics = tune_margin_regressor(
        data=data,
        config=config,
        output_dir=output_dir,
    )
    tuned_regressor: MarginRegressor = regressor_artifact.model

    # Build and calibrate PaceModulatedMarginClassifier
    classifier = PaceModulatedMarginClassifier(
        regressor=tuned_regressor,
        base_pace=DEFAULT_BASE_PACE,
        pace_col=PACE_COLUMN_NAME,
        fit_sigma=True,
    )

    # Calibrate sigma_0 on validation set using unscaled pace and validation outcomes
    raw_val_pace = None
    if isinstance(data.margin.X_val, pd.DataFrame) and PACE_COLUMN_NAME in data.margin.X_val.columns:
        raw_val_pace = data.margin.X_val[PACE_COLUMN_NAME].to_numpy(dtype=np.float32)

    X_val_input = (
        data.margin.X_val_processed
        if tuned_regressor.model_type == "ridge"
        else data.margin.X_val
    )

    classifier.fit(
        X=X_val_input,
        y=data.y_val,
        pace=raw_val_pace,
    )

    val_probs = classifier.predict_proba(X_val_input, pace=raw_val_pace)
    val_metrics = evaluate_classification_metrics(data.y_val, val_probs[:, 1])
    val_metrics["Calibrated_Sigma"] = round(classifier.sigma_0_, 4)
    val_metrics["Baseline_Sigma"] = round(tuned_regressor.residual_std_, 4)
    val_metrics["Margin_RMSE"] = margin_metrics["RMSE"]

    logger.info(
        f"      Pace-Modulated CDF Val -> Log Loss: {val_metrics['Log_Loss']:.4f} | "
        f"Accuracy: {val_metrics['Accuracy']:.3%} | "
        f"ROC-AUC: {val_metrics['ROC_AUC']:.4f} | "
        f"Brier: {val_metrics['Brier_Score']:.4f} | "
        f"Sigma: {val_metrics['Calibrated_Sigma']:.2f}"
    )

    if output_dir is not None:
        save_json(val_metrics, output_dir / PACE_MARGIN_METRICS_FILE)

    artifact = ModelArtifacts(
        feature_set=data.margin,
        model=classifier,
        final_model=classifier,
    )

    return artifact, val_metrics
