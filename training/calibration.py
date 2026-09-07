"""
Probability calibration module implementing Platt Scaling and Beta Calibration.

This module provides leak-free probability calibration tools for binary classification:
- Platt Scaling (Sigmoid calibration): Standard monotonic logistic mapping of logits:
      logit(P(Y=1|p)) = a * logit(p) + c,   with a >= 0.
- Beta Calibration (Kull et al., 2017): Asymmetric bivariate calibration based on
  Beta distributions, allowing flexible tail corrections for underdogs vs. favorites:
      logit(P(Y=1|p)) = a * ln(p) - b * ln(1 - p) + c,   with a >= 0, b >= 0.

It includes model-selection routines to automatically evaluate both techniques
on leak-free cross-validated probability outputs and select the lowest log-loss calibrator.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

CALIBRATION_EPSILON = 1e-12


# ======================================================
# Out-of-Fold Generation
# ======================================================

def generate_oof_predictions(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: TimeSeriesSplit,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates strictly chronological out-of-fold probability predictions.

    TimeSeriesSplit does not partition the initial training fold, so standard
    cross_val_predict fails. This helper manually iterates through the chronological
    folds to collect non-overlapping, leak-free out-of-fold validation probabilities.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y, dtype=np.float64)

    oof_probs: List[np.ndarray] = []
    oof_targets: List[np.ndarray] = []

    for train_idx, val_idx in cv.split(X_arr):
        fold_est = clone(estimator)
        fold_est.fit(X_arr[train_idx], y_arr[train_idx])
        probs = fold_est.predict_proba(X_arr[val_idx])[:, 1]
        oof_probs.append(probs)
        oof_targets.append(y_arr[val_idx])

    return np.concatenate(oof_probs), np.concatenate(oof_targets)


# ======================================================
# Beta Calibrator Core
# ======================================================

class BetaCalibrator(BaseEstimator, ClassifierMixin):
    """
    Bivariate logistic regression on log-odds features enforcing monotonicity.

    Mathematical Formulation:
        logit(P(Y=1|p)) = a * ln(p) - b * ln(1 - p) + c
    with constraints a >= 0, b >= 0 to ensure monotonicity.
    When a = b, this reduces to standard Platt sigmoid scaling.

    Reference:
        Kull, M., Silva Filho, T., & Flach, P. (2017). Beta calibration: a well-founded
        and easily implemented alternative on binary classification. AISTATS 2017.
    """

    def __init__(self) -> None:
        self.a_: float = 1.0
        self.b_: float = 1.0
        self.c_: float = 0.0
        self.is_fitted_: bool = False

    @staticmethod
    def _transform_probabilities(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts log-probability basis features ln(p) and -ln(1-p)."""
        p_clipped = np.clip(p, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
        z1 = np.log(p_clipped)
        z2 = -np.log(1.0 - p_clipped)
        return z1, z2

    def fit_from_probabilities(
        self,
        p_raw: np.ndarray,
        y: Union[pd.Series, np.ndarray],
    ) -> "BetaCalibrator":
        """Fits Beta calibration parameters a, b, c directly from raw probabilities."""
        y_arr = np.asarray(y, dtype=np.float64)
        z1, z2 = self._transform_probabilities(p_raw)

        def _objective(params: np.ndarray) -> float:
            a, b, c = params
            logits = a * z1 + b * z2 + c
            probs = np.clip(expit(logits), CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
            return float(log_loss(y_arr, probs))

        # Initial point is identity calibration: a=1, b=1, c=0
        initial_guess = [1.0, 1.0, 0.0]
        bounds = [(0.0, None), (0.0, None), (None, None)]

        result = minimize(
            _objective,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-10, "maxiter": 500},
        )

        if result.success:
            self.a_, self.b_, self.c_ = [float(v) for v in result.x]
        else:
            self.a_, self.b_, self.c_ = 1.0, 1.0, 0.0

        self.is_fitted_ = True
        return self

    def predict_proba(self, p_raw: np.ndarray) -> np.ndarray:
        """Transforms raw probabilities into calibrated probability array of shape (N, 2)."""
        if not self.is_fitted_:
            raise RuntimeError("BetaCalibrator is not fitted yet.")

        p_arr = np.asarray(p_raw, dtype=np.float64)
        if p_arr.ndim == 2 and p_arr.shape[1] == 2:
            p_arr = p_arr[:, 1]
        elif p_arr.ndim == 2 and p_arr.shape[1] == 1:
            p_arr = p_arr.ravel()

        z1, z2 = self._transform_probabilities(p_arr)
        calibrated_logits = self.a_ * z1 + self.b_ * z2 + self.c_
        calibrated_p = np.clip(expit(calibrated_logits), CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)

        return np.column_stack([1.0 - calibrated_p, calibrated_p])


# ======================================================
# Beta Calibrated Classifier Wrapper
# ======================================================

class BetaCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible classifier wrapper that applies Beta Calibration
    using out-of-fold cross-validation probabilities to prevent data leakage.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        cv: Optional[TimeSeriesSplit] = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.cv = cv
        self.calibrator_ = BetaCalibrator()
        self.estimator_: Optional[Any] = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.is_fitted_: bool = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_params: Any,
    ) -> "BetaCalibratedClassifier":
        """Fits base estimator on full data and Beta calibrator on out-of-fold probabilities."""
        if self.base_estimator is None:
            raise ValueError("base_estimator cannot be None.")

        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        cv_splitter = self.cv if self.cv is not None else TimeSeriesSplit(n_splits=5)

        # 1. Out-of-fold probability generation across splits
        oof_probs, oof_targets = generate_oof_predictions(
            estimator=self.base_estimator,
            X=X,
            y=y_arr,
            cv=cv_splitter,
        )

        # 2. Fit Beta calibrator on leak-free OOF predictions
        self.calibrator_ = BetaCalibrator().fit_from_probabilities(oof_probs, oof_targets)

        # 3. Fit base estimator on full training dataset
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_arr, **fit_params)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns calibrated probability distribution of shape (N, 2)."""
        if not self.is_fitted_ or self.estimator_ is None:
            raise RuntimeError("BetaCalibratedClassifier is not fitted yet.")

        raw_probs = self.estimator_.predict_proba(X)[:, 1]
        return self.calibrator_.predict_proba(raw_probs)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns binary predictions thresholded at 0.50."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(np.int32)


# ======================================================
# Calibration Model Selection Routine
# ======================================================

def _fit_platt_log_loss(
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
) -> Tuple[float, float, float]:
    """Fits Platt scaling (a * logit(p) + c) on OOF probabilities and returns (a, c, log_loss)."""
    eps = CALIBRATION_EPSILON
    p_clipped = np.clip(oof_probs, eps, 1.0 - eps)
    logits = logit(p_clipped)

    def _obj(params: np.ndarray) -> float:
        a, c = params
        p_cal = np.clip(expit(a * logits + c), eps, 1.0 - eps)
        return float(log_loss(oof_targets, p_cal))

    res = minimize(_obj, [1.0, 0.0], bounds=[(0.0, None), (None, None)], method="L-BFGS-B")
    a, c = (float(res.x[0]), float(res.x[1])) if res.success else (1.0, 0.0)
    best_loss = float(res.fun) if res.success else float(log_loss(oof_targets, p_clipped))
    return a, c, best_loss


def _fit_beta_log_loss(
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fits Beta calibration on OOF probabilities and returns (a, b, c, log_loss)."""
    calibrator = BetaCalibrator().fit_from_probabilities(oof_probs, oof_targets)
    probs = calibrator.predict_proba(oof_probs)[:, 1]
    loss = float(log_loss(oof_targets, probs))
    return calibrator.a_, calibrator.b_, calibrator.c_, loss


def calibrate_estimator_with_model_selection(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    tscv: TimeSeriesSplit,
) -> Tuple[Any, Dict[str, Union[str, float, Dict[str, float]]]]:
    """
    Compares Platt (Sigmoid) scaling vs. Beta Calibration using leak-free
    cross-validated out-of-fold probability predictions.

    Selects whichever technique produces lower cross-validated log-loss.
    Returns:
        calibrated_estimator: Fitted CalibratedClassifierCV or BetaCalibratedClassifier.
        selection_metadata: Dictionary detailing selected method and diagnostic losses.
    """
    # 1. Generate leak-free out-of-fold predictions
    oof_probs, oof_targets = generate_oof_predictions(
        estimator=estimator,
        X=X_train,
        y=y_train,
        cv=tscv,
    )

    raw_loss = float(log_loss(oof_targets, np.clip(oof_probs, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)))

    # 2. Evaluate Platt and Beta calibration on the same OOF validation partition
    platt_a, platt_c, platt_loss = _fit_platt_log_loss(oof_probs, oof_targets)
    beta_a, beta_b, beta_c, beta_loss = _fit_beta_log_loss(oof_probs, oof_targets)

    # 3. Model selection based on log loss
    if beta_loss < (platt_loss - 1e-4):
        logger.info(
            "        Selected Beta Calibration (OOF LogLoss: %.4f vs Platt: %.4f | Raw: %.4f)",
            beta_loss, platt_loss, raw_loss,
        )
        selected_model = BetaCalibratedClassifier(base_estimator=estimator, cv=tscv)
        selected_model.fit(X_train, y_train)
        return selected_model, {
            "method": "beta",
            "raw_log_loss": round(raw_loss, 4),
            "platt_log_loss": round(platt_loss, 4),
            "beta_log_loss": round(beta_loss, 4),
            "selected_log_loss": round(beta_loss, 4),
            "params": {"a": round(beta_a, 4), "b": round(beta_b, 4), "c": round(beta_c, 4)},
        }

    logger.info(
        "        Selected Platt Sigmoid Calibration (OOF LogLoss: %.4f vs Beta: %.4f | Raw: %.4f)",
        platt_loss, beta_loss, raw_loss,
    )
    selected_model = CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=tscv,
        n_jobs=-1,
    )
    selected_model.fit(X_train, y_train)
    return selected_model, {
        "method": "platt",
        "raw_log_loss": round(raw_loss, 4),
        "platt_log_loss": round(platt_loss, 4),
        "beta_log_loss": round(beta_loss, 4),
        "selected_log_loss": round(platt_loss, 4),
        "params": {"a": round(platt_a, 4), "c": round(platt_c, 4)},
    }

