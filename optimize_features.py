"""
Model-Specific Sequential Backward Selection (SBS) Optimizer.

Features:
- Independent Feature Spaces: Prunes features individually for MLP, XGB, and LR.
- Partial Fast-Tracking: Retrains ONLY the model being tested; uses cached predictions for the others.
- Model-Specific Importance: Ranks features independently per algorithm using SNR (Mean/Std).
- Asymmetric Checkpointing: Outputs a JSON dictionary with 3 distinct removal lists.
"""

import json
import logging
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from training.config import TrainingConfig, TrainingArtifacts, FeatureSet, TrainingData
from training.data import load_and_prep_data, scale_features
from training.ensemble import learn_ensemble_weights
from training.tuning import tune_base_models
from training.utils import setup_logger, unwrap_base_estimator, save_joblib

# ======================================================
# Configuration
# ======================================================
MAX_FAILURES_BEFORE_STOP = 3  # Stops after 3 consecutive batch rounds with 0 accepted drops
WEAKEST_BATCH_SIZE_PER_MODEL = 6  # Tests 6 candidates per model (18 per batch)
PERMUTATION_REPEATS = 5  # Stable permutation evaluation
MIN_IMPROVEMENT = 1e-4  # Minimum required log loss improvement to prevent fitting validation noise
AUTO_RESUME = True

DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "optimization_checkpoints"


def apply_model_specific_reduction(master_data: TrainingData, removals: dict) -> TrainingData:
    """Drops features dynamically based on model-specific removal lists."""
    
    def reduce_fs(fs: FeatureSet, cols_to_drop: list) -> FeatureSet:
        valid_drops = [c for c in cols_to_drop if c in fs.X_train.columns]
        if not valid_drops:
            return fs
        
        return FeatureSet(
            X_train=fs.X_train.drop(columns=valid_drops),
            X_val=fs.X_val.drop(columns=valid_drops),
            X_test=fs.X_test.drop(columns=valid_drops),
            feature_names=[f for f in fs.feature_names if f not in valid_drops]
        )

    reduced_data = TrainingData(
        mlp=reduce_fs(master_data.mlp, removals["mlp"]),
        xgb=reduce_fs(master_data.xgb, removals["xgb"]),
        lr=reduce_fs(master_data.lr, removals["lr"]),
        y_train=master_data.y_train,
        y_val=master_data.y_val,
        y_test=master_data.y_test,
        summary=master_data.summary
    )
    
    scale_features(reduced_data.mlp)
    scale_features(reduced_data.lr)
    return reduced_data


def get_model_importance(model, X_val, y_val, feature_names, random_seed) -> list:
    """
    Calculates Permutation Importance for a single model.

    Features are ranked primarily by mean importance ascending (lowest/most negative first):
    - Negative mean: shuffling improved validation log loss (harmful / overfitted feature).
    - Zero mean: shuffling had negligible impact (uninformative / redundant feature).
    - Positive mean: shuffling degraded validation log loss (predictive / useful feature).
    Tie-breaking uses standard deviation ascending (lower variance first).
    """
    res = permutation_importance(
        model, X_val, y_val, scoring="neg_log_loss", 
        n_repeats=PERMUTATION_REPEATS, random_state=random_seed, n_jobs=-1
    )
    
    scores = {}
    for name, m, s in zip(feature_names, res.importances_mean, res.importances_std):
        scores[name] = (float(m), float(s))
            
    return [feat[0] for feat in sorted(scores.items(), key=lambda x: (x[1][0], x[1][1]))]


def compute_independent_importances(mlp_cal, xgb_cal, lr_cal, reduced_data, random_seed) -> dict:
    """Returns sorted weakest features independently for each model architecture."""
    return {
        "mlp": get_model_importance(mlp_cal, reduced_data.mlp.X_val_processed, reduced_data.y_val, reduced_data.mlp.feature_names, random_seed),
        "xgb": get_model_importance(xgb_cal, reduced_data.xgb.X_val, reduced_data.y_val, reduced_data.xgb.feature_names, random_seed),
        "lr": get_model_importance(lr_cal, reduced_data.lr.X_val_processed, reduced_data.y_val, reduced_data.lr.feature_names, random_seed)
    }


def fit_fast_model(target_model_name, base_estimator, X_train, y_train):
    """Fits an accelerated proxy model for ultra-fast candidate evaluation without shape errors."""
    if target_model_name == "lr":
        model = clone(base_estimator).set_params(max_iter=100, warm_start=False)
        model.fit(X_train, y_train)
        return model
    elif target_model_name == "mlp":
        model = clone(base_estimator).set_params(max_iter=25, early_stopping=True, n_iter_no_change=3)
        model.fit(X_train, y_train)
        return model
    elif target_model_name == "xgb":
        model = clone(base_estimator).set_params(tree_method="hist", n_estimators=80, n_jobs=-1, eval_metric="logloss")
        model.fit(X_train, y_train)
        return model
    return clone(base_estimator).fit(X_train, y_train)


def partial_fast_track(
    target_model_name: str, 
    reduced_data: TrainingData, 
    base_estimators: dict, 
    cached_probs: dict, 
    frozen_weights: np.ndarray,
    *args,
    **kwargs,
) -> float:
    """
    ULTRA-FAST TEST: Retrains ONLY the target model with accelerated proxy params.
    Uses cached predictions for the others.
    """
    new_probs = deepcopy(cached_probs)
    
    if target_model_name == "mlp":
        raw_model = fit_fast_model("mlp", base_estimators["mlp"], reduced_data.mlp.X_train_processed, reduced_data.y_train)
        new_probs["mlp"] = raw_model.predict_proba(reduced_data.mlp.X_val_processed)[:, 1]
    elif target_model_name == "xgb":
        raw_model = fit_fast_model("xgb", base_estimators["xgb"], reduced_data.xgb.X_train, reduced_data.y_train)
        new_probs["xgb"] = raw_model.predict_proba(reduced_data.xgb.X_val)[:, 1]
    elif target_model_name == "lr":
        raw_model = fit_fast_model("lr", base_estimators["lr"], reduced_data.lr.X_train_processed, reduced_data.y_train)
        new_probs["lr"] = raw_model.predict_proba(reduced_data.lr.X_val_processed)[:, 1]

    weights = np.array(frozen_weights, dtype=float).copy()
    idx_map = {"mlp": 0, "xgb": 1, "lr": 2}
    t_idx = idx_map.get(target_model_name, 0)
    if weights[t_idx] < 0.15:
        weights[t_idx] = 0.15
        weights = weights / weights.sum()

    stacked = np.column_stack((new_probs["mlp"], new_probs["xgb"], new_probs["lr"]))
    blended = np.clip(np.dot(stacked, weights), 1e-15, 1 - 1e-15)
    return float(log_loss(reduced_data.y_val, blended))


def fit_and_calibrate_single_model(
    target_model_name: str, 
    base_estimator, 
    reduced_data: TrainingData, 
    cv_folds: int = 3
):
    """Fits and probability-calibrates ONLY the target model on reduced_data."""
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    if target_model_name == "mlp":
        base = clone(base_estimator).set_params(max_iter=80, early_stopping=True, n_iter_no_change=5)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=tscv, n_jobs=-1)
        cal.fit(reduced_data.mlp.X_train_processed, reduced_data.y_train)
        probs = cal.predict_proba(reduced_data.mlp.X_val_processed)[:, 1]
    elif target_model_name == "xgb":
        base = clone(base_estimator).set_params(tree_method="hist", n_estimators=150, n_jobs=-1, eval_metric="logloss")
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=tscv, n_jobs=-1)
        cal.fit(reduced_data.xgb.X_train, reduced_data.y_train)
        probs = cal.predict_proba(reduced_data.xgb.X_val)[:, 1]
    elif target_model_name == "lr":
        base = clone(base_estimator)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=tscv, n_jobs=-1)
        cal.fit(reduced_data.lr.X_train_processed, reduced_data.y_train)
        probs = cal.predict_proba(reduced_data.lr.X_val_processed)[:, 1]
    else:
        raise ValueError(f"Unknown target model name: {target_model_name}")

    standalone_loss = float(log_loss(reduced_data.y_val, probs))
    return cal, probs, standalone_loss


def full_track_accept(reduced_data, base_estimators, cv_folds=3):
    """FULL ACCEPTANCE: Recalibrates all models, relearns weights, caches new baseline probabilities."""
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    mlp_base = clone(base_estimators["mlp"]).set_params(max_iter=80, early_stopping=True, n_iter_no_change=5)
    xgb_base = clone(base_estimators["xgb"]).set_params(tree_method="hist", n_estimators=150, n_jobs=-1, eval_metric="logloss")
    lr_base = clone(base_estimators["lr"])

    mlp_cal = CalibratedClassifierCV(mlp_base, method="sigmoid", cv=tscv, n_jobs=-1)
    xgb_cal = CalibratedClassifierCV(xgb_base, method="sigmoid", cv=tscv, n_jobs=-1)
    lr_cal = CalibratedClassifierCV(lr_base, method="sigmoid", cv=tscv, n_jobs=-1)

    mlp_cal.fit(reduced_data.mlp.X_train_processed, reduced_data.y_train)
    xgb_cal.fit(reduced_data.xgb.X_train, reduced_data.y_train)
    lr_cal.fit(reduced_data.lr.X_train_processed, reduced_data.y_train)

    temp_artifacts = TrainingArtifacts(config=TrainingConfig(), output_dir=CHECKPOINT_DIR)
    temp_artifacts.mlp.model, temp_artifacts.mlp.feature_set = mlp_cal, reduced_data.mlp
    temp_artifacts.xgb.model, temp_artifacts.xgb.feature_set = xgb_cal, reduced_data.xgb
    temp_artifacts.lr.model,  temp_artifacts.lr.feature_set  = lr_cal,  reduced_data.lr

    weights, formula = learn_ensemble_weights(
        temp_artifacts.mlp, temp_artifacts.xgb, temp_artifacts.lr, reduced_data.y_val
    )
    
    cached_probs = {
        "mlp": mlp_cal.predict_proba(reduced_data.mlp.X_val_processed)[:, 1],
        "xgb": xgb_cal.predict_proba(reduced_data.xgb.X_val)[:, 1],
        "lr": lr_cal.predict_proba(reduced_data.lr.X_val_processed)[:, 1]
    }
    
    stacked = np.column_stack((cached_probs["mlp"], cached_probs["xgb"], cached_probs["lr"]))
    blended = np.clip(np.dot(stacked, weights), 1e-15, 1 - 1e-15)
    full_logloss = log_loss(reduced_data.y_val, blended)
    
    return mlp_cal, xgb_cal, lr_cal, weights, formula, cached_probs, full_logloss


def save_lean_checkpoint(removals, best_logloss, weights, formula, config, iter_num, mlp_cal, xgb_cal, lr_cal, standalone_losses=None):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "removals": removals,
        "logloss": best_logloss,
        "ensemble_weights": weights,
        "best_formula": formula,
        "current_iteration": iter_num,
        "mlp_model": mlp_cal,
        "xgb_model": xgb_cal,
        "lr_model": lr_cal,
        "standalone_losses": standalone_losses,
    }
    save_joblib(state, CHECKPOINT_DIR / "best_sbs_state.pkl")
    
    with open(CHECKPOINT_DIR / "current_removals.json", "w") as f:
        json.dump({
            "best_logloss": best_logloss, 
            "formula": formula,
            "iteration": iter_num,
            "removals": removals,
            "standalone_losses": standalone_losses,
        }, f, indent=4)


def get_base_estimators(config: TrainingConfig, checkpoint_dir: Path, data: TrainingData) -> dict:
    """Instantiates base estimators using tuned hyperparameters if available, else tunes them."""
    search_dirs = [checkpoint_dir] + sorted(Path("models").glob("run_*"), reverse=True)
    
    mlp_params = None
    xgb_params = None
    lr_params = None
    
    for d in search_dirs:
        if (d / "mlp_best_params.json").exists() and mlp_params is None:
            with open(d / "mlp_best_params.json") as f:
                mlp_params = json.load(f)
        if (d / "xgb_best_params.json").exists() and xgb_params is None:
            with open(d / "xgb_best_params.json") as f:
                xgb_params = json.load(f)
        if (d / "lr_best_params.json").exists() and lr_params is None:
            with open(d / "lr_best_params.json") as f:
                lr_params = json.load(f)

    if mlp_params and xgb_params and lr_params:
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded existing tuned base parameters from {search_dirs[0]}.")
        mlp = MLPClassifier(**mlp_params, random_state=config.random_seed)
        xgb = XGBClassifier(**xgb_params, random_state=config.random_seed, tree_method="hist", max_bin=256, n_jobs=-1, eval_metric="logloss")
        lr = LogisticRegression(**lr_params, random_state=config.random_seed)
        return {"mlp": mlp, "xgb": xgb, "lr": lr}
    
    logger = logging.getLogger(__name__)
    logger.info("No cached hyperparameters found; tuning base models...")
    mlp_art, xgb_art, lr_art = tune_base_models(data, config, checkpoint_dir)
    return {
        "mlp": unwrap_base_estimator(mlp_art.model),
        "xgb": unwrap_base_estimator(xgb_art.model),
        "lr": unwrap_base_estimator(lr_art.model),
    }


def format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s" if mins > 0 else f"{secs}s"


def load_resume_state(master_data: TrainingData | None = None):
    """Loads the most recent optimization checkpoint if available and compatible."""
    state_path = CHECKPOINT_DIR / "best_sbs_state.pkl"
    if not AUTO_RESUME or not state_path.exists():
        return None

    try:
        state = joblib.load(state_path)
        if master_data is not None and "removals" in state:
            for model_name in ["mlp", "xgb", "lr"]:
                fs = getattr(master_data, model_name)
                for r in state["removals"].get(model_name, []):
                    if r not in fs.X_train.columns:
                        return None
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.warning("Could not load optimization checkpoint: %s", exc)
        return None

    return state


def main():
    logger = setup_logger(Path(__file__).resolve().parent)
    global_start_time = time.perf_counter()
    
    logger.info("\n=========================================")
    logger.info("  MODEL-SPECIFIC BACKWARD SELECTION (v5) ")
    logger.info("=========================================\n")
    
    config = TrainingConfig()
    removals = {"mlp": [], "xgb": [], "lr": []}
    master_data = load_and_prep_data(DATA_DIR, config)
    checkpoint_state = load_resume_state(master_data) if AUTO_RESUME else None

    # ==========================================
    # PHASE 1: Baseline / Resume Setup
    # ==========================================
    phase1_start = time.perf_counter()
    logger.info("[Phase 1] Establishing Heterogeneous Baseline...")

    warm_start = checkpoint_state is not None

    if warm_start:
        logger.info("[Resume] Restoring optimization state from previous checkpoint.")
        removals = checkpoint_state.get("removals", removals)
        best_full_logloss = float(checkpoint_state.get("logloss", np.inf))
        current_weights = checkpoint_state.get("ensemble_weights")
        current_formula = checkpoint_state.get("best_formula", {"MLP": 1/3, "XGBoost": 1/3, "Logistic Regression": 1/3})
        mlp_cal = checkpoint_state.get("mlp_model")
        xgb_cal = checkpoint_state.get("xgb_model")
        lr_cal = checkpoint_state.get("lr_model")
        iteration = int(checkpoint_state.get("current_iteration", 0)) + 1
        consecutive_failures = 0

        current_data = apply_model_specific_reduction(master_data, removals)
        current_cached_probs = {
            "mlp": mlp_cal.predict_proba(current_data.mlp.X_val_processed)[:, 1],
            "xgb": xgb_cal.predict_proba(current_data.xgb.X_val)[:, 1],
            "lr": lr_cal.predict_proba(current_data.lr.X_val_processed)[:, 1],
        }
        current_standalone_losses = checkpoint_state.get("standalone_losses") or {
            "mlp": float(log_loss(current_data.y_val, current_cached_probs["mlp"])),
            "xgb": float(log_loss(current_data.y_val, current_cached_probs["xgb"])),
            "lr":  float(log_loss(current_data.y_val, current_cached_probs["lr"])),
        }
        base_estimators = {
            "mlp": unwrap_base_estimator(mlp_cal),
            "xgb": unwrap_base_estimator(xgb_cal),
            "lr": unwrap_base_estimator(lr_cal),
        }

        logger.info(f"[Resume] Restored removals: {json.dumps(removals, indent=4)}")
        logger.info(f"[Resume] Restored validation log loss: {best_full_logloss:.5f}")
        logger.info(f"[Resume] Standalone Losses: MLP: {current_standalone_losses['mlp']:.5f} | XGB: {current_standalone_losses['xgb']:.5f} | LR: {current_standalone_losses['lr']:.5f}")
        logger.info(f"[Resume] Resume iteration continues at {iteration}")
    else:
        temp_artifacts = TrainingArtifacts(config=config, output_dir=CHECKPOINT_DIR)
        temp_artifacts.data = apply_model_specific_reduction(master_data, removals)
        
        base_estimators = get_base_estimators(config, CHECKPOINT_DIR, temp_artifacts.data)

        # Full-Track Baseline maps the calibrated ground truth
        mlp_cal, xgb_cal, lr_cal, current_weights, current_formula, current_cached_probs, best_full_logloss = full_track_accept(
            temp_artifacts.data, base_estimators, config.cv_folds
        )
        current_standalone_losses = {
            "mlp": float(log_loss(temp_artifacts.data.y_val, current_cached_probs["mlp"])),
            "xgb": float(log_loss(temp_artifacts.data.y_val, current_cached_probs["xgb"])),
            "lr":  float(log_loss(temp_artifacts.data.y_val, current_cached_probs["lr"])),
        }
        iteration = 1
        consecutive_failures = 0
        
        logger.info(f"Baseline Full-Track LogLoss: {best_full_logloss:.5f}")
        logger.info(f"Baseline Standalone Losses: MLP: {current_standalone_losses['mlp']:.5f} | XGB: {current_standalone_losses['xgb']:.5f} | LR: {current_standalone_losses['lr']:.5f}")
        logger.info(f"Baseline Ensemble Formula: {current_formula}")
        logger.info(f"Baseline completed in {format_time(time.perf_counter() - phase1_start)}\n")
        
        save_lean_checkpoint(
            removals, best_full_logloss, current_weights, 
            current_formula, config, 0, mlp_cal, xgb_cal, lr_cal,
            current_standalone_losses
        )

    # ==========================================
    # PHASE 2: Independent Dynamic Loop
    # ==========================================
    logger.info("[Phase 2] Beginning Independent Elimination Loop...")
    current_data = apply_model_specific_reduction(master_data, removals)
    if not warm_start:
        base_estimators = {
            "mlp": unwrap_base_estimator(mlp_cal),
            "xgb": unwrap_base_estimator(xgb_cal),
            "lr": unwrap_base_estimator(lr_cal),
        }

    tested_in_current_state = {m: set() for m in ["mlp", "xgb", "lr"]}
    need_recompute_importances = True
    weakest_features = {}
    current_fast_baselines = {}
    
    while consecutive_failures < MAX_FAILURES_BEFORE_STOP:
        if need_recompute_importances:
            logger.info(f"\n--- Computing Independent Importances (Iter {iteration}) ---")
            current_data = apply_model_specific_reduction(master_data, removals)
            weakest_features = compute_independent_importances(
                mlp_cal, xgb_cal, lr_cal, current_data, config.random_seed
            )
            for m in ["mlp", "xgb", "lr"]:
                current_fast_baselines[m] = partial_fast_track(
                    m, current_data, base_estimators, current_cached_probs, current_weights
                )
            need_recompute_importances = False
        
        # Build Candidate Queue across ALL models
        model_queues = {}
        for model_name in ["mlp", "xgb", "lr"]:
            valid_features = [
                f for f in weakest_features.get(model_name, []) 
                if f not in removals[model_name] and f not in tested_in_current_state[model_name]
            ]
            model_queues[model_name] = valid_features[:WEAKEST_BATCH_SIZE_PER_MODEL]
        
        candidates = []
        # Interleave: MLP-1, XGB-1, LR-1, MLP-2, XGB-2, LR-2...
        for i in range(WEAKEST_BATCH_SIZE_PER_MODEL):
            for model_name in ["mlp", "xgb", "lr"]:
                if model_name in model_queues and i < len(model_queues[model_name]):
                    candidates.append((model_name, model_queues[model_name][i]))
        
        if not candidates:
            logger.info("No more candidate features available to test across models. Stopping optimization.")
            break

        logger.info(f"Generated {len(candidates)} interleaved candidates for evaluation (Iter {iteration}).")
        
        batch_accepted_count = 0
        
        for target_model, feature in candidates:
            iter_start = time.perf_counter()
            logger.info(f"\nTesting removal of [{feature}] from [{target_model.upper()}]")
            
            tested_in_current_state[target_model].add(feature)

            test_removals = deepcopy(removals)
            test_removals[target_model].append(feature)
            reduced_data = apply_model_specific_reduction(master_data, test_removals)
            
            test_fast_logloss = partial_fast_track(
                target_model, reduced_data, base_estimators, current_cached_probs, current_weights
            )
            
            fast_baseline = current_fast_baselines[target_model]
            fast_improvement = fast_baseline - test_fast_logloss
            
            # Fast-track coarse filter: require non-negative improvement before running full calibration
            if fast_improvement > 0.0:
                logger.info(f"  [GATE PASSED] Partial Fast-Track improved by {fast_improvement:.6f}. Running Full-Track...")
                
                # Calibrate ONLY the target model whose feature set changed
                new_cal, new_target_probs, test_standalone_loss = fit_and_calibrate_single_model(
                    target_model, base_estimators[target_model], reduced_data, cv_folds=3
                )
                
                # Evaluate ensemble log loss with FROZEN baseline weights (prevents weight gaming)
                candidate_probs = deepcopy(current_cached_probs)
                candidate_probs[target_model] = new_target_probs
                
                stacked = np.column_stack((candidate_probs["mlp"], candidate_probs["xgb"], candidate_probs["lr"]))
                blended = np.clip(np.dot(stacked, current_weights), 1e-15, 1 - 1e-15)
                test_full_logloss = float(log_loss(reduced_data.y_val, blended))
                
                full_improvement = best_full_logloss - test_full_logloss
                standalone_improvement = current_standalone_losses[target_model] - test_standalone_loss
                
                # Strict acceptance rules to ensure only non-useful features are pruned:
                # 1. Ensemble log loss with frozen weights improves by at least MIN_IMPROVEMENT,
                #    AND target model standalone loss does not degrade noticeably (tolerance of 1e-4), OR
                # 2. Target model standalone loss improves by at least MIN_IMPROVEMENT
                #    AND ensemble log loss does not degrade.
                is_accepted = (
                    (full_improvement >= MIN_IMPROVEMENT and standalone_improvement >= -1e-4) or
                    (standalone_improvement >= MIN_IMPROVEMENT and full_improvement >= -1e-5)
                )
                
                if is_accepted:
                    # ACCEPTED
                    best_full_logloss = test_full_logloss
                    current_cached_probs[target_model] = new_target_probs
                    current_standalone_losses[target_model] = test_standalone_loss
                    if target_model == "mlp":
                        mlp_cal = new_cal
                    elif target_model == "xgb":
                        xgb_cal = new_cal
                    elif target_model == "lr":
                        lr_cal = new_cal
                    
                    removals = test_removals
                    batch_accepted_count += 1
                    
                    # Update fast baselines with newly accepted feature state
                    current_data = apply_model_specific_reduction(master_data, removals)
                    for m in ["mlp", "xgb", "lr"]:
                        current_fast_baselines[m] = partial_fast_track(
                            m, current_data, base_estimators, current_cached_probs, current_weights
                        )
                    
                    save_lean_checkpoint(
                        removals, best_full_logloss, current_weights, 
                        current_formula, config, iteration, mlp_cal, xgb_cal, lr_cal,
                        current_standalone_losses
                    )
                    
                    logger.info(f"  [ACCEPTED] Ensemble LogLoss Δ: {full_improvement:+.6f} | Standalone [{target_model.upper()}] Δ: {standalone_improvement:+.6f}")
                    logger.info(f"  ✔ Candidate completed in {format_time(time.perf_counter() - iter_start)}")
                    
                    logger.info(f"\n--- OPTIMIZATION UPDATE ---")
                    logger.info(f"Dropped:           [{feature}] from [{target_model.upper()}]")
                    logger.info(f"Current LogLoss:   {best_full_logloss:.5f}")
                    logger.info(f"Standalone Losses: MLP: {current_standalone_losses['mlp']:.5f} | XGB: {current_standalone_losses['xgb']:.5f} | LR: {current_standalone_losses['lr']:.5f}")
                    logger.info(f"Current Ensemble:  MLP: {current_formula.get('MLP', 0.0):.2f} | XGB: {current_formula.get('XGBoost', 0.0):.2f} | LR: {current_formula.get('Logistic Regression', 0.0):.2f}")
                    logger.info(f"---------------------------\n")
                else:
                    logger.info(f"  [REJECTED] Insufficient improvement (Ensemble Δ: {full_improvement:+.6f}, Standalone Δ: {standalone_improvement:+.6f})")
            else:
                logger.info(f"  [REJECTED] Fast-Track failed to improve: {fast_improvement:.6f}")
                
            logger.info(f"  ✖ Candidate completed in {format_time(time.perf_counter() - iter_start)}")

        if batch_accepted_count > 0:
            logger.info(f"\n=== BATCH COMPLETE: Pruned {batch_accepted_count} features in Iteration {iteration} ===")
            # Re-tune ensemble weights once per batch across the updated models
            current_data = apply_model_specific_reduction(master_data, removals)
            temp_artifacts = TrainingArtifacts(config=config, output_dir=CHECKPOINT_DIR)
            temp_artifacts.mlp.model, temp_artifacts.mlp.feature_set = mlp_cal, current_data.mlp
            temp_artifacts.xgb.model, temp_artifacts.xgb.feature_set = xgb_cal, current_data.xgb
            temp_artifacts.lr.model,  temp_artifacts.lr.feature_set  = lr_cal,  current_data.lr

            new_weights, new_formula = learn_ensemble_weights(
                temp_artifacts.mlp, temp_artifacts.xgb, temp_artifacts.lr, current_data.y_val
            )
            current_weights = new_weights
            current_formula = new_formula
            
            # Recalculate best ensemble loss with new weights
            stacked = np.column_stack((current_cached_probs["mlp"], current_cached_probs["xgb"], current_cached_probs["lr"]))
            blended = np.clip(np.dot(stacked, current_weights), 1e-15, 1 - 1e-15)
            best_full_logloss = float(log_loss(current_data.y_val, blended))
            
            save_lean_checkpoint(
                removals, best_full_logloss, current_weights, 
                current_formula, config, iteration, mlp_cal, xgb_cal, lr_cal,
                current_standalone_losses
            )
            
            consecutive_failures = 0
            tested_in_current_state = {m: set() for m in ["mlp", "xgb", "lr"]}
            need_recompute_importances = True
        else:
            consecutive_failures += 1
            logger.info(f"\n=== BATCH COMPLETE: 0 features accepted in Iteration {iteration} (Failures: {consecutive_failures}/{MAX_FAILURES_BEFORE_STOP}) ===")
            need_recompute_importances = False
            
        if consecutive_failures >= MAX_FAILURES_BEFORE_STOP:
            logger.info(f"\nHit {MAX_FAILURES_BEFORE_STOP} consecutive failure rounds. Triggering Early Stopping.")
            break
        
        iteration += 1

    # ==========================================
    # PHASE 3: Summary & Artifact Export
    # ==========================================
    total_elapsed = time.perf_counter() - global_start_time
    logger.info("\n=========================================")
    logger.info("               SBS COMPLETE              ")
    logger.info("=========================================")
    logger.info(f"Total time elapsed: {format_time(total_elapsed)}")
    logger.info(f"Final Calibrated LogLoss: {best_full_logloss:.5f}")
    logger.info(f"Final Formula: {current_formula}")
    logger.info(f"Final Removals:\n{json.dumps(removals, indent=4)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / "final_removals.json", "w") as f:
        json.dump({
            "final_logloss": best_full_logloss,
            "final_formula": current_formula,
            "removals": removals,
            "standalone_losses": current_standalone_losses,
        }, f, indent=4)
    logger.info(f"Saved final removals to {CHECKPOINT_DIR / 'final_removals.json'}\n")

if __name__ == "__main__":
    main()