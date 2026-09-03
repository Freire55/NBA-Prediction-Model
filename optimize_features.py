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

from training.config import TrainingConfig, TrainingArtifacts, FeatureSet, TrainingData
from training.data import load_and_prep_data, scale_features
from training.ensemble import learn_ensemble_weights
from training.tuning import tune_base_models
from training.utils import setup_logger, unwrap_base_estimator, save_joblib

# ======================================================
# Configuration
# ======================================================
MAX_FAILURES_BEFORE_STOP = 30
WEAKEST_BATCH_SIZE_PER_MODEL = 10  # Tests 10 from MLP, 10 from XGB, 10 from LR per iteration
PERMUTATION_REPEATS = 15
MIN_IMPROVEMENT = 1e-4
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
    """Calculates SNR Permutation Importance for a single model."""
    res = permutation_importance(
        model, X_val, y_val, scoring="neg_log_loss", 
        n_repeats=PERMUTATION_REPEATS, random_state=random_seed, n_jobs=-1
    )
    
    snr_scores = {
        name: m / (s + 1e-9) 
        for name, m, s in zip(feature_names, res.importances_mean, res.importances_std)
    }
    return [feat[0] for feat in sorted(snr_scores.items(), key=lambda x: x[1])]


def compute_independent_importances(mlp_cal, xgb_cal, lr_cal, reduced_data, random_seed) -> dict:
    """Returns sorted weakest features independently for each model architecture."""
    return {
        "mlp": get_model_importance(mlp_cal, reduced_data.mlp.X_val_processed, reduced_data.y_val, reduced_data.mlp.feature_names, random_seed),
        "xgb": get_model_importance(xgb_cal, reduced_data.xgb.X_val, reduced_data.y_val, reduced_data.xgb.feature_names, random_seed),
        "lr": get_model_importance(lr_cal, reduced_data.lr.X_val_processed, reduced_data.y_val, reduced_data.lr.feature_names, random_seed)
    }


def partial_fast_track(target_model_name, reduced_data, base_estimators, cached_probs, frozen_weights) -> float:
    """
    ULTRA-FAST TEST: Retrains ONLY the target model. Uses cached predictions for the others.
    """
    new_probs = deepcopy(cached_probs)
    
    if target_model_name == "mlp":
        raw_model = clone(base_estimators["mlp"]).fit(reduced_data.mlp.X_train_processed, reduced_data.y_train)
        new_probs["mlp"] = raw_model.predict_proba(reduced_data.mlp.X_val_processed)[:, 1]
    elif target_model_name == "xgb":
        raw_model = clone(base_estimators["xgb"]).fit(reduced_data.xgb.X_train, reduced_data.y_train)
        new_probs["xgb"] = raw_model.predict_proba(reduced_data.xgb.X_val)[:, 1]
    elif target_model_name == "lr":
        raw_model = clone(base_estimators["lr"]).fit(reduced_data.lr.X_train_processed, reduced_data.y_train)
        new_probs["lr"] = raw_model.predict_proba(reduced_data.lr.X_val_processed)[:, 1]

    stacked = np.column_stack((new_probs["mlp"], new_probs["xgb"], new_probs["lr"]))
    blended = np.clip(np.dot(stacked, frozen_weights), 1e-15, 1 - 1e-15)
    return log_loss(reduced_data.y_val, blended)


def full_track_accept(reduced_data, base_estimators, cv_folds):
    """FULL ACCEPTANCE: Recalibrates all models, relearns weights, caches new baseline probabilities."""
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    mlp_cal = CalibratedClassifierCV(clone(base_estimators["mlp"]), method="sigmoid", cv=tscv, n_jobs=-1)
    xgb_cal = CalibratedClassifierCV(clone(base_estimators["xgb"]), method="sigmoid", cv=tscv, n_jobs=-1)
    lr_cal = CalibratedClassifierCV(clone(base_estimators["lr"]), method="sigmoid", cv=tscv, n_jobs=-1)

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


def save_lean_checkpoint(removals, best_logloss, weights, formula, config, iter_num, mlp_cal, xgb_cal, lr_cal):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "removals": removals,
        "logloss": best_logloss,
        "ensemble_weights": weights,
        "best_formula": formula,
        "current_iteration": iter_num,
        "mlp_model": mlp_cal,
        "xgb_model": xgb_cal,
        "lr_model": lr_cal
    }
    save_joblib(state, CHECKPOINT_DIR / "best_sbs_state.pkl")
    
    with open(CHECKPOINT_DIR / "current_removals.json", "w") as f:
        json.dump({
            "best_logloss": best_logloss, 
            "formula": formula,
            "iteration": iter_num,
            "removals": removals
        }, f, indent=4)


def format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s" if mins > 0 else f"{secs}s"


def load_resume_state():
    """Loads the most recent optimization checkpoint if available."""
    state_path = CHECKPOINT_DIR / "best_sbs_state.pkl"
    if not AUTO_RESUME or not state_path.exists():
        return None

    try:
        state = joblib.load(state_path)
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
    checkpoint_state = load_resume_state() if AUTO_RESUME else None

    # ==========================================
    # PHASE 1: Baseline / Resume Setup
    # ==========================================
    phase1_start = time.perf_counter()
    logger.info("[Phase 1] Establishing Heterogeneous Baseline...")

    master_data = load_and_prep_data(DATA_DIR, config)
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
        base_estimators = {
            "mlp": unwrap_base_estimator(mlp_cal),
            "xgb": unwrap_base_estimator(xgb_cal),
            "lr": unwrap_base_estimator(lr_cal),
        }

        logger.info(f"[Resume] Restored removals: {json.dumps(removals, indent=4)}")
        logger.info(f"[Resume] Restored validation log loss: {best_full_logloss:.5f}")
        logger.info(f"[Resume] Resume iteration continues at {iteration}")
    else:
        temp_artifacts = TrainingArtifacts(config=config, output_dir=CHECKPOINT_DIR)
        temp_artifacts.data = apply_model_specific_reduction(master_data, removals)
        
        temp_artifacts.mlp, temp_artifacts.xgb, temp_artifacts.lr = tune_base_models(
            temp_artifacts.data, config, CHECKPOINT_DIR
        )
        
        base_estimators = {
            "mlp": unwrap_base_estimator(temp_artifacts.mlp.model),
            "xgb": unwrap_base_estimator(temp_artifacts.xgb.model),
            "lr": unwrap_base_estimator(temp_artifacts.lr.model)
        }

        # Full-Track Baseline maps the calibrated ground truth
        mlp_cal, xgb_cal, lr_cal, current_weights, current_formula, current_cached_probs, best_full_logloss = full_track_accept(
            temp_artifacts.data, base_estimators, config.cv_folds
        )
        iteration = 1
        consecutive_failures = 0
        
        logger.info(f"Baseline Full-Track LogLoss: {best_full_logloss:.5f}")
        logger.info(f"Baseline completed in {format_time(time.perf_counter() - phase1_start)}\n")
        
        save_lean_checkpoint(
            removals, best_full_logloss, current_weights, 
            current_formula, config, 0, mlp_cal, xgb_cal, lr_cal
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
            # Establish matched uncalibrated fast-track baselines for the current feature set
            for m in ["mlp", "xgb", "lr"]:
                current_fast_baselines[m] = partial_fast_track(
                    m, current_data, base_estimators, current_cached_probs, current_weights
                )
            need_recompute_importances = False
        
        # ========================================================
        # Build Candidate Queue via Round-Robin Interleaving
        # ========================================================
        model_queues = {}
        for model_name, proper_name in [("mlp", "MLP"), ("xgb", "XGBoost"), ("lr", "Logistic Regression")]:
            # Only test features for models that the ensemble actually uses
            if current_formula.get(proper_name, 0.0) > 0.01:
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
            logger.info("No more candidate features available to test across active models. Stopping optimization.")
            break

        logger.info(f"Generated {len(candidates)} interleaved candidates for evaluation (Iter {iteration}).")
        # ========================================================
        
        batch_accepted = False
        
        for target_model, feature in candidates:
            iter_start = time.perf_counter()
            logger.info(f"\nTesting removal of [{feature}] from [{target_model.upper()}]")
            
            # Track that this feature has been evaluated in the current state
            tested_in_current_state[target_model].add(feature)

            # Temporarily apply removal
            test_removals = deepcopy(removals)
            test_removals[target_model].append(feature)
            reduced_data = apply_model_specific_reduction(master_data, test_removals)
            
            # PARTIAL FAST TRACK: Extremely fast, only retrains the affected model
            test_fast_logloss = partial_fast_track(
                target_model, reduced_data, base_estimators, current_cached_probs, current_weights
            )
            
            # Compare against the target model's fast baseline on the same uncalibrated scale
            fast_baseline = current_fast_baselines[target_model]
            fast_improvement = fast_baseline - test_fast_logloss
            
            if fast_improvement > MIN_IMPROVEMENT:
                logger.info(f"  [GATE PASSED] Partial Fast-Track improved by {fast_improvement:.5f}. Running Full-Track...")
                
                # FULL TRACK ACCEPTANCE
                t_mlp_cal, t_xgb_cal, t_lr_cal, t_weights, t_formula, t_cached_probs, test_full_logloss = full_track_accept(
                    reduced_data, base_estimators, config.cv_folds
                )
                
                full_improvement = best_full_logloss - test_full_logloss
                
                if full_improvement > MIN_IMPROVEMENT:
                    # ACCEPTED
                    best_full_logloss = test_full_logloss
                    current_weights = t_weights
                    current_formula = t_formula
                    current_cached_probs = t_cached_probs
                    mlp_cal, xgb_cal, lr_cal = t_mlp_cal, t_xgb_cal, t_lr_cal
                    
                    removals = test_removals
                    consecutive_failures = 0
                    batch_accepted = True
                    
                    # State changed: reset tested pool and recompute importances & fast baselines
                    tested_in_current_state = {m: set() for m in ["mlp", "xgb", "lr"]}
                    need_recompute_importances = True

                    save_lean_checkpoint(
                        removals, best_full_logloss, current_weights, 
                        current_formula, config, iteration, mlp_cal, xgb_cal, lr_cal
                    )
                    
                    logger.info(f"  [ACCEPTED] Full-Track LogLoss improved by {full_improvement:.5f}")
                    logger.info(f"  ✔ Iteration completed in {format_time(time.perf_counter() - iter_start)}")
                    
                    logger.info(f"\n--- OPTIMIZATION UPDATE ---")
                    logger.info(f"Dropped:           [{feature}] from [{target_model.upper()}]")
                    logger.info(f"Current LogLoss:   {best_full_logloss:.5f}")
                    logger.info(f"Current Ensemble:  MLP: {current_formula.get('MLP', 0.0):.2f} | XGB: {current_formula.get('XGBoost', 0.0):.2f} | LR: {current_formula.get('Logistic Regression', 0.0):.2f}")
                    logger.info(f"---------------------------\n")
                    
                    break # Break to recompute independent importances with new feature space
                else:
                    logger.info(f"  [REJECTED] Full-Track failed to improve: {full_improvement:.5f}")
                    consecutive_failures += 1
            else:
                logger.info(f"  [REJECTED] Fast-Track failed to improve: {fast_improvement:.5f}")
                consecutive_failures += 1
                
            logger.info(f"  ✖ Iteration completed in {format_time(time.perf_counter() - iter_start)}")
            
            if consecutive_failures >= MAX_FAILURES_BEFORE_STOP:
                logger.info(f"\nHit {MAX_FAILURES_BEFORE_STOP} consecutive failures. Triggering Early Stopping.")
                break
        
        iteration += 1

    # ==========================================
    # PHASE 3: Summary
    # ==========================================
    total_elapsed = time.perf_counter() - global_start_time
    logger.info("\n=========================================")
    logger.info("               SBS COMPLETE              ")
    logger.info("=========================================\n")
    logger.info(f"Total time elapsed: {format_time(total_elapsed)}")
    logger.info(f"Final Calibrated LogLoss: {best_full_logloss:.5f}")
    logger.info(f"Final Removals:\n{json.dumps(removals, indent=4)}")

if __name__ == "__main__":
    main()