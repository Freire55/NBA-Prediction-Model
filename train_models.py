"""
Main orchestrator for the NBA prediction training and evaluation pipeline.

This pipeline executes a strictly chronological, leak-free machine learning workflow:
1. Data Ingestion & Scaling: Chronological train/val/test partitioning and feature scaling.
2. Hyperparameter Tuning & Calibration: TimeSeriesSplit log-loss searches and Platt calibration.
3. Continuous Margin & Pace Calibration: Expected margin regressor and pace-modulated normal CDF.
4. Ensemble Learning: Simplex-constrained convex weight optimization (SLSQP).
5. Full Retraining: Refitting tuned architectures on the combined historical period.
6. Explainability: Permutation importances, tree split gains, linear coefficients, and SHAP.
7. Out-of-Sample Evaluation: Multi-metric assessment on held-out test seasons.
8. Artifact Serialization: Checkpointing models, scalers, and diagnostic metadata.
"""

from dataclasses import asdict
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from training.config import (
    TrainingArtifacts,
    TrainingConfig,
    get_experiment_metadata,
)
from training.data import load_and_prep_data, scale_features
from training.ensemble import learn_ensemble_weights
from training.evaluation import (
    evaluate_all_models,
    print_metrics,
    save_metrics,
)
from training.explainability import generate_explanations
from training.margin import tune_pace_margin_classifier
from training.plots import (
    plot_calibration,
    plot_confusion,
    plot_margin_residual_distribution,
    plot_roc_curve,
)
from training.training import retrain_on_full_data
from training.tuning import tune_base_models
from training.utils import (
    PipelineStage,
    save_joblib,
    save_json,
    setup_logger,
)

TOTAL_PIPELINE_STAGES = 6


# ======================================================
# Directory Management & Output Setup
# ======================================================

def create_output_directory() -> Tuple[Path, Path]:
    """Creates the timestamped experiment directory under models/."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    output_dir = project_root / "models" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir


# ======================================================
# Artifact Serialization
# ======================================================

def save_json_artifacts(
    config: TrainingConfig,
    artifacts: TrainingArtifacts,
    ensemble_formula: Dict[str, float],
    metrics_dict: Dict[str, Dict[str, float]],
    output_dir: Path,
    data_dir: Path,
) -> None:
    """Saves structured experiment configuration, metadata, and evaluation results as JSON."""
    json_artifacts = {
        "training_config.json": config.to_dict(),
        "metadata.json": get_experiment_metadata(data_dir),
        "dataset_summary.json": asdict(artifacts.data.summary),
        "ensemble_formula.json": ensemble_formula,
        "test_metrics.json": metrics_dict,
    }
    for filename, payload in json_artifacts.items():
        save_json(payload, output_dir / filename)


def save_model_artifacts(artifacts: TrainingArtifacts) -> None:
    """Serializes fitted models, scalers, and feature signatures using joblib."""
    joblib_artifacts = {
        "mlp_model.pkl": artifacts.mlp.final_model,
        "xgb_model.pkl": artifacts.xgb.final_model,
        "catboost_model.pkl": artifacts.catboost.final_model,
        "lr_model.pkl": artifacts.lr.final_model,
        "mlp_scaler.pkl": artifacts.mlp.feature_set.scaler,
        "lr_scaler.pkl": artifacts.lr.feature_set.scaler,
        "ensemble_weights.pkl": artifacts.ensemble_weights,
        "mlp_features.pkl": artifacts.data.summary.mlp_feature_names,
        "xgb_features.pkl": artifacts.data.summary.xgb_feature_names,
        "catboost_features.pkl": artifacts.data.summary.catboost_feature_names,
        "lr_features.pkl": artifacts.data.summary.lr_feature_names,
    }

    if artifacts.margin.final_model is not None:
        joblib_artifacts["margin_model.pkl"] = artifacts.margin.final_model
        joblib_artifacts["margin_scaler.pkl"] = artifacts.margin.feature_set.scaler
        joblib_artifacts["margin_features.pkl"] = artifacts.data.summary.margin_feature_names

    for filename, obj in joblib_artifacts.items():
        if obj is not None:
            save_joblib(obj, artifacts.output_dir / filename)


# ======================================================
# Diagnostic Plotting
# ======================================================

def generate_evaluation_plots(
    artifacts: TrainingArtifacts,
    ensemble_probs: pd.Series,
    ensemble_preds: pd.Series,
    ensemble_metrics: Dict[str, float],
    config: TrainingConfig,
    output_dir: Path,
) -> None:
    """Generates ROC curve, calibration curve, confusion matrix, and margin residual plots."""
    y_test = artifacts.data.y_test

    plot_roc_curve(y_test, ensemble_probs, ensemble_metrics["ROC_AUC"], output_dir)
    plot_calibration(y_test, ensemble_probs, ensemble_metrics["Brier_Score"], config.calibration_bins, output_dir)
    plot_confusion(y_test, ensemble_preds, ensemble_metrics["Accuracy"], output_dir)

    # Margin residual distribution plot if margin model and margin targets are available
    if (
        artifacts.margin.final_model is not None
        and artifacts.data.y_margin_test is not None
        and hasattr(artifacts.margin.final_model, "predict_margin")
    ):
        reg_type = getattr(getattr(artifacts.margin.final_model, "regressor", None), "model_type", "ridge")
        X_test = (
            artifacts.margin.feature_set.X_test_processed
            if reg_type == "ridge"
            else artifacts.margin.feature_set.X_test
        )
        margin_preds = artifacts.margin.final_model.predict_margin(X_test)
        plot_margin_residual_distribution(artifacts.data.y_margin_test, margin_preds, output_dir)


def print_completion_summary(logger: logging.Logger, output_dir: Path) -> None:
    """Logs the final pipeline run completion banner and checklist of generated artifacts."""
    logger.info("=========================================")
    logger.info("          TRAINING COMPLETE              ")
    logger.info("=========================================\n")
    logger.info("Artifacts Generated in: %s/%s/", output_dir.parent.name, output_dir.name)

    artifacts = [
        "Models (.pkl)",
        "Metrics & Configurations (.json)",
        "Feature Rankings (.csv)",
        "training.log",
        "01_model_comparison.csv",
        "02_calibration_curve.png",
        "03_roc_curve.png",
        "04_confusion_matrix.png",
        "05_xgb_feature_importance.png",
        "06_mlp_feature_importance.png",
        "07_lr_coefficients.png",
        "08_xgb_shap_summary.png",
        "09_margin_coefficients.png",
        "10_margin_residuals.png",
    ]
    for artifact in artifacts:
        logger.info("  ✔ %s", artifact)


# ======================================================
# Stage Orchestration
# ======================================================

def run_stage_data_prep(artifacts: TrainingArtifacts, data_dir: Path, config: TrainingConfig) -> None:
    """Stage 1: Ingests datasets, sets up feature matrices, and fits initial scalers."""
    with PipelineStage(1, TOTAL_PIPELINE_STAGES, "Data preparation & scaling"):
        artifacts.data = load_and_prep_data(data_dir, config)
        scale_features(artifacts.data.mlp)
        scale_features(artifacts.data.lr)
        if artifacts.data.margin is not None:
            scale_features(artifacts.data.margin)


def run_stage_tuning(artifacts: TrainingArtifacts, config: TrainingConfig, output_dir: Path) -> None:
    """Stage 2: Hyperparameter optimization and model-selected calibration across all architectures."""
    with PipelineStage(2, TOTAL_PIPELINE_STAGES, "Hyperparameter tuning & probability calibration"):
        (
            artifacts.mlp,
            artifacts.xgb,
            artifacts.catboost,
            artifacts.lr,
        ) = tune_base_models(artifacts.data, config, output_dir)
        logging.getLogger(__name__).info("      Base models successfully tuned and calibrated.")

        if artifacts.data.margin is not None and artifacts.data.y_margin_train is not None:
            logging.getLogger(__name__).info("      Tuning Pace-Modulated Margin Regressor...")
            artifacts.margin, _ = tune_pace_margin_classifier(artifacts.data, config, output_dir)


def run_stage_ensemble(artifacts: TrainingArtifacts) -> Dict[str, float]:
    """Stage 3: Solves constrained optimization for optimal ensemble blending weights."""
    with PipelineStage(3, TOTAL_PIPELINE_STAGES, "Learning ensemble weighting"):
        weights, formula = learn_ensemble_weights(
            mlp=artifacts.mlp,
            xgb=artifacts.xgb,
            lr=artifacts.lr,
            y_val=artifacts.data.y_val,
            catboost=artifacts.catboost,
            margin=artifacts.margin,
        )
        artifacts.ensemble_weights = weights
        artifacts.ensemble_formula = formula
        return formula


def run_stage_retraining(artifacts: TrainingArtifacts) -> None:
    """Stage 4: Retrains best estimator specifications on the combined train+val window."""
    with PipelineStage(4, TOTAL_PIPELINE_STAGES, "Retraining on full historical data"):
        retrain_on_full_data(artifacts)


def run_stage_explainability(artifacts: TrainingArtifacts) -> None:
    """Stage 5: Computes permutation importances, linear weights, and Tree SHAP values."""
    with PipelineStage(5, TOTAL_PIPELINE_STAGES, "Generating SHAP & feature importances"):
        generate_explanations(artifacts)


def run_stage_evaluation(
    artifacts: TrainingArtifacts,
    config: TrainingConfig,
    output_dir: Path,
) -> Tuple[Dict[str, Dict[str, float]], pd.Series, pd.Series]:
    """Stage 6: Evaluates predictions on the held-out prospective test seasons."""
    with PipelineStage(6, TOTAL_PIPELINE_STAGES, "Final evaluation & plotting"):
        metrics_dict, ensemble_preds, ensemble_probs = evaluate_all_models(artifacts)
        print_metrics(metrics_dict)
        save_metrics(metrics_dict, output_dir)

        ensemble_metrics = metrics_dict["Ensemble"]
        generate_evaluation_plots(
            artifacts,
            ensemble_probs,
            ensemble_preds,
            ensemble_metrics,
            config,
            output_dir,
        )
        return metrics_dict, ensemble_preds, ensemble_probs


# ======================================================
# Main Execution Entry Point
# ======================================================

def main() -> None:
    """Executes the complete NBA prediction model training pipeline."""
    data_dir, output_dir = create_output_directory()
    logger = setup_logger(output_dir)

    logger.info("\n=========================================")
    logger.info("      NBA PREDICTION TRAINING PIPELINE   ")
    logger.info("=========================================\n")

    config = TrainingConfig()
    artifacts = TrainingArtifacts(config=config, output_dir=output_dir)

    run_stage_data_prep(artifacts, data_dir, config)
    run_stage_tuning(artifacts, config, output_dir)
    ensemble_formula = run_stage_ensemble(artifacts)
    run_stage_retraining(artifacts)
    run_stage_explainability(artifacts)
    metrics_dict, _, _ = run_stage_evaluation(artifacts, config, output_dir)

    save_json_artifacts(
        config=config,
        artifacts=artifacts,
        ensemble_formula=ensemble_formula,
        metrics_dict=metrics_dict,
        output_dir=output_dir,
        data_dir=data_dir,
    )
    save_model_artifacts(artifacts)
    print_completion_summary(logger, output_dir)


if __name__ == "__main__":
    main()