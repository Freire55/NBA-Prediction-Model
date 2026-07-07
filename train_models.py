"""
Main entry point for the NBA prediction training pipeline.

This script orchestrates the complete machine learning workflow:

1. Data preparation
2. Hyperparameter tuning
3. Ensemble weight learning
4. Final model retraining
5. Explainability generation
6. Final evaluation
7. Artifact serialization

Each stage is timed and logged, with all outputs stored in a
timestamped experiment directory.
"""

from datetime import datetime
from pathlib import Path

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
from training.plots import (
    plot_calibration,
    plot_confusion,
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


def create_output_directory() -> tuple[Path, Path]:
    """Creates the timestamped experiment directory."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data"

    output_dir = (
        project_root
        / "models"
        / f"run_{timestamp}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, output_dir


def save_json_artifacts(
    config,
    dataset_summary,
    ensemble_formula,
    metrics_dict,
    output_dir,
):
    """Saves JSON metadata produced during training."""

    json_artifacts = {
        "training_config.json": config.to_dict(),
        "metadata.json": get_experiment_metadata(),
        "dataset_summary.json": dataset_summary,
        "ensemble_formula.json": ensemble_formula,
        "test_metrics.json": metrics_dict,
    }

    for filename, data in json_artifacts.items():
        save_json(data, output_dir / filename)


def save_model_artifacts(artifacts):
    """Serializes trained models and supporting objects."""

    joblib_artifacts = {
        "mlp_model.pkl": artifacts.mlp_final,
        "xgb_model.pkl": artifacts.xgb_final,
        "lr_model.pkl": artifacts.lr_final,
        "scaler.pkl": artifacts.scaler_full,
        "ensemble_weights.pkl": artifacts.ensemble_weights,
        "feature_order.pkl": artifacts.features,
    }

    for filename, obj in joblib_artifacts.items():
        save_joblib(obj, artifacts.output_dir / filename)


def print_completion_summary(logger, output_dir):
    """Logs the training completion summary."""

    logger.info("=========================================")
    logger.info("          TRAINING COMPLETE              ")
    logger.info("=========================================\n")

    logger.info(
        f"Artifacts Generated in: {output_dir.parent.name}/{output_dir.name}/"
    )

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
    ]

    for artifact in artifacts:
        logger.info(f"  ✔ {artifact}")


def main() -> None:
    """
    Executes the complete NBA prediction model training pipeline.
    """

    data_dir, output_dir = create_output_directory()

    logger = setup_logger(output_dir)

    logger.info("\n=========================================")
    logger.info("      NBA PREDICTION TRAINING PIPELINE   ")
    logger.info("=========================================\n")

    config = TrainingConfig()
    artifacts = TrainingArtifacts(
        config=config,
        output_dir=output_dir,
    )

    # ======================================================
    # Data Preparation
    # ======================================================

    with PipelineStage(
        1,
        TOTAL_PIPELINE_STAGES,
        "Data preparation & scaling",
    ):
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            features,
            dataset_summary,
        ) = load_and_prep_data(data_dir, config)

        X_train_scaled, X_val_scaled, scaler_val = scale_features(
            X_train,
            X_val,
        )

        artifacts.X_train = X_train
        artifacts.y_train = y_train
        artifacts.X_val = X_val
        artifacts.y_val = y_val
        artifacts.X_test = X_test
        artifacts.y_test = y_test
        artifacts.features = features
        artifacts.X_train_scaled = X_train_scaled
        artifacts.X_val_scaled = X_val_scaled
        artifacts.scaler_val = scaler_val

    # ======================================================
    # Hyperparameter Tuning
    # ======================================================

    with PipelineStage(
        2,
        TOTAL_PIPELINE_STAGES,
        "Hyperparameter tuning base models",
    ):
        (
            artifacts.mlp_model,
            artifacts.xgb_model,
            artifacts.lr_model,
        ) = tune_base_models(
            artifacts.X_train_scaled,
            artifacts.y_train,
            artifacts.X_train,
            config,
            output_dir,
        )

    # ======================================================
    # Ensemble Learning
    # ======================================================

    with PipelineStage(
        3,
        TOTAL_PIPELINE_STAGES,
        "Learning ensemble weighting",
    ):
        (
            artifacts.ensemble_weights,
            ensemble_formula,
        ) = learn_ensemble_weights(
            artifacts.mlp_model,
            artifacts.xgb_model,
            artifacts.lr_model,
            artifacts.X_val_scaled,
            artifacts.X_val,
            artifacts.y_val,
        )

    # ======================================================
    # Retraining
    # ======================================================

    with PipelineStage(
        4,
        TOTAL_PIPELINE_STAGES,
        "Retraining on full historical data",
    ):
        (
            artifacts.mlp_final,
            artifacts.xgb_final,
            artifacts.lr_final,
            artifacts.scaler_full,
            artifacts.X_train_full,
            artifacts.X_test_scaled_full,
        ) = retrain_on_full_data(
            artifacts.X_train,
            artifacts.X_val,
            artifacts.y_train,
            artifacts.y_val,
            artifacts.X_test,
            artifacts.mlp_model,
            artifacts.xgb_model,
            artifacts.lr_model,
            output_dir,
        )

    # ======================================================
    # Explainability
    # ======================================================

    with PipelineStage(
        5,
        TOTAL_PIPELINE_STAGES,
        "Generating SHAP & feature importances",
    ):
        generate_explanations(artifacts)

    # ======================================================
    # Evaluation
    # ======================================================

    with PipelineStage(
        6,
        TOTAL_PIPELINE_STAGES,
        "Final evaluation & plotting",
    ):
        (
            metrics_dict,
            ensemble_preds,
            ensemble_probs,
        ) = evaluate_all_models(artifacts)

        print_metrics(metrics_dict)
        save_metrics(metrics_dict, output_dir)

        ensemble_metrics = metrics_dict["Ensemble"]

        plot_roc_curve(
            artifacts.y_test,
            ensemble_probs,
            ensemble_metrics["ROC_AUC"],
            output_dir,
        )

        plot_calibration(
            artifacts.y_test,
            ensemble_probs,
            ensemble_metrics["Brier_Score"],
            config.calibration_bins,
            output_dir,
        )

        plot_confusion(
            artifacts.y_test,
            ensemble_preds,
            ensemble_metrics["Accuracy"],
            output_dir,
        )

    save_json_artifacts(
        config,
        dataset_summary,
        ensemble_formula,
        metrics_dict,
        output_dir,
    )

    save_model_artifacts(artifacts)

    print_completion_summary(logger, output_dir)


if __name__ == "__main__":
    main()