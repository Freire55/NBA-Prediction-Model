"""
Generates feature importance and model explainability artifacts.

This module computes feature importance for each trained model,
generates SHAP explanations for the XGBoost model, and exports
combined feature rankings together with visualization plots.

Outputs:
    05_xgb_feature_importance.png
    06_mlp_feature_importance.png
    07_lr_coefficients.png
    08_xgb_shap_summary.png
    combined_feature_rankings.csv
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from training.config import TrainingArtifacts
from training.plots import plot_horizontal_bar
from training.utils import save_plot, unwrap_base_estimator

# ======================================================
# Constants
# ======================================================

XGB_IMPORTANCE_PLOT = "05_xgb_feature_importance.png"
MLP_IMPORTANCE_PLOT = "06_mlp_feature_importance.png"
LR_COEFFICIENTS_PLOT = "07_lr_coefficients.png"
SHAP_PLOT = "08_xgb_shap_summary.png"

COMBINED_RANKINGS_FILE = "combined_feature_rankings.csv"

# ======================================================
# Logging
# ======================================================

logger = logging.getLogger(__name__)


# ======================================================
# Explainability Functions
# ======================================================

def generate_mlp_importance(
    artifacts: TrainingArtifacts,
) -> pd.DataFrame:
    """
    Computes permutation feature importance for the trained MLP model.
    """
    logger.info("      Evaluating MLP Permutation Importance...")

    permutation = permutation_importance(
        artifacts.mlp_final,
        artifacts.X_val_scaled,
        artifacts.y_val,
        n_repeats=artifacts.config.permutation_repeats,
        random_state=artifacts.config.random_seed,
        n_jobs=-1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "Feature": artifacts.features,
                "Importance": permutation.importances_mean,
            }
        )
        .sort_values("Importance", ascending=False)
    )

    plot_horizontal_bar(
        importance_df,
        "MLP Permutation Importance (Top 15)",
        MLP_IMPORTANCE_PLOT,
        sort_col="Importance",
        output_dir=artifacts.output_dir,
    )

    return importance_df


def generate_xgb_importance(
    artifacts: TrainingArtifacts,
) -> pd.DataFrame:
    """
    Extracts the built-in feature importance values from XGBoost.
    """
    logger.info("      Extracting XGBoost Tree Importance...")

    xgb_raw = unwrap_base_estimator(artifacts.xgb_final)

    importance_df = (
        pd.DataFrame(
            {
                "Feature": artifacts.features,
                "Importance": xgb_raw.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
    )

    plot_horizontal_bar(
        importance_df,
        "XGBoost Feature Importance (Top 15)",
        XGB_IMPORTANCE_PLOT,
        sort_col="Importance",
        output_dir=artifacts.output_dir,
    )

    return importance_df


def generate_lr_coefficients(
    artifacts: TrainingArtifacts,
) -> pd.DataFrame:
    """
    Extracts and ranks Logistic Regression coefficients by absolute value.
    """
    logger.info("      Extracting Logistic Regression Coefficients...")

    coefficients_df = pd.DataFrame(
        {
            "Feature": artifacts.features,
            "Weight": artifacts.lr_final.coef_[0],
        }
    )

    coefficients_df["Abs_Weight"] = (
        coefficients_df["Weight"].abs()
    )

    coefficients_df = coefficients_df.sort_values(
        "Abs_Weight",
        ascending=False,
    )

    plot_horizontal_bar(
        coefficients_df,
        "Logistic Regression Coefficients (Absolute Top 15)",
        LR_COEFFICIENTS_PLOT,
        sort_col="Abs_Weight",
        xlabel="Absolute Coefficient",
        output_dir=artifacts.output_dir,
    )

    return coefficients_df


def generate_shap(
    artifacts: TrainingArtifacts,
) -> None:
    """
    Generates a global SHAP summary plot for the XGBoost model.
    """
    logger.info("      Generating SHAP explanations for XGBoost...")

    sample_size = min(
        artifacts.config.n_shap_samples,
        len(artifacts.X_train_full),
    )

    X_explain = artifacts.X_train_full.sample(
        n=sample_size,
        random_state=artifacts.config.random_seed,
    )

    explainer = shap.TreeExplainer(unwrap_base_estimator(artifacts.xgb_final))
    shap_values = explainer.shap_values(X_explain)

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_explain,
        show=False,
    )

    plt.title("Global SHAP Summary\n(XGBoost)")

    save_plot(
        artifacts.output_dir,
        SHAP_PLOT,
    )


# ======================================================
# Orchestration
# ======================================================

def generate_explanations(
    artifacts: TrainingArtifacts,
) -> None:
    """
    Generates all explainability artifacts, including feature
    importance rankings, SHAP explanations, and combined rankings.
    """
    mlp_importance = generate_mlp_importance(artifacts)
    xgb_importance = generate_xgb_importance(artifacts)
    lr_coefficients = generate_lr_coefficients(artifacts)

    combined = (
        mlp_importance[
            ["Feature", "Importance"]
        ].rename(columns={"Importance": "MLP"})
        .merge(
            xgb_importance[
                ["Feature", "Importance"]
            ].rename(columns={"Importance": "XGBoost"}),
            on="Feature",
            how="outer",
        )
        .merge(
            lr_coefficients[
                ["Feature", "Abs_Weight"]
            ].rename(
                columns={
                    "Abs_Weight": "Logistic_Regression"
                }
            ),
            on="Feature",
            how="outer",
        )
    )

    combined.to_csv(
        artifacts.output_dir / COMBINED_RANKINGS_FILE,
        index=False,
    )

    generate_shap(artifacts)