"""
Feature importance and model interpretability generation for NBA models.

This module extracts architecture-specific feature importances and global SHAP
values to explain the predictive drivers behind model forecasts:
- MLP: Permutation feature importance across chronological validation games
- XGBoost: MDI (Mean Decrease in Impurity) tree importance and SHAP TreeExplainer
- Logistic Regression: L2-regularized standardized regression coefficients
- Margin Regressor: Point differential regression weights (Ridge/XGB)

Outputs:
    05_xgb_feature_importance.png
    06_mlp_feature_importance.png
    07_lr_coefficients.png
    08_xgb_shap_summary.png
    09_margin_coefficients.png (if margin model present)
    combined_feature_rankings.csv
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from training.config import TrainingArtifacts
from training.margin import MarginRegressor, PaceModulatedMarginClassifier
from training.plots import plot_horizontal_bar
from training.utils import save_plot, unwrap_base_estimator

# ======================================================
# Output Filenames
# ======================================================

XGB_IMPORTANCE_PLOT = "05_xgb_feature_importance.png"
MLP_IMPORTANCE_PLOT = "06_mlp_feature_importance.png"
LR_COEFFICIENTS_PLOT = "07_lr_coefficients.png"
SHAP_PLOT = "08_xgb_shap_summary.png"
MARGIN_IMPORTANCE_PLOT = "09_margin_coefficients.png"
CATBOOST_IMPORTANCE_PLOT = "11_catboost_feature_importance.png"

COMBINED_RANKINGS_FILE = "combined_feature_rankings.csv"

logger = logging.getLogger(__name__)


# ======================================================
# Individual Model Interpretability
# ======================================================

def generate_mlp_importance(
    artifacts: TrainingArtifacts,
) -> pd.DataFrame:
    """
    Evaluates permutation importance for the Multi-Layer Perceptron.

    Shuffles one feature column at a time on validation data and records the
    resulting deterioration in predictive log-loss or accuracy.
    """
    logger.info("      Evaluating MLP Permutation Importance...")

    X_val = artifacts.mlp.feature_set.X_val_processed
    y_val = artifacts.data.y_val

    # Replace potential NaNs/Infs with 0 to safeguard permutation scorer
    X_val_clean = pd.DataFrame(X_val).fillna(0).replace([np.inf, -np.inf], 0).values

    permutation = permutation_importance(
        artifacts.mlp.final_model,
        X_val_clean,
        y_val,
        n_repeats=artifacts.config.permutation_repeats,
        random_state=artifacts.config.random_seed,
        n_jobs=-1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "Feature": artifacts.mlp.feature_set.feature_names,
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
    """Extracts native Gini gain / tree split importance from XGBoost."""
    logger.info("      Extracting XGBoost Tree Importance...")

    xgb_raw = unwrap_base_estimator(artifacts.xgb.final_model)

    importance_df = (
        pd.DataFrame(
            {
                "Feature": artifacts.xgb.feature_set.feature_names,
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
    """Ranks Logistic Regression coefficients by absolute magnitude."""
    logger.info("      Extracting Logistic Regression Coefficients...")

    lr_raw = unwrap_base_estimator(artifacts.lr.final_model)

    coefficients_df = pd.DataFrame(
        {
            "Feature": artifacts.lr.feature_set.feature_names,
            "Weight": lr_raw.coef_[0],
        }
    )
    coefficients_df["Abs_Weight"] = coefficients_df["Weight"].abs()
    coefficients_df = coefficients_df.sort_values("Abs_Weight", ascending=False)

    plot_horizontal_bar(
        coefficients_df,
        "Logistic Regression Coefficients (Absolute Top 15)",
        LR_COEFFICIENTS_PLOT,
        sort_col="Abs_Weight",
        xlabel="Absolute Coefficient",
        output_dir=artifacts.output_dir,
    )

    return coefficients_df


def generate_margin_importance(
    artifacts: TrainingArtifacts,
) -> Optional[pd.DataFrame]:
    """Extracts weights or feature importances from the continuous Margin Regressor."""
    if artifacts.margin.final_model is None or artifacts.margin.feature_set is None:
        return None

    logger.info("      Extracting Margin Regressor Coefficients...")
    model = artifacts.margin.final_model
    reg = model.regressor if isinstance(model, PaceModulatedMarginClassifier) else model

    if not isinstance(reg, MarginRegressor):
        return None

    if reg.model_type == "ridge":
        ridge_est = reg.estimator_
        weights = ridge_est.coef_
        df = pd.DataFrame(
            {
                "Feature": reg.feature_names_ or artifacts.margin.feature_set.feature_names,
                "Weight": weights,
                "Abs_Weight": np.abs(weights),
            }
        ).sort_values("Abs_Weight", ascending=False)

        plot_horizontal_bar(
            df,
            "Margin Ridge Regression Coefficients (Top 15)",
            MARGIN_IMPORTANCE_PLOT,
            sort_col="Abs_Weight",
            xlabel="Absolute Coefficient (PTS Differential)",
            output_dir=artifacts.output_dir,
        )
        return df

    elif reg.model_type == "xgb":
        xgb_est = reg.estimator_
        df = pd.DataFrame(
            {
                "Feature": reg.feature_names_ or artifacts.margin.feature_set.feature_names,
                "Importance": xgb_est.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        plot_horizontal_bar(
            df,
            "Margin XGBoost Importance (Top 15)",
            MARGIN_IMPORTANCE_PLOT,
            sort_col="Importance",
            xlabel="Gain Importance",
            output_dir=artifacts.output_dir,
        )
        return df

    return None


def generate_catboost_importance(
    artifacts: TrainingArtifacts,
) -> Optional[pd.DataFrame]:
    """Extracts native feature importance from the fitted CatBoost classifier."""
    if artifacts.catboost.final_model is None or artifacts.catboost.feature_set is None:
        return None

    logger.info("      Extracting CatBoost Feature Importance...")
    cb_raw = unwrap_base_estimator(artifacts.catboost.final_model)

    if hasattr(cb_raw, "get_feature_importance"):
        importances = cb_raw.get_feature_importance()
    elif hasattr(cb_raw, "feature_importances_"):
        importances = cb_raw.feature_importances_
    else:
        return None

    importance_df = (
        pd.DataFrame(
            {
                "Feature": artifacts.catboost.feature_set.feature_names,
                "Importance": importances,
            }
        )
        .sort_values("Importance", ascending=False)
    )

    plot_horizontal_bar(
        importance_df,
        "CatBoost Feature Importance (Top 15)",
        CATBOOST_IMPORTANCE_PLOT,
        sort_col="Importance",
        output_dir=artifacts.output_dir,
    )

    return importance_df


def generate_shap(
    artifacts: TrainingArtifacts,
) -> None:
    """Generates a global SHAP beeswarm summary plot for the XGBoost classifier."""
    logger.info("      Generating SHAP explanations for XGBoost...")

    sample_size = min(
        artifacts.config.n_shap_samples,
        len(artifacts.xgb.feature_set.X_train_full),
    )

    X_explain = artifacts.xgb.feature_set.X_train_full.sample(
        n=sample_size,
        random_state=artifacts.config.random_seed,
    )

    explainer = shap.TreeExplainer(unwrap_base_estimator(artifacts.xgb.final_model))
    shap_values = explainer.shap_values(X_explain, check_additivity=False)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.title("Global SHAP Summary\n(XGBoost)")
    save_plot(artifacts.output_dir, SHAP_PLOT)


# ======================================================
# Orchestration
# ======================================================

def generate_explanations(
    artifacts: TrainingArtifacts,
) -> None:
    """
    Generates all explainability artifacts across architectures and exports
    a consolidated multi-model feature ranking CSV.
    """
    mlp_importance = generate_mlp_importance(artifacts)
    xgb_importance = generate_xgb_importance(artifacts)
    catboost_importance = generate_catboost_importance(artifacts)
    lr_coefficients = (
        generate_lr_coefficients(artifacts)
        if artifacts.lr.final_model is not None and artifacts.lr.feature_set is not None
        else None
    )
    margin_importance = generate_margin_importance(artifacts)

    combined = (
        mlp_importance[["Feature", "Importance"]]
        .rename(columns={"Importance": "MLP"})
        .merge(
            xgb_importance[["Feature", "Importance"]].rename(columns={"Importance": "XGBoost"}),
            on="Feature",
            how="outer",
        )
    )

    if catboost_importance is not None:
        combined = combined.merge(
            catboost_importance[["Feature", "Importance"]].rename(columns={"Importance": "CatBoost"}),
            on="Feature",
            how="outer",
        )

    if lr_coefficients is not None:
        combined = combined.merge(
            lr_coefficients[["Feature", "Abs_Weight"]].rename(
                columns={"Abs_Weight": "Logistic_Regression"}
            ),
            on="Feature",
            how="outer",
        )

    if margin_importance is not None:
        val_col = "Abs_Weight" if "Abs_Weight" in margin_importance.columns else "Importance"
        combined = combined.merge(
            margin_importance[["Feature", val_col]].rename(columns={val_col: "Margin"}),
            on="Feature",
            how="outer",
        )

    combined.to_csv(
        artifacts.output_dir / COMBINED_RANKINGS_FILE,
        index=False,
    )

    generate_shap(artifacts)