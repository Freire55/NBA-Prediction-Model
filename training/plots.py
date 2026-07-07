"""
Visualization utilities for the NBA prediction training pipeline.

This module centralizes all plotting routines used during model
evaluation and explainability. It generates standardized figures
for feature importance, ROC curves, calibration curves, and
confusion matrices, saving each plot to the experiment output
directory.

Functions:
    plot_horizontal_bar() : Feature importance visualization.
    plot_roc_curve()      : Receiver Operating Characteristic curve.
    plot_calibration()    : Probability calibration curve.
    plot_confusion()      : Confusion matrix visualization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
)

from training.utils import save_plot

# ======================================================
# Plot Configuration
# ======================================================

DEFAULT_BAR_COLOR = "royalblue"
ROC_COLOR = "darkorange"
BASELINE_COLOR = "gray"

DEFAULT_FIGSIZE = (10, 8)
STANDARD_FIGSIZE = (8, 6)
SQUARE_FIGSIZE = (8, 8)

DEFAULT_TOP_FEATURES = 15


# ======================================================
# Plotting Functions
# ======================================================

def plot_horizontal_bar(
    df_imp: pd.DataFrame,
    title: str,
    filename: str,
    sort_col: str,
    output_dir: Path,
    xlabel: str = "Importance",
    top_n: int = DEFAULT_TOP_FEATURES,
) -> None:
    """
    Creates a standardized horizontal bar chart for feature rankings.

    Args:
        df_imp:
            DataFrame containing feature names and importance values.
        title:
            Plot title.
        filename:
            Output image filename.
        sort_col:
            Column used to determine feature ranking.
        output_dir:
            Directory where the figure is saved.
        xlabel:
            Label for the x-axis.
        top_n:
            Number of top-ranked features to display.
    """
    plt.figure(figsize=DEFAULT_FIGSIZE)

    top_df = (
        df_imp.head(top_n)
        .sort_values(by=sort_col, ascending=True)
    )

    plt.barh(
        top_df["Feature"],
        top_df[sort_col],
        color=DEFAULT_BAR_COLOR,
    )

    plt.title(title)
    plt.xlabel(xlabel)

    save_plot(output_dir, filename)


def plot_roc_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    auc_score: float,
    output_dir: Path,
) -> None:
    """
    Generates the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true:
            Ground-truth labels.
        probabilities:
            Predicted probabilities for the positive class.
        auc_score:
            Precomputed ROC-AUC score.
        output_dir:
            Directory where the figure is saved.
    """
    fpr, tpr, _ = roc_curve(y_true, probabilities)

    plt.figure(figsize=STANDARD_FIGSIZE)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        color=ROC_COLOR,
        label=f"Ensemble ROC (AUC = {auc_score:.2f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=BASELINE_COLOR,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"Weighted Ensemble ROC Curve\n"
        f"Test Seasons: 2021+\n"
        f"AUC = {auc_score:.3f}"
    )
    plt.legend(loc="lower right")

    save_plot(output_dir, "03_roc_curve.png")


def plot_calibration(
    y_true: pd.Series,
    probabilities: np.ndarray,
    brier_score: float,
    n_bins: int,
    output_dir: Path,
) -> None:
    """
    Generates the probability calibration curve.

    Args:
        y_true:
            Ground-truth labels.
        probabilities:
            Predicted probabilities.
        brier_score:
            Computed Brier score.
        n_bins:
            Number of calibration bins.
        output_dir:
            Directory where the figure is saved.
    """
    prob_true, prob_pred = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
    )

    plt.figure(figsize=SQUARE_FIGSIZE)

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=1,
        label="Standard Ensemble",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=BASELINE_COLOR,
        label="Perfectly Calibrated",
    )

    plt.title(
        f"Ensemble Calibration Curve\n"
        f"Test Seasons: 2021+\n"
        f"Brier Score = {brier_score:.3f}"
    )

    plt.xlabel("Predicted Probability")
    plt.ylabel("True Win Rate")
    plt.legend()

    save_plot(output_dir, "02_calibration_curve.png")


def plot_confusion(
    y_true: pd.Series,
    predictions: np.ndarray,
    accuracy: float,
    output_dir: Path,
) -> None:
    """
    Generates the confusion matrix for the ensemble classifier.

    Args:
        y_true:
            Ground-truth labels.
        predictions:
            Predicted class labels.
        accuracy:
            Classification accuracy.
        output_dir:
            Directory where the figure is saved.
    """
    cm = confusion_matrix(y_true, predictions)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Away Win", "Home Win"],
    ).plot(cmap=plt.cm.Blues)

    plt.title(
        f"Weighted Ensemble Confusion Matrix\n"
        f"Test Seasons: 2021+\n"
        f"Accuracy = {accuracy * 100:.1f}%"
    )

    save_plot(output_dir, "04_confusion_matrix.png")