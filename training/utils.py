"""
Utility functions for the NBA prediction training pipeline.

This module provides reusable utilities shared across the training
package, including logging setup, artifact serialization, plot
saving, and a context manager for timing pipeline stages.

Functions:
    setup_logger()
    save_json()
    save_joblib()
    save_plot()

Classes:
    PipelineStage
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

from training.config import NumpyEncoder

# ======================================================
# Logging Configuration
# ======================================================

TRAINING_LOG_FILE = "training.log"

CONSOLE_LOG_FORMAT = "%(message)s"

FILE_LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ======================================================
# Serialization Utilities
# ======================================================

def setup_logger(output_dir: Path) -> logging.Logger:
    """
    Configures the root logger to write both to the console and to
    a training log file.

    Args:
        output_dir:
            Directory where the log file is written.

    Returns:
        Configured root logger.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if setup_logger() is called twice.
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_LOG_FORMAT)
    )

    file_handler = logging.FileHandler(
        output_dir / TRAINING_LOG_FILE
    )
    file_handler.setFormatter(
        logging.Formatter(
            FILE_LOG_FORMAT,
            datefmt=FILE_DATE_FORMAT,
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


def save_json(data: dict, filepath: Path) -> None:
    """Saves a dictionary as a formatted JSON file."""
    with open(filepath, "w") as file:
        json.dump(data, file, cls=NumpyEncoder, indent=4)


def save_joblib(obj: Any, filepath: Path) -> None:
    """
    Serializes an object using Joblib.

    Args:
        obj:
            Object to serialize.
        filepath:
            Destination file.
    """
    joblib.dump(obj, filepath)


def save_plot(output_dir: Path, filename: str) -> None:
    """
    Saves the current matplotlib figure and closes it.

    Args:
        output_dir:
            Directory where the figure is saved.
        filename:
            Output image filename.
    """
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


# ======================================================
# Model Utilities
# ======================================================

def unwrap_base_estimator(model: Any) -> Any:
    """
    Returns the underlying fitted estimator from a CalibratedClassifierCV,
    or the model itself if it is not a calibrated wrapper.

    CalibratedClassifierCV (fit with a cross-validation splitter rather
    than cv="prefit") does not expose attributes like
    `feature_importances_` directly, and tools such as SHAP's
    TreeExplainer need a concrete tree model rather than the wrapper.
    This helper reaches into the first fitted fold estimator so that
    feature-importance and SHAP code can keep working unchanged
    regardless of whether calibration is applied upstream.

    Note: when the wrapper was fit with an internal cross-validation
    splitter, each fold has its own fitted estimator; this returns the
    first one as a representative model for interpretability purposes.
    It is not meant to reconstruct a single "canonical" final model.

    Args:
        model:
            A fitted estimator, potentially wrapped in
            CalibratedClassifierCV.

    Returns:
        The underlying fitted estimator suitable for introspection.
    """
    if isinstance(model, CalibratedClassifierCV):
        calibrated_fold = model.calibrated_classifiers_[0]
        return getattr(
            calibrated_fold,
            "estimator",
            getattr(calibrated_fold, "base_estimator", None),
        )
    return model


# ======================================================
# Pipeline Timing
# ======================================================

class PipelineStage:
    """
    Context manager that logs the execution time of a pipeline stage.

    Example:
        with PipelineStage(1, 6, "Training"):
            ...
    """

    def __init__(
        self,
        step_num: int,
        total_steps: int,
        title: str,
    ) -> None:
        self.step_num = step_num
        self.total_steps = total_steps
        self.title = title
        self.logger = logging.getLogger(__name__)
        self.start_time = 0.0

    def __enter__(self) -> "PipelineStage":
        """Starts timing and logs the stage header."""
        self.logger.info(
            f"[{self.step_num}/{self.total_steps}] {self.title}"
        )
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Logs the elapsed execution time."""
        elapsed = time.perf_counter() - self.start_time

        minutes, seconds = divmod(int(elapsed), 60)
        time_str = (
            f"{minutes}m {seconds}s"
            if minutes > 0
            else f"{seconds}s"
        )

        if exc_type is None:
            self.logger.info(
                f"  ✔ completed in {time_str}\n"
            )
        else:
            self.logger.error(
                f"  ✖ failed after {time_str}\n"
            )