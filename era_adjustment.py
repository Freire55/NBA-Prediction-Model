"""
Applies season-based era normalization to historical NBA statistics.

This script converts numeric box score statistics into season-relative
z-scores, allowing player and team performances to be compared fairly
across different NBA eras. Each statistic is normalized independently
within each season.

Input:
    data/raw_historical_nba.csv

Output:
    data/era_adjusted_nba.csv
"""

import logging
from pathlib import Path
import pandas as pd

# ======================================================
# Constants & Configuration
# ======================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = "raw_historical_nba.csv"
OUTPUT_FILE = "era_adjusted_nba.csv"

EXCLUDED_COLUMNS = {
    "SEASON_ID",
    "TEAM_ID",
    "GAME_ID",
    "MIN",
    "VIDEO_AVAILABLE",
}

DEFAULT_Z_SCORE = 0.0

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ======================================================
# Functions
# ======================================================

def load_raw_historical_data(filepath: Path) -> pd.DataFrame:
    """Loads raw historical team box scores from disk."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"'{filepath.name}' not found at {filepath}. Run fetch_history.py first."
        )
    return pd.read_csv(filepath)


def calculate_era_adjusted_zscores(
    df: pd.DataFrame, excluded_columns: set[str] = EXCLUDED_COLUMNS
) -> pd.DataFrame:
    """
    Computes season-relative z-scores for all numeric box score statistics.

    Each statistic is grouped by SEASON_ID, centered to mean 0, and scaled by std.
    """
    stats_to_normalize = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]

    z_scores = (
        df.groupby("SEASON_ID")[stats_to_normalize]
        .transform(lambda values: (values - values.mean()) / values.std())
        .fillna(DEFAULT_Z_SCORE)
    )

    z_scores.columns = [f"Z_{column}" for column in stats_to_normalize]
    return pd.concat([df, z_scores], axis=1)


def save_era_adjusted_data(df: pd.DataFrame, output_path: Path) -> None:
    """Saves the era-adjusted dataset to disk."""
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved era-adjusted dataset to '{output_path.name}'.")


def main() -> None:
    """Executes the era-adjustment pipeline."""
    df = load_raw_historical_data(DATA_DIR / INPUT_FILE)
    adjusted_df = calculate_era_adjusted_zscores(df)
    save_era_adjusted_data(adjusted_df, DATA_DIR / OUTPUT_FILE)


if __name__ == "__main__":
    main()