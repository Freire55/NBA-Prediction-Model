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
import numpy as np
import pandas as pd

# ======================================================
# Constants & Configuration
# ======================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
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
    Computes leak-free era-adjusted z-scores for all numeric box score statistics.

    To eliminate full-season lookahead bias, each season S_i is standardized using
    the empirical mean and standard deviation from the prior chronological season S_{i-1}.
    For the initial baseline season, its own mean and standard deviation are used.
    """
    stats_to_normalize = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]

    seasons = sorted(df["SEASON_ID"].unique())
    season_stats = df.groupby("SEASON_ID")[stats_to_normalize].agg(["mean", "std"])

    z_frames = []
    for i, season in enumerate(seasons):
        season_df = df[df["SEASON_ID"] == season][stats_to_normalize]
        ref_season = seasons[i - 1] if i > 0 else seasons[0]

        means = season_stats.loc[ref_season, (slice(None), "mean")].values
        stds = season_stats.loc[ref_season, (slice(None), "std")].values
        stds = np.where(np.isnan(stds) | (stds <= 0), 1.0, stds)

        z_vals = (season_df.values - means) / stds
        z_frame = pd.DataFrame(
            z_vals,
            index=season_df.index,
            columns=[f"Z_{col}" for col in stats_to_normalize],
        )
        z_frames.append(z_frame)

    z_scores = pd.concat(z_frames).sort_index().fillna(DEFAULT_Z_SCORE)
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