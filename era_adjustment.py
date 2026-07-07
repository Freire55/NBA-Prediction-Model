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


# ======================================================
# Load Data
# ======================================================

try:
    df = pd.read_csv(DATA_DIR / INPUT_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"'{INPUT_FILE}' not found. Run fetch_history.py first."
    )

# ======================================================
# Era Adjustment
# ======================================================

# Identify numeric statistics that should be normalized.
stats_to_normalize = [
    column
    for column in df.select_dtypes(include="number").columns
    if column not in EXCLUDED_COLUMNS
]

# Compute season-relative z-scores for every statistic.
z_scores = (
    df.groupby("SEASON_ID")[stats_to_normalize]
    .transform(lambda values: (values - values.mean()) / values.std())
    .fillna(DEFAULT_Z_SCORE)
)

# Rename generated columns.
z_scores.columns = [f"Z_{column}" for column in stats_to_normalize]

# Append normalized statistics.
df = pd.concat([df, z_scores], axis=1)

# ======================================================
# Save Output
# ======================================================

output_path = DATA_DIR / OUTPUT_FILE

df.to_csv(output_path, index=False)

print(
    f"Successfully saved era-adjusted dataset to '{OUTPUT_FILE}'."
)