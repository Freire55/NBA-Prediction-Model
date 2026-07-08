"""
Generates ML-based player embeddings using Dimensionality Reduction (PCA).

This script reads the raw player game logs, calculates per-minute advanced
statistics, smooths them using an exponentially weighted moving average (EWMA),
and compresses them into a 2-dimensional latent space (Embeddings).

To prevent target leakage, the PCA basis and Scaler are fit strictly on 
historical training seasons and applied forward to validation/test seasons.

Output:
    data/player_embeddings.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ======================================================
# Configuration
# ======================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = "raw_player_game_logs.csv"
OUTPUT_FILE = "player_embeddings.csv"

# Align this exactly with `train_end` in config.py!
TRAIN_END_SEASON = "22018" 

EWMA_HALFLIFE = 20 

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def parse_minutes(minutes: pd.Series) -> pd.Series:
    time_split = minutes.astype(str).str.split(":", expand=True)
    if 1 not in time_split.columns:
        time_split[1] = 0
    return (
        pd.to_numeric(time_split[0], errors="coerce").fillna(0)
        + pd.to_numeric(time_split[1], errors="coerce").fillna(0) / 60.0
    )

def main():
    logger.info("Loading raw player logs for embedding generation...")
    df = pd.read_csv(DATA_DIR / INPUT_FILE)

    df["PLAYER_ID"] = df["PLAYER_ID"].astype(str)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON_ID"] = df["SEASON_ID"].astype(str)
    
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_DATE"], keep="last")

    df["MINUTES"] = parse_minutes(df["MIN"])
    valid_mins = df["MINUTES"] >= 1.0

    # 1. Expand the feature space (14 distinct per-minute metrics)
    logger.info("Calculating per-minute productivity vectors...")
    
    base_stats = [
        "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", 
        "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"
    ]
    
    stat_cols = []
    for stat in base_stats:
        col_name = f"{stat}_PER_MIN"
        df.loc[valid_mins, col_name] = df[stat] / df["MINUTES"]
        stat_cols.append(col_name)

    df[stat_cols] = df[stat_cols].fillna(0)

    # 2. Create historical, leak-free rolling profiles
    logger.info("Building historical profiles (Exponential Moving Average)...")
    
    def calculate_leak_free_ewma(series):
        return series.shift(1).ewm(halflife=EWMA_HALFLIFE, min_periods=1).mean()

    player_groups = df.groupby("PLAYER_ID")
    for col in stat_cols:
        df[f"ROLLING_{col}"] = player_groups[col].transform(calculate_leak_free_ewma)

    rolling_cols = [f"ROLLING_{col}" for col in stat_cols]
    df = df.dropna(subset=rolling_cols).copy()

    # 3. Fit PCA chronologically to prevent future leakage
    logger.info(f"Training PCA basis strictly on seasons <= {TRAIN_END_SEASON}...")
    
    # Isolate training era to fit the transformations
    train_mask = df["SEASON_ID"] <= TRAIN_END_SEASON
    
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, rolling_cols].values)
    X_scaled = scaler.transform(df[rolling_cols].values)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled[train_mask])
    
    embeddings = pca.transform(X_scaled)

    # 4. Assign agnostic embedding names
    df["EMBED_1"] = embeddings[:, 0]
    df["EMBED_2"] = embeddings[:, 1]

    output_df = df[["PLAYER_ID", "GAME_DATE", "EMBED_1", "EMBED_2"]]
    
    output_path = DATA_DIR / OUTPUT_FILE
    output_df.to_csv(output_path, index=False)
    
    variance_explained = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"Success! Generated embeddings for {len(output_df):,} player appearances.")
    logger.info(f"The 2 Principal Components capture {variance_explained:.1f}% of historical playstyle variance.")

if __name__ == "__main__":
    main()