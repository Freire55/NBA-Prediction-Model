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

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from feature_engineering_players import calculate_game_score

# ======================================================
# Configuration
# ======================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = "raw_player_game_logs.csv"
OUTPUT_FILE = "player_embeddings.csv"

# Align this exactly with `train_end` in config.py!
TRAIN_END_SEASON = "22018"
EWMA_HALFLIFE = 20
N_COMPONENTS = 8

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_minutes(minutes: pd.Series) -> pd.Series:
    """Converts MM:SS minute representations to decimal minutes."""
    time_split = minutes.astype(str).str.split(":", expand=True)
    if 1 not in time_split.columns:
        time_split[1] = 0
    return (
        pd.to_numeric(time_split[0], errors="coerce").fillna(0)
        + pd.to_numeric(time_split[1], errors="coerce").fillna(0) / 60.0
    )


def calculate_ts_perc(df: pd.DataFrame) -> pd.Series:
    """Calculates True Shooting Percentage (TS%) for each player-game."""
    numerator = df["PTS"]
    denominator = 2 * (df["FGA"] + 0.44 * df["FTA"])
    denominator = denominator.replace(0, np.nan)  
    return (numerator / denominator).fillna(0)


def calculate_efg_perc(df: pd.DataFrame) -> pd.Series:
    """Calculates Effective Field Goal Percentage (eFG%) for each player-game."""
    numerator = df["FGM"] + 0.5 * df["FG3M"]
    denominator = df["FGA"].replace(0, np.nan)
    return (numerator / denominator).fillna(0)


def calculate_tov_perc(df: pd.DataFrame) -> pd.Series:
    """Calculates Turnover Percentage (TOV%) for each player-game."""
    numerator = df["TOV"]
    denominator = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).fillna(0)


def calculate_fantasy_score(df: pd.DataFrame) -> pd.Series:
    """Calculates a simple fantasy score for each player-game."""
    return (
        df["PTS"]
        + 1.2 * df["FGM"]
        - 0.7 * df["FGA"]
        - 0.4 * (df["FTA"] - df["FTM"])
        + 0.7 * df["OREB"]
        + 0.7 * df["DREB"]
        + 1.5 * df["AST"]
        + 2 * df["STL"]
        + 2 * df["BLK"]
        - 1 * df["TOV"]
    )


def calculate_ast_to_tov_ratio(df: pd.DataFrame) -> pd.Series:
    """Calculates Assist-to-Turnover Ratio (AST/TO) for each player-game."""
    denominator = df["TOV"].replace(0, np.nan)
    return (df["AST"] / denominator).fillna(0)


def calculate_usg_proxy(df: pd.DataFrame) -> pd.Series:
    """Calculates a Usage Proxy (Offensive Load per minute) for each player-game."""
    numerator = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    denominator = df["MINUTES"].replace(0, np.nan)
    return (numerator / denominator).fillna(0)


def calculate_pie_proxy(df: pd.DataFrame) -> pd.Series:
    """Calculates a Proxy for Player Impact per minute."""
    player_stats = (
        df["PTS"] + df["FGM"] + df["FTM"] - df["FGA"] - df["FTA"] 
        + df["DREB"] + (df["OREB"] / 2) + df["AST"] + df["STL"] 
        + (df["BLK"] / 2) - df["PF"] - df["TOV"]
    )
    denominator = df["MINUTES"].replace(0, np.nan)
    return (player_stats / denominator).fillna(0)


def load_and_clean_player_logs(filepath: Path) -> pd.DataFrame:
    """Loads and chronologically sorts player game logs, parsing minutes."""
    df = pd.read_csv(filepath)
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(str)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON_ID"] = df["SEASON_ID"].astype(str)
    
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_DATE"], keep="last")
    df["MINUTES"] = parse_minutes(df["MIN"])
    return df


def compute_per_minute_productivity_vectors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Calculates 14 per-minute base stats and 8 advanced efficiency metrics."""
    valid_mins = df["MINUTES"] >= 5.0
    base_stats = [
        "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", 
        "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"
    ]
    
    stat_cols = []
    for stat in base_stats:
        col_name = f"{stat}_PER_MIN"
        df.loc[valid_mins, col_name] = df[stat] / df["MINUTES"]
        stat_cols.append(col_name)

    df["TS_PERC"] = calculate_ts_perc(df)
    df["EFG_PERC"] = calculate_efg_perc(df)
    df["TOV_PERC"] = calculate_tov_perc(df)
    df["FANTASY_SCORE"] = calculate_fantasy_score(df)
    df["GAME_SCORE"] = calculate_game_score(df)
    df["USG_PROXY"] = calculate_usg_proxy(df)
    df["AST_TO_TOV_RATIO"] = calculate_ast_to_tov_ratio(df)
    df["PIE_PROXY"] = calculate_pie_proxy(df)

    advanced_stats = [
        "TS_PERC", "EFG_PERC", "TOV_PERC", "FANTASY_SCORE", 
        "GAME_SCORE", "USG_PROXY", "PIE_PROXY", "AST_TO_TOV_RATIO"
    ]
    stat_cols.extend(advanced_stats)
    df[stat_cols] = df[stat_cols].fillna(0)
    return df, stat_cols


def build_leak_free_ewma_profiles(
    df: pd.DataFrame, stat_cols: list[str], halflife: int = EWMA_HALFLIFE
) -> tuple[pd.DataFrame, list[str]]:
    """Builds leak-free rolling EWMA profiles for each player appearance."""
    def calculate_leak_free_ewma(series):
        return series.shift(1).ewm(halflife=halflife, min_periods=1).mean()

    player_groups = df.groupby("PLAYER_ID")
    rolling_df = player_groups[stat_cols].transform(calculate_leak_free_ewma)
    rolling_df.columns = [f"ROLLING_{col}" for col in stat_cols]
    df = pd.concat([df, rolling_df], axis=1)

    rolling_cols = [f"ROLLING_{col}" for col in stat_cols]
    df = df.dropna(subset=rolling_cols).copy()
    return df, rolling_cols


def fit_chronological_pca(
    df: pd.DataFrame,
    rolling_cols: list[str],
    train_end_season: str = TRAIN_END_SEASON,
    n_components: int = N_COMPONENTS,
) -> tuple[pd.DataFrame, list[str], PCA]:
    """Fits PCA strictly on training seasons and generates latent embeddings."""
    train_mask = df["SEASON_ID"] <= train_end_season

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, rolling_cols].values)
    X_scaled = scaler.transform(df[rolling_cols].values)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled[train_mask])
    embeddings = pca.transform(X_scaled)

    embed_cols = []
    for i in range(1, n_components + 1):
        col_name = f"EMBED_{i}"
        df[col_name] = embeddings[:, i - 1]
        embed_cols.append(col_name)

    return df, embed_cols, pca


def fit_chronological_nmf(
    df: pd.DataFrame,
    rolling_cols: list[str],
    train_end_season: str = TRAIN_END_SEASON,
    n_components: int = N_COMPONENTS,
) -> tuple[pd.DataFrame, list[str], NMF, list[str]]:
    """Fits NMF strictly on non-negative metrics from training seasons and generates latent embeddings."""
    non_negative_stats = [
        col for col in rolling_cols
        if not any(bad in col for bad in ["GAME_SCORE", "FANTASY_SCORE", "PIE_PROXY"])
    ]
    train_mask = df["SEASON_ID"] <= train_end_season

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(df.loc[train_mask, non_negative_stats].values)

    X_train = np.clip(scaler.transform(df.loc[train_mask, non_negative_stats].values), 0.0, None)
    X_all = np.clip(scaler.transform(df[non_negative_stats].values), 0.0, None)

    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=42,
        max_iter=1000,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    nmf.fit(X_train)
    embeddings = nmf.transform(X_all)

    embed_cols = []
    for i in range(1, n_components + 1):
        col_name = f"EMBED_{i}"
        df[col_name] = embeddings[:, i - 1]
        embed_cols.append(col_name)

    return df, embed_cols, nmf, non_negative_stats


def fit_chronological_hybrid(
    df: pd.DataFrame,
    rolling_cols: list[str],
    train_end_season: str = TRAIN_END_SEASON,
    n_pca: int = 4,
    n_nmf: int = 4,
) -> tuple[pd.DataFrame, list[str], PCA, NMF, list[str]]:
    """Fits PCA (top n_pca dims) and NMF (n_nmf archetypes) strictly on training seasons."""
    train_mask = df["SEASON_ID"] <= train_end_season

    # 1. PCA on standardized features
    scaler_pca = StandardScaler()
    scaler_pca.fit(df.loc[train_mask, rolling_cols].values)
    X_pca_train = scaler_pca.transform(df.loc[train_mask, rolling_cols].values)
    X_pca_all = scaler_pca.transform(df[rolling_cols].values)

    pca = PCA(n_components=n_pca, random_state=42)
    pca.fit(X_pca_train)
    embeddings_pca = pca.transform(X_pca_all)

    # 2. NMF on non-negative features scaled to [0, 1]
    non_negative_stats = [
        col for col in rolling_cols
        if not any(bad in col for bad in ["GAME_SCORE", "FANTASY_SCORE", "PIE_PROXY"])
    ]
    scaler_nmf = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler_nmf.fit(df.loc[train_mask, non_negative_stats].values)
    X_nmf_train = np.clip(scaler_nmf.transform(df.loc[train_mask, non_negative_stats].values), 0.0, None)
    X_nmf_all = np.clip(scaler_nmf.transform(df[non_negative_stats].values), 0.0, None)

    nmf = NMF(
        n_components=n_nmf,
        init="nndsvda",
        random_state=42,
        max_iter=1000,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    nmf.fit(X_nmf_train)
    embeddings_nmf = nmf.transform(X_nmf_all)

    combined_embeddings = np.hstack([embeddings_pca, embeddings_nmf])
    total_components = n_pca + n_nmf
    embed_cols = []
    for i in range(1, total_components + 1):
        col_name = f"EMBED_{i}"
        df[col_name] = combined_embeddings[:, i - 1]
        embed_cols.append(col_name)

    return df, embed_cols, pca, nmf, non_negative_stats


def save_embeddings(df: pd.DataFrame, embed_cols: list[str], output_path: Path) -> None:
    """Saves player embeddings to CSV."""
    output_df = df[["PLAYER_ID", "GAME_DATE"] + embed_cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Success! Generated {len(embed_cols)}D embeddings for {len(output_df):,} player appearances.")


def log_pca_variance_summary(pca: PCA, n_components: int) -> None:
    """Logs detailed PCA explained variance breakdown."""
    cum_var = 0.0
    logger.info("=" * 65)
    logger.info(f"PCA EXPLAINED VARIANCE BREAKDOWN ({n_components} DIMENSIONS):")
    logger.info("=" * 65)
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        cum_var += var
        logger.info(f"  Dimension {i} (EMBED_{i}): {var * 100:6.2f}% of variance  |  Cumulative: {cum_var * 100:6.2f}%")
    logger.info("-" * 65)
    logger.info(f"  Total Variance ({n_components} dimensions): {sum(pca.explained_variance_ratio_[:n_components]) * 100:.2f}%")
    logger.info("=" * 65)


def log_nmf_loadings(nmf: NMF, feature_names: list[str], offset: int = 0) -> None:
    """Logs top feature loadings for each NMF factor."""
    clean_names = [col.replace("ROLLING_", "") for col in feature_names]
    logger.info("=" * 65)
    logger.info(f"NMF FACTOR LOADINGS ({nmf.n_components} ARCHETYPES):")
    logger.info("=" * 65)
    for i in range(nmf.n_components):
        top_indices = np.argsort(nmf.components_[i])[::-1][:4]
        loadings_str = ", ".join([f"{clean_names[idx]} ({nmf.components_[i, idx]:.3f})" for idx in top_indices])
        logger.info(f"  Archetype {i+1} (EMBED_{offset + i + 1}): {loadings_str}")
    logger.info("=" * 65)


def main() -> None:
    """Executes the player embedding pipeline with selected method."""
    parser = argparse.ArgumentParser(description="Generate player embeddings using PCA, NMF, or Hybrid.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["pca", "nmf", "hybrid"],
        default="hybrid",
        help="Embedding dimensionality reduction method (default: hybrid)",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=N_COMPONENTS,
        help="Number of embedding components (default: 8)",
    )
    args = parser.parse_args()

    logger.info("Loading raw player logs for embedding generation...")
    df = load_and_clean_player_logs(DATA_DIR / INPUT_FILE)

    logger.info("Calculating per-minute productivity vectors...")
    df, stat_cols = compute_per_minute_productivity_vectors(df)

    logger.info("Building historical profiles (Exponential Moving Average)...")
    df, rolling_cols = build_leak_free_ewma_profiles(df, stat_cols)

    if args.method == "pca":
        logger.info(f"Fitting PCA ({args.components}D) strictly on seasons <= {TRAIN_END_SEASON}...")
        df, embed_cols, pca = fit_chronological_pca(df, rolling_cols, n_components=args.components)
        save_embeddings(df, embed_cols, DATA_DIR / OUTPUT_FILE)
        log_pca_variance_summary(pca, args.components)
    elif args.method == "nmf":
        logger.info(f"Fitting NMF ({args.components}D) strictly on seasons <= {TRAIN_END_SEASON}...")
        df, embed_cols, nmf, non_neg_stats = fit_chronological_nmf(df, rolling_cols, n_components=args.components)
        save_embeddings(df, embed_cols, DATA_DIR / OUTPUT_FILE)
        log_nmf_loadings(nmf, non_neg_stats)
    elif args.method == "hybrid":
        n_pca = args.components // 2
        n_nmf = args.components - n_pca
        logger.info(f"Fitting Hybrid PCA-{n_pca} + NMF-{n_nmf} strictly on seasons <= {TRAIN_END_SEASON}...")
        df, embed_cols, pca, nmf, non_neg_stats = fit_chronological_hybrid(
            df, rolling_cols, n_pca=n_pca, n_nmf=n_nmf
        )
        save_embeddings(df, embed_cols, DATA_DIR / OUTPUT_FILE)
        log_pca_variance_summary(pca, n_pca)
        log_nmf_loadings(nmf, non_neg_stats, offset=n_pca)


if __name__ == "__main__":
    main()