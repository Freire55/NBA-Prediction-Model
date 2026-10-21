"""
Production Prediction Pipeline for NBA Daily Matchups.

Generates calibrated win probabilities, point spreads, and tactical matchup
breakdowns for live/upcoming games and completed historical/yesterday games.
Outputs a standardized, mobile-ready JSON contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from nba_api.stats.endpoints import scoreboardv3
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("predict_today")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROFILES_CACHE_PATH = DATA_DIR / "latest_team_profiles.pkl"


# ======================================================
# Model Discovery & Loading
# ======================================================

def find_latest_model_dir(base_dir: Path = MODELS_DIR) -> Path:
    """Discovers the most recent timestamped model training directory."""
    runs = sorted(
        [d for d in base_dir.glob("run_*") if d.is_dir() and (d / "ensemble_weights.pkl").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"No valid model run directories found in {base_dir}")
    latest_run = runs[-1]
    logger.info(f"Discovered latest model run: {latest_run.name}")
    return latest_run


class LoadedModelArtifacts:
    """Encapsulates fitted estimators, scalers, feature sets, and ensemble weights."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        logger.info(f"Loading trained models and scalers from {run_dir}...")

        # Feature lists
        self.mlp_feats: List[str] = joblib.load(run_dir / "mlp_features.pkl")
        self.xgb_feats: List[str] = joblib.load(run_dir / "xgb_features.pkl")
        self.cb_feats: List[str] = joblib.load(run_dir / "catboost_features.pkl")
        self.margin_feats: List[str] = joblib.load(run_dir / "margin_features.pkl")

        # Scalers
        self.mlp_scaler = joblib.load(run_dir / "mlp_scaler.pkl")
        self.margin_scaler = joblib.load(run_dir / "margin_scaler.pkl")

        # Fitted Estimators
        self.mlp_model = joblib.load(run_dir / "mlp_model.pkl")
        self.xgb_model = joblib.load(run_dir / "xgb_model.pkl")
        self.cb_model = joblib.load(run_dir / "catboost_model.pkl")
        self.margin_model = joblib.load(run_dir / "margin_model.pkl")

        # Ensemble weights: [MLP, XGBoost, CatBoost, Pace Margin]
        self.weights: np.ndarray = joblib.load(run_dir / "ensemble_weights.pkl")

        self.all_required_features = sorted(
            list(set(self.mlp_feats) | set(self.xgb_feats) | set(self.cb_feats) | set(self.margin_feats))
        )
        logger.info(
            f"Successfully loaded models ({len(self.all_required_features)} union features). "
            f"Ensemble weights: MLP={self.weights[0]:.3f}, XGB={self.weights[1]:.3f}, "
            f"CB={self.weights[2]:.3f}, Margin={self.weights[3]:.3f}"
        )


# ======================================================
# Team Profile Extraction & Caching
# ======================================================

def extract_and_cache_team_profiles(
    parquet_path: Path = DATA_DIR / "ml_ready_matchups_players.parquet",
    cache_path: Path = PROFILES_CACHE_PATH,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Extracts each team's latest rolling state and active roster personnel embeddings.
    Caches to disk for instant sub-second lookup.
    """
    if cache_path.exists() and not force_refresh:
        logger.info(f"Loading cached team profiles from {cache_path}...")
        return joblib.load(cache_path)

    logger.info("Building team rolling profiles and latent embeddings from history...")

    if not parquet_path.exists():
        parquet_path = parquet_path.with_suffix(".csv")
        df = pd.read_csv(parquet_path)
    else:
        df = pd.read_parquet(parquet_path)

    all_teams = sorted(list(set(df["HOME_TEAM_ABBREVIATION"].unique())))
    team_raw_stats: Dict[str, Dict[str, Any]] = {}
    team_id_map: Dict[str, int] = {}

    for team in all_teams:
        h_games = df[df["HOME_TEAM_ABBREVIATION"] == team]
        a_games = df[df["AWAY_TEAM_ABBREVIATION"] == team]
        last_h = h_games.index[-1] if len(h_games) > 0 else -1
        last_a = a_games.index[-1] if len(a_games) > 0 else -1

        stats = {}
        if last_h > last_a:
            row = df.loc[last_h]
            for col in df.columns:
                if col.startswith("HOME_") and col != "HOME_WIN":
                    stats[col[5:]] = row[col]
        else:
            row = df.loc[last_a]
            for col in df.columns:
                if col.startswith("AWAY_"):
                    stats[col[5:]] = row[col]

        team_raw_stats[team] = stats
        team_id_map[team] = int(stats.get("TEAM_ID", 0))

    # Build active roster personnel embeddings
    from feature_engineering.feature_engineering_players import (
        build_active_roster_aggregations,
        compute_player_expected_impacts,
        compute_player_rolling_metrics,
        compute_player_single_game_metrics,
        load_input_datasets,
    )

    logs_df, _, embeddings_df = load_input_datasets()
    logs_df = compute_player_single_game_metrics(logs_df)
    logs_df = compute_player_rolling_metrics(logs_df)
    logs_df, embed_cols = compute_player_expected_impacts(logs_df, embeddings_df)
    roster_agg = build_active_roster_aggregations(logs_df, embed_cols)

    team_roster: Dict[str, Dict[str, Any]] = {}
    for team, tid in team_id_map.items():
        t_roster = roster_agg[roster_agg["TEAM_ID"] == tid]
        if len(t_roster) > 0:
            team_roster[team] = t_roster.sort_values("GAME_DATE").iloc[-1].to_dict()

    # Load altitude map
    alt_df = pd.read_csv(DATA_DIR / "team_altitudes.csv").set_index("TEAM_ABBREVIATION")
    alt_map = alt_df["ALTITUDE_FT"].to_dict()

    payload = {
        "team_raw_stats": team_raw_stats,
        "team_roster": team_roster,
        "alt_map": alt_map,
        "cached_at": datetime.now().isoformat(),
    }

    joblib.dump(payload, cache_path)
    logger.info(f"Cached {len(team_raw_stats)} team profiles to {cache_path}")
    return payload


# ======================================================
# Matchup Vector Synthesis
# ======================================================

def build_matchup_features(
    home_team: str,
    away_team: str,
    profiles: Dict[str, Any],
    all_required_features: List[str],
) -> Tuple[pd.DataFrame, float]:
    """
    Constructs a complete leak-free 394-feature row for a prospective or completed matchup.
    """
    team_raw = profiles["team_raw_stats"]
    team_roster = profiles["team_roster"]
    alt_map = profiles["alt_map"]

    if home_team not in team_raw:
        raise ValueError(f"Home team '{home_team}' not found in team profiles.")
    if away_team not in team_raw:
        raise ValueError(f"Away team '{away_team}' not found in team profiles.")

    h_stats = team_raw[home_team]
    a_stats = team_raw[away_team]
    h_roster = team_roster.get(home_team, {})
    a_roster = team_roster.get(away_team, {})

    row: Dict[str, float] = {}

    # 1. Base team features & Direct Deltas
    for f in all_required_features:
        if f.startswith("HOME_"):
            stat = f[5:]
            row[f] = float(h_stats.get(stat, 0.0))
        elif f.startswith("AWAY_"):
            stat = f[5:]
            row[f] = float(a_stats.get(stat, 0.0))
        elif f.startswith("DELTA_"):
            stat = f[6:]
            if stat in h_stats and stat in a_stats:
                row[f] = float(h_stats[stat]) - float(a_stats[stat])
        elif f.startswith("EMBED_"):
            parts = f.split("_")
            if "RAW" in parts:
                col = f"EMBED_{parts[3]}_{parts[4]}"
            elif "MEAN" in parts:
                col = f"EXPECTED_EMBED_{parts[2]}_WEIGHTED_MEAN"
            else:
                col = f"EXPECTED_EMBED_{parts[2]}_{parts[3]}"
            row[f] = float(h_roster.get(col, 0.0)) - float(a_roster.get(col, 0.0))

    # 2. Domain & Formulaic Deltas
    row["DELTA_ELO"] = float(h_stats.get("PRE_GAME_ELO", 1500.0)) - float(a_stats.get("PRE_GAME_ELO", 1500.0))
    row["DELTA_CIRCADIAN_FATIGUE"] = float(a_stats.get("CIRCADIAN_FATIGUE_INDEX", 0.0)) - float(
        h_stats.get("CIRCADIAN_FATIGUE_INDEX", 0.0)
    )
    row["DELTA_TRAVEL_7D"] = float(a_stats.get("TRAVEL_7D", 0.0)) - float(h_stats.get("TRAVEL_7D", 0.0))
    row["DELTA_TZ_CIRCADIAN_PENALTY"] = float(a_stats.get("TZ_CIRCADIAN_PENALTY", 0.0)) - float(
        h_stats.get("TZ_CIRCADIAN_PENALTY", 0.0)
    )
    row["DELTA_FOUR_FACTORS_NET_EFG"] = float(h_stats.get("FOUR_FACTOR_EFG_ROLLING_8", 0.5)) - float(
        a_stats.get("FOUR_FACTOR_EFG_ROLLING_8", 0.5)
    )
    row["DELTA_ROSTER_AVAILABILITY_240"] = float(h_stats.get("ROSTER_AVAILABILITY_PCT_240", 1.0)) - float(
        a_stats.get("ROSTER_AVAILABILITY_PCT_240", 1.0)
    )
    row["DELTA_MISSING_PLAYER_TALENT"] = float(a_stats.get("MISSING_PLAYER_TALENT_DEFICIT", 0.0)) - float(
        h_stats.get("MISSING_PLAYER_TALENT_DEFICIT", 0.0)
    )
    row["DELTA_MISSING_MINUTES_SHARE"] = float(a_stats.get("MISSING_MINUTES_SHARE", 0.0)) - float(
        h_stats.get("MISSING_MINUTES_SHARE", 0.0)
    )
    row["DELTA_MISSING_PLAYER_TALENT_SHARE"] = float(a_stats.get("MISSING_PLAYER_TALENT_SHARE", 0.0)) - float(
        h_stats.get("MISSING_PLAYER_TALENT_SHARE", 0.0)
    )

    # 3. Tactical Clashes
    tov_clash = (
        float(h_stats.get("DEF_TOV_RATE", 0.14)) * float(a_stats.get("FOUR_FACTOR_TOV_ROLLING_8", 0.14))
    ) - (
        float(a_stats.get("DEF_TOV_RATE", 0.14)) * float(h_stats.get("FOUR_FACTOR_TOV_ROLLING_8", 0.14))
    )
    glass_clash = (
        float(h_stats.get("FOUR_FACTOR_OREB_ROLLING_8", 0.25)) * (1.0 - float(a_stats.get("DEF_REB_RATE", 0.75)))
    ) - (
        float(a_stats.get("FOUR_FACTOR_OREB_ROLLING_8", 0.25)) * (1.0 - float(h_stats.get("DEF_REB_RATE", 0.75)))
    )
    ftr_clash = (
        float(h_stats.get("FOUR_FACTOR_FTR_ROLLING_8", 0.25)) * float(a_stats.get("DEF_FTR", 0.24))
    ) - (
        float(a_stats.get("FOUR_FACTOR_FTR_ROLLING_8", 0.25)) * float(h_stats.get("DEF_FTR", 0.24))
    )
    three_clash = (
        float(a_stats.get("OPP_3PA_RATE", 0.35)) * float(h_stats.get("FOUR_FACTOR_EFG_ROLLING_8", 0.50))
    ) - (
        float(h_stats.get("OPP_3PA_RATE", 0.35)) * float(a_stats.get("FOUR_FACTOR_EFG_ROLLING_8", 0.50))
    )
    row["DELTA_TURNOVER_PRESSURE_CLASH"] = tov_clash
    row["DELTA_GLASS_DOMINANCE"] = glass_clash
    row["DELTA_FTR_CLASH"] = ftr_clash
    row["DELTA_3PT_EXPLOITATION"] = three_clash
    row["DELTA_TACTICAL_CLASH_ADVANTAGE"] = (
        2.0 * tov_clash + 1.5 * glass_clash + 1.0 * ftr_clash + 1.0 * three_clash
    )

    # 4. Schedule, Altitude, and Tempo
    row["REST_ADVANTAGE"] = float(h_stats.get("REST_DAYS", 3)) - float(a_stats.get("REST_DAYS", 3))
    row["HOME_B2B"] = float(h_stats.get("B2B", 0))
    row["AWAY_B2B"] = float(a_stats.get("B2B", 0))
    row["SEASON_YEAR"] = float(h_stats.get("SEASON_YEAR", 2026))

    h_alt = alt_map.get(home_team, 0.0)
    a_alt = alt_map.get(away_team, 0.0)
    eff_diff = max(0.0, (h_alt - a_alt) - 1000.0)
    alt_adv = 1.0 - np.exp(-((eff_diff / 3000.0) ** 2))
    row["ALTITUDE_ADVANTAGE"] = float(alt_adv)
    row["ALTITUDE_B2B_PENALTY"] = float(alt_adv * row["AWAY_B2B"])

    expected_pace = (
        float(h_stats.get("ROLLING_PACE_8", 100.0)) + float(a_stats.get("ROLLING_PACE_8", 100.0))
    ) / 2.0
    row["MATCHUP_EXPECTED_PACE"] = expected_pace

    # Fill any remaining NaNs safely
    df_row = pd.DataFrame([row]).reindex(columns=all_required_features).fillna(0.0)
    return df_row, expected_pace


# ======================================================
# Inference & Ensemble Orchestration
# ======================================================

def predict_single_matchup(
    home_team: str,
    away_team: str,
    artifacts: LoadedModelArtifacts,
    profiles: Dict[str, Any],
) -> Dict[str, Any]:
    """Generates calibrated probability forecasts and point spreads across all submodels."""
    feat_df, expected_pace = build_matchup_features(
        home_team=home_team,
        away_team=away_team,
        profiles=profiles,
        all_required_features=artifacts.all_required_features,
    )

    # MLP
    X_mlp = artifacts.mlp_scaler.transform(feat_df[artifacts.mlp_feats])
    p_mlp = float(artifacts.mlp_model.predict_proba(X_mlp)[0, 1])

    # XGBoost
    X_xgb = feat_df[artifacts.xgb_feats]
    p_xgb = float(artifacts.xgb_model.predict_proba(X_xgb)[0, 1])

    # CatBoost
    X_cb = feat_df[artifacts.cb_feats]
    p_cb = float(artifacts.cb_model.predict_proba(X_cb)[0, 1])

    # Pace-Modulated Margin Regressor & CDF
    X_margin = artifacts.margin_scaler.transform(feat_df[artifacts.margin_feats])
    pace_vec = np.array([expected_pace], dtype=np.float32)
    p_margin = float(artifacts.margin_model.predict_proba(X_margin, pace=pace_vec)[0, 1])
    projected_margin = float(artifacts.margin_model.predict_margin(X_margin, pace=pace_vec)[0])

    # Simplex SLSQP Convex Ensemble
    w = artifacts.weights
    p_ens = float(w[0] * p_mlp + w[1] * p_xgb + w[2] * p_cb + w[3] * p_margin)

    # Confidence Tiering
    diff_from_tossup = abs(p_ens - 0.50)
    if diff_from_tossup >= 0.25:
        confidence = "VERY_HIGH"
    elif diff_from_tossup >= 0.15:
        confidence = "HIGH"
    elif diff_from_tossup >= 0.07:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    projected_winner = home_team if p_ens >= 0.50 else away_team
    winner_prob = p_ens if p_ens >= 0.50 else (1.0 - p_ens)

    # Over/Under Pace indication (League mean is ~99.5)
    if expected_pace >= 102.0:
        pace_bias = "HIGH_TEMPO_OVER"
    elif expected_pace <= 97.5:
        pace_bias = "LOW_TEMPO_UNDER"
    else:
        pace_bias = "AVERAGE_TEMPO"

    return {
        "projected_winner": projected_winner,
        "win_probability": round(winner_prob, 4),
        "confidence_tier": confidence,
        "projected_home_margin": round(projected_margin, 2),
        "expected_pace": round(expected_pace, 1),
        "pace_bias": pace_bias,
        "ensemble_home_prob": round(p_ens, 4),
        "submodel_breakdown": {
            "mlp_home_prob": round(p_mlp, 4),
            "xgboost_home_prob": round(p_xgb, 4),
            "catboost_home_prob": round(p_cb, 4),
            "pace_margin_home_prob": round(p_margin, 4),
            "margin_projected_pts": round(projected_margin, 2),
        },
    }


# ======================================================
# Scoreboard Processing & Contract Formatting
# ======================================================

def fetch_and_predict_date(
    target_date: str,
    artifacts: LoadedModelArtifacts,
    profiles: Dict[str, Any],
) -> Dict[str, Any]:
    """Fetches NBA official scoreboard and generates predictions for all scheduled games."""
    logger.info(f"Querying NBA ScoreboardV3 for {target_date}...")
    sb = scoreboardv3.ScoreboardV3(game_date=target_date)
    sb_dict = sb.get_dict().get("scoreboard", {})
    raw_games = sb_dict.get("games", [])

    logger.info(f"Retrieved {len(raw_games)} games for {target_date}.")

    predictions_list: List[Dict[str, Any]] = []
    completed_count = 0
    correct_count = 0
    brier_sum = 0.0
    abs_margin_err_sum = 0.0

    team_raw = profiles["team_raw_stats"]

    for g in raw_games:
        game_id = g.get("gameId")
        status_text = g.get("gameStatusText", "").strip()
        is_final = "final" in status_text.lower()

        h_info = g.get("homeTeam", {})
        a_info = g.get("awayTeam", {})

        h_tri = h_info.get("teamTricode")
        a_tri = a_info.get("teamTricode")
        h_score = h_info.get("score")
        a_score = a_info.get("score")

        try:
            pred = predict_single_matchup(
                home_team=h_tri,
                away_team=a_tri,
                artifacts=artifacts,
                profiles=profiles,
            )
        except Exception as e:
            logger.warning(f"Error predicting {a_tri} @ {h_tri}: {e}")
            continue

        h_elo = round(float(team_raw.get(h_tri, {}).get("PRE_GAME_ELO", 1500.0)), 1)
        a_elo = round(float(team_raw.get(a_tri, {}).get("PRE_GAME_ELO", 1500.0)), 1)

        # Actual result scoring if game is completed
        actual_result = None
        if is_final and h_score is not None and a_score is not None:
            completed_count += 1
            h_pts = int(h_score)
            a_pts = int(a_score)
            actual_margin = h_pts - a_pts
            actual_winner = h_tri if actual_margin > 0 else a_tri
            is_correct = (pred["projected_winner"] == actual_winner)

            if is_correct:
                correct_count += 1

            # Brier Score: (p - y)^2 where y=1 if home won
            y_home = 1.0 if actual_margin > 0 else 0.0
            p_home = pred["ensemble_home_prob"]
            brier_sum += (p_home - y_home) ** 2

            margin_err = abs(pred["projected_home_margin"] - actual_margin)
            abs_margin_err_sum += margin_err

            actual_result = {
                "winner": actual_winner,
                "home_score": h_pts,
                "away_score": a_pts,
                "actual_margin": actual_margin,
                "prediction_correct": is_correct,
                "margin_error_pts": round(margin_err, 2),
            }

        game_contract = {
            "game_id": game_id,
            "game_time_utc": g.get("gameTimeUTC"),
            "game_time_et": g.get("gameEt", status_text),
            "status": status_text,
            "is_final": is_final,
            "home_team": {
                "team_id": h_info.get("teamId"),
                "tricode": h_tri,
                "city": h_info.get("teamCity"),
                "name": h_info.get("teamName"),
                "score": h_score if is_final else None,
                "elo": h_elo,
                "win_probability": pred["ensemble_home_prob"],
                "projected_spread": round(-pred["projected_home_margin"], 1),
            },
            "away_team": {
                "team_id": a_info.get("teamId"),
                "tricode": a_tri,
                "city": a_info.get("teamCity"),
                "name": a_info.get("teamName"),
                "score": a_score if is_final else None,
                "elo": a_elo,
                "win_probability": round(1.0 - pred["ensemble_home_prob"], 4),
                "projected_spread": round(pred["projected_home_margin"], 1),
            },
            "prediction": {
                "projected_winner": pred["projected_winner"],
                "winner_probability": pred["win_probability"],
                "confidence_tier": pred["confidence_tier"],
                "projected_home_margin": pred["projected_home_margin"],
                "expected_pace": pred["expected_pace"],
                "pace_bias": pred["pace_bias"],
            },
            "ensemble_breakdown": pred["submodel_breakdown"],
            "actual_result": actual_result,
        }

        predictions_list.append(game_contract)

    # Summary section
    summary = {
        "total_games": len(predictions_list),
        "completed_games": completed_count,
        "accuracy": round(correct_count / completed_count, 4) if completed_count > 0 else None,
        "correct_predictions": correct_count if completed_count > 0 else None,
        "brier_score": round(brier_sum / completed_count, 4) if completed_count > 0 else None,
        "mean_margin_error": round(abs_margin_err_sum / completed_count, 2) if completed_count > 0 else None,
    }

    contract = {
        "generated_at": datetime.now().isoformat(),
        "target_date": target_date,
        "model_run": artifacts.run_dir.name,
        "summary": summary,
        "predictions": predictions_list,
    }

    return contract


# ======================================================
# Pretty Table Printing for Console / Inspection
# ======================================================

def print_predictions_table(payload: Dict[str, Any]) -> None:
    """Renders an ASCII table of matchups, projections, and evaluation results."""
    target_date = payload["target_date"]
    games = payload["predictions"]
    summary = payload["summary"]

    print("\n" + "=" * 115)
    print(f" NBA PREDICTION MODEL | TARGET DATE: {target_date} | MODEL RUN: {payload['model_run']}")
    print("=" * 115)

    if not games:
        print("  No games found for this date.\n")
        return

    header = (
        f"{'MATCHUP':<16} | {'STATUS':<11} | {'PROJ WINNER':<12} | {'CONF':<9} | "
        f"{'PROB':<7} | {'SPREAD':<7} | {'MLP':<6} | {'XGB':<6} | {'CB':<6} | {'MAR':<6} | {'ACTUAL RESULT'}"
    )
    print(header)
    print("-" * 115)

    for g in games:
        h = g["home_team"]["tricode"]
        a = g["away_team"]["tricode"]
        matchup = f"{a} @ {h}"
        status = g["status"][:11]

        pred = g["prediction"]
        winner = pred["projected_winner"]
        conf = pred["confidence_tier"]
        prob = f"{pred['winner_probability']*100:.1f}%"
        spread = f"{pred['projected_home_margin']:+.1f}"

        sub = g["ensemble_breakdown"]
        mlp_p = f"{sub['mlp_home_prob']*100:.0f}%"
        xgb_p = f"{sub['xgboost_home_prob']*100:.0f}%"
        cb_p = f"{sub['catboost_home_prob']*100:.0f}%"
        mar_p = f"{sub['pace_margin_home_prob']*100:.0f}%"

        actual_str = "-"
        if g["actual_result"]:
            act = g["actual_result"]
            mark = "CORRECT" if act["prediction_correct"] else "MISS"
            actual_str = (
                f"{act['winner']} won ({act['home_score']}-{act['away_score']}) [{mark}] "
                f"Err: {act['margin_error_pts']:+.1f} pts"
            )

        row_str = (
            f"{matchup:<16} | {status:<11} | {winner:<12} | {conf:<9} | "
            f"{prob:<7} | {spread:<7} | {mlp_p:<6} | {xgb_p:<6} | {cb_p:<6} | {mar_p:<6} | {actual_str}"
        )
        print(row_str)

    print("-" * 115)
    if summary["completed_games"] > 0:
        acc_pct = summary["accuracy"] * 100
        print(
            f" PERFORMANCE EVALUATION: {summary['correct_predictions']}/{summary['completed_games']} Correct "
            f"({acc_pct:.1f}% Accuracy) | Brier Score: {summary['brier_score']:.4f} | "
            f"Mean Margin Error: {summary['mean_margin_error']:.1f} pts"
        )
        print("=" * 115 + "\n")
    else:
        print(f" Total Upcoming Matchups: {summary['total_games']}")
        print("=" * 115 + "\n")


# ======================================================
# CLI Orchestrator
# ======================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA Production Daily Prediction Script")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format (defaults to current date)",
    )
    parser.add_argument(
        "--include-yesterday",
        action="store_true",
        default=True,
        help="Also fetch and score yesterday's completed games",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to specific model run directory (defaults to latest in models/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory to save JSON contracts",
    )
    parser.add_argument(
        "--refresh-profiles",
        action="store_true",
        default=False,
        help="Force recomputation of team profiles and embeddings cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine dates
    if args.date:
        today_date = args.date
    else:
        # Default to today's ET date (approximate with UTC-4)
        today_date = (datetime.now(timezone.utc) - timedelta(hours=4)).strftime("%Y-%m-%d")

    yesterday_date = (datetime.strptime(today_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Discover and load model artifacts
    model_dir = Path(args.model_dir) if args.model_dir else find_latest_model_dir()
    artifacts = LoadedModelArtifacts(model_dir)

    # Extract/load team profiles
    profiles = extract_and_cache_team_profiles(force_refresh=args.refresh_profiles)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process Yesterday (Completed Games)
    if args.include_yesterday:
        yesterday_payload = fetch_and_predict_date(
            target_date=yesterday_date,
            artifacts=artifacts,
            profiles=profiles,
        )
        print_predictions_table(yesterday_payload)
        yesterday_json_path = output_dir / "predictions_yesterday.json"
        with open(yesterday_json_path, "w") as f:
            json.dump(yesterday_payload, f, indent=2)
        logger.info(f"Saved yesterday's predictions & evaluation to {yesterday_json_path}")

    # Process Today (Upcoming Games)
    today_payload = fetch_and_predict_date(
        target_date=today_date,
        artifacts=artifacts,
        profiles=profiles,
    )
    print_predictions_table(today_payload)
    today_json_path = output_dir / "predictions_today.json"
    with open(today_json_path, "w") as f:
        json.dump(today_payload, f, indent=2)
    logger.info(f"Saved today's live predictions contract to {today_json_path}")


if __name__ == "__main__":
    main()
