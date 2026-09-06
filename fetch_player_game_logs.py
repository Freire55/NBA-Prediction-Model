"""
Fetches historical NBA regular season player game logs from the NBA Stats API.

Saves the combined multi-season player logs to data/raw_player_game_logs.csv.
"""

from datetime import datetime
import logging
from pathlib import Path
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

DATA_DIR = Path(__file__).resolve().parent / "data"
START_YEAR = 2000
MAX_RETRIES = 3
REQUEST_DELAY = 2.0
RETRY_DELAY = 5.0

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_historical_seasons(start_year: int = START_YEAR) -> list[str]:
    """Generates the list of NBA season strings (e.g., '2000-01') up to the current season."""
    current_year = datetime.now().year
    current_month = datetime.now().month
    end_year = current_year if current_month >= 10 else current_year - 1

    seasons = []
    for year in range(start_year, end_year + 1):
        next_year = str(year + 1)[-2:]
        seasons.append(f"{year}-{next_year}")
    return seasons


def fetch_player_season_logs(
    season: str,
    max_retries: int = MAX_RETRIES,
    request_delay: float = REQUEST_DELAY,
    retry_delay: float = RETRY_DELAY,
) -> pd.DataFrame:
    """Fetches individual player game logs for a single season with retry backoff."""
    logger.info(f"Fetching player logs for {season}...")
    for attempt in range(max_retries):
        try:
            game_log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
            )
            df = game_log.get_data_frames()[0]
            time.sleep(request_delay)
            return df
        except Exception as err:
            logger.warning(f"Attempt {attempt + 1} failed for {season}: {err}")
            time.sleep(retry_delay)
    raise ConnectionError(f"Failed to fetch player logs for {season} after {max_retries} attempts.")


def fetch_all_player_game_logs(seasons: list[str]) -> pd.DataFrame:
    """Iterates through seasons and concatenates all fetched player game logs."""
    all_logs = []
    for season in seasons:
        df = fetch_player_season_logs(season)
        all_logs.append(df)

    if not all_logs:
        raise ValueError("No player game logs were fetched.")
    return pd.concat(all_logs, ignore_index=True)


def save_player_game_logs(df: pd.DataFrame, output_path: Path) -> None:
    """Saves raw individual player game logs to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Success! Saved {len(df):,} individual player performances to '{output_path.name}'.")


def main() -> None:
    """Orchestrates historical NBA player game log retrieval."""
    seasons = get_historical_seasons()
    master_logs_df = fetch_all_player_game_logs(seasons)
    save_player_game_logs(master_logs_df, DATA_DIR / "raw_player_game_logs.csv")


if __name__ == "__main__":
    main()