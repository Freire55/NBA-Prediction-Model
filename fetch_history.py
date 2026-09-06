"""
Fetches historical NBA regular season team game logs from the NBA Stats API.

Saves the combined multi-season game logs to data/raw_historical_nba.csv.
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


def fetch_season_game_log(
    season: str,
    max_retries: int = MAX_RETRIES,
    request_delay: float = REQUEST_DELAY,
    retry_delay: float = RETRY_DELAY,
) -> pd.DataFrame:
    """Fetches team regular season game logs for a single season with retry backoff."""
    logger.info(f"Fetching {season}...")
    for attempt in range(max_retries):
        try:
            game_log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="T",
            )
            df = game_log.get_data_frames()[0]
            time.sleep(request_delay)
            return df
        except Exception as err:
            logger.warning(f"Attempt {attempt + 1} failed for {season}: {err}")
            time.sleep(retry_delay)
    raise ConnectionError(f"Failed to fetch {season} after {max_retries} attempts.")


def fetch_all_historical_seasons(seasons: list[str]) -> pd.DataFrame:
    """Iterates through seasons and concatenates all fetched team game logs."""
    all_games = []
    for season in seasons:
        df = fetch_season_game_log(season)
        all_games.append(df)

    if not all_games:
        raise ValueError("No game logs were fetched.")
    return pd.concat(all_games, ignore_index=True)


def save_historical_data(df: pd.DataFrame, output_path: Path) -> None:
    """Saves raw historical team box scores to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Success! Saved {len(df):,} total games to '{output_path.name}'.")


def main() -> None:
    """Orchestrates historical NBA game log retrieval."""
    seasons = get_historical_seasons()
    master_df = fetch_all_historical_seasons(seasons)
    save_historical_data(master_df, DATA_DIR / "raw_historical_nba.csv")


if __name__ == "__main__":
    main()