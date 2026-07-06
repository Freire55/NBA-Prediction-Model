import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time
from datetime import datetime
from pathlib import Path

# Set up the data folder
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Find the current season
current_year = datetime.now().year
current_month = datetime.now().month
# Before October, use the prior season
end_year = current_year if current_month >= 10 else current_year - 1

# Build the season list
seasons = []
for year in range(2000, end_year + 1):
    next_year = str(year + 1)[-2:]
    if next_year == "00":
        next_year = "00"
    seasons.append(f"{year}-{next_year}")

all_games = []

# Download each season
for season in seasons:
    print(f"Fetching team data for the {season} season...")
    try:
        # Keep it to the regular season
        game_log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
        df = game_log.get_data_frames()[0]
        all_games.append(df)
        
        # Pause between calls
        time.sleep(2)
    except Exception as e:
        print(f"Error fetching {season}: {e}")

# Save the combined file
if all_games:
    master_df = pd.concat(all_games, ignore_index=True)
    output_file = DATA_DIR / "raw_historical_nba.csv"
    master_df.to_csv(output_file, index=False)
    print(f"Success! Saved {len(master_df)} total games to 'raw_historical_nba.csv'.")
else:
    print("Failed to fetch any data.")