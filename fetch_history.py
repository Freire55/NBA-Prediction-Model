import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Figure out the current NBA season based on today's date
current_year = datetime.now().year
current_month = datetime.now().month
end_year = current_year if current_month >= 10 else current_year - 1

# Build a list of season strings like '2000-01', '2001-02'
seasons = []
for year in range(2000, end_year + 1):
    next_year = str(year + 1)[-2:]
    if next_year == "00":
        next_year = "00"
    seasons.append(f"{year}-{next_year}")

all_games = []

# Loop through each season and download the basic box scores
for season in seasons:
    print(f"Fetching data for the {season} season...")
    try:
        game_log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
        df = game_log.get_data_frames()[0]
        all_games.append(df)
        # Sleep for 2 seconds so we don't spam the NBA API and get temporarily banned
        time.sleep(2)
    except Exception as e:
        print(f"Error fetching {season}: {e}")

# Combine all the individual seasons into one massive spreadsheet and save it
if all_games:
    master_df = pd.concat(all_games, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "raw_historical_nba.csv"
    master_df.to_csv(output_file, index=False)
    print(f"Success! Saved {len(master_df)} total games to '{output_file.name}'.")
else:
    print("Failed to fetch any data.")