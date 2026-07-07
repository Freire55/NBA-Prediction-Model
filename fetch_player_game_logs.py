import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

current_year = datetime.now().year
current_month = datetime.now().month
end_year = current_year if current_month >= 10 else current_year - 1

seasons = []
for year in range(2000, end_year + 1):
    next_year = str(year + 1)[-2:]
    if next_year == "00":
        next_year = "00"
    seasons.append(f"{year}-{next_year}")

all_player_logs = []

# Download player logs
max_retries = 3
for season in seasons:
    print(f"Fetching {season}...")
    
    for attempt in range(max_retries):
        try:
            game_log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
            df = game_log.get_data_frames()[0]
            all_player_logs.append(df)
            time.sleep(2)
            break 
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {season}: {e}")
            time.sleep(5) 
    else:
        raise ConnectionError(f"Failed to fetch {season} after {max_retries} attempts.")

if all_player_logs:
    master_logs_df = pd.concat(all_player_logs, ignore_index=True)
    output_file = DATA_DIR / "raw_player_game_logs.csv"
    master_logs_df.to_csv(output_file, index=False)
    print(f"Success! Saved {len(master_logs_df)} individual game performances to 'raw_player_game_logs.csv'.")
else:
    print("Failed to fetch player game logs.")