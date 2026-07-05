import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Figure out the current NBA season
current_year = datetime.now().year
current_month = datetime.now().month
end_year = current_year if current_month >= 10 else current_year - 1

# Build a list of season strings AND their internal NBA ID (e.g., '22000')
seasons = []
for year in range(2000, end_year + 1):
    next_year = str(year + 1)[-2:]
    if next_year == "00":
        next_year = "00"
    seasons.append((f"{year}-{next_year}", f"2{year}"))

all_advanced = []

# Loop through each season and grab the advanced team ratings (Offensive/Defensive Rating, etc)
for season_str, season_id in seasons:
    print(f"Fetching advanced data for the {season_str} season...")
    try:
        advanced_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season_str,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Advanced' 
        )
        df = advanced_stats.get_data_frames()[0]
        # Manually attach the season ID so we can match it back to our box scores later
        df['SEASON_ID'] = season_id
        all_advanced.append(df)
        time.sleep(2)
    except Exception as e:
        print(f"An error occurred: {e}")

# Combine and save to CSV
if all_advanced:
    advanced_df = pd.concat(all_advanced, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "advanced_team_stats.csv"
    advanced_df.to_csv(output_file, index=False)
    print(f"Success! Saved advanced stats to '{output_file.name}'.")