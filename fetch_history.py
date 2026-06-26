import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time

print("Initializing NBA historical data fetch...")

# 1. Create a list of seasons from 2000 to 2023
seasons = []
for year in range(2000, 2024):
    # The API requires the 'YYYY-YY' format (e.g., '2000-01', '2009-10')
    next_year = str(year + 1)[-2:]
    
    # Handle the transition from 1999-2000 to 2000-01 formatting smoothly
    if next_year == "00":
        next_year = "00"
        
    seasons.append(f"{year}-{next_year}")

all_games = []

# 2. Loop through the seasons and fetch data
for season in seasons:
    print(f"Fetching data for the {season} season...")
    try:
        game_log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
        df = game_log.get_data_frames()[0]
        all_games.append(df)
        
        # CRITICAL: Pause for 2 seconds to avoid being IP blocked by the NBA
        time.sleep(2)
    except Exception as e:
        print(f"Error fetching {season}: {e}")

# 3. Combine all individual season tables into one massive table
if all_games:
    print("Concatenating all seasons into one master dataset...")
    # pandas.concat merges our list of 24 dataframes into one continuous table
    master_df = pd.concat(all_games, ignore_index=True)
    
    # 4. Save to your local computer as a CSV file
    master_df.to_csv("raw_historical_nba.csv", index=False)
    print(f"Success! Saved {len(master_df)} total games to 'raw_historical_nba.csv'.")
else:
    print("Failed to fetch any data.")