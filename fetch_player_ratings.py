import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Calculate the seasons we need to fetch, just like our team scripts
current_year = datetime.now().year
current_month = datetime.now().month
end_year = current_year if current_month >= 10 else current_year - 1

seasons = []
for year in range(2000, end_year + 1):
    next_year = str(year + 1)[-2:]
    if next_year == "00":
        next_year = "00"
    seasons.append((f"{year}-{next_year}", f"2{year}"))

all_player_adv = []

# Fetch the season-long advanced ratings for every player in the league
for season_str, season_id in seasons:
    print(f"Fetching advanced player ratings for {season_str}...")
    try:
        # measure_type_detailed_defense='Advanced' gives us PIE, Net Rating, True Shooting, etc.
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Advanced'
        )
        df = player_stats.get_data_frames()[0]
        df['SEASON_ID'] = season_id
        all_player_adv.append(df)
        time.sleep(2) # Be polite to the API
    except Exception as e:
        print(f"Error fetching {season_str}: {e}")

if all_player_adv:
    player_adv_df = pd.concat(all_player_adv, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "advanced_player_ratings.csv"
    player_adv_df.to_csv(output_file, index=False)
    print(f"Success! Saved {len(player_adv_df)} player seasons to '{output_file.name}'.")
else:
    print("Failed to fetch player ratings.")