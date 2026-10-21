"""
Data extraction package for historical NBA team and player box scores.
"""

from data_extraction.fetch_history import fetch_season_game_log, get_historical_seasons
from data_extraction.fetch_player_game_logs import fetch_player_season_logs

__all__ = [
    "fetch_season_game_log",
    "get_historical_seasons",
    "fetch_player_season_logs",
]
