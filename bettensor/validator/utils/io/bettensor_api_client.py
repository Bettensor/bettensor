# File: bettensor/utils/bettensor_api.py

import requests
from datetime import datetime, timedelta, timezone
import bittensor as bt
import json
import os
from ..database.database_manager import DatabaseManager
from .base_api_client import BaseAPIClient


class BettensorAPIClient(BaseAPIClient):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()

        self.base_url = "https://dev-bettensor-api.azurewebsites.net/"
        self.db_manager = db_manager
        self.games = []

    def fetch_all_game_data(self):
        """
        fetch and update game data from bettensor API. overridden from BaseAPIClient

        """

        # Get the last update date from the database
        last_update_date = self.load_last_update_date()

        # Fetch the games from the API
        games = self.get_games(last_update_date)

        # Update and save the last_update_date on successful response
        new_last_update = datetime.now(timezone.utc)
        self.save_last_update_date(new_last_update)

        return games

    def get_games(self, last_update_date=None, items_per_page=100):
        if last_update_date is None:
            last_update_date = self.last_update_date

        all_games = []
        page_index = 0
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=15)
        ).isoformat()  # 15 days ago

        while True:
            games = self._get_games_page(
                start_date, items_per_page, page_index, last_update_date
            )
            if not games:
                break
            all_games.extend(games)
            page_index += 1
        return all_games

    def get_games_page(self, start_date, items_per_page, page_index, last_update_date):
        params = {
            "PageIndex": page_index,
            "ItemsPerPage": items_per_page,
            "SortOrder": "StartDate",
            "LastUpdateDate": last_update_date.isoformat(),
            "StartDate": start_date,
            "LeagueFilter": "true",
        }

        try:
            data = self.get_data("Games/TeamGames/Search", params)
            return data
        except Exception as e:
            bt.logging.error(f"Error fetching games from API: {e}")
            return None

    def transform_game_data(self, game):
        # Existing implementation
        return {
            "home": game.get("team_a"),
            "away": game.get("team_b"),
            "game_id": game.get("external_id"),
            "date": game.get("date"),
            "odds": {
                "average_home_odds": game.get("team_a_odds"),
                "average_away_odds": game.get("team_b_odds"),
                "average_tie_odds": game.get("tie_odds"),
            },
            "sport": game.get("sport").lower(),
            "league": game.get("league"),
        }

    def get_data(self, endpoint, params):
        url = self.base_url + endpoint
        response = requests.get(url, params=params)
        return response.json()
