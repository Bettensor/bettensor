# File: bettensor/utils/bettensor_api.py

import requests
from datetime import datetime, timedelta, timezone
import bittensor as bt
import json
import os
from ..database.database_manager import DatabaseManager
from .base_api_client import BaseAPIClient
import aiohttp


class BettensorAPIClient(BaseAPIClient):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()

        self.base_url = "https://dev-bettensor-api.azurewebsites.net/"
        self.db_manager = db_manager
        self.games = []
        self.api_key = os.getenv("BETTENSOR_API_KEY", None)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.session = aiohttp.ClientSession()  # Use aiohttp for async requests

    async def fetch_all_game_data(self, last_update_date):
        """
        fetch and update game data from bettensor API. overridden from BaseAPIClient

        """
        if last_update_date is None:
            # set to 1 week ago
            last_update_date = datetime.now(timezone.utc) - timedelta(days=15)

        # Fetch the games from the API
        bt.logging.info(f"Fetching games from API with last update date: {last_update_date}")
        games = await self.get_games(last_update_date)
        # print one game
        # if len(games) > 0:
        #     bt.logging.trace(f"Games: {games[0]}")

        return games

    async def get_games(self, last_update_date=None, items_per_page=100):
        """Get all games from the API with pagination."""
        all_games = []
        page_index = 0
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=15)
        ).isoformat()  # 15 days ago

        while True:
            # Add await here
            games = await self.get_games_page(
                start_date, items_per_page, page_index, last_update_date
            )
            if not games:
                break
            all_games.extend(games)
            page_index += 1
        return all_games

    async def get_games_page(self, start_date, items_per_page, page_index, last_update_date):
        """Get a single page of games from the API."""
        params = {
            "PageIndex": page_index,
            "ItemsPerPage": items_per_page,
            "SortOrder": "StartDate",
            "LastUpdateDate": last_update_date.isoformat() if last_update_date else None,
            "StartDate": start_date,
            "LeagueFilter": "true",
        }

        try:
            # Add await here
            response = await self.session.get(
                f"{self.base_url}/Games/TeamGames/Search",
                params=params,
                headers=self.headers
            )
            # Add await here
            data = await response.json()
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

    async def get_data(self, endpoint, params):
        url = self.base_url + endpoint
        response = await self.session.get(url, params=params)
        return response.json()
