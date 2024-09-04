import requests
from datetime import datetime, timedelta
import bittensor as bt

class BettensorAPIClient:
    def __init__(self):
        self.base_url = "https://dev-bettensor-api.azurewebsites.net/Games/TeamGames/Search"

    def get_games(self, start_date=None, end_date=None, sports_filter="Soccer,Baseball", items_per_page=100):
        if start_date is None:
            start_date = datetime.utcnow()
        if end_date is None:
            end_date = start_date + timedelta(days=7)

        params = {
            "PageIndex": 1,
            "ItemsPerPage": items_per_page,
            "SortOrder": "StartDate",
            "MinPredictions": 0,
            "StartDate": start_date.isoformat(),
            "EndDate": end_date.isoformat(),
            "SportsFilter": sports_filter,
            "AwaitingResult": "true"
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            bt.logging.error(f"Error fetching games from new API: {e}")
            return None

    def transform_game_data(self, game):
        return {
            "home": game.get("teamA"),
            "away": game.get("teamB"),
            "game_id": game.get("externalId"),
            "date": game.get("date"),
            "odds": {
                "average_home_odds": game.get("teamAOdds"),
                "average_away_odds": game.get("teamBOdds"),
                "average_tie_odds": game.get("drawOdds")
            },
            "sport": game.get("sport").lower(),
            "league": game.get("league")
        }