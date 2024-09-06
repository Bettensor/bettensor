import requests
import time
from typing import Dict, Any, Optional
import bittensor as bt
from datetime import datetime, timezone

class APIClient:
    def __init__(self, rapid_api_key: str, bet365_api_key: str):
        self.rapid_api_key = rapid_api_key
        self.bet365_api_key = bet365_api_key
        self.last_request_time = 0
        self.min_request_interval = 1  # Minimum 1 second between requests

    def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Implement basic rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)

        bt.logging.debug(f"Sending request to {url} with params: {params}")

        try:
            response = requests.get(url, params=params)
            self.last_request_time = time.time()

            bt.logging.debug(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                return response.json()
            else:
                bt.logging.error(f"Failed to fetch data. Status code: {response.status_code}")
                bt.logging.error(f"Response content: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Request failed: {str(e)}")
            return None

    def get_baseball_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        url = "https://api-baseball.p.rapidapi.com/games"
        params = {"id": game_id}
        return self._make_request(url, params, "baseball")

    def get_soccer_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
        params = {"id": game_id}
        return self._make_request(url, params, "soccer")

    def get_upcoming_nfl_events(self) -> Optional[Dict[str, Any]]:
        url = "https://api.b365api.com/v1/bet365/upcoming"
        params = {"sport_id": 12, "token": self.bet365_api_key}
        return self._make_request(url, params)

    def get_nfl_odds(self, event_id: str) -> Optional[Dict[str, Any]]:
        url = "https://api.b365api.com/v3/bet365/prematch"
        params = {"FI": event_id, "token": self.bet365_api_key}
        return self._make_request(url, params)

    def process_nfl_games(self) -> list:
        api_response = self.get_upcoming_nfl_events()
        if not api_response:
            bt.logging.error("Failed to fetch upcoming NFL events")
            return []

        nfl_games = []
        for game in api_response.get('results', []):
            if game.get('league', {}).get('name') == "NFL":
                game_id = game['id']
                home_team = game['home']['name']
                away_team = game['away']['name']
                game_time = datetime.fromtimestamp(int(game['time']), tz=timezone.utc).isoformat()

                bt.logging.debug(f"Fetching odds for {away_team} @ {home_team} ({game_time})")
                odds_data = self.get_nfl_odds(game_id)

                if odds_data:
                    moneyline_odds = self.get_moneyline_odds(odds_data, home_team, away_team)
                    if moneyline_odds:
                        nfl_games.append({
                            "home": home_team,
                            "away": away_team,
                            "game_id": str(game_id),
                            "date": game_time,
                            "odds": {
                                "average_home_odds": float(moneyline_odds[home_team]['odds']),
                                "average_away_odds": float(moneyline_odds[away_team]['odds']),
                                "average_tie_odds": None
                            },
                            "sport": "nfl",
                            "league": "NFL",
                        })
                        bt.logging.debug(f"Processed game: {home_team} vs {away_team}")
                    else:
                        bt.logging.warning(f"No moneyline odds found for game {game_id}")
                else:
                    bt.logging.warning(f"Failed to fetch odds for game {game_id}")

        bt.logging.info(f"Processed {len(nfl_games)} NFL games")
        return nfl_games

    def get_moneyline_odds(self, odds_data: Dict[str, Any], home_team: str, away_team: str) -> Dict[str, Dict[str, Any]]:
        moneyline_odds = {}
        for market in odds_data.get('results', [{}])[0].get('main', {}).get('sp', {}).values():
            if market.get('name') == 'Game Lines':
                for odd in market.get('odds', []):
                    if odd.get('name') == 'Money Line':
                        team = home_team if odd['header'] == "1" else away_team
                        moneyline_odds[team] = {
                            'odds': odd['odds'],
                            'implied_probability': f"{(1 / float(odd['odds'])) * 100:.2f}%"
                        }
        return moneyline_odds

    def get_nfl_result(self, event_id: str) -> Optional[Dict[str, Any]]:
        url = "https://api.b365api.com/v1/bet365/result"
        params = {"event_id": event_id, "token": self.nfl_api_token}
        return self._make_request(url, params)