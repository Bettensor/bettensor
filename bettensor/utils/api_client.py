import requests
from typing import Dict, Any, Optional

class APIClient:
    def __init__(self, rapid_api_key: str):
        self.rapid_api_key = rapid_api_key
        self.api_hosts = {
            "baseball": "api-baseball.p.rapidapi.com",
            "soccer": "api-football-v1.p.rapidapi.com",
        }

    def _make_request(self, url: str, params: Dict[str, Any], sport: str) -> Optional[Dict[str, Any]]:
        headers = {
            "x-rapidapi-host": self.api_hosts[sport],
            "x-rapidapi-key": self.rapid_api_key,
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None

    def get_baseball_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        url = "https://api-baseball.p.rapidapi.com/games"
        params = {"id": game_id}
        return self._make_request(url, params, "baseball")

    def get_soccer_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
        params = {"id": game_id}
        return self._make_request(url, params, "soccer")