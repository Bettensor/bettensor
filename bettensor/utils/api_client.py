import requests
import time
from typing import Dict, Any, Optional
import bittensor as bt

class APIClient:
    def __init__(self, rapid_api_key: str):
        self.rapid_api_key = rapid_api_key
        self.api_hosts = {
            "baseball": "api-baseball.p.rapidapi.com",
            "soccer": "api-football-v1.p.rapidapi.com",
        }
        self.last_request_time = 0
        self.min_request_interval = 1  # Minimum 1 second between requests

    def _make_request(self, url: str, params: Dict[str, Any], sport: str) -> Optional[Dict[str, Any]]:
        headers = {
            "x-rapidapi-host": self.api_hosts[sport],
            "x-rapidapi-key": self.rapid_api_key,
        }

        # Implement basic rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)

        bt.logging.debug(f"Sending request to {url} with params: {params}")
        bt.logging.debug(f"Headers: {headers}")

        try:
            response = requests.get(url, headers=headers, params=params)
            self.last_request_time = time.time()

            bt.logging.trace(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                bt.logging.error("Unauthorized: Check your API key")
            elif response.status_code == 429:
                bt.logging.error("Rate limit exceeded")
                # Implement exponential backoff here if needed
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
