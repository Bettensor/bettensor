import datetime
import json
import os
import requests
import time
import logging
from typing import Dict, Any

class BaseAPIClient:
    '''
    Base class for handling all sports data API requests.
    '''
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.last_update_file = "last_update.json"
        self.last_update_date = self.load_last_update_date()
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key})
        self.logger = logging.getLogger(__name__)

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None, max_retries: int = 3) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def get_data(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        return self._make_request(endpoint, params)







    ################################## Last Update Tracking ##################################

    def load_last_update_date(self):
        # Existing implementation
        if os.path.exists(self.last_update_file):
            with open(self.last_update_file, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data['last_update_date'])
        return datetime.utcnow() - datetime.timedelta(days=7)  # Default to a week ago if no file exists

    def save_last_update_date(self, date):
        # Existing implementation
        with open(self.last_update_file, 'w') as f:
            json.dump({'last_update_date': date.isoformat()}, f)
