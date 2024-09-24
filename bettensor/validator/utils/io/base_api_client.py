import json
import bittensor as bt
from datetime import datetime

class BaseAPIClient:
    """
    Base class for handling all sports data API requests.
    """

    def __init__(self):
        self.last_update_file = "last_update.json"
        self.last_update_date = self.load_last_update_date()

    def fetch_and_update_game_data(self):
        """
        fetch and update game data from external APIs.
        Override this method in the child classes.
        """

        pass

    def fetch_and_update_single_game_data(self, external_id):
        """
        fetch and update single game data from external APIs.
        Override this method in the child classes.
        """

        pass

    def load_last_update_date(self):
        """
        load the last update date from the last_update.json file.
        """
        try:
            with open(self.last_update_file, "r") as f:
                data = json.load(f)
                timestamp = data.get("timestamp")
                if timestamp:
                    return datetime.fromisoformat(timestamp)
                return None
        except Exception as e:
            bt.logging.error(f"Error loading last update date: {e}")
            return None

    def save_last_update_date(self, last_update_date):
        """
        save the last update date to the last_update.json file.
        """
        with open(self.last_update_file, "w") as f:
            timestamp = last_update_date.isoformat()
            json.dump({"timestamp": timestamp}, f)

        pass
