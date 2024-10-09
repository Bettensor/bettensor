import requests
import json
from datetime import datetime
import bittensor as bt
from argparse import ArgumentParser
from bettensor.validator.utils.database.database_manager import DatabaseManager


class WebsiteHandler:
    def __init__(self, validator):
        # Initialize the parser and validator
        self.validator = validator

    def create_keys_table(self):
        """
        Creates keys table in db
        """
        query = """
        CREATE TABLE IF NOT EXISTS keys (
            hotkey TEXT PRIMARY KEY,
            coldkey TEXT
        )
        """
        self.validator.db_manager.execute_query(query)

    def get_or_update_coldkey(self, hotkey: str) -> str:
        """
        Retrieves coldkey from metagraph if it doesn't exist in keys table
        """
        self.validator.initialize_connection()  # Ensure subtensor is initialized

        query = "SELECT coldkey FROM keys WHERE hotkey = ?"
        result = self.validator.db_manager.fetch_one(query, (hotkey,))

        if result:
            return result['coldkey']
        else:
        # If not found, fetch from Metagraph and insert into the database
            self.validator.sync_metagraph()
            for neuron in self.validator.metagraph.neurons:
                if neuron.hotkey == hotkey:
                    coldkey = neuron.coldkey
                insert_query = "INSERT INTO keys (hotkey, coldkey) VALUES (?, ?)"
                self.validator.db_manager.execute_query(insert_query, (hotkey, coldkey))
                return coldkey

        # If coldkey is not found, insert "dummy_coldkey"
        insert_query = "INSERT INTO keys (hotkey, coldkey) VALUES (?, ?)"
        self.validator.db_manager.execute_query(insert_query, (hotkey, "dummy_coldkey"))
        return "dummy_coldkey"

    def fetch_predictions_from_db(self):
        """
        Fetch predictions from the SQLite3 database.

        :return: List of dictionaries containing prediction data
        """
        query = """
        SELECT prediction_id, game_id, miner_uid, prediction_date, predicted_outcome,
            predicted_odds, team_a, team_b, wager, team_a_odds, team_b_odds, tie_odds,
            model_name, confidence_score, outcome, payout, sent_to_site
        FROM predictions
        WHERE sent_to_site = 0
        """
        
        try:
            predictions = self.validator.db_manager.fetch_all(query)
            return predictions
        except Exception as e:
            bt.logging.error(f"Error fetching predictions: {e}")
            return []

    def update_sent_status(self, prediction_ids):
        query = "UPDATE predictions SET sent_to_site = 1 WHERE predictionID = ?"
        try:
            self.validator.db_manager.executemany(query, [(pid,) for pid in prediction_ids])
            bt.logging.info(f"Updated sent_to_site status for {len(prediction_ids)} predictions")
        except Exception as e:
            bt.logging.error(f"Error updating sent_to_site status: {e}")

    def send_predictions(self, predictions):
        """
        Send predictions to the Bettensor API.

        :param predictions: List of dictionaries containing prediction data
        :return: API response status code
        """
        url = "https://www.bettensor.com/API/Predictions/"

        transformed_data = []

        for prediction in predictions:
            hotkey = prediction["miner_hotkey"]
            coldkey = self.get_or_update_coldkey(hotkey)

        metadata = {
            "miner_uid": prediction.get("miner_uid"),
            "miner_hotkey": hotkey,
            "miner_coldkey": coldkey,
            "prediction_date": prediction["prediction_date"],
            "predicted_outcome": prediction["predicted_outcome"],
            "wager": prediction["wager"],
            "predicted_odds": prediction["predicted_odds"],
            "team_a_odds": prediction["team_a_odds"],
            "team_b_odds": prediction["team_b_odds"],
            "tie_odds": prediction["tie_odds"],
            "is_model_prediction": prediction["is_model_prediction"],
            "outcome": prediction["outcome"],
            "payout": prediction["payout"],
            "subtensor_network": self.validator.subtensor.network,
        }

        transformed_prediction = {
            "externalGameId": prediction["game_id"],
            "minerHotkey": hotkey,
            "minerColdkey": coldkey,
            "predictionDate": prediction["prediction_date"],
            "predictedOutcome": prediction["predicted_outcome"],
            "wager": prediction["wager"],
            "predictionOdds": prediction["predicted_odds"],
            "metaData": metadata,
        }

        try:
            date = datetime.strptime(prediction["prediction_date"], "%Y-%m-%d %H:%M:%S")
            transformed_prediction["prediction_date"] = date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            pass

            transformed_data.append(transformed_prediction)
        
            headers = {"Content-Type": "application/json"}

        bt.logging.info(f"Sending {len(transformed_data)} predictions to API")
        bt.logging.debug(f"First prediction (for debugging): {json.dumps(transformed_data[0], indent=2)}")

        try:
            response = requests.post(url, data=json.dumps(transformed_data), headers=headers)
            if response.status_code in [200, 201]:
                self.update_sent_status([p["prediction_id"] for p in predictions])
                bt.logging.info(f"Response status code: {response.status_code}")
                bt.logging.debug(f"Response content: {response.text}")
                return response.status_code

        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Error sending predictions: {e}")
        return None

    def fetch_and_send_predictions(self):
        """
        Fetch predictions from the database and send them to the API.

        :return: API response status code
        """
        network = self.validator.subtensor.network
        bt.logging.info(f"Network: {network}")
        if network == "finney":
            self.create_keys_table()  # Ensure the keys table exists
            predictions = self.fetch_predictions_from_db()
            if predictions:
                bt.logging.debug("Sending predictions to the Bettensor website.")
                return self.send_predictions(predictions)
            else:
                bt.logging.info("No new predictions found in the database.")
        else:
            bt.logging.info("Not on Finney network, skipping prediction sending.")
        return None