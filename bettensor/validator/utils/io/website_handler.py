import traceback
import requests
import json
from datetime import datetime
import bittensor as bt
from argparse import ArgumentParser
from bettensor.validator.utils.database.database_manager import DatabaseManager
from dateutil import parser
import pytz


class WebsiteHandler:
    def __init__(self, validator):
        # Initialize the parser and validator
        self.validator = validator

    def ensure_keys_table(self):
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
            coldkey = None  # Initialize coldkey

            for neuron in self.validator.metagraph.neurons:
                if neuron.hotkey == hotkey:
                    coldkey = neuron.coldkey
                    insert_query = "INSERT INTO keys (hotkey, coldkey) VALUES (?, ?)"
                    self.validator.db_manager.execute_query(insert_query, (hotkey, coldkey))
                    return coldkey  # Exit after finding and inserting the coldkey

            # If coldkey is not found after iterating through neurons
            coldkey = "dummy_coldkey"
            insert_query = "INSERT INTO keys (hotkey, coldkey) VALUES (?, ?)"
            self.validator.db_manager.execute_query(insert_query, (hotkey, coldkey))
            return coldkey

    def fetch_predictions_from_db(self):
        """
        Fetch predictions from the SQLite3 database and organize them by miner_uid.

        :return: Dictionary with miner_uid as keys and lists of prediction dictionaries as values
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
            # Check if predictions is a list of dictionaries
            if not isinstance(predictions, list) or not all(isinstance(p, dict) for p in predictions):
                bt.logging.error("Invalid predictions format. Expected list of dictionaries.")
                return {}
            
            organized_predictions = {}
            for prediction in predictions:
                miner_uid = prediction.get('miner_uid')
                if miner_uid not in organized_predictions:
                    organized_predictions[miner_uid] = []
                organized_predictions[miner_uid].append(prediction)
            
            return organized_predictions
        except Exception as e:
            bt.logging.error(f"Error fetching predictions: {e}")
            return {}

    def update_sent_status(self, prediction_ids):
        query = "UPDATE predictions SET sent_to_site = 1 WHERE prediction_id = ?"
        try:
            self.validator.db_manager.executemany(query, [(pid,) for pid in prediction_ids])
            bt.logging.info(f"Updated sent_to_site status for {len(prediction_ids)} predictions")
        except Exception as e:
            bt.logging.error(f"Error updating sent_to_site status: {e}")

    def send_predictions(self, predictions_by_miner_uid):
        """
        Send organized predictions to the Bettensor API.

        :param predictions_by_miner_uid: Dictionary with miner_uid as keys and lists of prediction data
        :return: API response status code
        """
        url = "https://www.bettensor.com/API/Predictions/"
        headers = {"Content-Type": "application/json"}
        network = self.validator.subtensor.network

        for miner_uid, predictions in predictions_by_miner_uid.items():
            try:
                # Fetch or update coldkey once per miner_uid
                hotkey = self.validator.metagraph.hotkeys[miner_uid]
                if hotkey:
                    coldkey = self.get_or_update_coldkey(hotkey)
                else:
                    bt.logging.warning(f"Invalid miner_uid: {miner_uid}. Setting coldkey to 'dummy_coldkey'.")
                    coldkey = "dummy_coldkey"
                # fetch miner stats by miner_uid

                miner_stats = self.validator.db_manager.fetch_one("SELECT * FROM miner_stats WHERE miner_uid = ?", (miner_uid,))

                transformed_data = []
                for prediction in predictions:
                    try:
                        if not isinstance(prediction, dict):
                            bt.logging.error(f"Invalid prediction format for miner_uid {miner_uid}: Expected dict, got {type(prediction).__name__}. Skipping prediction.")
                            continue  # Skip this prediction if it's not a dictionary

                        metadata = {
                            "miner_uid": miner_uid,
                            "miner_hotkey": hotkey,
                            "miner_coldkey": coldkey,
                            "miner_stats": miner_stats,
                            "prediction_date": prediction.get("prediction_date"),
                            "predicted_outcome": prediction.get("predicted_outcome"),
                            "wager": prediction.get("wager"),
                            "predicted_odds": prediction.get("predicted_odds"),
                            "team_a_odds": prediction.get("team_a_odds"),
                            "team_b_odds": prediction.get("team_b_odds"),
                            "tie_odds": prediction.get("tie_odds"),
                            "is_model_prediction": prediction.get("is_model_prediction"),
                            "outcome": prediction.get("outcome"),
                            "payout": prediction.get("payout"),
                            "subtensor_network": network,
                        }

                        transformed_prediction = {
                            "externalGameId": str(prediction.get("game_id")),
                            "minerHotKey": hotkey,
                            "minerColdKey": coldkey,
                            "predictionDate": self._format_prediction_date(prediction.get("prediction_date")),
                            "predictedOutcome": prediction.get("predicted_outcome"),
                            "wager": prediction.get("wager"),
                            "modelName": prediction.get("model_name"),
                            "predictionOdds": prediction.get("predicted_odds"),
                            "metaData": json.dumps(metadata),
                        }

                        transformed_data.append(transformed_prediction)

                    except Exception as e:
                        bt.logging.error(f"Error processing prediction_id {prediction.get('prediction_id', 'Unknown')} for miner_uid {miner_uid}: {e}")
                        bt.logging.error(f"Error traceback: {traceback.format_exc()}")
                        continue  # Skip this prediction and continue with the next

                if not transformed_data:
                    bt.logging.info(f"No valid predictions to send for miner_uid: {miner_uid}")
                    continue

                bt.logging.info(f"Sending {len(transformed_data)} predictions to API for miner_uid: {miner_uid}")
                bt.logging.debug(f"First prediction (for debugging): {json.dumps(transformed_data[0], indent=2)}")

                response = requests.post(url, data=json.dumps(transformed_data), headers=headers)
                if response.status_code in [200, 201]:
                    prediction_ids = [p["prediction_id"] for p in predictions if isinstance(p, dict) and "prediction_id" in p]
                    self.update_sent_status(prediction_ids)
                    bt.logging.info(f"Response status code for miner_uid {miner_uid}: {response.status_code}")
                    bt.logging.debug(f"Response content: {response.text}")
                else:
                    bt.logging.error(f"Failed to send predictions for miner_uid {miner_uid}. Status code: {response.status_code}, Response: {response.text}")

            except Exception as e:
                bt.logging.error(f"Error sending predictions for miner_uid {miner_uid}: {e}")
                bt.logging.error(f"Error traceback: {traceback.format_exc()}")

        return None

    def _format_prediction_date(self, prediction_date):
        """
        Ensure prediction_date is in ISO 8601 format with UTC timezone.

        :param prediction_date: Original prediction date string
        :return: Formatted date string
        """
        try:
            # First, try to parse the date assuming it's already in ISO format
            date = parser.isoparse(prediction_date)
            
            # Ensure the date is timezone-aware (use UTC if no timezone)
            if date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)
            
            # Format to ISO 8601 with 'Z' indicating UTC
            return date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        except ValueError:
            # If parsing fails, attempt to parse with a more flexible parser
            try:
                date = parser.parse(prediction_date)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=pytz.UTC)
                return date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            except ValueError:
                bt.logging.warning(f"Invalid date format for prediction_date '{prediction_date}'. Using current UTC time.")
                return datetime.now(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def fetch_and_send_predictions(self):
        """
        Fetch predictions from the database and send them to the API.

        :return: API response status code
        """
        try:
            network = self.validator.subtensor.network
            bt.logging.info(f"Network: {network}")
            if network == "finney":
                self.ensure_keys_table()  # Ensure the keys table exists
                predictions = self.fetch_predictions_from_db()
                if predictions:
                    bt.logging.debug("Sending predictions to the Bettensor website.")
                    return self.send_predictions(predictions)
                else:
                    bt.logging.info("No new predictions found in the database.")
            else:
                bt.logging.info("Not on Finney network, skipping prediction sending.")
            return None
        except Exception as e:
            bt.logging.error(f"Error fetching and sending predictions: {e}")
            bt.logging.error(f"Error traceback: {traceback.format_exc()}")