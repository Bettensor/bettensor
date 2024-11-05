"""
Class to handle and process all incoming miner data.
"""


import asyncio
import datetime
from datetime import datetime, timezone, timedelta
import traceback
from typing import Dict
import bittensor as bt
import torch
from bettensor.protocol import GameData, TeamGame, TeamGamePrediction
from bettensor.validator.utils.database.database_manager import DatabaseManager


"""
Miner Data Methods, Extends the Bettensor Validator Class

"""


class MinerDataMixin:
    def insert_predictions(self, processed_uids, predictions):
        """
        Inserts new predictions into the database in batches
        """
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            bt.logging.trace(f"insert_predictions called with {len(predictions)} predictions")

            return_dict = {}
            valid_predictions = []

            # Begin single transaction for the batch
            self.db_manager.begin_transaction()

            try:
                for miner_uid, prediction_dict in predictions.items():
                    bt.logging.trace(f"insert_predictions processing uid: {miner_uid}, prediction_dict size: {len(prediction_dict)}")
                    
                    for prediction_id, res in prediction_dict.items():
                        try:
                            if int(miner_uid) not in processed_uids:
                                bt.logging.trace(f"uid {miner_uid} not in processed_uids, skipping")
                                return_dict[prediction_id] = (False, "UID not in processed_uids")
                                continue

                            hotkey = self.metagraph.hotkeys[int(miner_uid)]
                            
                            # Check if prediction exists
                            count_result = self.db_manager.fetch_one(
                                "SELECT COUNT(*) FROM predictions WHERE prediction_id = ?",
                                (prediction_id,),
                            )
                            
                            if count_result and count_result > 0:
                                bt.logging.debug(f"Prediction {prediction_id} already exists, skipping.")
                                return_dict[prediction_id] = (True, "Prediction already exists for this validator")
                                continue

                            # Check wager
                            if res.wager <= 0:
                                bt.logging.warning(f"Skipping prediction with non-positive wager: {res.wager} for UID {miner_uid}")
                                return_dict[prediction_id] = (False, "Prediction with non-positive wager - nice try")
                                continue

                            # Validate game exists
                            query = "SELECT sport, league, event_start_date, team_a, team_b, team_a_odds, team_b_odds, tie_odds, outcome FROM game_data WHERE external_id = ?"
                            result = self.db_manager.fetch_one(query, (res.game_id,))
                            bt.logging.debug(f"Result: {result}")

                            if not result:
                                bt.logging.debug(f"Game {res.game_id} not found in game_data for prediction {prediction_id} from uid {miner_uid}, skipping")
                                return_dict[prediction_id] = (False, "Game not found in validator game_data")
                                continue

                            # Extract game data
                            sport = result['sport']
                            league = result['league']
                            event_start_date = result['event_start_date']
                            team_a = result['team_a']
                            team_b = result['team_b']
                            team_a_odds = result['team_a_odds']
                            team_b_odds = result['team_b_odds']
                            tie_odds = result['tie_odds']
                            outcome = result['outcome']

                            # Convert predicted outcome
                            if res.predicted_outcome == team_a:
                                predicted_outcome = 0
                            elif res.predicted_outcome == team_b:
                                predicted_outcome = 1
                            elif res.predicted_outcome.lower() == "tie":
                                predicted_outcome = 2
                            else:
                                bt.logging.debug(f"Invalid predicted_outcome: {res.predicted_outcome}. Skipping this prediction.")
                                return_dict[prediction_id] = (False, "Invalid predicted_outcome")
                                continue

                            # Check if game started
                            if current_time >= event_start_date:
                                bt.logging.debug(f"Prediction not inserted: game {res.game_id} has already started.")
                                return_dict[prediction_id] = (False, "Game has already started")
                                continue

                            # Check daily wager limit
                            current_total_wager = self.db_manager.fetch_one(
                                "SELECT SUM(wager) as total FROM predictions WHERE miner_uid = ? AND DATE(prediction_date) = DATE(?)",
                                (miner_uid, current_time),
                            )
                            
                            # Get the sum value from the result and handle None case
                            current_total_wager = float(current_total_wager['total'] if current_total_wager and current_total_wager['total'] is not None else 0)
                            
                            if current_total_wager + res.wager > 1000:
                                bt.logging.debug(f"Prediction would exceed daily limit. Current: ${current_total_wager}, Attempted: ${res.wager}")
                                return_dict[prediction_id] = (False, "Prediction would exceed daily limit")
                                continue

                            # Get odds for predicted outcome
                            outcome_to_odds = {0: team_a_odds, 1: team_b_odds, 2: tie_odds}
                            predicted_odds = outcome_to_odds.get(predicted_outcome, res.predicted_odds)

                            # Add to valid predictions for batch insert
                            valid_predictions.append((
                                prediction_id, res.game_id, miner_uid, current_time,
                                predicted_outcome, predicted_odds, team_a, team_b,
                                res.wager, team_a_odds, team_b_odds, tie_odds,
                                outcome, res.model_name, res.confidence_score
                            ))
                            return_dict[prediction_id] = (True, "Prediction validated successfully")

                        except Exception as e:
                            bt.logging.error(f"Error processing prediction {prediction_id}: {str(e)}")
                            bt.logging.error(f"Traceback: {traceback.format_exc()}")
                            return_dict[prediction_id] = (False, f"Error processing prediction: {str(e)}")
                            continue

                # Batch insert all valid predictions
                if valid_predictions:
                    self.db_manager.executemany(
                        """
                        INSERT INTO predictions (
                            prediction_id, game_id, miner_uid, prediction_date,
                            predicted_outcome, predicted_odds, team_a, team_b,
                            wager, team_a_odds, team_b_odds, tie_odds,
                            outcome, model_name, confidence_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        valid_predictions
                    )

                # Commit the transaction
                self.db_manager.commit_transaction()

                # Add to entropy system and update return dict for successful predictions
                for pred_values in valid_predictions:
                    prediction_id = pred_values[0]
                    self.scoring_system.entropy_system.add_prediction(
                        prediction_id, pred_values[2], pred_values[1],
                        pred_values[4], pred_values[8], pred_values[5],
                        pred_values[3]
                    )
                    return_dict[prediction_id] = (True, "Prediction inserted successfully")
                    bt.logging.info(f"Prediction inserted successfully: {prediction_id}")

            except Exception as e:
                bt.logging.error(f"Error in batch processing: {str(e)}")
                self.db_manager.rollback_transaction()
                # Update return dict for failed batch
                for pred_values in valid_predictions:
                    return_dict[pred_values[0]] = (False, f"Batch insertion failed: {str(e)}")

            # Send confirmations for each miner
            for miner_uid, prediction_dict in predictions.items():
                self.scoring_system.entropy_system.save_state()
                bt.logging.info(f"Sending confirmation synapse to miner {miner_uid}")
                self.send_confirmation_synapse_async(int(miner_uid), return_dict)

        except Exception as e:
            bt.logging.error(f"miner_data.py | insert_predictions | An error occurred: {e}")
            bt.logging.error(f"miner_data.py | insert_predictions | Traceback: {traceback.format_exc()}")
            if self.db_manager.is_transaction_active():
                self.db_manager.rollback_transaction()

    def send_confirmation_synapse_async(self, miner_uid, predictions):
        self.executor.submit(self.send_confirmation_synapse(miner_uid, predictions))

    def send_confirmation_synapse(self, miner_uid, predictions):
        """
        Sends a confirmation synapse to the miner

        Args:
            miner_uid: the uid of the miner
            predictions: a dictionary with uids as keys and TeamGamePrediction objects as values
        """
        confirmation_dict = {
            str(pred_id): {"success": success, "message": message}
            for pred_id, (success, message) in predictions.items()
        }

        #bt.logging.info(f"Full confirmation_dict before adding miner stats: {confirmation_dict}")

        #get miner stats for uid
        miner_stats = self.db_manager.fetch_one("SELECT * FROM miner_stats WHERE miner_uid = ?", (miner_uid,))
        if miner_stats is None:
            bt.logging.warning(f"No miner_stats found for miner_uid: {miner_uid}")
            confirmation_dict['miner_stats'] = {}
        else:
            # handle None values in miner_stats 
            for key, value in miner_stats.items():
                if value is None:
                    miner_stats[key] = 0 
            confirmation_dict['miner_stats'] = miner_stats

        #bt.logging.info(f"confirmation_dict after adding miner stats: {confirmation_dict}")
        synapse = GameData.create(
            db_path=self.db_path,
            wallet=self.wallet,
            subnet_version=self.subnet_version,
            neuron_uid=miner_uid,
            synapse_type="confirmation",
            confirmation_dict=confirmation_dict,
        )

        #get axon for uid
        axon = self.metagraph.axons[miner_uid]


        bt.logging.info(f"Sending confirmation synapse to miner {miner_uid}, axon: {axon}")
        try:
            self.dendrite.query(
                axons=axon,
                synapse=synapse,
                timeout=self.timeout,
                deserialize=True,
            )
            bt.logging.info(f"Confirmation synapse sent to miner {miner_uid}")
        except Exception as e:
            bt.logging.error(f"miner_data.py | send_confirmation_synapse | An error occurred: {e}")
            bt.logging.error(f"miner_data.py | send_confirmation_synapse | Traceback: {traceback.format_exc()}")
            raise

        

    def process_prediction(self, processed_uids: torch.tensor, synapses: list) -> list:
        """
        processes responses received by miners

        Args:
            processed_uids: list of uids that have been processed
            synapses: list of deserialized synapses
        """
        bt.logging.trace(f"enter process_prediction")
        predictions = {}
        try:
            for synapse in synapses:
                #bt.logging.trace(f"Processing synapse type: {type(synapse)}")
                
                if not hasattr(synapse, 'prediction_dict') or not hasattr(synapse, 'metadata'):
                    bt.logging.warning(f"Invalid synapse object: {synapse}")
                    continue
                
                prediction_dict = synapse.prediction_dict
                metadata = synapse.metadata

                if metadata and hasattr(metadata, "neuron_uid"):
                    uid = metadata.neuron_uid
                    # Ensure prediction_dict is not None before processing
                    if prediction_dict is not None:
                        predictions[uid] = prediction_dict
                    else:
                        bt.logging.trace(
                        f"prediction from miner {uid} is empty and will be skipped."
                    )
                else:
                    bt.logging.warning(
                        "metadata is missing or does not contain neuron_uid."
                    )
                
            self.insert_predictions(processed_uids, predictions)

        except Exception as e:
            bt.logging.error(f"miner_data.py | process_prediction | An error occurred: {e}")
            bt.logging.error(f"miner_data.py | process_prediction | Traceback: {traceback.format_exc()}")
            raise

    def update_recent_games(self):
        bt.logging.info("miner_data.py update_recent_games called")
        current_time = datetime.now(timezone.utc)
        five_hours_ago = current_time - timedelta(hours=5)

        recent_games = self.db_manager.fetch_all(
            """
            SELECT external_id, team_a, team_b, sport, league, event_start_date
            FROM game_data
            WHERE event_start_date < ? AND outcome = 'Unfinished'
            """,
            (five_hours_ago.isoformat(),),
        )
        bt.logging.info("Recent games: ")
        bt.logging.info(recent_games)

        for game in recent_games:
            external_id, team_a, team_b, sport, league, event_start_date = game
            game_info = {
                "external_id": external_id,
                "team_a": team_a,
                "team_b": team_b,
                "sport": sport,
                "league": league,
                "event_start_date": event_start_date,
            }
            bt.logging.info("Game info: ")
            bt.logging.info(game_info)
            numeric_outcome = self.api_client.determine_winner(game_info)
            bt.logging.info("Outcome: ")
            bt.logging.info(numeric_outcome)

            if numeric_outcome is not None:
                # Update the game outcome in the database
                self.api_client.update_game_outcome(external_id, numeric_outcome)

        self.db_manager.commit_transaction()
        bt.logging.info(f"Checked {len(recent_games)} games for updates")

    def prepare_game_data_for_entropy(self, predictions):
        game_data = []
        for game_id, game_predictions in predictions.items():
            current_odds = self.get_current_odds(game_id)
            game_data.append(
                {
                    "id": game_id,
                    "predictions": game_predictions,
                    "current_odds": current_odds,
                }
            )
        return game_data

    def get_recent_games(self):
        """retrieves recent games from the database"""
        two_days_ago = (
            datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            - datetime.timedelta(hours=48)
        ).isoformat()
        return self.db_manager.fetch_all(
            "SELECT id, team_a, team_b, external_id FROM game_data WHERE event_start_date >= ? AND outcome = 'Unfinished'",
            (two_days_ago,),
        )

    def get_current_odds(self, game_id):
        try:
            # Query to fetch the current odds for the given game_id
            query = """
            SELECT team_a_odds, team_b_odds, tie_odds
            FROM game_data
            WHERE id = ? OR external_id = ?
            """
            result = self.db_manager.fetchone(query, (game_id, game_id))
            if result:
                home_odds, away_odds, tie_odds = result
                return [home_odds, away_odds, tie_odds]
            else:
                bt.logging.warning(f"No odds found for game_id: {game_id}")
                return [0.0, 0.0, 0.0]  # Return default values if no odds are found
        except Exception as e:
            bt.logging.error(f"Database error in get_current_odds: {e}")
            return [0.0, 0.0, 0.0]  # Return default values in case of database error

    def fetch_local_game_data(self, current_timestamp: str) -> Dict[str, TeamGame]:
        # Calculate timestamp for 15 days ago
        fifteen_days_ago = (
            datetime.fromisoformat(current_timestamp) - timedelta(days=15)
        ).isoformat()

        query = """
            SELECT external_id, team_a, team_b, sport, league, create_date, last_update_date, event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie
            FROM game_data
            WHERE event_start_date > ? OR (event_start_date BETWEEN ? AND ?)
        """

        rows = self.db_manager.fetch_all(
            query, (current_timestamp, fifteen_days_ago, current_timestamp)
        )

        gamedata_dict = {}
        for row in rows:
            team_game = TeamGame(
                game_id=row["external_id"],  # External ID from API
                team_a=row["team_a"],
                team_b=row["team_b"],
                sport=row["sport"],
                league=row["league"],
                create_date=row["create_date"],
                last_update_date=row["last_update_date"],
                event_start_date=row["event_start_date"],
                active=bool(row["active"]),
                outcome=row["outcome"],
                team_a_odds=row["team_a_odds"],
                team_b_odds=row["team_b_odds"],
                tie_odds=row["tie_odds"],
                can_tie=bool(row["can_tie"]),
            )
            gamedata_dict[row["external_id"]] = team_game

        return gamedata_dict

