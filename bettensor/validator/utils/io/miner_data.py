"""
Class to handle and process all incoming miner data.
"""


import datetime
from datetime import datetime, timezone, timedelta
from typing import Dict
import bittensor as bt
import torch
from bettensor.protocol import TeamGame, TeamGamePrediction
from bettensor.validator.utils.database.database_manager import DatabaseManager


"""
Miner Data Methods, Extends the Bettensor Validator Class

"""


class MinerDataMixin:
    def insert_predictions(self, processed_uids, predictions):
        """
        Inserts new predictions into the database

        Args:
        processed_uids: list of uids that have been processed
        predictions: a dictionary with uids as keys and TeamGamePrediction objects as values
        """
        current_time = datetime.now(timezone.utc).isoformat()

        # Get today's date in UTC
        today_utc = datetime.now(timezone.utc).date().isoformat()

        for uid, prediction_dict in predictions.items():
            for prediction_id, res in prediction_dict.items():
                if int(uid) not in processed_uids:
                    continue

                # Get today's date in UTC
                today_utc = datetime.now(timezone.utc).isoformat()

                hotkey = self.metagraph.hotkeys[int(uid)]
                prediction_id = res.prediction_id
                game_id = res.game_id
                miner_uid = uid
                prediction_date = today_utc
                predicted_outcome = res.predicted_outcome
                wager = res.wager

                if wager <= 0:
                    bt.logging.warning(
                        f"Skipping prediction with non-positive wager: {wager} for UID {uid}"
                    )
                    continue

                # Check if the predictionID already exists
                if (
                    self.db_manager.fetchone(
                        "SELECT COUNT(*) FROM predictions WHERE prediction_id = ?",
                        (prediction_id,),
                    )[0]
                    > 0
                ):
                    bt.logging.debug(
                        f"Prediction {prediction_id} already exists, skipping."
                    )
                    continue

                query = "SELECT sport, league, event_start_date, team_a, team_b, team_a_odds, team_b_odds, tie_odds, outcome FROM game_data WHERE id = ?"
                result = self.db_manager.fetchone(query, (game_id,))

                if not result:
                    continue

                (
                    sport,
                    league,
                    event_start_date,
                    team_a,
                    team_b,
                    team_a_odds,
                    team_b_odds,
                    tie_odds,
                    outcome,
                ) = result

                # Convert predictedOutcome to numeric value
                if predicted_outcome == team_a:
                    predicted_outcome = 0
                elif predicted_outcome == team_b:
                    predicted_outcome = 1
                elif predicted_outcome.lower() == "tie":
                    predicted_outcome = 2
                else:
                    bt.logging.debug(
                        f"Invalid predicted_outcome: {predicted_outcome}. Skipping this prediction."
                    )
                    continue

                # Check if the game has already started
                if current_time >= event_start_date:
                    bt.logging.debug(
                        f"Prediction not inserted: game {game_id} has already started."
                    )
                    continue

                self.db_manager.begin_transaction()
                try:
                    # Calculate total wager for the date, excluding the current prediction
                    current_total_wager = (
                        self.db_manager.fetchone(
                            "SELECT SUM(wager) FROM predictions WHERE miner_uid = ? AND DATE(prediction_date) = DATE(?)",
                            (miner_uid, prediction_date),
                        )[0]
                        or 0
                    )
                    new_total_wager = current_total_wager + wager

                    if new_total_wager > 1000:
                        bt.logging.debug(
                            f"Prediction for miner {miner_uid} would exceed daily limit. Current total: ${current_total_wager}, Attempted wager: ${wager}"
                        )
                        self.db_manager.rollback_transaction()
                        continue  # Skip this prediction but continue processing others

                    # Insert new prediction
                    self.db_manager.execute_query(
                        """
                        INSERT INTO predictions (prediction_id, game_id, miner_uid, prediction_date, predicted_outcome, team_a, team_b, wager, team_a_odds, team_b_odds, tie_odds, can_overwrite, outcome)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            prediction_id,
                            game_id,
                            miner_uid,
                            prediction_date,
                            predicted_outcome,
                            team_a,
                            team_b,
                            wager,
                            team_a_odds,
                            team_b_odds,
                            tie_odds,
                            False,
                            outcome,
                        ),
                    )
                    self.db_manager.commit_transaction()
                except Exception as e:
                    self.db_manager.rollback_transaction()
                    bt.logging.error(f"An error occurred: {e}")

        # After inserting predictions, update entropy scores
        # game_data = self.prepare_game_data_for_entropy(predictions)
        # self.entropy_system.update_ebdr_scores(game_data)

    def process_prediction(
        self, processed_uids: torch.tensor, predictions: list
    ) -> list:
        """
        processes responses received by miners

        Args:
            processed_uids: list of uids that have been processed
            predictions: list of deserialized synapses
        """

        predictions_dict = {}

        for synapse in predictions:
            if len(synapse) >= 3:
                game_data_dict = synapse[0]
                prediction_dict: Dict[str, TeamGamePrediction] = synapse[1]
                metadata = synapse[2]
                error = synapse[3] if len(synapse) > 3 else None
                error = synapse[3] if len(synapse) > 3 else None

                if metadata and hasattr(metadata, "neuron_uid"):
                    uid = metadata.neuron_uid

                    # Ensure prediction_dict is not None before processing
                    if prediction_dict is not None:
                        predictions_dict[uid] = prediction_dict
                    else:
                        bt.logging.trace(
                            f"prediction from miner {uid} is empty and will be skipped."
                        )
                else:
                    bt.logging.warning(
                        "metadata is missing or does not contain neuron_uid."
                    )
            else:
                bt.logging.warning(
                    "synapse data is incomplete or not in the expected format."
                )
        self.insert_predictions(processed_uids, predictions_dict)

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
            SELECT game_id, team_a, team_b, sport, league, external_id, create_date, last_update_date, event_start_date, active, outcome, team_a_odds, team_b_odds, tie_odds, can_tie
            FROM game_data
            WHERE event_start_date > ? OR (event_start_date BETWEEN ? AND ?)
        """

        rows = self.db_manager.fetch_all(
            query, (current_timestamp, fifteen_days_ago, current_timestamp)
        )

        gamedata_dict = {}
        for row in rows:
            team_game = TeamGame(
                game_id=row[0],  # External ID from API
                team_a=row[1],
                team_b=row[2],
                sport=row[3],
                league=row[4],
                create_date=row[6],
                last_update_date=row[7],
                event_start_date=row[8],
                active=bool(row[9]),
                outcome=row[10],
                team_a_odds=row[11],
                team_b_odds=row[12],
                tie_odds=row[13],
                can_tie=bool(row[14]),
            )
            gamedata_dict[row[0]] = team_game

        return gamedata_dict
