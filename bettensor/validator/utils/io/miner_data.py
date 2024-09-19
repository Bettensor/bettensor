"""
Class to handle and process all incoming miner data.
"""


import datetime
from typing import Dict
import bittensor as bt
import torch
from bettensor.protocol import TeamGame, TeamGamePrediction
from bettensor.validator.bettensor_validator import BettensorValidator
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
        current_time = (
            datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
        )

        # Get today's date in UTC
        today_utc = datetime.now(datetime.timezone.utc).date().isoformat()

        for uid, prediction_dict in predictions.items():
            for predictionID, res in prediction_dict.items():
                if int(uid) not in processed_uids:
                    continue

                # Get today's date in UTC
                today_utc = datetime.now(datetime.timezone.utc).isoformat()

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
        game_data = self.prepare_game_data_for_entropy(predictions)
        self.entropy_system.update_ebdr_scores(game_data)

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

        self.create_table()
        self.insert_predictions(processed_uids, predictions_dict)

    def update_game_outcome(self, game_id, numeric_outcome):
        """Updates the outcome of a game in the database"""
        try:
            self.db_manager.begin_transaction()
            affected_rows = self.db_manager.execute_query(
                "UPDATE game_data SET outcome = ?, active = 0 WHERE external_id = ?",
                (numeric_outcome, game_id),
            )
            if affected_rows == 0:
                bt.logging.trace(f"No game updated for external_id {game_id}")
            else:
                bt.logging.trace(
                    f"Updated game {game_id} with outcome: {numeric_outcome}"
                )
            self.db_manager.commit_transaction()
        except Exception as e:
            self.db_manager.rollback_transaction()
            bt.logging.error(f"Error updating game outcome: {e}")

    def update_recent_games(self):
        current_time = datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

        # Fetch games that have started at least 4 hours ago but don't have a final outcome yet
        four_hours_ago = current_time - datetime.timedelta(hours=4)
        recent_games = self.db_manager.fetchall(
            """
            SELECT id, team_a, team_b, external_id, event_start_date, sport
            FROM game_data
            WHERE event_start_date <= ? AND outcome = 'Unfinished'
            ORDER BY event_start_date
            """,
            (four_hours_ago.isoformat(),),
        )

        bt.logging.info(f"Checking {len(recent_games)} games for updates")

        for game in recent_games:
            game_id, team_a, team_b, external_id, start_time, sport = game
            start_time = datetime.fromisoformat(start_time).replace(
                tzinfo=datetime.timezone.utc
            )

            # Additional check to ensure the game has indeed started
            if start_time > current_time:
                continue

            bt.logging.trace(f"Checking {sport} game {external_id} for results")
            self.determine_winner((game_id, team_a, team_b, external_id))

        bt.logging.info("Recent games and predictions update process completed")

        # DEAD CODE

    # def calculate_total_wager(self, minerId, event_start_date, exclude_id=None):
    #     """calculates the total wager for a given miner and event start date"""
    #     query = """
    #         SELECT p.wager
    #         FROM predictions p
    #         JOIN game_data g ON p.game_id = g.id
    #         WHERE p.miner_uid = ? AND DATE(g.event_start_date) = DATE(?)
    #     """
    #     params = (minerId, event_start_date)

    #     if exclude_id:
    #         query += " AND p.game_id != ?"
    #         params += (exclude_id,)

    #     wagers = self.db_manager.fetchall(query, params)
    #     total_wager = sum([w[0] for w in wagers])

    #     return total_wager

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
        return self.db_manager.fetchall(
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

    def determine_winner(self, game_info):
        game_id, team_a, team_b, external_id = game_info

        sport = self.get_sport_from_db(external_id)
        if not sport:
            bt.logging.error(f"No game found with externalId {external_id}")
            return

        bt.logging.trace(f"Fetching {sport} game data for externalId: {external_id}")

        if sport == "baseball":
            game_data = self.api_client.get_baseball_game(str(external_id))
        elif sport == "soccer":
            game_data = self.api_client.get_soccer_game(str(external_id))
        elif sport.lower() == "nfl":
            game_data = self.api_client.get_nfl_result(str(external_id))
        else:
            bt.logging.error(f"Unsupported sport: {sport}")
            return

        if not game_data:
            bt.logging.error(f"Invalid or empty game data for {external_id}")
            return

        if sport.lower() == "nfl":
            if "results" not in game_data or not game_data["results"]:
                bt.logging.error(
                    f"Invalid or empty game data for NFL game {external_id}"
                )
                return
            game_response = game_data["results"][0]
            status = game_response.get("time_status")
            if status != "3":  # 3 means the game has finished
                bt.logging.trace(
                    f"NFL game {external_id} is not finished yet. Current status: {status}"
                )
                return
        else:
            if "response" not in game_data or not game_data["response"]:
                bt.logging.error(
                    f"Invalid or empty game data for {sport} game {external_id}"
                )
                return
            game_response = game_data["response"][0]

        # Check if the game has finished
        if sport == "baseball":
            status = game_response.get("status", {}).get("long")
            if status != "Finished":
                bt.logging.trace(
                    f"Baseball game {external_id} is not finished yet. Current status: {status}"
                )
                return
        elif sport == "soccer":
            status = game_response.get("fixture", {}).get("status", {}).get("long")
            if status not in [
                "Match Finished",
                "Match Finished After Extra Time",
                "Match Finished After Penalties",
            ]:
                bt.logging.trace(
                    f"Soccer game {external_id} is not finished yet. Current status: {status}"
                )
                return

        # Process scores and update game outcome
        self.process_game_result(sport, game_response, external_id, team_a, team_b)


def process_game_result(self, sport, game_response, external_id, team_a, team_b):
    # Handle NFL scores
    if sport.lower() == "nfl":
        # The NFL score is provided as a string like "20-27"
        scores = game_response.get("ss", "").split("-")
        if len(scores) == 2:
            home_score, away_score = map(int, scores)
        else:
            bt.logging.error(f"Invalid score format for NFL game {external_id}")
            return
    # Handle baseball and soccer scores
    elif sport == "baseball":
        home_score = game_response.get("scores", {}).get("home", {}).get("total")
        away_score = game_response.get("scores", {}).get("away", {}).get("total")
    elif sport == "soccer":
        home_score = game_response.get("goals", {}).get("home")
        away_score = game_response.get("goals", {}).get("away")
    elif sport.lower() == "nfl":
        scores = game_response.get("ss", "").split("-")
        if len(scores) == 2:
            home_score, away_score = map(int, scores)
        else:
            bt.logging.error(f"Invalid score format for NFL game {external_id}")
            return
    else:
        bt.logging.error(f"Unsupported sport: {sport}")
        return

    # Validate scores
    if home_score is None or away_score is None:
        bt.logging.error(f"Unable to extract scores for {sport} game {external_id}")
        return

    # Convert scores to integers for comparison
    home_score = int(home_score)
    away_score = int(away_score)

    # Determine game outcome: 0 for home win, 1 for away win, 2 for tie
    if home_score > away_score:
        numeric_outcome = 0
    elif away_score > home_score:
        numeric_outcome = 1
    else:
        numeric_outcome = 2

    bt.logging.trace(
        f"Game {external_id} result: {team_a} {home_score} - {away_score} {team_b}"
    )

    # Update the game outcome in the database
    self.update_game_outcome(external_id, numeric_outcome)


def get_sport_from_db(self, external_id):
    result = self.db_manager.fetchone(
        "SELECT sport FROM game_data WHERE external_id = ?", (external_id,)
    )
    return result[0] if result else None


def recalculate_all_profits(self):
    self.weight_setter.recalculate_daily_profits()


def fetch_local_game_data(self, current_timestamp: str) -> Dict[str, TeamGame]:
    # Calculate timestamp for 15 days ago
    fifteen_days_ago = (
        datetime.fromisoformat(current_timestamp) - datetime.timedelta(days=15)
    ).isoformat()

    query = """
        SELECT id, teamA, teamB, sport, league, externalId, createDate, lastUpdateDate, eventStartDate, active, outcome, teamAodds, teamBodds, tieOdds, canTie
        FROM game_data
        WHERE eventStartDate > ? OR (eventStartDate BETWEEN ? AND ?)
    """

    rows = self.db_manager.fetchall(
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
