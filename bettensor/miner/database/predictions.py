import datetime
from typing import Any, List, Dict
from bettensor.miner.database.outcome_handler import OutcomeHandler
from bettensor.protocol import TeamGamePrediction, TeamGame
from bettensor.miner.stats.miner_stats import MinerStateManager
import bittensor as bt
import uuid
from datetime import datetime, timezone, timedelta
import traceback

class PredictionsHandler:
    def __init__(self, db_manager, state_manager: MinerStateManager, miner_uid: str):
        """
        Initialize the Predictions handler.

        Args:
            db_manager: The database manager instance.
            state_manager (MinerStateManager): The miner state manager instance.
            miner_uid (str): The unique identifier of the miner.

        Behavior:
            - Sets up the database manager, state manager, and miner UID for handling predictions.
        """
        bt.logging.trace(f"Initializing Predictions handler for miner: {miner_uid}")
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.miner_uid = miner_uid
        self.new_prediction_window = timedelta(hours=2)
        self.outcome_handler = OutcomeHandler()
        bt.logging.trace("Predictions handler initialization complete")

    def process_predictions(self, updated_games: Dict[str, TeamGame], new_games: Dict[str, TeamGame]) -> Dict[str, TeamGamePrediction]:
        bt.logging.info(f"Processing predictions for {len(updated_games)} updated and {len(new_games)} new games")
        recent_predictions_dict = {}
        
        try:
            for game_id, game_data in updated_games.items():
                self._update_prediction_outcomes(game_data)
        
            current_time = datetime.now(timezone.utc)
            three_days_ago = current_time - timedelta(days=3)
            
            recent_predictions = self._get_recent_predictions(three_days_ago)
            
            for prediction in recent_predictions:
                recent_predictions_dict[prediction.predictionID] = prediction

            bt.logging.info(f"Prediction processing complete. Recent predictions: {len(recent_predictions_dict)}")
        except Exception as e:
            bt.logging.error(f"Error processing predictions: {e}")
            bt.logging.debug(f"Traceback: {traceback.format_exc()}")

        return recent_predictions_dict

    def _get_recent_predictions(self, start_date: datetime) -> List[TeamGamePrediction]:
        bt.logging.trace(f"Retrieving predictions since {start_date}")
        recent_predictions = []
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM predictions
                    WHERE minerID = ? AND predictionDate >= ?
                    ORDER BY predictionDate DESC
                """, (self.miner_uid, start_date.isoformat()))
                
                columns = [column[0] for column in cursor.description]
                for row in cursor.fetchall():
                    prediction_dict = dict(zip(columns, row))
                    prediction = TeamGamePrediction(**prediction_dict)
                    recent_predictions.append(prediction)
                
            bt.logging.trace(f"Retrieved {len(recent_predictions)} recent predictions")
        except Exception as e:
            bt.logging.error(f"Error retrieving recent predictions: {e}")
        
        return recent_predictions

    def _update_prediction_outcomes(self, game_data: TeamGame):
        bt.logging.trace(f"Updating prediction outcomes for game: {game_data.externalId}")
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM predictions WHERE teamGameID = ? AND minerID = ?", 
                    (game_data.externalId, self.miner_uid)
                )
                columns = [column[0] for column in cursor.description]
                predictions = [dict(zip(columns, row)) for row in cursor.fetchall()]

                for prediction_dict in predictions:
                    prediction_dict['wager'] = float(prediction_dict['wager'])
                    prediction_dict['teamAodds'] = float(prediction_dict['teamAodds'])
                    prediction_dict['teamBodds'] = float(prediction_dict['teamBodds'])
                    prediction_dict['tieOdds'] = float(prediction_dict['tieOdds']) if prediction_dict['tieOdds'] is not None else None
                    prediction_dict.pop('can_overwrite', None)

                    try:
                        prediction_obj = TeamGamePrediction(**prediction_dict)
                        if game_data.outcome != "Unfinished" and prediction_obj.outcome == "Unfinished":
                            self._update_prediction_outcome(prediction_obj, game_data)
                    except Exception as e:
                        bt.logging.error(f"Error creating TeamGamePrediction object: {e}")
                        bt.logging.error(f"Problematic prediction data: {prediction_dict}")
        except Exception as e:
            bt.logging.error(f"Error updating prediction outcomes for game {game_data.externalId}: {e}")

    def _update_prediction_outcome(self, prediction: TeamGamePrediction, game_data: TeamGame):
        """
        Update the outcome of an existing prediction based on game data.

        Args:
            prediction (TeamGamePrediction): The prediction to update.
            game_data (TeamGame): The updated game data.

        Behavior:
            - Updates the prediction outcome based on the game outcome.
            - Updates the miner's state based on the prediction result.
        """
        try:
            outcome = game_data.outcome
            if outcome != "Unfinished":
                if (outcome == 0 and prediction.predictedOutcome == prediction.teamA) or \
                   (outcome == 1 and prediction.predictedOutcome == prediction.teamB) or \
                   (outcome == 2 and prediction.predictedOutcome == "Tie"):
                    result = 'win'
                    earnings = prediction.wager * (prediction.teamAodds if outcome == 0 else 
                                                   prediction.teamBodds if outcome == 1 else 
                                                   prediction.tieOdds)
                else:
                    result = 'loss'
                    earnings = 0

                self.state_manager.update_on_game_result({'outcome': result, 'earnings': earnings})

                with self.db_manager.get_cursor() as cursor:
                    cursor.execute(
                        "UPDATE predictions SET outcome = ? WHERE predictionID = ?",
                        (result, prediction.predictionID)
                    )
        except Exception as e:
            bt.logging.error(f"Error updating prediction outcome for prediction {prediction.predictionID}: {e}")

    def get_recent_predictions(self, time_window: timedelta = None) -> Dict[str, TeamGamePrediction]:
        """
        Get recent predictions for the miner within a given time window.

        Args:
            time_window (timedelta, optional): The time window for recent predictions. Defaults to None.

        Returns:
            Dict[str, TeamGamePrediction]: A dictionary of recent predictions.

        Behavior:
            - Retrieves predictions for the miner within the given time window.
            - Returns a dictionary of predictions with their IDs as keys.
        """
        if time_window is None:
            time_window = self.new_prediction_window
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - time_window
        
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE minerID = ? AND predictionDate > ? AND outcome = 'Unfinished'
                ORDER BY predictionDate DESC
            """, (self.miner_uid, cutoff_time.isoformat()))
            
            predictions = cursor.fetchall()
        
        return {p[0]: TeamGamePrediction(*p) for p in predictions}

    def get_predictions(self, miner_uid) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve predictions for the miner.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of predictions.

        Behavior:
            - Retrieves predictions for the miner.
            - Returns a dictionary of predictions with their IDs as keys.
        """
        bt.logging.trace("Retrieving predictions")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT p.*, g.eventStartDate
                FROM predictions p
                JOIN games g ON p.teamGameID = g.externalId
                WHERE p.minerID = ?
                ORDER BY p.predictionDate DESC
            """, (miner_uid,))
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            predictions = {}
            for row in rows:
                prediction = dict(zip(columns, row))
                # Convert the numeric outcome to a string representation
                prediction['outcome'] = self.outcome_handler.convert_outcome(
                    prediction['outcome'],
                    prediction['predictedOutcome']
                )
                # Check if the game is more than a week old and still unfinished
                if prediction['outcome'] == "Unfinished":
                    event_start_date = datetime.fromisoformat(prediction['eventStartDate'].replace('Z', '+00:00'))
                    if datetime.now(event_start_date.tzinfo) - event_start_date > timedelta(days=7):
                        prediction['outcome'] = "Unknown"
                predictions[prediction['predictionID']] = prediction

        bt.logging.trace(f"Retrieved {len(predictions)} predictions")
        return predictions

    def add_prediction(self, prediction):
        bt.logging.info(f"Adding prediction: {prediction}")
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO predictions (
                        predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
                        wager, teamAodds, teamBodds, tieOdds, outcome, teamA, teamB
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction['predictionID'],
                    prediction['teamGameID'],
                    self.miner_uid,
                    prediction['predictionDate'],
                    prediction['predictedOutcome'],
                    prediction['wager'],
                    prediction['teamAodds'],
                    prediction['teamBodds'],
                    prediction['tieOdds'],
                    prediction['outcome'],
                    prediction['teamA'],
                    prediction['teamB']
                ))
            bt.logging.info(f"Successfully added prediction: {prediction['predictionID']}")
            
            # Update the state manager with the new prediction
            self.state_manager.update_on_prediction(prediction)
        except Exception as e:
            bt.logging.error(f"Error adding prediction {prediction['predictionID']}: {str(e)}")
            raise

    def get_prediction(self, prediction_id):
        """
        Get a prediction by its ID.

        Args:
            prediction_id (str): The ID of the prediction to retrieve.

        Returns:
            TeamGamePrediction: The retrieved prediction object.
            str: The teamA name.
            str: The teamB name.

        Behavior:
            - Retrieves the prediction from the database.
            - Returns the prediction object, teamA name, and teamB name.
        """
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT p.*, g.teamA, g.teamB 
                FROM predictions p
                JOIN games g ON p.teamGameID = g.gameID
                WHERE p.predictionID = ?
            """, (prediction_id,))
            row = cursor.fetchone()
            if row:
                prediction = TeamGamePrediction(
                    predictionID=row[0],
                    teamGameID=row[1],
                    minerID=row[2],
                    predictionDate=row[3],
                    predictedOutcome=row[4],
                    wager=row[5],
                    teamAodds=row[6],
                    teamBodds=row[7],
                    tieOdds=row[8],
                    outcome=row[9],
                    teamA=row[11],
                    teamB=row[12]
                )
                # We can access teamA and teamB here if needed, but we don't include them in the TeamGamePrediction object
                teamA = row[11]
                teamB = row[12]
                return prediction, teamA, teamB
        return None, None, None

    def update_prediction_team_names(self):
        """
        Update team names for predictions that have missing team names.

        Behavior:
            - Retrieves predictions with missing team names.
            - Looks up the corresponding game for each prediction.
            - Updates the prediction with the team names from the game.
        """
        bt.logging.info("Updating prediction team names")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT p.predictionID, p.teamGameID, g.teamA, g.teamB
                FROM predictions p
                JOIN games g ON p.teamGameID = g.gameID
                WHERE p.teamA IS NULL OR p.teamB IS NULL
            """)
            predictions_to_update = cursor.fetchall()

            for pred_id, game_id, team_a, team_b in predictions_to_update:
                cursor.execute("""
                    UPDATE predictions
                    SET teamA = ?, teamB = ?
                    WHERE predictionID = ?
                """, (team_a, team_b, pred_id))

        bt.logging.info(f"Updated team names for {len(predictions_to_update)} predictions")

    def update_predictions_from_games(self, updated_games: Dict[str, TeamGame]):
        bt.logging.debug("Updating predictions based on new game data")
        updated_predictions = []
        try:
            predictions_to_check = self.get_predictions_for_games(updated_games.keys())
            
            for pred in predictions_to_check:
                game = updated_games[pred['teamGameID']]
                bt.logging.trace(f"Checking game {game.externalId}: Current outcome: {game.outcome}, Prediction outcome: {pred['outcome']}")
                
                new_outcome = self.outcome_handler.convert_outcome(game.outcome, pred['predictedOutcome'])
                
                if new_outcome != pred['outcome']:
                    self.update_prediction_outcome(pred['predictionID'], new_outcome)
                    updated_pred = self.get_prediction_by_id(pred['predictionID'])
                    updated_predictions.append(updated_pred)
                    bt.logging.info(f"Updated prediction {pred['predictionID']} for game {game.externalId}: New outcome {new_outcome}")
            
            bt.logging.info(f"Updated {len(updated_predictions)} predictions")
            return updated_predictions
        except Exception as e:
            bt.logging.error(f"Error updating predictions from games: {e}")
            return []

    def get_predictions_for_games(self, game_ids):
        with self.db_manager.get_cursor() as cursor:
            placeholders = ','.join(['?'] * len(game_ids))
            cursor.execute(f"""
                SELECT * FROM predictions 
                WHERE teamGameID IN ({placeholders}) AND outcome IS NULL
            """, tuple(game_ids))
            return cursor.fetchall()

    def update_prediction_outcome(self, prediction_id, outcome):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                UPDATE predictions
                SET outcome = ?
                WHERE predictionID = ?
            """, (outcome, prediction_id))

    def get_prediction_by_id(self, prediction_id: str):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM predictions WHERE predictionID = ?", (prediction_id,))
            prediction = cursor.fetchone()
            if prediction:
                return dict(zip([column[0] for column in cursor.description], prediction))
        return None

    def fix_existing_prediction_outcomes(self):
        bt.logging.info("Starting to fix existing prediction outcomes")
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT p.predictionID, p.teamGameID, p.outcome as prediction_outcome, g.outcome as game_outcome
                    FROM predictions p
                    JOIN games g ON p.teamGameID = g.externalID
                    WHERE p.outcome != 'Unfinished' AND g.outcome = 'Unfinished'
                """)
                incorrect_predictions = cursor.fetchall()

                for pred_id, game_id, pred_outcome, game_outcome in incorrect_predictions:
                    bt.logging.info(f"Fixing prediction {pred_id} for game {game_id}: Resetting outcome from {pred_outcome} to Unfinished")
                    cursor.execute("""
                        UPDATE predictions
                        SET outcome = 'Unfinished'
                        WHERE predictionID = ?
                    """, (pred_id,))

            bt.logging.info(f"Fixed {len(incorrect_predictions)} incorrect prediction outcomes")
        except Exception as e:
            bt.logging.error(f"Error fixing existing prediction outcomes: {e}")