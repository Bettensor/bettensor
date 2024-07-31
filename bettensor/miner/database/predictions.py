import datetime
from typing import Any, List, Dict
from bettensor.protocol import TeamGamePrediction, TeamGame
from bettensor.miner.stats.miner_stats import MinerStateManager
import bittensor as bt
import uuid
from datetime import datetime, timezone, timedelta

class PredictionsHandler:
    def __init__(self, db_manager, state_manager: MinerStateManager, miner_uid: str):
        bt.logging.trace(f"Initializing Predictions handler for miner: {miner_uid}")
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.miner_uid = miner_uid
        self.new_prediction_window = timedelta(hours=2)
        bt.logging.trace("Predictions handler initialization complete")

    def process_predictions(self, updated_games: Dict[str, TeamGame], new_games: Dict[str, TeamGame]) -> Dict[str, TeamGamePrediction]:
        bt.logging.info(f"Processing predictions for {len(updated_games)} updated and {len(new_games)} new games")
        
        try:
            updated_predictions = self.process_game_results(updated_games)
            bt.logging.info(f"Number of updated predictions: {len(updated_predictions)}")
            recent_predictions = self.get_recent_predictions()
            bt.logging.info(f"Number of recent predictions: {len(recent_predictions)}")
            result = {pred.predictionID: pred for pred in recent_predictions}
            bt.logging.info(f"Prediction processing complete. Returning {len(result)} predictions")
            return result
        except Exception as e:
            bt.logging.error(f"Error processing predictions: {e}")
            return {}

    def process_game_results(self, updated_games: Dict[str, TeamGame]):
        bt.logging.info(f"Processing results for {len(updated_games)} games")
        updated_predictions = []

        try:
            for game_id, game_data in updated_games.items():
                bt.logging.debug(f"Processing game {game_data.externalId} with outcome: {game_data.outcome}")
                with self.db_manager.get_cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM predictions 
                        WHERE teamGameID = ? AND minerID = ?
                    """, (game_data.externalId, self.miner_uid))
                    predictions = cursor.fetchall()
                    bt.logging.debug(f"Found {len(predictions)} predictions for game {game_data.externalId}")

                    for pred in predictions:
                        pred_dict = dict(zip([column[0] for column in cursor.description], pred))
                        predicted_outcome = pred_dict['predictedOutcome']
                        wager = float(pred_dict['wager'])
                        bt.logging.debug(f"Processing prediction {pred_dict['predictionID']} with current outcome: {pred_dict['outcome']}")

                        if game_data.outcome != "Unfinished":
                            if (game_data.outcome == 0 and predicted_outcome == pred_dict['teamA']) or \
                               (game_data.outcome == 1 and predicted_outcome == pred_dict['teamB']) or \
                               (game_data.outcome == 2 and predicted_outcome == "Tie"):
                                result = 'Wager Won'
                                odds = float(pred_dict['teamAodds'] if game_data.outcome == 0 else 
                                             pred_dict['teamBodds'] if game_data.outcome == 1 else 
                                             pred_dict['tieOdds'])
                                earnings = wager * (odds - 1)
                            else:
                                result = 'Wager Lost'
                                earnings = -wager

                            cursor.execute("""
                                UPDATE predictions 
                                SET outcome = ? 
                                WHERE predictionID = ?
                            """, (result, pred_dict['predictionID']))

                            self.state_manager.update_on_game_result({
                                'outcome': result,
                                'earnings': earnings,
                                'wager': wager,
                                'prediction': TeamGamePrediction(**pred_dict)
                            })

                            pred_dict['outcome'] = result
                            bt.logging.debug(f"Updated prediction {pred_dict['predictionID']} outcome to {result}")
                        else:
                            result = 'Unfinished'
                            bt.logging.debug(f"Prediction {pred_dict['predictionID']} remains Unfinished")

                        updated_predictions.append(TeamGamePrediction(**pred_dict))

            bt.logging.info(f"Updated {len(updated_predictions)} predictions")
            return updated_predictions
        except Exception as e:
            bt.logging.error(f"Error processing game results: {e}")
            return []

    def get_recent_predictions(self, time_window: timedelta = None) -> List[TeamGamePrediction]:
        if time_window is None:
            time_window = self.new_prediction_window
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - time_window
        
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE minerID = ? AND predictionDate > ?
                ORDER BY predictionDate DESC
            """, (self.miner_uid, cutoff_time.isoformat()))
            
            columns = [column[0] for column in cursor.description]
            return [TeamGamePrediction(**dict(zip(columns, row))) for row in cursor.fetchall()]

    def get_predictions(self, miner_uid) -> Dict[str, Dict[str, Any]]:
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
            predictions = {row[0]: dict(zip(columns, row)) for row in cursor.fetchall()}

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

    def update_prediction_team_names(self):
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

    def get_prediction(self, prediction_id):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT p.*, g.teamA, g.teamB 
                FROM predictions p
                JOIN games g ON p.teamGameID = g.gameID
                WHERE p.predictionID = ?
            """, (prediction_id,))
            row = cursor.fetchone()
            if row:
                columns = [column[0] for column in cursor.description]
                prediction_dict = dict(zip(columns, row))
                prediction = TeamGamePrediction(**prediction_dict)
                return prediction, prediction_dict['teamA'], prediction_dict['teamB']
        return None, None, None