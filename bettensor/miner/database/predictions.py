import datetime
from typing import Any, List, Dict
from bettensor.protocol import TeamGamePrediction, TeamGame
from bettensor.miner.stats.miner_stats import MinerStateManager
import bittensor as bt
import uuid
from datetime import datetime, timezone, timedelta

class PredictionsHandler:
    def __init__(self, db_manager, state_manager: MinerStateManager, miner_uid: str):
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.miner_uid = miner_uid
        self.new_prediction_window = timedelta(hours=24)

    def process_predictions(self, updated_games: Dict[str, TeamGame], new_games: Dict[str, TeamGame]) -> Dict[str, TeamGamePrediction]:
        updated_predictions = self.process_game_results(updated_games)
        recent_predictions = self.get_recent_predictions()
        result = {pred.predictionID: pred for pred in recent_predictions}
        return result

    def process_game_results(self, updated_games: Dict[str, TeamGame]):
        updated_predictions = []
        batch_updates = []

        for game_id, game_data in updated_games.items():
            actual_outcome = self.determine_actual_outcome(game_data)
            
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM predictions 
                    WHERE teamGameID = ?
                """, (game_id,))
                predictions = cursor.fetchall()
                
                for pred in predictions:
                    pred_dict = dict(pred)
                    new_outcome = self.determine_new_outcome(pred_dict['prediction'], actual_outcome)
                    
                    if pred_dict['outcome'] != new_outcome:
                        batch_updates.append((new_outcome, pred_dict['predictionID']))
                        pred_dict['outcome'] = new_outcome
                        updated_predictions.append(TeamGamePrediction(**pred_dict))
                        
                        self.state_manager.update_on_game_result({
                            'outcome': new_outcome,
                            'earnings': self.calculate_earnings(pred_dict['wager'], pred_dict['prediction'], game_data.outcome),
                            'wager': pred_dict['wager'],
                            'prediction': TeamGamePrediction(**pred_dict)
                        })

        # Batch update
        if batch_updates:
            with self.db_manager.get_cursor() as cursor:
                cursor.executemany("""
                    UPDATE predictions 
                    SET outcome = ?
                    WHERE predictionID = ?
                """, batch_updates)

        return updated_predictions

    def determine_actual_outcome(self, game_data):
        if game_data.outcome == 0:
            return f"Team A Win ({game_data.teamA})"
        elif game_data.outcome == 1:
            return f"Team B Win ({game_data.teamB})"
        else:
            return "Draw"

    def calculate_earnings(self, wager, prediction, result):
        if prediction == result:
            odds = float(prediction.split('(')[1].split(')')[0])
            return wager * (odds - 1)
        else:
            return -wager

    def determine_new_outcome(self, prediction, actual_outcome):
        team = actual_outcome.split('(')[1].split(')')[0]
        if prediction == actual_outcome:
            return f"Wager Won ({team})"
        else:
            return f"Wager Lost ({team})"

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
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT p.*, g.eventStartDate, g.outcome as game_outcome
                FROM predictions p
                JOIN games g ON p.teamGameID = g.externalID
                WHERE p.minerID = ?
                ORDER BY p.predictionDate DESC
            """, (miner_uid,))
            columns = [column[0] for column in cursor.description]
            predictions = {row[0]: dict(zip(columns, row)) for row in cursor.fetchall()}

        return predictions

    def add_prediction(self, prediction):
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
            
            self.state_manager.update_on_prediction(prediction)
        except Exception as e:
            raise

    def update_prediction_team_names(self):
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

    def update_miner_uid(self, new_miner_uid):
        self.miner_uid = new_miner_uid