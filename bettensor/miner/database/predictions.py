import datetime
import traceback
from typing import Any, List, Dict
from bettensor.miner.database.database_manager import DatabaseManager
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
        #bt.logging.info(f"Processed predictions: {len(result)} predictions")
        #bt.logging.info(f"Sample processed prediction: {next(iter(result.values())) if result else 'No predictions'}")
        return result

    def process_game_results(self, updated_games: Dict[str, TeamGame]):
        updated_predictions = []
        batch_updates = []

        for game_id, game_data in updated_games.items():
            actual_outcome = self.determine_actual_outcome(game_data)
            
            query = """
                SELECT * FROM predictions 
                WHERE teamGameID = %s
            """
            predictions = self.db_manager.execute_query(query, params=(game_id,))
            
            for pred in predictions:
                pred_dict = dict(zip([column[0] for column in self.db_manager.execute_query("SELECT * FROM predictions LIMIT 0")], pred))
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
            query = """
                UPDATE predictions 
                SET outcome = %s
                WHERE predictionID = %s
            """
            self.db_manager.execute_query(query, params=batch_updates)

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
        
        query = """
            SELECT * FROM predictions 
            WHERE minerID = %s AND predictionDate > %s
            ORDER BY predictionDate DESC
        """
        predictions = self.db_manager.execute_query(query, params=(str(self.miner_uid), cutoff_time.isoformat()))
        
        columns = [column[0] for column in self.db_manager.execute_query("SELECT * FROM predictions LIMIT 0")]
        return [TeamGamePrediction(**dict(zip(columns, row))) for row in predictions]

    def get_predictions(self, miner_uid, limit=None):
        query = """
        SELECT *
        FROM predictions
        WHERE minerid = %s
        ORDER BY predictiondate DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            results = self.db_manager.execute_query(query, (miner_uid,))
            predictions = {}
            for row in results:
                #bt.logging.debug(f"Raw row data: {row}")  # Log raw row data for debugging
                prediction_id = row[0]
                prediction = {
                    'predictionID': row[0],
                    'teamGameID': row[1],
                    'minerID': row[2],
                    'predictionDate': row[3].isoformat() if isinstance(row[3], datetime) else str(row[3]),
                    'predictedOutcome': row[4],
                    'teamA': row[5],
                    'teamB': row[6],
                    'wager': float(row[7]) if row[7] is not None else None,
                    'teamAodds': float(row[8]) if row[8] is not None else None,
                    'teamBodds': float(row[9]) if row[9] is not None else None,
                    'tieOdds': float(row[10]) if row[10] is not None else None,
                    'outcome': row[11],
                }
                # Calculate wager odds
                if prediction['predictedOutcome'] == prediction['teamA']:
                    prediction['wagerOdds'] = prediction['teamAodds']
                elif prediction['predictedOutcome'] == prediction['teamB']:
                    prediction['wagerOdds'] = prediction['teamBodds']
                else:
                    prediction['wagerOdds'] = prediction['tieOdds']
                
                predictions[prediction_id] = prediction

            #bt.logging.info(f"Retrieved {len(predictions)} predictions for miner {miner_uid}")
            #bt.logging.info(f"Sample prediction: {next(iter(predictions.values())) if predictions else 'No predictions'}")
            return predictions
        except Exception as e:
            bt.logging.error(f"Error retrieving predictions: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def add_prediction(self, prediction):
        query = """
        INSERT INTO predictions (
            predictionID, teamGameID, minerID, predictionDate,
            predictedOutcome, teamA, teamB, wager, teamAodds,
            teamBodds, tieOdds, outcome
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            prediction['predictionID'],
            prediction['teamGameID'],  # This is now the externalId
            prediction['minerID'],
            prediction['predictionDate'],
            prediction['predictedOutcome'],
            prediction['teamA'],
            prediction['teamB'],
            prediction['wager'],
            prediction['teamAodds'],
            prediction['teamBodds'],
            prediction['tieOdds'],
            prediction['outcome']
        )
        
        try:
            self.db_manager.execute_query(query, params)
            #bt.logging.info(f"Added prediction: {prediction['predictionID']}")
        except Exception as e:
            bt.logging.error(f"Error adding prediction: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    def update_prediction_team_names(self):
        query1 = """
            SELECT p.predictionID, p.teamGameID, g.teamA, g.teamB
            FROM predictions p
            JOIN games g ON p.teamGameID = g.gameID
            WHERE p.teamA IS NULL OR p.teamB IS NULL
        """
        predictions_to_update = self.db_manager.execute_query(query1)

        for pred_id, game_id, team_a, team_b in predictions_to_update:
            query2 = """
                UPDATE predictions
                SET teamA = %s, teamB = %s
                WHERE predictionID = %s
            """
            self.db_manager.execute_query(query2, params=(team_a, team_b, pred_id))

    def get_prediction(self, prediction_id):
        query = """
            SELECT p.*, g.teamA, g.teamB 
            FROM predictions p
            JOIN games g ON p.teamGameID = g.gameID
            WHERE p.predictionID = %s
        """
        result = self.db_manager.execute_query(query, params=(prediction_id,))
        if result and result[0]:
            columns = [column[0] for column in self.db_manager.execute_query("SELECT p.*, g.teamA, g.teamB FROM predictions p JOIN games g ON p.teamGameID = g.gameID LIMIT 0")]
            prediction_dict = dict(zip(columns, result[0]))
            prediction = TeamGamePrediction(**prediction_dict)
            return prediction, prediction_dict['teamA'], prediction_dict['teamB']
        return None, None, None

    def update_miner_uid(self, new_miner_uid):
        self.miner_uid = new_miner_uid

    def print_all_predictions(self):
        query = """
        SELECT * FROM predictions
        ORDER BY predictionDate DESC
        LIMIT 50
        """
        
        try:
            results = self.db_manager.execute_query(query)
            if results:
                bt.logging.info("Recent predictions in the database:")
                for row in results:
                    bt.logging.info(f"Prediction ID: {row['predictionid']}, Game ID: {row['teamgameid']}, Miner ID: {row['minerid']}, Date: {row['predictiondate']}, Outcome: {row['outcome']}")
            else:
                bt.logging.info("No predictions found in the database.")
        except Exception as e:
            bt.logging.error(f"Error retrieving predictions: {str(e)}")