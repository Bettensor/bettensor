import traceback
from typing import Any, List, Dict, Optional
from bettensor.protocol import TeamGamePrediction, TeamGame
import bittensor as bt
from datetime import datetime, timezone, timedelta
from bettensor.miner.models.model_utils import SoccerPredictor
from fuzzywuzzy import process

class PredictionsHandler:
    def __init__(self, db_manager, state_manager, miner_hotkey: str):
        bt.logging.trace("Initializing PredictionsHandler")
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = state_manager.miner_uid
        self.new_prediction_window = timedelta(hours=24)
        self.soccer_predictor = SoccerPredictor(model_name='podos_soccer_model')
        self.update_predictions_with_hotkey()
        self.update_predictions_with_minerid()
        bt.logging.trace("PredictionsHandler initialization complete")

    def update_predictions_with_hotkey(self):
        bt.logging.trace("Updating predictions with miner hotkey")
        query = """
        UPDATE predictions
        SET minerHotkey = %s
        WHERE minerID = %s AND minerHotkey IS NULL
        """
        try:
            self.db_manager.execute_query(query, params=(self.miner_hotkey, self.miner_uid))
            bt.logging.info("Successfully updated predictions with miner hotkey")
        except Exception as e:
            bt.logging.error(f"Error updating predictions with hotkey: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    def update_predictions_with_minerid(self):
        bt.logging.trace("Updating predictions with miner ID")
        query = """
        UPDATE predictions
        SET minerID = %s
        WHERE minerID IS NULL
        """
        try:
            self.db_manager.execute_query(query, params=(self.miner_uid,))
            bt.logging.info("Successfully updated predictions with miner ID")
        except Exception as e:
            bt.logging.error(f"Error updating predictions with miner ID: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    def process_predictions(self, updated_games: Dict[str, TeamGame], new_games: Dict[str, TeamGame]) -> Dict[str, TeamGamePrediction]:
        bt.logging.trace(f"Processing predictions for {len(updated_games)} updated games and {len(new_games)} new games")
        updated_predictions = self.process_game_results(updated_games)
        recent_predictions = self.get_recent_predictions()

        soccer_games = {game_id: game for game_id, game in new_games.items() if game.sport == 'soccer'}

        if soccer_games:
            home_teams = [game.homeTeam for game in soccer_games.values()]
            away_teams = [game.awayTeam for game in soccer_games.values()]
            odds = [[game.teamAodds, game.tieOdds, game.teamBodds] for game in soccer_games.values()]

            encoded_teams = set(self.soccer_predictor.le.classes_)
            matched_home_teams = [self.get_best_match(team, encoded_teams) for team in home_teams]
            matched_away_teams = [self.get_best_match(team, encoded_teams) for team in away_teams]

            if None not in matched_home_teams + matched_away_teams:
                soccer_predictions = self.soccer_predictor.predict(matched_home_teams, matched_away_teams, odds)
            else:
                bt.logging.warning("Some teams not found in label encoder, skipping model predictions")

        result = {pred.predictionID: pred for pred in recent_predictions}
        bt.logging.trace(f"Processed {len(result)} predictions")
        return result

    def get_best_match(self, team_name, encoded_teams):
        match, score = process.extract0ne(team_name, encoded_teams)
        if score >= 80:
            return match
        else:
            return None

    def process_game_results(self, updated_games: Dict[str, TeamGame]):
        bt.logging.trace(f"Processing game results for {len(updated_games)} games")
        updated_predictions = []
        batch_updates = []

        for external_id, game_data in updated_games.items():
            actual_outcome = self.determine_actual_outcome(game_data)
            predictions = self.get_predictions_for_game(external_id)
            
            for pred in predictions:
                new_outcome = self.determine_new_outcome(pred.predictedOutcome, actual_outcome)
                
                if pred.outcome != new_outcome:
                    batch_updates.append((new_outcome, pred.predictionID))
                    pred.outcome = new_outcome
                    updated_predictions.append(pred)
                    
                    self.state_manager.update_on_game_result({
                        'outcome': new_outcome,
                        'earnings': self.calculate_earnings(pred.wager, pred.predictedOutcome, game_data.outcome),
                        'wager': pred.wager,
                        'prediction': pred.dict()
                    })

        if batch_updates:
            query = "UPDATE predictions SET outcome = %s WHERE predictionID = %s"
            self.db_manager.execute_batch(query, batch_updates)

        bt.logging.trace(f"Processed {len(updated_predictions)} predictions")
        return updated_predictions

    def determine_actual_outcome(self, game_data: TeamGame) -> str:
        bt.logging.trace(f"Determining actual outcome for game {game_data.id}")
        if game_data.outcome == 0:
            return f"Team A Win ({game_data.teamA})"
        elif game_data.outcome == 1:
            return f"Team B Win ({game_data.teamB})"
        else:
            return "Draw"

    def calculate_earnings(self, wager: float, prediction: str, result: int) -> float:
        bt.logging.trace(f"Calculating earnings for wager {wager}, prediction {prediction}, result {result}")
        if (prediction.startswith("Team A") and result == 0) or (prediction.startswith("Team B") and result == 1):
            odds = float(prediction.split('(')[1].split(')')[0])
            return wager * (odds - 1)
        else:
            return -wager

    def determine_new_outcome(self, prediction: str, actual_outcome: str) -> str:
        bt.logging.trace(f"Determining new outcome for prediction {prediction} and actual outcome {actual_outcome}")
        team = actual_outcome.split('(')[1].split(')')[0] if '(' in actual_outcome else ''
        if prediction == actual_outcome:
            return f"Wager Won ({team})"
        else:
            return f"Wager Lost ({team})"

    def get_recent_predictions(self, limit=100):
        bt.logging.trace(f"Getting recent predictions (limit: {limit})")
        query = """
        SELECT predictionid AS "predictionID", teamgameid AS "teamGameID", minerid AS "minerID", minerhotkey AS "minerHotkey", 
               predictiondate AS "predictionDate", predictedoutcome AS "predictedOutcome", teama AS "teamA", teamb AS "teamB", 
               wager, teamaodds AS "teamAodds", teambodds AS "teamBodds", tieodds AS "tieOdds", outcome
        FROM predictions
        WHERE minerHotkey = %s
        ORDER BY predictionDate DESC
        LIMIT %s
        """
        results = self.db_manager.execute_query(query, (self.miner_hotkey, limit))
        for row in results:
            row['predictionDate'] = row['predictionDate'].isoformat() if isinstance(row['predictionDate'], datetime) else row['predictionDate']
        return [TeamGamePrediction(**row) for row in results]

    def add_prediction(self, prediction: Dict[str, Any]):
        bt.logging.trace(f"Adding prediction: {prediction}")
        query = """
        INSERT INTO predictions (
            predictionID, teamGameID, minerID, minerHotkey, predictionDate, predictedOutcome,
            teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            prediction['predictionID'],
            prediction['teamGameID'],
            self.miner_uid,  # Ensure minerID is set
            self.miner_hotkey,  # Ensure minerHotkey is set
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
            self.state_manager.update_on_prediction(prediction)
            bt.logging.info(f"Prediction {prediction['predictionID']} added successfully")
            return {'status': 'success', 'message': f"Prediction {prediction['predictionID']} added successfully"}
        except Exception as e:
            bt.logging.error(f"Error adding prediction: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'message': f"Error adding prediction: {str(e)}"}

    def get_predictions_for_game(self, external_id: str) -> List[TeamGamePrediction]:
        bt.logging.trace(f"Getting predictions for game: {external_id}")
        query = """
        SELECT predictionid AS "predictionID", teamgameid AS "teamGameID", minerid AS "minerID", minerhotkey AS "minerHotkey", 
               predictiondate AS "predictionDate", predictedoutcome AS "predictedOutcome", teama AS "teamA", teamb AS "teamB", 
               wager, teamaodds AS "teamAodds", teambodds AS "teamBodds", tieodds AS "tieOdds", outcome
        FROM predictions
        WHERE teamGameID = %s
        """
        results = self.db_manager.execute_query(query, (external_id,))
        predictions = []
        for row in results:
            try:
                row['predictionDate'] = row['predictionDate'].isoformat() if isinstance(row['predictionDate'], datetime) else row['predictionDate']
                prediction = TeamGamePrediction(**row)
                predictions.append(prediction)
            except Exception as e:
                bt.logging.error(f"Error creating TeamGamePrediction: {e}")
                bt.logging.error(f"Row data: {row}")
        return predictions

    def update_prediction_outcome(self, prediction: TeamGamePrediction, game_outcome: str) -> Optional[TeamGamePrediction]:
        bt.logging.trace(f"Updating prediction outcome: {prediction.predictionID}, game outcome: {game_outcome}")
        new_outcome = self.determine_new_outcome(prediction.predictedOutcome, game_outcome)
        if prediction.outcome != new_outcome:
            query = "UPDATE predictions SET outcome = %s WHERE predictionID = %s"
            self.db_manager.execute_query(query, (new_outcome, prediction.predictionID))
            prediction.outcome = new_outcome
            return prediction
        return None

    def calculate_wager_odds(self, prediction):
        if prediction['predictedOutcome'] == prediction['Home']:
            return prediction['teamAodds']
        elif prediction['predictedOutcome'] == prediction['Away']:
            return prediction['teamBodds']
        elif prediction['predictedOutcome'] == 'Tie' and prediction['tieOdds'] is not None:
            return prediction['tieOdds']
        return 0.0

    def get_predictions_with_teams(self, miner_hotkey):
        query = """
        SELECT p.predictionid AS "predictionID", p.teamgameid AS "teamGameID", p.minerid AS "minerID", p.minerhotkey AS "minerHotkey", 
               p.predictiondate AS "predictionDate", p.predictedoutcome AS "predictedOutcome", g.teama AS "Home", g.teamb AS "Away", 
               p.wager, p.teamaodds AS "teamAodds", p.teambodds AS "teamBodds", p.tieodds AS "tieOdds", p.outcome
        FROM predictions p
        JOIN games g ON p.teamgameid = g.externalid
        WHERE p.minerHotkey = %s
        """
        try:
            results = self.db_manager.execute_query(query, (miner_hotkey,))
            for row in results:
                if row['predictionDate'] is not None:
                    row['predictionDate'] = row['predictionDate'].isoformat() if isinstance(row['predictionDate'], datetime) else row['predictionDate']
                row['wagerOdds'] = self.calculate_wager_odds(row)
            return {row['predictionID']: row for row in results}
        except Exception as e:
            bt.logging.error(f"Error getting predictions with teams: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def get_predictions(self, miner_hotkey):
        query = """
        SELECT predictionid AS "predictionID", teamgameid AS "teamGameID", minerid AS "minerID", minerhotkey AS "minerHotkey", 
               predictiondate AS "predictionDate", predictedoutcome AS "predictedOutcome", teama AS "teamA", teamb AS "teamB", 
               wager, teamaodds AS "teamAodds", teambodds AS "teamBodds", tieodds AS "tieOdds", outcome
        FROM predictions
        WHERE minerHotkey = %s
        """
        try:
            results = self.db_manager.execute_query(query, (miner_hotkey,))
            for row in results:
                if row['predictionDate'] is not None:
                    row['predictionDate'] = row['predictionDate'].isoformat() if isinstance(row['predictionDate'], datetime) else row['predictionDate']
                # Only call calculate_wager_odds if Home and Away fields are present
                if 'Home' in row and 'Away' in row:
                    row['wagerOdds'] = self.calculate_wager_odds(row)
            return {row['predictionID']: row for row in results}
        except Exception as e:
            bt.logging.error(f"Error getting predictions: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {}