import traceback
from typing import Any, List, Dict, Optional, Union
import uuid
from bettensor.protocol import TeamGamePrediction, TeamGame
import bittensor as bt
from datetime import datetime, timezone, timedelta
import psycopg2
from bettensor.miner.models.model_utils import SoccerPredictor
from fuzzywuzzy import process
from psycopg2.extras import RealDictCursor
from bettensor.miner.stats.miner_stats import MinerStatsHandler

class PredictionsHandler:
    def __init__(self, db_manager, state_manager, miner_hotkey: str):
        print("PredictionsHandler initialized")
        bt.logging.info("PredictionsHandler initialized")
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = state_manager.miner_uid
        self.new_prediction_window = timedelta(hours=24)
        self.stats_handler = MinerStatsHandler(state_manager)
        self.update_predictions_with_minerid()
        bt.logging.trace("PredictionsHandler initialization complete")

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

    def add_prediction(self, prediction: Dict[str, Any]):
        print(f"DEBUG: Adding prediction: {prediction}")
        # Deduct wager before adding prediction
        wager_amount = prediction.get('wager', 0)
        bt.logging.debug(f"Attempting to deduct wager: {wager_amount}")
        if not self.stats_handler.deduct_wager(wager_amount):
            bt.logging.warning(f"Insufficient funds to place wager of {wager_amount}")
            return {'status': 'error', 'message': f"Insufficient funds to place wager of {wager_amount}"}

        query = """
        INSERT INTO predictions (
            predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
            teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
            teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome
        """
        params = (
            prediction['predictionID'],
            prediction['teamGameID'],
            self.miner_uid,
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
            bt.logging.debug(f"Executing query: {query}")
            bt.logging.debug(f"Query parameters: {params}")
            result = self.db_manager.execute_query(query, params)
            print(f"DEBUG: Prediction added, result: {result}")
            if result:
                inserted_row = result[0] if isinstance(result, list) else result
                self.stats_handler.update_on_prediction({
                    'wager': prediction.get('wager', 0),
                    'predictionDate': prediction.get('predictionDate')
                })
                bt.logging.info(f"Prediction {prediction['predictionID']} added successfully: {inserted_row}")
                return {'status': 'success', 'message': f"Prediction {prediction['predictionID']} added successfully", 'data': inserted_row}
            else:
                bt.logging.error("No row returned after insertion")
                bt.logging.error(f"Database result: {result}")
                return {'status': 'error', 'message': "No row returned after insertion"}
        except Exception as e:
            print(f"DEBUG: Error adding prediction: {str(e)}")
            bt.logging.error(f"Error adding prediction: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'message': f"Error adding prediction: {str(e)}"}

    def get_predictions(self, miner_uid):
        query = """
        SELECT p.predictionID, p.teamGameID, p.minerID, p.predictionDate, p.predictedOutcome,
               p.teamA, p.teamB, p.wager, p.teamAodds, p.teamBodds, p.tieOdds, p.outcome,
               g.teamA as home, g.teamB as away
        FROM predictions p
        JOIN games g ON p.teamGameID = g.externalID
        WHERE p.minerID = %s
        """
        try:
            results = self.db_manager.execute_query(query, (miner_uid,))
            for row in results:
                if row['predictiondate'] is not None:
                    row['predictiondate'] = row['predictiondate'].isoformat() if isinstance(row['predictiondate'], datetime) else row['predictiondate']
            return {row['predictionid']: row for row in results}
        except Exception as e:
            bt.logging.error(f"Error getting predictions: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def process_predictions(self, updated_games: Dict[str, TeamGame], new_games: Dict[str, TeamGame]) -> Dict[str, TeamGamePrediction]:
        bt.logging.trace(f"Processing predictions for {len(updated_games)} updated games and {len(new_games)} new games")
        updated_predictions = self.process_game_results(updated_games)
        recent_predictions = self.get_recent_predictions()
        result = {pred.predictionID: pred for pred in recent_predictions}
        bt.logging.trace(f"Processed {len(result)} predictions")
        return result
    
    def process_model_predictions(self, games: Dict[str, TeamGame], sport: str) -> Dict[str, TeamGamePrediction]:
        if sport not in self.models:
            bt.logging.warning(f"Model for sport {sport} not found, skipping model predictions")
            return {}
        
        model = self.models[sport]
        predictions = {}

        home_teams = [game.teamA for game in games.values()]
        away_teams = [game.teamB for game in games.values()]
        odds = [[game.teamAodds, game.tieOdds, game.teamBodds] for game in games.values()]

        encoded_teams = set(model.le.classes_)
        matched_home_teams = [self.get_best_match(team, encoded_teams) for team in home_teams]
        matched_away_teams = [self.get_best_match(team, encoded_teams) for team in away_teams]

        if None not in matched_home_teams + matched_away_teams:
            model_predictions = model.predict_games(matched_home_teams, matched_away_teams, odds)
            
            for (game_id, game), prediction in zip(games.items(), model_predictions):
                pred_dict = {
                    'predictionID': str(uuid.uuid4()),
                    'teamGameID': game_id,
                    'minerID': self.miner_uid,
                    'predictionDate': datetime.now(timezone.utc).isoformat(),
                    'predictedOutcome': game.teamA if prediction['PredictedOutcome'] == 'Home Win' else game.teamB if prediction['PredictedOutcome'] == 'Away Win' else 'Tie',
                    'wager': prediction['recommendedWager'],
                    'teamAodds': game.teamAodds,
                    'teamBodds': game.teamBodds,
                    'tieOdds': game.tieOdds,
                    'outcome': 'unfinished',
                    'teamA': game.teamA,
                    'teamB': game.teamB
                }
                predictions[game_id] = TeamGamePrediction(**pred_dict)
                self.add_prediction(pred_dict)
        else:
            bt.logging.warning(f"Some teams not found in label encoder for {sport}, skipping model predictions.")

        return predictions

    def get_best_match(self, team_name, encoded_teams):
        match, score = process.extractOne(team_name, encoded_teams)
        if score >= 80:
            return match
        else:
            return None

    def process_game_results(self, updated_games: Dict[str, TeamGame]):
        bt.logging.trace(f"Processing game results for {len(updated_games)} games")
        updated_predictions = []
        batch_updates = []

        for external_id, game_data in updated_games.items():
            predictions = self.get_predictions_for_game(external_id)
            
            for pred in predictions:
                processed_prediction = self.process_game_outcome(pred, game_data)
                if processed_prediction: 
                    updated_predictions.append(processed_prediction)
                    if processed_prediction.outcome != "Unfinished":
                        self.stats_handler.update_on_game_result({
                            'outcome': processed_prediction.outcome,
                            'earnings': self.calculate_earnings(processed_prediction.wager, processed_prediction.predictedOutcome, game_data.outcome),
                            'wager': processed_prediction.wager
                        })

        if batch_updates:
            query = "UPDATE predictions SET outcome = %s WHERE predictionID = %s"
            self.db_manager.execute_batch(query, batch_updates)

        bt.logging.trace(f"Processed {len(updated_predictions)} predictions")
        return updated_predictions

    def process_game_outcome(self, prediction: TeamGamePrediction, game_data: TeamGame) -> Optional[TeamGamePrediction]:
        bt.logging.trace(f"Processing game outcome for prediction: {prediction.predictionID}, game: {game_data.id}")
        
        if game_data.outcome is None or game_data.outcome == "Unfinished":
            return None

        actual_outcome = self._map_game_outcome(game_data.outcome)
        predicted_outcome = self._map_predicted_outcome(prediction.predictedOutcome, game_data)

        bt.logging.debug(f"Actual outcome: {actual_outcome}, Predicted outcome: {predicted_outcome}")

        if actual_outcome == predicted_outcome:
            new_outcome = "Wager Won"
        else:
            new_outcome = "Wager Lost"

        if prediction.outcome != new_outcome:
            query = "UPDATE predictions SET outcome = %s WHERE predictionID = %s"
            self.db_manager.execute_query(query, (new_outcome, prediction.predictionID))
            prediction.outcome = new_outcome
            bt.logging.info(f"Updated prediction {prediction.predictionID} outcome to {new_outcome}")

            # Calculate earnings
            earnings = self._calculate_earnings(prediction, game_data, new_outcome)

            # Update stats
            self.stats_handler.update_on_game_result({
                'outcome': new_outcome,
                'earnings': earnings,
                'wager': prediction.wager
            })

            return prediction
        return None

    def _map_game_outcome(self, outcome):
        if outcome in [0, "0", "Team A Win"]:
            return "Team A Win"
        elif outcome in [1, "1", "Team B Win"]:
            return "Team B Win"
        elif outcome in [2, "2", "Tie"]:
            return "Tie"
        else:
            bt.logging.warning(f"Unknown game outcome: {outcome}")
            return "Unknown"

    def _map_predicted_outcome(self, outcome, game_data):
        if outcome in ["Team A Win", "Home Win"] or outcome == game_data.teamA:
            return "Team A Win"
        elif outcome in ["Team B Win", "Away Win"] or outcome == game_data.teamB:
            return "Team B Win"
        elif "Tie" in outcome:
            return "Tie"
        else:
            bt.logging.warning(f"Unknown predicted outcome: {outcome}")
            return "Unknown"

    def calculate_earnings(self, wager: float, prediction: Union[str, TeamGamePrediction], result: Union[int, str]) -> float:
        bt.logging.trace(f"Calculating earnings for wager {wager}, prediction {prediction}, result {result}")
        if result == "Unfinished":
            return 0.0
        
        if isinstance(prediction, str):
            predicted_team = prediction
            odds = 1.0  # Default odds
        else:
            predicted_team = prediction.predictedOutcome
            if predicted_team == prediction.teamA:
                odds = prediction.teamAodds
            elif predicted_team == prediction.teamB:
                odds = prediction.teamBodds
            else:
                odds = prediction.tieOdds

        bt.logging.trace(f"Predicted Team: {predicted_team}, Odds: {odds}")
        bt.logging.trace(f"Result: {result}")
        
        if (predicted_team == "Team A Win" and result == 0) or \
           (predicted_team == "Team B Win" and result == 1) or \
           (predicted_team == "Tie" and result == 2):
            earnings = wager * odds 
        else:
            earnings = 0

        # Update stats and state
        self.stats_handler.update_on_game_result({
            'outcome': 'Wager Won' if earnings > 0 else 'Wager Lost',
            'earnings': earnings,
            'wager': wager
        })

        return earnings

    def get_recent_predictions(self, limit=100):
        bt.logging.trace(f"Getting recent predictions (limit: {limit})")
        query = """
        SELECT predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
               teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome
        FROM predictions
        WHERE minerID = %s
        ORDER BY predictionDate DESC
        LIMIT %s
        """
        results = self.db_manager.execute_query(query, (self.miner_uid, limit))
        predictions = []
        for row in results:
            try:
                if row['predictiondate'] is not None:
                    row['predictiondate'] = row['predictiondate'].isoformat() if isinstance(row['predictiondate'], datetime) else row['predictiondate']
                prediction = TeamGamePrediction(
                    predictionID=row['predictionid'],
                    teamGameID=row['teamgameid'],
                    minerID=row['minerid'],
                    predictionDate=row['predictiondate'],
                    predictedOutcome=row['predictedoutcome'],
                    teamA=row['teama'],
                    teamB=row['teamb'],
                    wager=float(row['wager']),
                    teamAodds=float(row['teamaodds']),
                    teamBodds=float(row['teambodds']),
                    tieOdds=float(row['tieodds']) if row['tieodds'] is not None else None,
                    outcome=row['outcome']
                )
                predictions.append(prediction)
            except Exception as e:
                bt.logging.error(f"Error processing prediction: {e}")
                bt.logging.error(f"Row data: {row}")
        return predictions

    def get_predictions_for_game(self, external_id: str) -> List[TeamGamePrediction]:
        bt.logging.trace(f"Getting predictions for game: {external_id}")
        query = """
        SELECT predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
               teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome
        FROM predictions
        WHERE teamGameID = %s
        """
        try:
            results = self.db_manager.execute_query(query, (external_id,))
            predictions = []
            for row in results:
                try:
                    prediction = TeamGamePrediction(
                        predictionID=row['predictionid'],
                        teamGameID=row['teamgameid'],
                        minerID=row['minerid'],
                        predictionDate=row['predictiondate'],
                        predictedOutcome=row['predictedoutcome'],
                        teamA=row['teama'],
                        teamB=row['teamb'],
                        wager=float(row['wager']),
                        teamAodds=float(row['teamaodds']),
                        teamBodds=float(row['teambodds']),
                        tieOdds=float(row['tieodds']) if row['tieodds'] is not None else None,
                        outcome=row['outcome']
                    )
                    predictions.append(prediction)
                except Exception as e:
                    bt.logging.error(f"Error creating TeamGamePrediction: {e}")
                    bt.logging.error(f"Row data: {row}")
            return predictions
        except Exception as e:
            bt.logging.error(f"Error getting predictions for game: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return []

    def get_predictions_with_teams(self, miner_uid):
        bt.logging.debug(f"Getting predictions with teams for miner: {miner_uid}")
        query = """
        SELECT p.*, g.teamA as home, g.teamB as away
        FROM predictions p
        JOIN games g ON p.teamGameID = g.externalID
        WHERE p.minerID = %s
        ORDER BY p.predictionDate DESC
        """
        try:
            results = self.db_manager.execute_query(query, (miner_uid,))
            #print(f"Retrieved {len(results)} predictions with teams")
            return {row['predictionid']: row for row in results}
        except Exception as e:
            #print(f"Error getting predictions with teams: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return {}

    def calculate_payout(self, wager: float, predicted_outcome: str, team_a_odds: float, team_b_odds: float, tie_odds: float, outcome: str) -> float:
        if outcome == "Unfinished":
            return 0.0
        
        if predicted_outcome == "Team A Win":
            odds = team_a_odds
        elif predicted_outcome == "Team B Win":
            odds = team_b_odds
        else:
            odds = tie_odds

        if outcome == "Wager Won":
            return wager * odds
        else:
            return 0.0