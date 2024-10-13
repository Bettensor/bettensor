import traceback
from typing import Any, List, Dict, Optional, Union
import uuid
from bettensor.protocol import TeamGamePrediction, TeamGame
import warnings
from eth_utils.exceptions import ValidationError

warnings.filterwarnings("ignore", message="Network .* does not have a valid ChainId.*")
import bittensor as bt
from datetime import datetime, timezone, timedelta
import psycopg2
from bettensor.miner.models.model_utils import SoccerPredictor, NFLPredictor
from fuzzywuzzy import process
from psycopg2.extras import RealDictCursor
from bettensor.miner.stats.miner_stats import MinerStatsHandler
import numpy as np


class PredictionsHandler:
    def __init__(self, db_manager, state_manager=None, miner_hotkey: str = None):
        bt.logging.debug("PredictionsHandler initialized")
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = state_manager.miner_uid if state_manager else None
        self.new_prediction_window = timedelta(hours=24)
        self.stats_handler = MinerStatsHandler(state_manager)
        self.models = {
            "soccer": SoccerPredictor(
                model_name="podos_soccer_model",
                id=self.miner_uid,
                db_manager=self.db_manager,
                miner_stats_handler=self.stats_handler,
            ),
            "football": NFLPredictor(
                model_name="nfl_wager_model",
                id=self.miner_uid,
                db_manager=self.db_manager,
                miner_stats_handler=self.stats_handler,
            ),
        }
        self.update_predictions_with_minerid()
        bt.logging.trace("PredictionsHandler initialization complete")
        

    def update_predictions_with_minerid(self):
        if not self.miner_uid:
            bt.logging.debug("No miner_uid available, skipping prediction update")
            return
        bt.logging.trace("Updating predictions with miner ID")
        query = """
        UPDATE predictions
        SET miner_uid = %s
        WHERE miner_uid IS NULL
        """
        try:
            self.db_manager.execute_query(query, params=(self.miner_uid,))
            bt.logging.debug("Successfully updated predictions with miner ID")
        except Exception as e:
            bt.logging.error(f"Error updating predictions with miner ID: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    def add_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:

        # check miner cash to ensure they have enough to make the wager
        miner_cash = self.state_manager.get_stats()['miner_cash']
        if miner_cash < prediction['wager']:
            bt.logging.warning(f"Miner {self.miner_uid} does not have enough cash to make the wager,skipping prediction")
            return {"status": "error", "message": "Miner does not have enough cash to make the wager"}

        #bt.logging.debug(f"Adding prediction: {prediction}")
        prediction_id = str(uuid.uuid4())
        prediction["prediction_id"] = prediction_id

        # Calculate predicted_odds based on the predicted_outcome
        if prediction["predicted_outcome"] == prediction["team_a"]:
            predicted_odds = prediction["team_a_odds"]
        elif prediction["predicted_outcome"] == prediction["team_b"]:
            predicted_odds = prediction["team_b_odds"]
        else:
            predicted_odds = prediction["tie_odds"]

        query = """
        INSERT INTO predictions (prediction_id, game_id, miner_uid, prediction_date, predicted_outcome,
                                 team_a, team_b, wager, team_a_odds, team_b_odds, tie_odds,
                                 model_name, confidence_score, outcome, predicted_odds, payout)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            prediction_id,
            prediction["game_id"],
            prediction["miner_uid"],
            prediction["prediction_date"],
            prediction["predicted_outcome"],
            prediction["team_a"],
            prediction["team_b"],
            prediction["wager"],
            prediction["team_a_odds"],
            prediction["team_b_odds"],
            prediction["tie_odds"],
            prediction.get("model_name"),
            prediction.get("confidence_score"),
            "Unfinished",
            predicted_odds,
            0.0  # Initial payout is 0
        )

        self.db_manager.execute_query(query, params)
        #bt.logging.info(f"Prediction added with ID: {prediction_id}")
        
        return {"status": "success", "prediction_id": prediction_id}

    def update_prediction_sent(self, prediction_id: str, validator_confirmation_dict,validator_hotkey:str):

        ## validator_confirmation_dict is a dictionary of dictionaries with the following structure:
        ## {
        ##     "prediction_id": {
        ##          validators: {
        ##             "validator_hotkey1": {confirmed: True/False}
        ##             ...
        ##             }
        ##          }
        ##     }
        ## }
        ## Where the "sent" count is the number of validator_hotkeys for a given prediction_id,
        ## and the "confirmed" count is the number of validator_hotkeys that have confirmed the prediction. 
        ## if prediction_id not in validator_confirmation_dict, add prediction and validator hotkey with count 0
        ## this method doesn't update the confirmation count, it only updates the sent count
        is_new = False

        if prediction_id not in validator_confirmation_dict:
            validator_confirmation_dict[prediction_id] = {"validators":{validator_hotkey:{'confirmed':False}}}
            is_new = True
        # if prediction_id in validator_confirmation_dict and validator_hotkey not in validator_confirmation_dict[prediction_id], add validator_hotkey with count 0
        elif validator_hotkey not in validator_confirmation_dict[prediction_id]["validators"]:
            validator_confirmation_dict[prediction_id]["validators"][validator_hotkey] = {'confirmed':False}
            is_new = True

        query = """
        UPDATE predictions
        SET validators_sent_to = validators_sent_to + 1
        WHERE prediction_id = %s
        """
        if is_new:
            self.db_manager.execute_query(query, (prediction_id,))

    def update_prediction_confirmations(self, prediction_ids: List[str], validator_hotkey: str, validator_confirmation_dict: Dict[str, Any]) -> None:
        updated_predictions = []

        for prediction_id in prediction_ids:
            if (prediction_id in validator_confirmation_dict and
                validator_hotkey in validator_confirmation_dict[prediction_id]["validators"]):
                if not validator_confirmation_dict[prediction_id]["validators"][validator_hotkey]['confirmed']:
                    validator_confirmation_dict[prediction_id]["validators"][validator_hotkey]['confirmed'] = True
                    updated_predictions.append(prediction_id)
                    bt.logging.debug(f"Marked prediction {prediction_id} as confirmed for validator {validator_hotkey}")
                else:
                    bt.logging.debug(f"Prediction {prediction_id} already confirmed for validator {validator_hotkey}")
            elif prediction_id == "miner_stats":
                #  ignore miner_stats key, we will use this soon
                continue
            else:
                bt.logging.warning(f"Prediction ID {prediction_id} or validator {validator_hotkey} not found in confirmation dict.")

        if not updated_predictions:
            bt.logging.info("No new confirmations to update.")
            return

        placeholders = ",".join(["%s"] * len(updated_predictions))
        query = f"""
        UPDATE predictions
        SET validators_confirmed = validators_confirmed + 1
        WHERE prediction_id IN ({placeholders})
        AND validators_confirmed < validators_sent_to
        """
        params = updated_predictions

        try:
            self.db_manager.execute_query(query, params)
            bt.logging.debug(f"Updated validators_confirmed for prediction_ids: {updated_predictions}")
        except Exception as e:
            bt.logging.error(f"Error updating validators_confirmed: {e}")

        try:
            self.db_manager.execute_query(query, params)
            bt.logging.debug(f"Updated validators_confirmed for prediction_ids: {updated_predictions}")
        except Exception as e:
            bt.logging.error(f"Error updating validators_confirmed: {e}")

    def get_predictions(self, miner_uid):
        query = """
        SELECT p.prediction_id, p.game_id, p.miner_uid, p.prediction_date, p.predicted_outcome,
               p.team_a, p.team_b, p.wager, p.team_a_odds, p.team_b_odds, p.tie_odds, p.outcome,
               g.team_a as home, g.team_b as away, p.validators_sent_to, p.validators_confirmed,
               p.model_name, p.confidence_score
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.miner_uid = %s 
        ORDER BY p.prediction_date DESC 
        LIMIT 100
        """
        try:
            results = self.db_manager.execute_query(query, (miner_uid,))
            predictions = {}
            for row in results:
                if row["prediction_date"] is not None:
                    row["prediction_date"] = (
                        row["prediction_date"].isoformat()
                        if isinstance(row["prediction_date"], datetime)
                        else row["prediction_date"]
                    )
                predictions[row["prediction_id"]] = TeamGamePrediction(**row)
            return predictions
        except Exception as e:
            bt.logging.error(f"Error getting predictions: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def process_predictions(
        self, updated_games: Dict[str, TeamGame], new_games: Dict[str, TeamGame]
    ) -> Dict[str, TeamGamePrediction]:
        bt.logging.trace(
            f"Processing predictions for {len(updated_games)} updated games and {len(new_games)} new games"
        )
        updated_predictions = self.process_game_results(updated_games)
        recent_predictions = self.get_recent_predictions()
        result = {**recent_predictions, **updated_predictions}
        bt.logging.info(f"Processed {len(result)} predictions")
        return result

    def process_model_predictions(
        self, games: Dict[str, TeamGame], sport: str
    ) -> Dict[str, TeamGamePrediction]:
        if sport not in self.models:
            bt.logging.warning(
                f"Model for sport {sport} not found, skipping model predictions"
            )
            return {}

        model = self.models[sport]
        predictions = {}

        if sport == "soccer":
            encoded_teams = set(model.le.classes_)
        else:  # Football
            encoded_teams = model.bet365_teams

        matched_games = []
        for game_id, game in games.items():
            home_match = self.get_best_match(game.team_a, encoded_teams, sport)
            away_match = self.get_best_match(game.team_b, encoded_teams, sport)

            if home_match and away_match:
                matched_games.append(
                    {
                        "game_id": game_id,
                        "home_team": home_match,
                        "away_team": away_match,
                        "odds": [game.team_a_odds, game.tie_odds, game.team_b_odds],
                        "original_game": game,
                    }
                )

        bt.logging.debug(f"Matched games: {len(matched_games)} out of {len(games)}")

        if matched_games:
            home_teams = [game["home_team"] for game in matched_games]
            away_teams = [game["away_team"] for game in matched_games]
            odds = [game["odds"] for game in matched_games]

            model_predictions = model.predict_games(home_teams, away_teams, odds)

            for game_data, prediction in zip(matched_games, model_predictions):
                game = game_data["original_game"]
                model_name = "NFL Model" if sport == "football" else "Soccer Model"
                pred_dict = {
                    "prediction_id": str(uuid.uuid4()),
                    "game_id": game_data["game_id"],
                    "miner_uid": self.miner_uid,
                    "prediction_date": datetime.now(timezone.utc).isoformat(),
                    "predicted_outcome": game.team_a if prediction["PredictedOutcome"] == "Home Win" else game.team_b if prediction["PredictedOutcome"] == "Away Win" else "Tie",
                    "predicted_odds": game.team_a_odds if prediction["PredictedOutcome"] == "Home Win" else game.team_b_odds if prediction["PredictedOutcome"] == "Away Win" else game.tie_odds,
                    "team_a": game.team_a,
                    "team_b": game.team_b,
                    "wager": float(prediction["recommendedWager"]),
                    "team_a_odds": float(game.team_a_odds),
                    "team_b_odds": float(game.team_b_odds),
                    "tie_odds": float(game.tie_odds) if game.tie_odds is not None else None,
                    "model_name": model_name,
                    "confidence_score": float(prediction["ConfidenceScore"]),
                    "outcome": "Pending",
                    "payout": 0.0, #init payout to 0
                }
                predictions[game_data["game_id"]] = TeamGamePrediction(**pred_dict)

                #sort predictions by confidence score
                

                self.add_prediction(pred_dict)

            # Set the made_daily_predictions flag to True
            model.made_daily_predictions = True
            bt.logging.info(f"Set made_daily_predictions to True for {sport} model")

        else:
            bt.logging.warning(f"No games found with matching team names for {sport}")

        return predictions

    def get_best_match(self, team_name, encoded_teams, sport):
        match, score = process.extractOne(team_name, encoded_teams)
        if score >= self.models[sport].fuzzy_match_percentage:
            return match
        else:
            return None

    def process_game_results(self, game_results: Dict[str, TeamGame]):
        bt.logging.trace(f"Processing game results for {len(game_results)} games")
        updated_predictions = {}
        for game_id, game in game_results.items():
            # bt.logging.debug(f"Processing prediction for game {game_id}")
            query = """
            SELECT prediction_id, game_id, miner_uid, prediction_date, predicted_outcome,
                   team_a, team_b, wager, team_a_odds, team_b_odds, tie_odds, outcome,
                   model_name, confidence_score
            FROM predictions
            WHERE game_id = %s AND outcome = 'Unfinished'
            """
            results = self.db_manager.execute_query(query, (game_id,))
            for row in results:
                try:
                    prediction_data = {
                        "prediction_id": row["prediction_id"],
                        "game_id": row["game_id"],
                        "miner_uid": row["miner_uid"],
                        "prediction_date": row["prediction_date"].isoformat()
                        if isinstance(row["prediction_date"], datetime)
                        else row["prediction_date"],
                        "predicted_outcome": row["predicted_outcome"],
                        "team_a": row["team_a"],
                        "team_b": row["team_b"],
                        "wager": float(row["wager"]),
                        "team_a_odds": float(row["team_a_odds"]),
                        "team_b_odds": float(row["team_b_odds"]),
                        "tie_odds": float(row["tie_odds"])
                        if row["tie_odds"] is not None
                        else None,
                        "outcome": row["outcome"],
                        "model_name": row["model_name"],
                        "confidence_score": float(row["confidence_score"])
                        if row["confidence_score"] is not None and row["confidence_score"] != -1
                        else None,
                        "predicted_odds": float(row["team_a_odds"])
                        if row["predicted_outcome"] == row["team_a"]
                        else float(row["team_b_odds"])
                        if row["predicted_outcome"] == row["team_b"]
                        else float(row["tie_odds"])
                        if row["tie_odds"] is not None
                        else None,
                        "payout": 0.0,  # Initialize payout to 0
                    }
                    prediction = TeamGamePrediction(**prediction_data)
                except Exception as e:
                    bt.logging.error(f"Error creating TeamGamePrediction: {e}")
                    bt.logging.error(f"Row data: {row}")
                    continue

                bt.logging.trace(
                    f"Processing game outcome for prediction: {prediction.prediction_id}, game: {game.game_id}"
                )

                actual_outcome = self._map_game_outcome(game.outcome)
                predicted_outcome = self._map_predicted_outcome(
                    prediction.predicted_outcome, game
                )

                bt.logging.debug(
                    f"Actual outcome: {actual_outcome}, Predicted outcome: {predicted_outcome}"
                )

                if actual_outcome == "Unfinished":
                    bt.logging.debug(
                        f"Game {game_id} is still unfinished. Keeping prediction {prediction.prediction_id} as Unfinished."
                    )
                    continue
                elif actual_outcome == "Unknown":
                    bt.logging.warning(
                        f"Unknown game outcome '{game.outcome}' for game {game_id}. Keeping prediction {prediction.prediction_id} as Unfinished."
                    )
                    continue

                new_outcome = (
                    "Wager Won" if actual_outcome == predicted_outcome else "Wager Lost"
                )
                bt.logging.info(
                    f"Updated prediction {prediction.prediction_id} outcome to {new_outcome}"
                )

                self.update_prediction_outcome(prediction.prediction_id, new_outcome)
                prediction.outcome = new_outcome  # Update the prediction object
                updated_predictions[prediction.prediction_id] = prediction

                # Calculate earnings
                self._calculate_earnings(prediction, game)

        return updated_predictions

    def process_game_outcome(
        self, prediction: TeamGamePrediction, game_data: TeamGame
    ) -> Optional[TeamGamePrediction]:
        bt.logging.trace(
            f"Processing game outcome for prediction: {prediction.prediction_id}, game: {game_data.game_id}"
        )

        # Check if the prediction already has a non-"Unfinished" outcome
        if prediction.outcome != "Unfinished":
            bt.logging.debug(
                f"Prediction {prediction.prediction_id} already processed. Skipping."
            )
            return prediction

        actual_outcome = self._map_game_outcome(game_data.outcome)
        predicted_outcome = self._map_predicted_outcome(
            prediction.predicted_outcome, game_data
        )

        bt.logging.debug(
            f"Actual outcome: {actual_outcome}, Predicted outcome: {predicted_outcome}"
        )

        if actual_outcome == "Unknown":
            new_outcome = "Unfinished"
        elif actual_outcome == predicted_outcome:
            new_outcome = "Wager Won"
        else:
            new_outcome = "Wager Lost"

        if new_outcome != prediction.outcome:
            query = "UPDATE predictions SET outcome = %s WHERE prediction_id = %s"
            self.db_manager.execute_query(query, (new_outcome, prediction.prediction_id))
            prediction.outcome = new_outcome
            bt.logging.info(
                f"Updated prediction {prediction.prediction_id} outcome to {new_outcome}"
            )

            if new_outcome != "Unfinished":
                # Calculate earnings
                earnings = self.calculate_earnings(
                    prediction.wager, prediction, game_data.outcome
                )

                # Update stats
                self.stats_handler.update_on_game_result(
                    {
                        "outcome": new_outcome,
                        "earnings": earnings,
                        "wager": prediction.wager,
                    }
                )

        return prediction

    def _map_game_outcome(self, outcome):
        if outcome in [0, "0", "Team A Win"]:
            return "Team A Win"
        elif outcome in [1, "1", "Team B Win"]:
            return "Team B Win"
        elif outcome in [2, "2", "Tie"]:
            return "Tie"
        elif outcome in [3, "3", "Unfinished", None]:
            return "Unfinished"
        else:
            bt.logging.warning(f"Unknown game outcome: {outcome}")
            return "Unknown"

    def _map_predicted_outcome(self, outcome, game_data):
        if isinstance(game_data, dict):
            team_a = game_data.get("team_a") 
            team_b = game_data.get("team_b")
        else:
            team_a = getattr(game_data, "team_a", None)
            team_b = getattr(game_data, "team_b", None)

        if outcome in ["Team A Win", "Home Win"] or outcome == team_a:
            return "Team A Win"
        elif outcome in ["Team B Win", "Away Win"] or outcome == team_b:
            return "Team B Win"
        elif "Tie" in outcome:
            return "Tie"
        else:
            bt.logging.warning(f"Unknown predicted outcome: {outcome}")
            return "Unknown"

    def calculate_earnings(
        self,
        wager: float,
        prediction: Union[str, TeamGamePrediction],
        result: Union[int, str],
    ) -> float:
        bt.logging.trace(
            f"Calculating earnings for wager {wager}, prediction {prediction}, result {result}"
        )
        if result == "Unfinished":
            return 0.0

        if isinstance(prediction, str):
            predicted_team = prediction
            odds = 1.0  # Default odds
        else:
            predicted_team = prediction.predicted_outcome
            if predicted_team == prediction.team_a:
                odds = prediction.team_a_odds
            elif predicted_team == prediction.team_b:
                odds = prediction.team_b_odds
            else:
                odds = prediction.tie_odds

        bt.logging.trace(f"Predicted Team: {predicted_team}, Odds: {odds}")
        bt.logging.trace(f"Result: {result}")

        if (
            (predicted_team == "Team A Win" and result == 0)
            or (predicted_team == "Team B Win" and result == 1)
            or (predicted_team == "Tie" and result == 2)
        ):
            earnings = wager * odds
        else:
            earnings = 0

        # Update stats and state
        self.stats_handler.update_on_game_result(
            {
                "outcome": "Wager Won" if earnings > 0 else "Wager Lost",
                "earnings": earnings,
                "wager": wager,
            }
        )

        return earnings

    def get_recent_predictions(self, limit: int = 100):
        query = """
        SELECT p.prediction_id, p.game_id, p.miner_uid, p.prediction_date, p.predicted_outcome,
               p.team_a, p.team_b, p.wager, p.team_a_odds, p.team_b_odds, p.tie_odds,
               p.model_name, p.confidence_score, p.outcome, p.predicted_odds, p.payout
        FROM predictions p
        WHERE p.miner_uid = %s
        ORDER BY p.prediction_date DESC
        LIMIT %s
        """
        results = self.db_manager.execute_query(query, (self.miner_uid, limit))
        predictions = {}
        for row in results:
            try:
                if row["prediction_date"] is not None:
                    row["prediction_date"] = (
                        row["prediction_date"].isoformat()
                        if isinstance(row["prediction_date"], datetime)
                        else row["prediction_date"]
                    )
                prediction = TeamGamePrediction(
                    prediction_id=row["prediction_id"],
                    game_id=row["game_id"],
                    miner_uid=row["miner_uid"],
                    prediction_date=row["prediction_date"],
                    predicted_outcome=row["predicted_outcome"],
                    team_a=row["team_a"],
                    team_b=row["team_b"],
                    wager=float(row["wager"]),
                    team_a_odds=float(row["team_a_odds"]),
                    team_b_odds=float(row["team_b_odds"]),
                    tie_odds=float(row["tie_odds"])
                    if row["tie_odds"] is not None
                    else None,
                    outcome=row["outcome"],
                    model_name=row["model_name"],
                    confidence_score=float(row["confidence_score"]) if row["confidence_score"] is not None else None,
                    predicted_odds=float(row["predicted_odds"]) if row["predicted_odds"] is not None else -1,
                    payout=float(row["payout"]) if row["payout"] is not None else 0.0
                )
                predictions[row["prediction_id"]] = prediction
            except Exception as e:
                bt.logging.error(f"Error processing prediction: {e}")
                bt.logging.error(f"Row data: {row}")
        bt.logging.info(f"Retrieved {len(predictions)} recent predictions")
        return predictions

    def get_predictions_for_game(self, game_id: str) -> List[TeamGamePrediction]:

        query = """
        SELECT prediction_id, game_id, miner_uid, prediction_date, predicted_outcome,
               team_a, team_b, wager, team_a_odds, team_b_odds, tie_odds, 
               model_name, confidence_score, outcome, predicted_odds, payout
        FROM predictions
        WHERE game_id = %s
        """
        try:
            results = self.db_manager.execute_query(query, (game_id,))
            predictions = []
            for row in results:
                try:
                    prediction = TeamGamePrediction(
                        prediction_id=row["prediction_id"],
                        game_id=row["game_id"],
                        miner_uid=row["miner_uid"],
                        prediction_date=row["prediction_date"],
                        predicted_outcome=row["predicted_outcome"],
                        team_a=row["team_a"],
                        team_b=row["team_b"],
                        wager=float(row["wager"]),
                        team_a_odds=float(row["team_a_odds"]),
                        team_b_odds=float(row["team_b_odds"]),
                        tie_odds=float(row["tie_odds"]) if row["tie_odds"] is not None else None,
                        model_name=row["model_name"],
                        confidence_score=float(row["confidence_score"]) if row["confidence_score"] is not None else None,
                        outcome=row["outcome"],
                        predicted_odds=float(row["predicted_odds"]) if row["predicted_odds"] is not None else -1,
                        payout=float(row["payout"]) if row["payout"] is not None else 0.0
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
        SELECT p.*, g.team_a as home, g.team_b as away
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.miner_uid = %s
        ORDER BY p.prediction_date DESC
        """
        try:
            results = self.db_manager.execute_query(query, (miner_uid,))
            # print(f"Retrieved {len(results)} predictions with teams")
            return {row["prediction_id"]: row for row in results}
        except Exception as e:
            # print(f"Error getting predictions with teams: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return {}

    def calculate_payout(
        self,
        wager: float,
        predicted_outcome: str,
        team_a_odds: float,
        team_b_odds: float,
        tie_odds: float,
        outcome: str,
    ) -> float:
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

    def check_and_correct_prediction_outcomes(self):
        bt.logging.info("Starting to check and correct prediction outcomes")
        query = """
        SELECT p.prediction_id, p.game_id, p.outcome as prediction_outcome, 
               p.predicted_outcome, g.outcome as game_outcome, 
               p.team_a, p.team_b, p.team_a_odds, p.team_b_odds, p.tie_odds
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.outcome != 'Unfinished'
        """
        try:
            results = self.db_manager.execute_query(query)
            bt.logging.info(f"Found {len(results)} predictions to check")
            corrections = []
            for row in results:
                prediction_id = row["prediction_id"]
                prediction_outcome = row["prediction_outcome"]
                predicted_outcome = row["predicted_outcome"]
                game_outcome = self._map_game_outcome(row["game_outcome"])

                if game_outcome == "Unfinished" or game_outcome is None:
                    corrections.append((prediction_id, "Unfinished"))
                    bt.logging.debug(
                        f"Resetting prediction {prediction_id} to Unfinished as game is not finished"
                    )
                elif prediction_outcome not in ["Wager Won", "Wager Lost"]:
                    corrections.append((prediction_id, "Unfinished"))
                    bt.logging.debug(
                        f"Resetting prediction {prediction_id} to Unfinished due to invalid outcome: {prediction_outcome}"
                    )
                else:
                    # Check if the prediction outcome is correct
                    actual_outcome = self._map_game_outcome(game_outcome)
                    predicted_outcome = self._map_predicted_outcome(
                        predicted_outcome, row
                    )

                    if actual_outcome == "Unknown":
                        corrections.append((prediction_id, "Unfinished"))
                        bt.logging.debug(
                            f"Resetting prediction {prediction_id} to Unfinished due to unknown game outcome"
                        )
                    elif actual_outcome == predicted_outcome:
                        if prediction_outcome != "Wager Won":
                            corrections.append((prediction_id, "Wager Won"))
                            bt.logging.debug(
                                f"Correcting prediction {prediction_id} to Wager Won"
                            )
                    else:
                        if prediction_outcome != "Wager Lost":
                            corrections.append((prediction_id, "Wager Lost"))
                            bt.logging.debug(
                                f"Correcting prediction {prediction_id} to Wager Lost"
                            )

            if corrections:
                update_query = (
                    "UPDATE predictions SET outcome = %s WHERE prediction_id = %s"
                )
                self.db_manager.execute_batch(
                    update_query,
                    [(outcome, pred_id) for pred_id, outcome in corrections],
                )
                bt.logging.info(f"Corrected {len(corrections)} prediction outcomes")
            else:
                bt.logging.info("No prediction outcomes needed correction")

        except Exception as e:
            bt.logging.error(
                f"Error checking and correcting prediction outcomes: {str(e)}"
            )
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

        bt.logging.info("Finished checking and correcting prediction outcomes")

    def update_prediction_outcome(self, prediction_id: str, new_outcome: str):
        bt.logging.debug(
            f"Updating prediction {prediction_id} outcome to {new_outcome}"
        )
        query = "UPDATE predictions SET outcome = %s WHERE prediction_id = %s"
        self.db_manager.execute_query(query, (new_outcome, prediction_id))
        bt.logging.info(f"Updated prediction {prediction_id} outcome to {new_outcome}")

    def _calculate_earnings(self, prediction: TeamGamePrediction, game: TeamGame):
        bt.logging.trace(
            f"Calculating earnings for prediction {prediction.prediction_id}"
        )

        earnings = 0
        if prediction.outcome == "Wager Won":
            if prediction.predicted_outcome == game.team_a:
                odds = prediction.team_a_odds
            elif prediction.predicted_outcome == game.team_b:
                odds = prediction.team_b_odds
            else:  # Tie
                odds = prediction.tie_odds

            earnings = prediction.wager * odds
        elif prediction.outcome == "Wager Lost":
            earnings = 0  # No change in earnings for a lost wager
        else:
            bt.logging.warning(
                f"Unexpected outcome {prediction.outcome} for prediction {prediction.prediction_id}"
            )
            return

        bt.logging.info(
            f"Earnings for prediction {prediction.prediction_id}: {earnings}"
        )

        # Update miner stats
        self.stats_handler.update_miner_earnings(earnings)

        # Update cash (only add earnings for wins, don't subtract wager)
        self.stats_handler.update_miner_cash(earnings)

        if prediction.outcome == "Wager Won":
            self.stats_handler.increment_miner_wins()
        elif prediction.outcome == "Wager Lost":
            self.stats_handler.increment_miner_losses()

        self.stats_handler.update_win_loss_ratio()

    def is_nfl_model_on(self):
        return self.models['football'].nfl_model_on if 'football' in self.models else False

    def is_soccer_model_on(self):
        return self.models['soccer'].soccer_model_on if 'soccer' in self.models else False