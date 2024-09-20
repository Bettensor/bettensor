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
            "nfl": NFLPredictor(
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
        SET minerID = %s
        WHERE minerID IS NULL
        """
        try:
            self.db_manager.execute_query(query, params=(self.miner_uid,))
            bt.logging.debug("Successfully updated predictions with miner ID")
        except Exception as e:
            bt.logging.error(f"Error updating predictions with miner ID: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    def add_prediction(self, prediction: Dict[str, Any]):
        # print(f"DEBUG: Adding prediction: {prediction}")
        # Deduct wager before adding prediction
        wager_amount = prediction.get("wager", 0)
        bt.logging.debug(f"Attempting to deduct wager: {wager_amount}")
        if not self.stats_handler.deduct_wager(wager_amount):
            bt.logging.warning(f"Insufficient funds to place wager of {wager_amount}")
            return {
                "status": "error",
                "message": f"Insufficient funds to place wager of {wager_amount}",
            }

        query = """
        INSERT INTO predictions (
            predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
            teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome,
            validators_sent_to, validators_confirmed
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 0)
        RETURNING predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
            teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome,
            validators_sent_to, validators_confirmed
        """
        params = (
            prediction["predictionID"],
            prediction["teamGameID"],
            self.miner_uid,
            prediction["predictionDate"],
            prediction["predictedOutcome"],
            prediction["teamA"],
            prediction["teamB"],
            prediction["wager"],
            prediction["teamAodds"],
            prediction["teamBodds"],
            prediction["tieOdds"],
            prediction["outcome"],
        )

        try:
            bt.logging.debug(f"Executing query: {query}")
            bt.logging.debug(f"Query parameters: {params}")
            result = self.db_manager.execute_query(query, params)
            # print(f"DEBUG: Prediction added, result: {result}")
            if result:
                inserted_row = result[0] if isinstance(result, list) else result
                self.stats_handler.update_on_prediction(
                    {
                        "wager": prediction.get("wager", 0),
                        "predictionDate": prediction.get("predictionDate"),
                    }
                )
                bt.logging.debug(
                    f"Prediction {prediction['predictionID']} added successfully: {inserted_row}"
                )
                return {
                    "status": "success",
                    "message": f"Prediction {prediction['predictionID']} added successfully",
                    "data": inserted_row,
                }
            else:
                bt.logging.error("No row returned after insertion")
                bt.logging.error(f"Database result: {result}")
                return {"status": "error", "message": "No row returned after insertion"}
        except Exception as e:
            # print(f"DEBUG: Error adding prediction: {str(e)}")
            bt.logging.error(f"Error adding prediction: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": f"Error adding prediction: {str(e)}"}

    def update_prediction_sent(self, prediction_id: str):
        query = """
        UPDATE predictions
        SET validators_sent_to = validators_sent_to + 1
        WHERE predictionID = %s
        """
        self.db_manager.execute_query(query, (prediction_id,))

    def update_prediction_confirmations(self, prediction_ids: List[str], validator_hotkey: str):
        placeholders = ','.join(['%s'] * len(prediction_ids))
        query = f"""
        UPDATE predictions
        SET validators_confirmed = validators_confirmed + 1
        WHERE predictionID IN ({placeholders})
        AND validators_confirmed < validators_sent_to
        """
        params = prediction_ids
        self.db_manager.execute_query(query, params)

    def get_predictions(self, miner_uid):
        query = """
        SELECT p.predictionID, p.teamGameID, p.minerID, p.predictionDate, p.predictedOutcome,
               p.teamA, p.teamB, p.wager, p.teamAodds, p.teamBodds, p.tieOdds, p.outcome,
               g.teamA as home, g.teamB as away
        FROM predictions p
        JOIN games g ON p.teamGameID = g.externalID
        WHERE p.minerID = %s 
        ORDER BY p.predictionDate DESC 
        LIMIT 100
        """
        try:
            results = self.db_manager.execute_query(query, (miner_uid,))
            predictions = {}
            for row in results:
                if row["predictiondate"] is not None:
                    row["predictiondate"] = (
                        row["predictiondate"].isoformat()
                        if isinstance(row["predictiondate"], datetime)
                        else row["predictiondate"]
                    )
                predictions[row["predictionid"]] = TeamGamePrediction(**row)
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
        else:  # NFL
            encoded_teams = model.bet365_teams

        matched_games = []
        for game_id, game in games.items():
            if sport == "soccer":
                home_match = self.get_best_match(game.teamA, encoded_teams, sport)
                away_match = self.get_best_match(game.teamB, encoded_teams, sport)
            else:
                home_match = model.db_manager.predictions_handler.get_best_match(
                    game.teamA, encoded_teams, sport
                )
                away_match = model.db_manager.predictions_handler.get_best_match(
                    game.teamB, encoded_teams, sport
                )

            if home_match and away_match:
                matched_games.append(
                    {
                        "game_id": game_id,
                        "home_team": home_match,
                        "away_team": away_match,
                        "odds": [game.teamAodds, game.tieOdds, game.teamBodds],
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
                pred_dict = {
                    "predictionID": str(uuid.uuid4()),
                    "teamGameID": game_data["game_id"],
                    "minerID": self.miner_uid,
                    "predictionDate": datetime.now(timezone.utc).isoformat(),
                    "predictedOutcome": game.teamA
                    if prediction["PredictedOutcome"] == "Home Win"
                    else game.teamB
                    if prediction["PredictedOutcome"] == "Away Win"
                    else "Tie",
                    "wager": float(
                        prediction["recommendedWager"]
                    ),  # Convert to Python float
                    "teamAodds": float(game.teamAodds),  # Convert to Python float
                    "teamBodds": float(game.teamBodds),  # Convert to Python float
                    "tieOdds": float(game.tieOdds)
                    if game.tieOdds is not None
                    else None,  # Convert to Python float
                    "outcome": "unfinished",
                    "teamA": game.teamA,
                    "teamB": game.teamB,
                }
                predictions[game_data["game_id"]] = TeamGamePrediction(**pred_dict)
                self.add_prediction(pred_dict)
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
            SELECT predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
                   teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome
            FROM predictions
            WHERE teamGameID = %s AND outcome = 'Unfinished'
            """
            results = self.db_manager.execute_query(query, (game_id,))
            for row in results:
                try:
                    prediction_data = {
                        "predictionID": row["predictionid"],
                        "teamGameID": row["teamgameid"],
                        "minerID": row["minerid"],
                        "predictionDate": row["predictiondate"].isoformat()
                        if isinstance(row["predictiondate"], datetime)
                        else row["predictiondate"],
                        "predictedOutcome": row["predictedoutcome"],
                        "teamA": row["teama"],
                        "teamB": row["teamb"],
                        "wager": float(row["wager"]),
                        "teamAodds": float(row["teamaodds"]),
                        "teamBodds": float(row["teambodds"]),
                        "tieOdds": float(row["tieodds"])
                        if row["tieodds"] is not None
                        else None,
                        "outcome": row["outcome"],
                    }
                    prediction = TeamGamePrediction(**prediction_data)
                except Exception as e:
                    bt.logging.error(f"Error creating TeamGamePrediction: {e}")
                    bt.logging.error(f"Row data: {row}")
                    continue

                bt.logging.trace(
                    f"Processing game outcome for prediction: {prediction.predictionID}, game: {game.id}"
                )

                actual_outcome = self._map_game_outcome(game.outcome)
                predicted_outcome = self._map_predicted_outcome(
                    prediction.predictedOutcome, game
                )

                bt.logging.debug(
                    f"Actual outcome: {actual_outcome}, Predicted outcome: {predicted_outcome}"
                )

                if actual_outcome == "Unfinished":
                    bt.logging.debug(
                        f"Game {game_id} is still unfinished. Keeping prediction {prediction.predictionID} as Unfinished."
                    )
                    continue
                elif actual_outcome == "Unknown":
                    bt.logging.warning(
                        f"Unknown game outcome '{game.outcome}' for game {game_id}. Keeping prediction {prediction.predictionID} as Unfinished."
                    )
                    continue

                new_outcome = (
                    "Wager Won" if actual_outcome == predicted_outcome else "Wager Lost"
                )
                bt.logging.info(
                    f"Updated prediction {prediction.predictionID} outcome to {new_outcome}"
                )

                self.update_prediction_outcome(prediction.predictionID, new_outcome)
                prediction.outcome = new_outcome  # Update the prediction object
                updated_predictions[prediction.predictionID] = prediction

                # Calculate earnings
                self._calculate_earnings(prediction, game)

        return updated_predictions

    def process_game_outcome(
        self, prediction: TeamGamePrediction, game_data: TeamGame
    ) -> Optional[TeamGamePrediction]:
        bt.logging.trace(
            f"Processing game outcome for prediction: {prediction.predictionID}, game: {game_data.id}"
        )

        # Check if the prediction already has a non-"Unfinished" outcome
        if prediction.outcome != "Unfinished":
            bt.logging.debug(
                f"Prediction {prediction.predictionID} already processed. Skipping."
            )
            return prediction

        actual_outcome = self._map_game_outcome(game_data.outcome)
        predicted_outcome = self._map_predicted_outcome(
            prediction.predictedOutcome, game_data
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
            query = "UPDATE predictions SET outcome = %s WHERE predictionID = %s"
            self.db_manager.execute_query(query, (new_outcome, prediction.predictionID))
            prediction.outcome = new_outcome
            bt.logging.info(
                f"Updated prediction {prediction.predictionID} outcome to {new_outcome}"
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
        elif outcome in [
            "Unfinished",
            "unfinished",
            None,
        ]:  # Added None as a possible "Unfinished" state
            return "Unfinished"
        else:
            bt.logging.warning(f"Unknown game outcome: {outcome}")
            return "Unknown"

    def _map_predicted_outcome(self, outcome, game_data):
        if isinstance(game_data, dict):
            team_a = game_data.get("teama") or game_data.get("teamA")
            team_b = game_data.get("teamb") or game_data.get("teamB")
        else:
            team_a = getattr(game_data, "teamA", None)
            team_b = getattr(game_data, "teamB", None)

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
            predicted_team = prediction.predictedOutcome
            if predicted_team == prediction.teamA:
                odds = prediction.teamAodds
            elif predicted_team == prediction.teamB:
                odds = prediction.teamBodds
            else:
                odds = prediction.tieOdds

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

    def get_recent_predictions(self, limit=100):
        bt.logging.trace(f"Getting recent predictions (limit: {limit})")
        query = """
        SELECT predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
               teamA, teamB, wager, teamAodds, teamBodds, tieOdds, outcome
        FROM predictions
        WHERE minerID = %s AND outcome = 'Unfinished'
        ORDER BY predictionDate DESC
        LIMIT %s
        """
        results = self.db_manager.execute_query(query, (self.miner_uid, limit))
        predictions = {}
        for row in results:
            try:
                if row["predictiondate"] is not None:
                    row["predictiondate"] = (
                        row["predictiondate"].isoformat()
                        if isinstance(row["predictiondate"], datetime)
                        else row["predictiondate"]
                    )
                prediction = TeamGamePrediction(
                    predictionID=row["predictionid"],
                    teamGameID=row["teamgameid"],
                    minerID=row["minerid"],
                    predictionDate=row["predictiondate"],
                    predictedOutcome=row["predictedoutcome"],
                    teamA=row["teama"],
                    teamB=row["teamb"],
                    wager=float(row["wager"]),
                    teamAodds=float(row["teamaodds"]),
                    teamBodds=float(row["teambodds"]),
                    tieOdds=float(row["tieodds"])
                    if row["tieodds"] is not None
                    else None,
                    outcome=row["outcome"],
                )
                predictions[row["predictionid"]] = prediction
            except Exception as e:
                bt.logging.error(f"Error processing prediction: {e}")
                bt.logging.error(f"Row data: {row}")
        bt.logging.info(f"Retrieved {len(predictions)} recent predictions")
        return predictions

    def get_predictions_for_game(self, external_id: str) -> List[TeamGamePrediction]:
        # bt.logging.trace(f"Getting predictions for game: {external_id}")
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
                        predictionID=row["predictionid"],
                        teamGameID=row["teamgameid"],
                        minerID=row["minerid"],
                        predictionDate=row["predictiondate"],
                        predictedOutcome=row["predictedoutcome"],
                        teamA=row["teama"],
                        teamB=row["teamb"],
                        wager=float(row["wager"]),
                        teamAodds=float(row["teamaodds"]),
                        teamBodds=float(row["teambodds"]),
                        tieOdds=float(row["tieodds"])
                        if row["tieodds"] is not None
                        else None,
                        outcome=row["outcome"],
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
            # print(f"Retrieved {len(results)} predictions with teams")
            return {row["predictionid"]: row for row in results}
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
        SELECT p.predictionID, p.teamGameID, p.outcome as prediction_outcome, 
               p.predictedOutcome, g.outcome as game_outcome, 
               p.teamA, p.teamB, p.teamAodds, p.teamBodds, p.tieOdds
        FROM predictions p
        JOIN games g ON p.teamGameID = g.externalID
        WHERE p.outcome != 'Unfinished'
        """
        try:
            results = self.db_manager.execute_query(query)
            bt.logging.info(f"Found {len(results)} predictions to check")
            corrections = []
            for row in results:
                prediction_id = row["predictionid"]
                prediction_outcome = row["prediction_outcome"]
                predicted_outcome = row["predictedoutcome"]
                game_outcome = row["game_outcome"]

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
                    "UPDATE predictions SET outcome = %s WHERE predictionID = %s"
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
        query = "UPDATE predictions SET outcome = %s WHERE predictionID = %s"
        self.db_manager.execute_query(query, (new_outcome, prediction_id))
        bt.logging.info(f"Updated prediction {prediction_id} outcome to {new_outcome}")

    def _calculate_earnings(self, prediction: TeamGamePrediction, game: TeamGame):
        bt.logging.trace(
            f"Calculating earnings for prediction {prediction.predictionID}"
        )

        earnings = 0
        if prediction.outcome == "Wager Won":
            if prediction.predictedOutcome == game.teamA:
                odds = prediction.teamAodds
            elif prediction.predictedOutcome == game.teamB:
                odds = prediction.teamBodds
            else:  # Tie
                odds = prediction.tieOdds

            earnings = prediction.wager * odds
        elif prediction.outcome == "Wager Lost":
            earnings = 0  # No change in earnings for a lost wager
        else:
            bt.logging.warning(
                f"Unexpected outcome {prediction.outcome} for prediction {prediction.predictionID}"
            )
            return

        bt.logging.info(
            f"Earnings for prediction {prediction.predictionID}: {earnings}"
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
