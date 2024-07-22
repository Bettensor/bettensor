import datetime
from typing import List, Dict
from bettensor.protocol import TeamGamePrediction, TeamGame
from bettensor.miner.database.database_manager import DatabaseManager
from bettensor.utils.miner_stats import MinerStateManager
import bittensor as bt

class PredictionsHandler:
    def __init__(self, db_manager: DatabaseManager, state_manager: MinerStateManager, miner_uid: str):
        """
        Initialize the PredictionsHandler.

        Args:
            db_manager (DatabaseManager): The database manager instance.
            state_manager (MinerStateManager): The miner state manager instance.
            miner_uid (str): The unique identifier of the miner.

        Behavior:
            - Sets up the database manager, state manager, and miner UID for handling predictions.
        """
        bt.logging.trace(f"Initializing PredictionsHandler for miner: {miner_uid}")
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.miner_uid = miner_uid
        bt.logging.trace("PredictionsHandler initialization complete")

    def process_predictions(self, updated_games: Dict[str, TeamGame], new_games: Dict[str, TeamGame]) -> Dict[str, TeamGamePrediction]:
        """
        Process predictions for updated and new games.

        Args:
            updated_games (Dict[str, TeamGame]): A dictionary of updated games.
            new_games (Dict[str, TeamGame]): A dictionary of new games.

        Returns:
            Dict[str, TeamGamePrediction]: A dictionary of new predictions.

        Behavior:
            - Updates outcomes for existing predictions based on updated games.
            - Creates new predictions for new games.
            - Returns only the newly created predictions.
        """
        bt.logging.trace(f"Processing predictions for {len(updated_games)} updated games and {len(new_games)} new games")
        new_prediction_dict = {}
        
        for game_id, game_data in updated_games.items():
            self._update_prediction_outcome(game_data)
        
        for game_id, game_data in new_games.items():
            new_prediction = self._get_or_create_prediction(game_data)
            if new_prediction:
                new_prediction_dict[new_prediction.predictionID] = new_prediction

        bt.logging.trace(f"Prediction processing complete. New predictions: {len(new_prediction_dict)}")
        return new_prediction_dict

    def _update_prediction_outcome(self, game_data: TeamGame):
        """
        Update the outcome of an existing prediction based on game data.

        Args:
            game_data (TeamGame): The updated game data.

        Behavior:
            - Retrieves the existing prediction for the game.
            - If the game has a final outcome, updates the prediction outcome.
            - Updates the miner's state based on the prediction result.
        """
        bt.logging.trace(f"Updating prediction outcome for game: {game_data.externalId}")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM predictions WHERE teamGameID = ?", 
                (game_data.externalId,)
            )
            prediction = cursor.fetchone()

            if prediction:
                outcome = game_data.outcome
                if outcome != "Unfinished":
                    prediction_obj = TeamGamePrediction(*prediction)
                    if (outcome == 0 and prediction_obj.predictedOutcome == prediction_obj.teamA) or \
                       (outcome == 1 and prediction_obj.predictedOutcome == prediction_obj.teamB) or \
                       (outcome == 2 and prediction_obj.predictedOutcome == "Tie"):
                        result = 'win'
                        earnings = prediction_obj.wager * (prediction_obj.teamAodds if outcome == 0 else 
                                                           prediction_obj.teamBodds if outcome == 1 else 
                                                           prediction_obj.tieOdds)
                    else:
                        result = 'loss'
                        earnings = 0

                    self.state_manager.update_on_game_result({'outcome': result, 'earnings': earnings})

                    cursor.execute(
                        "UPDATE predictions SET outcome = ? WHERE predictionID = ?",
                        (result, prediction_obj.predictionID)
                    )

        bt.logging.trace(f"Prediction outcome updated for game: {game_data.externalId}")

    def _get_or_create_prediction(self, game_data: TeamGame) -> TeamGamePrediction:
        """
        Get an existing prediction or create a new one for a game.

        Args:
            game_data (TeamGame): The game data to create a prediction for.

        Returns:
            TeamGamePrediction: A new prediction if one doesn't exist, None otherwise.

        Behavior:
            - Checks if a prediction already exists for the game.
            - If it doesn't exist, creates a new prediction.
            - Saves the new prediction to the database.
        """
        bt.logging.trace(f"Getting or creating prediction for game: {game_data.externalId}")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM predictions WHERE teamGameID = ?", 
                (game_data.externalId,)
            )
            prediction = cursor.fetchone()

            if prediction:
                bt.logging.trace(f"Prediction retrieved for game: {game_data.externalId}")
                return None  # Prediction already exists, don't create a new one
            else:
                # Create a new prediction
                new_prediction = self._create_new_prediction(game_data)
                self._save_prediction(new_prediction)
                bt.logging.trace(f"Prediction created for game: {game_data.externalId}")
                return new_prediction

    def _create_new_prediction(self, game_data: TeamGame) -> TeamGamePrediction:
        """
        Create a new prediction for a game.

        Args:
            game_data (TeamGame): The game data to create a prediction for.

        Returns:
            TeamGamePrediction: A new prediction object.

        Behavior:
            - Generates a new prediction based on the game data.
            - Uses a placeholder prediction logic (always predicts teamA).
        """
        bt.logging.trace(f"Creating new prediction for game: {game_data.externalId}")
        # Implement your prediction logic here
        # This is a placeholder implementation
        prediction = TeamGamePrediction(
            predictionID=f"{game_data.externalId}_{self.miner_uid}",
            teamGameID=game_data.externalId,
            minerID=self.miner_uid,
            predictionDate=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            predictedOutcome=game_data.teamA,  # Placeholder: always predict teamA
            teamA=game_data.teamA,
            teamB=game_data.teamB,
            wager=1.0,  # Placeholder wager
            teamAodds=game_data.teamAodds,
            teamBodds=game_data.teamBodds,
            tieOdds=game_data.tieOdds,
            can_overwrite=True,
            outcome="Unfinished"
        )
        bt.logging.trace(f"New prediction created for game: {game_data.externalId}")
        return prediction

    def _save_prediction(self, prediction: TeamGamePrediction):
        """
        Save a prediction to the database.

        Args:
            prediction (TeamGamePrediction): The prediction to save.

        Behavior:
            - Inserts the prediction into the database.
            - Updates the miner's state with the new prediction.
        """
        bt.logging.trace(f"Saving prediction: {prediction.predictionID}")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO predictions (
                    predictionID, teamGameID, minerID, predictionDate, predictedOutcome,
                    teamA, teamB, wager, teamAodds, teamBodds, tieOdds, canOverwrite, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.predictionID,
                prediction.teamGameID,
                prediction.minerID,
                prediction.predictionDate,
                prediction.predictedOutcome,
                prediction.teamA,
                prediction.teamB,
                prediction.wager,
                prediction.teamAodds,
                prediction.teamBodds,
                prediction.tieOdds,
                prediction.can_overwrite,
                prediction.outcome
            ))
            cursor.connection.commit()

        self.state_manager.update_on_prediction({'wager': prediction.wager})
        bt.logging.trace(f"Prediction saved: {prediction.predictionID}")

    def get_predictions(self, filters=None) -> Dict[str, TeamGamePrediction]:
        """
        Retrieve predictions from the database.

        Args:
            filters (optional): Filters to apply when retrieving predictions.

        Returns:
            Dict[str, TeamGamePrediction]: A dictionary of predictions, keyed by predictionID.

        Behavior:
            - Retrieves predictions from the database.
            - Applies filters if provided.
            - Returns predictions sorted by prediction date (most recent first).
        """
        bt.logging.trace("Retrieving predictions")
        with self.db_manager.get_cursor() as cursor:
            if filters:
                # TODO: Implement filter logic here
                pass
            else:
                cursor.execute("SELECT * FROM predictions")
            
            predictions_raw = cursor.fetchall()

        prediction_dict = {}
        for prediction in predictions_raw:
            single_prediction = TeamGamePrediction(*prediction)
            prediction_dict[single_prediction.predictionID] = single_prediction

        bt.logging.trace(f"Retrieved {len(prediction_dict)} predictions")
        return dict(sorted(prediction_dict.items(), key=lambda item: item[1].predictionDate, reverse=True))