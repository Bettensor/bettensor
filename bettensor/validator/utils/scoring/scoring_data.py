import pytz
import bittensor as bt
import numpy as np
from datetime import datetime, timedelta
from bettensor.validator.utils.database.database_manager import DatabaseManager
from typing import List, Tuple, Dict


class ScoringData:
    def __init__(self, db_manager, num_miners):
        self.db_manager = db_manager
        self.num_miners = num_miners

    def preprocess_for_scoring(self, date_str):
        bt.logging.debug(f"Preprocessing for scoring on date: {date_str}")

        # Step 1: Get closed games for that day (Outcome != 3)
        closed_games = self._fetch_closed_game_data(date_str)

        if not closed_games:
            bt.logging.warning("No closed games found for the given date.")
            return np.array([]), np.array([]), np.array([])

        game_ids = [game["external_id"] for game in closed_games]

        # Step 2: Get all predictions for each of those closed games
        predictions = self._fetch_predictions(game_ids)

        # Step 3: Ensure predictions have their payout calculated and outcome updated
        self._update_predictions_with_payout(predictions, closed_games)

        # Step 4: Structure prediction data into the format necessary for scoring
        structured_predictions = np.array(
            [
                [
                    int(pred["miner_uid"]),
                    int(pred["game_id"]),
                    int(pred["predicted_outcome"]),
                    float(pred["predicted_odds"]),
                    float(pred["payout"]) if pred["payout"] is not None else 0.0,
                    float(pred["wager"]),
                ]
                for pred in predictions
                if pred["game_id"] in game_ids
            ]
        )

        results = np.array([game["outcome"] for game in closed_games])

        closing_line_odds = np.array(
            [
                [
                    float(game["team_a_odds"]),
                    float(game["team_b_odds"]),
                    float(game["tie_odds"]) if game["tie_odds"] is not None else 0.0,
                ]
                for game in closed_games
            ]
        )

        bt.logging.debug(
            f"Structured predictions shape: {structured_predictions.shape}"
        )
        bt.logging.debug(f"Closing line odds shape: {closing_line_odds.shape}")
        bt.logging.debug(f"Results shape: {results.shape}")

        return structured_predictions, closing_line_odds, results

    def _fetch_closed_game_data(self, date_str):
        """
        Fetch all closed games for the given date.
        """
        query = """
            SELECT 
                game_id,
                external_id,
                team_a,
                team_b,
                team_a_odds,
                team_b_odds,
                tie_odds,
                can_tie,
                event_start_date,
                create_date,
                last_update_date,
                sport,
                league,
                outcome,
                active
            FROM game_data
            WHERE DATE(event_start_date) = DATE(?) AND active = 0
        """
        return self.db_manager.fetch_all(query, (date_str,))

    def _fetch_predictions(self, game_ids):
        query = """
        SELECT * FROM predictions
        WHERE game_id IN ({})
        """.format(
            ",".join(["?"] * len(game_ids))
        )
        return self.db_manager.fetch_all(query, game_ids)

    def _update_predictions_with_payout(self, predictions, closed_games):
        closed_game_ids = {game["external_id"] for game in closed_games}

        game_outcomes = {game["external_id"]: game["outcome"] for game in closed_games}
        for pred in predictions:
            game_id = pred["game_id"]
            outcome = game_outcomes.get(game_id)
            if outcome is not None and pred["payout"] is None:
                wager = float(pred["wager"])
                predicted_outcome = pred["predicted_outcome"]
                if int(predicted_outcome) == int(outcome):
                    payout = wager * float(pred["predicted_odds"])
                    # bt.logging.debug(f"Prediction {pred['prediction_id']} is correct, setting payout to {payout}")
                else:
                    payout = 0.0
                self.db_manager.execute_query(
                    """
                    UPDATE predictions
                    SET payout = ?, outcome = ?
                    WHERE prediction_id = ?
                    """,
                    (payout, outcome, pred["prediction_id"]),
                )

    def validate_data_integrity(self):
        """Validate that all predictions reference closed games with valid outcomes."""
        invalid_predictions = self.db_manager.fetch_all(
            """
            SELECT p.prediction_id, p.game_id, g.outcome
            FROM predictions p
            LEFT JOIN game_data g ON p.game_id = g.external_id
            WHERE g.outcome IS NULL OR g.external_id IS NULL
            """,
            (),
        )

        if invalid_predictions:
            bt.logging.error(f"Invalid predictions found: {len(invalid_predictions)}")
            bt.logging.error(
                f"Sample of invalid predictions: {invalid_predictions[:5]}"
            )
            raise ValueError(
                "Data integrity check failed: Some predictions reference invalid or open games."
            )
        else:
            bt.logging.debug("All predictions reference valid closed games.")
