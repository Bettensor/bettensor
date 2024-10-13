import traceback
import pytz
import bittensor as bt
import numpy as np
from datetime import datetime, timedelta
from bettensor.validator.utils.database.database_manager import DatabaseManager
from typing import List, Tuple, Dict
from collections import defaultdict


class ScoringData:
    def __init__(self, db_manager, num_miners, validator):
        self.db_manager = db_manager
        self.num_miners = num_miners
        self.validator = validator
        self.miner_stats = defaultdict(lambda: {
            'clv': 0.0,
            'roi': 0.0,
            'entropy': 0.0,
            'sortino': 0.0,
            'composite_daily': 0.0,
            'tier_scores': {}
        })
        self.init_miner_stats()

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
        predictions = self._update_predictions_with_payout(predictions, closed_games)

       
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

        bt.logging.debug(f"Structured predictions: {structured_predictions}")

        results = np.array(
            [
                [
                    int(game["external_id"]),
                    int(game["outcome"]),   
                ]
                for game in closed_games
            ]
        )

        closing_line_odds = np.array(
            [
                [
                    int(game["external_id"]),
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

        bt.logging.debug(f"First 5 structured predictions: {structured_predictions[:5]}")
        bt.logging.debug(f"First 5 closing line odds: {closing_line_odds[:5]}")
        bt.logging.debug(f"First 5 results: {results[:5]}")

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
            WHERE DATE(event_start_date) = DATE(?) AND outcome != 3
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
        

        game_outcomes = {game["external_id"]: game["outcome"] for game in closed_games}
        for pred in predictions:
            game_id = pred["game_id"]
            miner_uid = pred["miner_uid"]
            outcome = game_outcomes.get(game_id)
            if outcome is not None and pred["payout"] is None:
                wager = float(pred["wager"])
                predicted_outcome = pred["predicted_outcome"]
                if int(predicted_outcome) == int(outcome):
                    payout = wager * float(pred["predicted_odds"])
                    bt.logging.debug(f"Correct prediction for miner {miner_uid}: Payout set to {payout}")
                else:
                    payout = 0.0
                    bt.logging.debug(f"Incorrect prediction for miner {miner_uid}: Payout set to 0.0")
                self.db_manager.execute_query(
                    """
                    UPDATE predictions
                    SET payout = ?, outcome = ?
                    WHERE prediction_id = ?
                    """,
                    (payout, outcome, pred["prediction_id"]),
                )
                pred["payout"] = payout
                pred["outcome"] = outcome
        return predictions
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

    def init_miner_stats(self, num_miners: int = 256):
            """
            Populate miner_stats table with initial zero values for all miners if the table is empty.

            Args:
                db_manager: The database manager object.
                num_miners (int): The number of miners to initialize. Defaults to 256.
            """
            bt.logging.trace("Initializing Miner Stats")
            try:
                # Check if the miner_stats table is empty
                count = self.db_manager.fetch_one("SELECT COUNT(*) FROM miner_stats")["COUNT(*)"]
                bt.logging.trace(f"Miner stats count: {count}")
                
                if count == 0:
                    bt.logging.info("Initializing miner_stats table with zero values.")
                    
                    # Prepare the insert query
                    insert_query = """
                    INSERT INTO miner_stats (
                        miner_hotkey, miner_coldkey, miner_uid, miner_rank, miner_status,
                        miner_cash, miner_current_incentive, miner_current_tier,
                        miner_current_scoring_window, miner_current_composite_score,
                        miner_current_sharpe_ratio, miner_current_sortino_ratio,
                        miner_current_roi, miner_current_clv_avg, miner_last_prediction_date,
                        miner_lifetime_earnings, miner_lifetime_wager_amount,
                        miner_lifetime_roi, miner_lifetime_predictions,
                        miner_lifetime_wins, miner_lifetime_losses, miner_win_loss_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    # Prepare batch of initial values for all miners
                    initial_values: List[tuple] = [
                        (f"hotkey_{i}", f"coldkey_{i}", i, 0, "active",
                        0.0, 0.0, 0, 0, 0.0,
                        0.0, 0.0, 0.0, 0.0, None,
                        0.0, 0.0, 0.0, 0, 0, 0, 0.0)
                        for i in range(num_miners)
                    ]
                    
                    # Execute batch insert
                    count = self.db_manager.executemany(insert_query, initial_values)
                    bt.logging.trace(f"Inserted {count} rows into miner_stats table.")

                    
                    bt.logging.info(f"Successfully initialized {count} miners in miner_stats table.")
                else:
                    bt.logging.info("miner_stats table is not empty. Skipping initialization.")
            
            except Exception as e:
                bt.logging.error(f"Error initializing miner_stats: {str(e)}")
                raise

    def update_miner_stats(self, current_day):
        """
        Update miner statistics based on the current day's scores.
        
        Args:
            current_day (int): The current day index.
        """
        try:
            self.db_manager.begin_transaction()
            bt.logging.info(f"Updating miner stats for day {current_day}...")

            #get hotkey and coldkey from metagraph
            for miner_uid in range(self.num_miners):
                hotkey = self.validator.metagraph.hotkeys[miner_uid]
                coldkey = self.validator.metagraph.coldkeys[miner_uid]  

                update_keys_query = """
                    UPDATE miner_stats
                    SET
                        miner_hotkey = ?,
                        miner_coldkey = ?
                    WHERE miner_uid = ?
                """
                self.db_manager.execute_query(update_keys_query, (hotkey, coldkey, miner_uid))

            # Fetch and update lifetime statistics
            update_lifetime_query = """
                UPDATE miner_stats
                SET
                    miner_lifetime_earnings = (
                        SELECT COALESCE(SUM(payout), 0)
                        FROM predictions p
                        WHERE p.miner_uid = miner_stats.miner_uid
                    ),
                    miner_lifetime_wager_amount = (
                        SELECT COALESCE(SUM(wager), 0)
                        FROM predictions p
                        WHERE p.miner_uid = miner_stats.miner_uid
                    ),
                    miner_lifetime_predictions = (
                        SELECT COUNT(*)
                        FROM predictions p
                        WHERE p.miner_uid = miner_stats.miner_uid
                    ),
                    miner_lifetime_wins = (
                        SELECT COUNT(*)
                        FROM predictions p
                        WHERE p.miner_uid = miner_stats.miner_uid
                        AND p.payout > 0
                    ),
                    miner_lifetime_losses = (
                        SELECT COUNT(*)
                        FROM predictions p
                        WHERE p.miner_uid = miner_stats.miner_uid
                        AND p.payout = 0
                    ),
                    miner_last_prediction_date = (
                        SELECT MAX(p.prediction_date)
                        FROM predictions p
                        WHERE p.miner_uid = miner_stats.miner_uid
                    )
            """
            self.db_manager.execute_query(update_lifetime_query)
            bt.logging.debug("Updated lifetime statistics for miners.")

            # Fetch scores for the current day, excluding 'daily' scores
            fetch_scores_query = """
                SELECT 
                    s.miner_uid,
                    s.day_id,
                    s.score_type,
                    s.composite_score
                FROM 
                    scores s
                WHERE 
                    s.day_id = ? 
                    AND s.score_type LIKE 'tier_%'
            """
            scores = self.db_manager.fetch_all(fetch_scores_query, (current_day,))

            bt.logging.info(f"Fetched {len(scores)} tier-specific score records for day {current_day}.")

            # Update miner stats based on tier-specific composite scores
            for record in scores:
                miner_uid = record["miner_uid"]
                day_id = record["day_id"]
                score_type = record["score_type"]
                composite_score = record["composite_score"]

                # Extract tier number from score_type, e.g., 'tier_1' -> 1
                tier_number = int(score_type.split('_')[1]) if score_type.startswith('tier_') else None

                if tier_number is not None:
                    # Example logic: Assign composite_score to the corresponding tier
                    # Adjust this logic based on your actual requirements
                    if 'tier_scores' not in self.miner_stats[miner_uid]:
                        self.miner_stats[miner_uid]['tier_scores'] = {}
                    self.miner_stats[miner_uid]['tier_scores'][tier_number] = composite_score
                    #bt.logging.debug(f"Updated miner_uid {miner_uid} with tier_{tier_number} composite score: {composite_score}")

            # Handle daily scores separately
            fetch_daily_scores_query = """
                SELECT 
                    miner_uid,
                    day_id,
                    clv_score,
                    roi_score,
                    entropy_score,
                    sortino_score,
                    composite_score
                FROM 
                    scores
                WHERE 
                    day_id = ? 
                    AND score_type = 'daily'
            """
            daily_scores = self.db_manager.fetch_all(fetch_daily_scores_query, (current_day,))

            bt.logging.info(f"Fetched {len(daily_scores)} daily score records for day {current_day}.")

            for record in daily_scores:
                miner_uid = record["miner_uid"]
                clv = record["clv_score"] if record["clv_score"] is not None else 0.0
                roi = record["roi_score"] if record["roi_score"] is not None else 0.0
                entropy = record["entropy_score"] if record["entropy_score"] is not None else 0.0
                sortino = record["sortino_score"] if record["sortino_score"] is not None else 0.0
                composite_daily = record["composite_score"] if record["composite_score"] is not None else 0.0

                # Update miner_stats with daily scores
                self.miner_stats[miner_uid]['clv'] = clv
                self.miner_stats[miner_uid]['roi'] = roi
                self.miner_stats[miner_uid]['entropy'] = entropy
                self.miner_stats[miner_uid]['sortino'] = sortino
                self.miner_stats[miner_uid]['composite_daily'] = composite_daily

                #bt.logging.debug(f"Updated miner_uid {miner_uid} with daily scores: CLV={clv}, ROI={roi}, Entropy={entropy}, Sortino={sortino}, Composite Daily={composite_daily}")

            # Update derived lifetime statistics
            update_derived_lifetime_query = """
                UPDATE miner_stats
                SET
                    miner_win_loss_ratio = CASE 
                        WHEN miner_lifetime_losses > 0 
                        THEN CAST(miner_lifetime_wins AS REAL) / miner_lifetime_losses 
                        ELSE miner_lifetime_wins 
                    END,
                    miner_lifetime_roi = CASE
                        WHEN (miner_lifetime_wager_amount - 
                              (SELECT COALESCE(SUM(p.wager), 0) 
                               FROM predictions p 
                               WHERE p.miner_uid = miner_stats.miner_uid 
                                 AND p.outcome = 3)) > 0
                        THEN (miner_lifetime_earnings - 
                              (miner_lifetime_wager_amount - 
                               (SELECT COALESCE(SUM(p.wager), 0) 
                                FROM predictions p 
                                WHERE p.miner_uid = miner_stats.miner_uid 
                                  AND p.outcome = 3))
                             ) / 
                             (miner_lifetime_wager_amount - 
                              (SELECT COALESCE(SUM(p.wager), 0) 
                               FROM predictions p 
                               WHERE p.miner_uid = miner_stats.miner_uid 
                                 AND p.outcome = 3))
                        ELSE 0
                    END
            """
            self.db_manager.execute_query(update_derived_lifetime_query)
            bt.logging.debug("Updated derived lifetime statistics for miners.")

            self.db_manager.commit_transaction()
            bt.logging.info("Miner stats update transaction committed successfully.")

        except Exception as e:
            self.db_manager.rollback_transaction()
            bt.logging.error(f"Error updating miner stats: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise

        finally:
            # Double-check the results
            final_check_query = """
                SELECT COUNT(*) as count 
                FROM miner_stats 
                WHERE miner_current_composite_score > 0
            """
            final_check = self.db_manager.fetch_one(final_check_query, ())
            bt.logging.info(f"Final check - Miner stats with non-zero composite score: {final_check['count']}")
            bt.logging.info("Miner stats update process completed.")

