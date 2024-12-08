"""
BetTensor Scoring Module. 

This module handles the scoring of miners based on their data. The scoring is intended to select for miners that deploy positive expected value strategies, with some degree of risk management. 
We mostly determine +EV through closing line value analysis.

Inputs: 
- Miner Predictions

Outputs: 
- A NumPy array of the composite scores for all miners, indexed by miner_uid. 
"""


from collections import defaultdict
import json
import numpy as np
import bittensor as bt
from datetime import datetime, timezone, timedelta, date
from typing import List, Dict
import time
import traceback
import asyncio

from bettensor.validator.utils.database.database_manager import DatabaseManager
from .scoring_data import ScoringData
from .entropy_system import EntropySystem

class ScoringSystem:
    def __init__(
        self,
        db_manager: DatabaseManager,
        num_miners: int = 256,
        max_days: int = 45,
        reference_date: datetime = datetime(
            year=2024, month=9, day=30, tzinfo=timezone.utc
        ),
        validator = None,
        
        
    ):
        """
        Initialize the ScoringSystem.

        Args:
            db_path (str): Path to the database.
            num_miners (int): Number of miners.
            max_days (int): Maximum number of days to track.
            reference_date (datetime, optional): Reference date for scoring. Defaults to January 1, 2024.
        """
        #numpy max integer
        max_int = np.iinfo(np.int64).max
        np.set_printoptions(threshold=max_int)
        self.num_miners = num_miners
        self.max_days = max_days
        self.num_tiers = (
            7  # 5 tiers + 2 for invalid UIDs (0) and empty network slots (-1)
        )
        self.valid_uids = set()  # Initialize as an empty set
        self.validator = validator
        self.reference_date = reference_date
        self.invalid_uids = []
        self.epsilon = 1e-8  # Small constant to prevent division by zero
        self.db_manager = db_manager
        # Initialize tier configurations
        self.tier_configs = [
            {
                "window": 0,
                "min_wager": 0,
                "capacity": int(num_miners * 1),
                "incentive": 0,
            },  # Tier -1 for empty slots
            {
                "window": 0,
                "min_wager": 0,
                "capacity": int(num_miners * 1),
                "incentive": 0,
            },  # Tier 0 for invalid UIDs
            {
                "window": 3,
                "min_wager": 0,
                "capacity": int(num_miners * 1.0),
                "incentive": 0.1,
            },  # Tier 1
            {
                "window": 7,
                "min_wager": 4000,
                "capacity": int(num_miners * 0.2),
                "incentive": 0.15,
            },  # Tier 2
            {
                "window": 15,
                "min_wager": 10000,
                "capacity": int(num_miners * 0.2),
                "incentive": 0.2,
            },  # Tier 3
            {
                "window": 30,
                "min_wager": 20000,
                "capacity": int(num_miners * 0.1),
                "incentive": 0.25,
            },  # Tier 4
            {
                "window": 45,
                "min_wager": 35000,
                "capacity": int(num_miners * 0.05),
                "incentive": 0.3,
            },  # Tier 5
        ]
        
        self.tier_mapping = {
            0: "daily",  # Daily score
            1: "tier_1",
            2: "tier_2",
            3: "tier_3",
            4: "tier_4",
            5: "tier_5"
        }

        # Initialize score arrays
        self.clv_scores = np.zeros((num_miners, max_days))
        self.roi_scores = np.zeros((num_miners, max_days))
        self.sortino_scores = np.zeros((num_miners, max_days))
        self.amount_wagered = np.zeros((num_miners, max_days))
        self.entropy_scores = np.zeros((num_miners, max_days))
        self.tiers = np.ones((num_miners, max_days), dtype=int)

        # Initialize composite scores. The last dimension is for a daily score [0], and the other 5 are a "tier score"
        # tier score is calculated as a rolling average over the scoring window of that tier. every miner gets a tier score for every tier

        self.composite_scores = np.zeros((num_miners, max_days, 6))

        # Scoring weights
        self.clv_weight = 0.30
        self.roi_weight = 0.30
        self.sortino_weight = 0.30
        self.entropy_weight = 0.10
        self.entropy_window = self.max_days



        
        self.entropy_system = EntropySystem(num_miners, max_days, db_manager=db_manager)
        self.incentives = []

        self.current_day = 0
        self.current_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)  # Initialize to start of current UTC day
        self.last_update_date = None  # Will store the last day processed as a date object (no time)

        self.scoring_data = ScoringData(self)



    async def populate_amount_wagered(self):
        """
        Populate the amount_wagered array with the cumulative sum of wagers over the scoring window.
        Each day's value represents the total amount wagered in the previous window_size days,
        properly aligned with the current_day index in the circular buffer.
        """
        bt.logging.info("Populating amount_wagered from database predictions...")
        try:
            # Calculate the start date based on current_date and max_days
            start_date = self.current_date - timedelta(days=self.max_days)
            
            # Fetch daily wagers within the window
            query = """
                SELECT 
                    miner_uid,
                    DATE(prediction_date) as pred_date,
                    SUM(wager) as daily_wager
                FROM predictions
                WHERE prediction_date > ? AND prediction_date <= ?
                GROUP BY miner_uid, DATE(prediction_date)
                ORDER BY prediction_date ASC
            """
            params = (start_date.isoformat(), self.current_date.isoformat())
            daily_wagers = await self.db_manager.fetch_all(query, params)
            
            # Reset arrays
            self.amount_wagered.fill(0)
            daily_amounts = np.zeros_like(self.amount_wagered)
            
            # Map dates to circular buffer indices
            for wager in daily_wagers:
                miner_id = int(wager["miner_uid"])
                wager_amount = float(wager["daily_wager"])
                pred_date = datetime.strptime(wager["pred_date"], '%Y-%m-%d').date()
                
                # Calculate days from prediction to current date
                days_ago = (self.current_date.date() - pred_date).days
                
                # Skip if outside our window
                if not (0 <= days_ago < self.max_days):
                    continue
                    
                # Map to circular buffer index
                buffer_index = (self.current_day - days_ago) % self.max_days
                
                if 0 <= miner_id < self.num_miners:
                    daily_amounts[miner_id, buffer_index] = wager_amount
            
            # Calculate cumulative sums for each day in the buffer
            max_window = self.tier_configs[-1]["window"]  # 45 days
            
            for miner_id in range(self.num_miners):
                for day_offset in range(self.max_days):
                    # Calculate the actual day index in the circular buffer
                    day_idx = (self.current_day - day_offset) % self.max_days
                    
                    # Calculate window start index for this day
                    window_start = (day_idx - max_window + 1) % self.max_days
                    
                    # Sum the daily amounts within the window
                    if window_start <= day_idx:
                        # Window doesn't wrap around
                        self.amount_wagered[miner_id, day_idx] = daily_amounts[miner_id, window_start:day_idx + 1].sum()
                    else:
                        # Window wraps around the circular buffer
                        self.amount_wagered[miner_id, day_idx] = (
                            daily_amounts[miner_id, window_start:].sum() +
                            daily_amounts[miner_id, :day_idx + 1].sum()
                        )
            
            bt.logging.info(f"amount_wagered populated successfully for current_day={self.current_day}")
            bt.logging.debug(f"Total amount wagered: {self.amount_wagered.sum():.2f}")
            
        except Exception as e:
            bt.logging.error(f"Error populating amount_wagered: {e}")
            raise

    def advance_day(self, current_date: date):
        bt.logging.debug(f"Attempting to advance day with current_date: {current_date}, last_update_date: {self.last_update_date}")
        
        if self.last_update_date is None:
            self.last_update_date = current_date
            bt.logging.info(f"Set last_update_date to {self.last_update_date} without advancing day.")
            return

        days_passed = (current_date - self.last_update_date).days
        bt.logging.debug(f"Days passed since last update: {days_passed}")

        if days_passed > 0:
            for day in range(1, days_passed + 1):
                # Advance to the next day in the circular buffer
                previous_day = self.current_day
                self.current_day = (self.current_day + 1) % self.max_days
                bt.logging.debug(f"Advancing to day_id={self.current_day}")

                # Copy previous day's cumulative values instead of resetting
                self.amount_wagered[:, self.current_day] = self.amount_wagered[:, previous_day]
                bt.logging.debug(f"Copied amount_wagered from day_id={previous_day} to day_id={self.current_day}")

                # Carry over tier information from the previous day
                self.tiers[:, self.current_day] = self.tiers[:, previous_day]
                bt.logging.debug(f"Copied tiers from day_id={previous_day} to day_id={self.current_day}")

            # Update the last_update_date and current_date
            self.last_update_date = current_date
            self.current_date = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
            bt.logging.info(f"Advanced {days_passed} day(s). New current_day index: {self.current_day}")

            if days_passed > 1:
                self.handle_downtime(days_passed)
        else:
            bt.logging.info("No new day to advance. current_day remains unchanged.")

    def handle_downtime(self, days_passed):
        bt.logging.warning(
            f"System was down for {days_passed - 1} day(s). Adjusting scores..."
        )

        # For each day of downtime, we'll copy the last known scores and tiers
        for i in range(1, days_passed):
            prev_day = (self.current_day - days_passed + i) % self.max_days
            current_day = (self.current_day - days_passed + i + 1) % self.max_days

            self.clv_scores[:, current_day] = self.clv_scores[:, prev_day]
            self.roi_scores[:, current_day] = self.roi_scores[:, prev_day]
            self.entropy_scores[:, current_day] = self.entropy_scores[:, prev_day]
            self.composite_scores[:, current_day] = self.composite_scores[:, prev_day]
            self.tiers[:, current_day] = self.tiers[:, prev_day]

        bt.logging.info("Downtime handling complete.")

    async def update_scores(self, predictions, closing_line_odds, results):
        """
        Update the scores for the current day based on predictions, closing line odds, and results.

        Args:
            predictions (np.ndarray): Array of miner predictions.
            closing_line_odds (np.ndarray): Array of closing line odds.
            results (np.ndarray): Array of game results.
        """
        bt.logging.info(f"Updating scores for day {self.current_day}")
        try:
            if predictions.size > 0 and closing_line_odds.size > 0 and results.size > 0:
                self._update_raw_scores(predictions, closing_line_odds, results)
                await self._update_composite_scores()
                self.log_score_summary()
            else:
                bt.logging.warning("No data available for score update.")
        except Exception as e:
            bt.logging.error(f"Error updating scores: {str(e)}")
            raise

    def _update_raw_scores(self, predictions, closing_line_odds, results):
        bt.logging.trace(f"Predictions shape: {predictions.shape}")
        bt.logging.trace(f"Closing line odds shape: {closing_line_odds.shape}")
        bt.logging.trace(f"Results shape: {results.shape}")

        # Extract unique external_ids from predictions
        external_ids = np.unique(predictions[:, 1]).astype(int).tolist()
        bt.logging.debug(f"Unique external_ids: {external_ids}")

        # Calculate CLV scores for all predictions
        clv_scores = self._calculate_clv_scores(predictions, closing_line_odds)
        self.clv_scores[:, self.current_day] = clv_scores

        # Calculate ROI scores for all predictions
        roi_scores = self._calculate_roi_scores(predictions, results)
        self.roi_scores[:, self.current_day] = roi_scores

        # Calculate Sortino scores for all predictions
        sortino_scores = self._calculate_risk_scores(predictions, results)
        self.sortino_scores[:, self.current_day] = sortino_scores

        # Get previous day's cumulative values
        previous_day = (self.current_day - 1) % self.max_days
        previous_cumulative = self.amount_wagered[:, previous_day].copy()

        # Group predictions by miner_id
        miner_predictions = defaultdict(list)
        for pred in predictions:
            miner_id = int(pred[0])
            miner_predictions[miner_id].append(pred)

        # Process wagers for each miner
        for miner_id, miner_preds in miner_predictions.items():
            daily_wager = 0.0
            for pred in miner_preds: 
                try:
                    wager = float(pred[5])
                    daily_wager += wager
                except (IndexError, ValueError) as e:
                    bt.logging.error(f"Error extracting wager for miner {miner_id}: {e}")

            # Add daily wager to previous cumulative total
            self.amount_wagered[miner_id, self.current_day] = previous_cumulative[miner_id] + daily_wager

        bt.logging.info(f"Updated cumulative amount_wagered for day {self.current_day}")
        bt.logging.debug(f"Amount wagered summary - min: {self.amount_wagered[:, self.current_day].min():.2f}, "
                         f"max: {self.amount_wagered[:, self.current_day].max():.2f}, "
                         f"mean: {self.amount_wagered[:, self.current_day].mean():.2f}")

        # Update entropy scores - now returns full array
        entropy_scores = self.entropy_system.get_current_ebdr_scores(
            self.current_date, self.current_day, external_ids
        )
        # Update entire entropy scores array
        self.entropy_scores = entropy_scores

        bt.logging.info(
            f"Entropy scores for current day - "
            f"min: {entropy_scores[:, self.current_day].min():.8f}, "
            f"max: {entropy_scores[:, self.current_day].max():.8f}, "
            f"mean: {entropy_scores[:, self.current_day].mean():.8f}, "
            f"non-zero: {(entropy_scores[:, self.current_day] != 0).sum()}"
        )

    def _meets_tier_requirements(self, miner, tier):
        """
        Check if a miner meets the requirements for a given tier based on min_wager.

        Args:
            miner (int): The miner's UID.
            tier (int): The tier to check requirements for.

        Returns:
            bool: True if the miner meets the tier requirements, False otherwise.
        """
        #bt.logging.debug(f"Checking if miner {miner} meets tier {tier} requirements")
        config = self.tier_configs[tier]
        cumulative_wager = self._get_cumulative_wager(miner, config["window"])
        meets_requirement = cumulative_wager >= config["min_wager"] 

        if meets_requirement:
            pass
            #bt.logging.debug(f"Miner {miner} meets tier {tier-1} requirements with cumulative wager {cumulative_wager}")
        else:
            pass
            # bt.logging.debug(f"Miner {miner} does NOT meet tier {tier-1} requirements with cumulative wager {cumulative_wager}")

        return meets_requirement

    async def manage_tiers(self, invalid_uids, valid_uids):  # Add async
        bt.logging.info("Managing tiers")

        try:
            current_tiers = self.tiers[:, self.current_day].copy()
            composite_scores_day = self.composite_scores[:, self.current_day, :]
            
            bt.logging.debug(f"Composite Scores Shape: {self.composite_scores.shape}")
            bt.logging.info(f"Current tiers before management: {np.bincount(current_tiers, minlength=self.num_tiers)}")

            # Step 1: Check for and perform demotions (top-down)
            for tier in range(self.num_tiers - 1, 1, -1):
                tier_miners = np.where(current_tiers == tier)[0]
                for miner in tier_miners:
                    if not self._meets_tier_requirements(miner, tier):
                        self._cascade_demotion(miner, tier, current_tiers, composite_scores_day)

            # Step 2: Promote and swap (using our new top-down implementation)
            await self._promote_and_swap(current_tiers, composite_scores_day, valid_uids)  # Add valid_uids parameter

            # Update tiers for the current day
            self.tiers[:, self.current_day] = current_tiers

            # Set invalid UIDs to tier 0
            self.tiers[list(invalid_uids), self.current_day] = 0

            bt.logging.info(f"Current tiers after management: {np.bincount(current_tiers, minlength=self.num_tiers)}")
            bt.logging.info("Tier management completed")

        except Exception as e:
            bt.logging.error(f"Error managing tiers: {str(e)}")
            raise

    async def _promote_and_swap(self, tiers, composite_scores_day, valid_uids):  # Add async and valid_uids
        """
        Promote miners to higher tiers and perform swaps when necessary, ensuring min_wager is respected.
        Working top-down to optimize tier distribution.
        """
        for tier in range(self.num_tiers - 1, 1, -1):  # Start from highest tier (5) down to tier 2
            current_tier_miners = np.where(tiers == tier)[0]
            # Only consider valid miners from tier 2 and above
            lower_tier_miners = np.where((tiers == tier - 1) & np.isin(np.arange(len(tiers)), list(valid_uids)))[0]
            
            
            # Check if there are open slots in this tier
            open_slots = self.tier_configs[tier]["capacity"] - len(current_tier_miners)
            
            if open_slots > 0:
                # Identify eligible miners from lower tier based on min_wager
                eligible_miners = [
                    miner for miner in lower_tier_miners
                    if self._meets_tier_requirements(miner, tier) and miner in valid_uids
                ]
                bt.logging.debug(
                    f"Tier {tier-1}: {len(eligible_miners)} eligible miners for {open_slots} openslots"
                )

                if eligible_miners:
                    # Sort eligible miners by composite scores descending
                    eligible_miners_sorted = sorted(
                        eligible_miners,
                        key=lambda x: composite_scores_day[x, tier - 2],
                        reverse=True
                    )

                    # Promote the best miners to fill open slots
                    promotions = eligible_miners_sorted[:open_slots]
                    for miner in promotions:
                        tiers[miner] = tier
                        bt.logging.info(f"Miner {miner} promoted to tier {tier-1}")

            else:
                # If tier is full, consider swaps with lower tier miners
                if len(lower_tier_miners) > 0:
                    # Sort current tier miners by score ascending (worst first)
                    current_sorted = sorted(
                        current_tier_miners,
                        key=lambda x: composite_scores_day[x, tier - 1]
                    )
                    
                    # Sort lower tier miners by score descending (best first)
                    lower_sorted = sorted(
                        lower_tier_miners,
                        key=lambda x: composite_scores_day[x, tier - 2],
                        reverse=True
                    )

                    # Check each potential swap
                    for lower_miner in lower_sorted:
                        if not self._meets_tier_requirements(lower_miner, tier) or lower_miner not in valid_uids:
                            continue
                            
                        # Compare with worst performing miner in current tier
                        for current_miner in current_sorted:
                            lower_score = composite_scores_day[lower_miner, tier - 2]
                            current_score = composite_scores_day[current_miner, tier - 1]

                            if lower_score > current_score:
                                # Swap tiers
                                tiers[lower_miner], tiers[current_miner] = tier, tier - 1
                                bt.logging.info(
                                    f"Swapped miner {lower_miner} (↑tier {tier}) with "
                                    f"miner {current_miner} (↓tier {tier-1})"
                                )
                                # Update sorted lists
                                current_sorted.remove(current_miner)
                                break
                            else:
                                # If best lower tier miner can't beat worst current tier,
                                # no need to check others
                                break

        # Log final tier distribution
        tier_counts = [np.sum(tiers == t) for t in range(1, self.num_tiers)]
        bt.logging.info(f"Final tier distribution: {tier_counts}")

   

    def _get_cumulative_wager(self, miner, window):
        """
        Get the cumulative wager for a miner from the current day's value.
        Since amount_wagered now stores cumulative values, we just need the current day's value.
        
        Args:
            miner (int): The miner's UID
            window (int): Number of days to look back (no longer used since values are cumulative)
            
        Returns:
            float: Total cumulative wager amount
        """
        # Simply return the current day's value since it's already cumulative
        wager = self.amount_wagered[miner, self.current_day]
        
        # bt.logging.debug(
        #     f"Miner {miner} cumulative wager: {wager:.2f} (current day {self.current_day})"
        # )
        return wager

    def _cascade_demotion(self, miner, current_tier, tiers, composite_scores):
        """
        Demote a miner from the current tier to the next lower tier without violating tier boundaries.

        Args:
            miner (int): The miner's UID.
            current_tier (int): The current tier of the miner.
            tiers (np.ndarray): The array representing current tiers of all miners.
            composite_scores (np.ndarray): The composite scores array for the current day.
        """
        bt.logging.debug(f"Demoting miner {miner} from tier {current_tier-1}")

        # Determine if the miner is valid
        is_valid_miner = miner in self.valid_uids

        # Calculate new tier
        new_tier = current_tier - 1

        if is_valid_miner:
            # Ensure valid miners are not demoted below tier 2
            new_tier = max(new_tier, 2)
        else:
            # Invalid miners can be demoted to tier 1 or 0
            new_tier = max(new_tier, 1)

        tiers[miner] = new_tier
        bt.logging.info(f"Miner {miner} demoted to tier {new_tier - 1}")

        # Recursively check if further demotion is needed
        if not self._meets_tier_requirements(miner, new_tier):
            self._cascade_demotion(miner, new_tier, tiers, composite_scores)

    async def reset_miner(self, miner_uid):
        """
        Completely reset a miner's stats and predictions.
        This is called when a hotkey changes UIDs or when a miner needs to be reset.
        """
        try:
            queries = [
                # Clear all stats for the miner
                ("""UPDATE miner_stats 
                    SET miner_hotkey = NULL,
                        miner_coldkey = NULL,
                        miner_rank = NULL,
                        miner_status = NULL,
                        miner_cash = 0,
                        miner_current_incentive = 0,
                        miner_current_tier = 1,
                        miner_current_scoring_window = 0,
                        miner_current_composite_score = NULL,
                        miner_current_sharpe_ratio = NULL,
                        miner_current_sortino_ratio = NULL,
                        miner_current_roi = NULL,
                        miner_current_clv_avg = NULL,
                        miner_last_prediction_date = NULL,
                        miner_lifetime_earnings = 0,
                        miner_lifetime_wager_amount = 0,
                        miner_lifetime_roi = 0,
                        miner_lifetime_predictions = 0,
                        miner_lifetime_wins = 0,
                        miner_lifetime_losses = 0,
                        miner_win_loss_ratio = 0
                    WHERE miner_uid = ?""", 
                 (miner_uid,)),
                 
                # Delete all predictions for this miner
                ("""DELETE FROM predictions 
                    WHERE miner_uid = ?""",
                 (miner_uid,)),
                  
            ]
            
            for query, params in queries:
                for attempt in range(3):  # Try each query up to 3 times
                    try:
                        await self.db_manager.execute_query(query, params)
                        break
                    except TimeoutError:
                        if attempt == 2:  # Last attempt
                            bt.logging.error(f"Failed to reset miner {miner_uid} after 3 attempts")
                            raise
                        time.sleep(1)  # Wait before retry
                        
            bt.logging.info(f"Successfully reset all data for miner {miner_uid}")
                    
        except Exception as e:
            bt.logging.error(f"Error resetting miner {miner_uid}: {str(e)}")
            raise

    def get_miner_history(self, miner_uid: int, score_type: str, days: int = None):
        """
        Get the score history for a specific miner and score type.

        Args:
            miner_uid (int): The UID of the miner.
            score_type (str): The type of score to retrieve ('clv', 'roi', 'sortino', 'entropy', 'composite', 'tier').
            days (int, optional): Number of days of history to return. If None, returns all available history.

        Returns:
            np.ndarray: An array containing the miner's score history.
        """
        score_array = getattr(self, f"{score_type}_scores", None)
        if score_array is None:
            raise ValueError(f"Invalid score type: {score_type}")

        if days is None:
            return score_array[miner_uid]
        else:
            return score_array[miner_uid, -days:]

    def log_score_summary(self):
        """
        Log a summary of the current scores, including the entire amount_wagered and tiers arrays for debugging.
        """
        bt.logging.info("=== Score Summary ===")
        for score_name, score_array in [
            ("CLV", self.clv_scores),
            ("ROI", self.roi_scores),
            ("Entropy", self.entropy_scores),
            ("Sortino", self.sortino_scores),
            ("Composite", self.composite_scores),
        ]:
            current_scores = self._get_array_for_day(score_array, self.current_day)
            bt.logging.info(
                f"{score_name} Scores - min: {current_scores.min():.4f}, "
                f"max: {current_scores.max():.4f}, "
                f"mean: {current_scores.mean():.4f}, "
                f"non-zero: {np.count_nonzero(current_scores)}"
            )
        
        # Debug: Print the entire amount_wagered array
        #bt.logging.info("=== Amount Wagered Array ===")
        #np.set_printoptions(threshold=np.inf)  # Remove threshold to print entire array
        #bt.logging.info(f"amount_wagered:\n{self.amount_wagered}")
        #np.set_printoptions(threshold=1000)  # Reset to default threshold

        # Debug: Print the entire tiers array
        #bt.logging.info("=== Tiers Array ===")
        #np.set_printoptions(threshold=np.inf)  # Remove threshold to print entire array
        #bt.logging.info(f"tiers:\n{self.tiers}")
        #np.set_printoptions(threshold=1000)  # Reset to default threshold

    def calculate_weights(self, day=None):
        """
        Calculate weights for all miners based on their tier and composite score.
        Ensures the weights array has a length of 256 with correct indices.
        Weights for invalid or empty UIDs are set to 0.
        
        Args:
            day (int, optional): The day index to calculate weights for. Defaults to current day.
        
        Returns:
            np.ndarray: The calculated weights for all miners
        """
        bt.logging.info("Calculating weights")
        
        try:
            weights = np.zeros(self.num_miners)

            tier_incentives = np.array([config["incentive"] for config in self.tier_configs[2:]])
            total_incentive = tier_incentives.sum()
            normalized_incentives = tier_incentives / total_incentive if total_incentive > 0 else np.zeros_like(tier_incentives)

            if day is None:
                day = self.current_day

            current_tiers = self.tiers[:, day]
            valid_miners = np.array(list(set(range(self.num_miners)) - self.invalid_uids))
            valid_miners = valid_miners[(current_tiers[valid_miners] >= 2) & (current_tiers[valid_miners] < self.num_tiers)]

            if not valid_miners.any():
                bt.logging.warning("No valid miners found. Returning zero weights.")
                return weights

            # Use each miner's composite score for their respective tier
            # Extract the composite scores based on current tier for each miner
            composite_scores = self.composite_scores[valid_miners, day, current_tiers[valid_miners]]

            # Apply non-linear normalization (e.g., exponential)
            exp_scores = np.exp(composite_scores)
            score_range = exp_scores.max() - exp_scores.min()
            if score_range > 0:
                normalized_scores = (exp_scores - exp_scores.min()) / score_range
            else:
                normalized_scores = np.ones_like(exp_scores) / len(exp_scores)  # Equal distribution if all scores are the same

            # Apply tier incentives
            for idx, tier in enumerate(range(2, self.num_tiers)):
                tier_miners = valid_miners[current_tiers[valid_miners] == tier]
                incentive_factor = normalized_incentives[idx]
                # Select normalized scores for miners in the current tier
                tier_scores = normalized_scores[current_tiers[valid_miners] == tier]
                weights[tier_miners] = tier_scores * incentive_factor * (1 + idx * 0.1)

            # Ensure weights sum to 1
            total_weight = weights.sum()
            if total_weight > 0:
                weights /= total_weight
            else:
                bt.logging.warning("Total weight is zero. Distributing weights equally among valid miners.")
                weights[list(self.valid_uids)] = 1 / len(self.valid_uids)

            # Double-check and log
            final_sum = weights.sum()
            bt.logging.info(f"Final weight sum: {final_sum:.6f}")
            if not np.isclose(final_sum, 1.0):
                bt.logging.warning(f"Weights sum is not exactly 1.0: {final_sum}")

            bt.logging.info(f"Min weight: {weights.min():.6f}, Max weight: {weights.max():.6f}")

            return weights

        except Exception as e:
            bt.logging.error(f"Error calculating weights: {str(e)}")
            raise

    async def scoring_run(self, date, invalid_uids, valid_uids):
        bt.logging.info(f"=== Starting scoring run for date: {date} ===")

        # Update invalid, valid, and empty UIDs
        self.invalid_uids = set(invalid_uids)
        self.valid_uids = set(valid_uids)
        self.empty_uids = set(range(self.num_miners)) - self.valid_uids - self.invalid_uids

        # Create boolean masks for each category
        empty_mask = np.zeros(self.num_miners, dtype=bool)
        empty_mask[list(self.empty_uids)] = True

        invalid_mask = np.zeros(self.num_miners, dtype=bool)
        invalid_mask[list(self.invalid_uids)] = True

        valid_mask = np.zeros(self.num_miners, dtype=bool)
        valid_mask[list(self.valid_uids)] = True

        # Set tiers using boolean masks
        if self.init:
            self.tiers[:, self.current_day] = 2  # Initialize valid UIDs to tier 2
            bt.logging.info(f"Assigned {len(self.valid_uids)} valid UIDs to tier 2.")
            self.init = False

        # Assign empty and invalid UIDs
        self.tiers[empty_mask, self.current_day] = 0
        self.tiers[invalid_mask, self.current_day] = 1

        # Ensure valid UIDs are at least in tier 2
        self.tiers[valid_mask & (self.tiers[:, self.current_day] < 2)] = 2  

        bt.logging.info(f"Assigned {len(self.empty_uids)} empty slots to tier 0.")
        bt.logging.info(f"Assigned {len(self.invalid_uids)} invalid UIDs to tier 1.")

        current_date = self._ensure_date(date)
        bt.logging.debug(f"Processing scoring_run for date: {current_date}")
        self.advance_day(current_date)  # Pass only the date part

        # Add this: Populate amount_wagered after advancing day
        await self.populate_amount_wagered()

        date_str = current_date.strftime('%Y-%m-%d %H:%M:%S')

        bt.logging.info(
            f"Current day: {self.current_day}, reference date: {self.reference_date}"
        )

        # Add this debugging code before calculating composite scores
        current_tiers = self.tiers[:, self.current_day]
        tier_distribution = [
            int(np.sum(current_tiers == tier))
            for tier in range(0, len(self.tier_configs) + 1)
        ]
        bt.logging.info(f"Current tier distribution: {tier_distribution}")

        (
            predictions,
            closing_line_odds,
            results,
        ) = await self.scoring_data.preprocess_for_scoring(date_str)

        bt.logging.info(
            f"Number of predictions: {predictions.shape[0] if predictions.size > 0 else 0}"
        )

        if predictions.size > 0 and closing_line_odds.size > 0 and results.size > 0:
            bt.logging.info("Updating scores...")
            await self.update_scores(predictions, closing_line_odds, results)
            bt.logging.info("Scores updated successfully.")

            total_wager = self.amount_wagered[:, self.current_day].sum()
            avg_wager = total_wager / self.num_miners
            bt.logging.info(f"Total wager for this run: {total_wager:.2f}")
            bt.logging.info(f"Average wager per miner: {avg_wager:.2f}")

            await self.manage_tiers(invalid_uids, valid_uids)

            # Calculate weights using the existing method
            weights = self.calculate_weights()

            # Update most_recent_weight in miner_stats table
            update_weights_query = """
                UPDATE miner_stats 
                SET most_recent_weight = ? 
                WHERE miner_uid = ?
            """
            weight_updates = [(float(weights[i]), i) for i in range(self.num_miners)]
            await self.db_manager.executemany(update_weights_query, weight_updates)
            bt.logging.info("Updated most_recent_weights in miner_stats table")

        else:
            bt.logging.warning(
                f"No predictions for date {date_str}. Using previous day's weights."
            )
            await self.manage_tiers(invalid_uids, valid_uids)

            previous_day = (self.current_day - 1) % self.max_days
            try:
                weights = self.calculate_weights(day=previous_day)
                bt.logging.info(f"Using weights from previous day: {previous_day}")
                
                # Update most_recent_weight in miner_stats table even for previous day's weights
                update_weights_query = """
                    UPDATE miner_stats 
                    SET most_recent_weight = ? 
                    WHERE miner_uid = ?
                """
                weight_updates = [(float(weights[i]), i) for i in range(self.num_miners)]
                await self.db_manager.executemany(update_weights_query, weight_updates)
                bt.logging.info("Updated most_recent_weights in miner_stats table using previous day's weights")
                
            except Exception as e:
                bt.logging.error(
                    f"Failed to retrieve weights from previous day: {e}. Assigning equal weights."
                )
                weights = np.zeros(self.num_miners)
                weights[list(self.valid_uids)] = 1 / len(self.valid_uids)
                
                # Update most_recent_weight with equal weights
                update_weights_query = """
                    UPDATE miner_stats 
                    SET most_recent_weight = ? 
                    WHERE miner_uid = ?
                """
                weight_updates = [(float(weights[i]), i) for i in range(self.num_miners)]
                await self.db_manager.executemany(update_weights_query, weight_updates)
                bt.logging.info("Updated most_recent_weights in miner_stats table with equal weights")

        # Assign invalid UIDs to tier 0
        weights[list(self.invalid_uids)] = 0

        # Renormalize weights
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            bt.logging.warning("Total weight is zero. Distributing weights equally among valid miners.")
            weights[list(self.valid_uids)] = 1 / len(self.valid_uids)

        bt.logging.info(f"Weight sum: {weights.sum():.6f}")
        bt.logging.info(
            f"Min weight: {weights.min():.6f}, Max weight: {weights.max():.6f}"
        )
        bt.logging.info(f"Weights: {weights}")
        # Log final tier distribution
        self.log_score_summary()

        bt.logging.info(f"=== Completed scoring run for date: {date_str} ===")

        # Save state at the end of each run
        await self.save_state()
        await self.scoring_data.update_miner_stats(self.current_day)

        # check that weights are length 256. 
        if len(weights) != 256:
            bt.logging.error(f"Weights are not length 256. They are length {len(weights)}")
            return None

        return weights

    def reset_date(self, new_date):
        """
        Reset the scoring system's date for testing purposes.

        Args:
            new_date (datetime): The new date to set as the current date.
        """
        # Ensure new_date is timezone-aware
        if new_date.tzinfo is None:
            new_date = new_date.replace(tzinfo=timezone.utc)

        self.current_date = new_date
        days_since_reference = (new_date - self.reference_date).days
        self.current_day = days_since_reference % self.max_days

        # Reset tiers to 1 for the current day, using modulo for wraparound
        self.tiers[:, self._get_day_index(self.current_day)] = 1

    def reset_all_miners_to_tier_1(self):
        """
        Reset all miners to tier 1.
        """
        self.tiers.fill(1)

    def _get_day_index(self, day):
        """
        Get the index of a day in the circular buffer.

        Args:
            day (int): The day to get the index for.

        Returns:
            int: The index of the day in the circular buffer.
        """
        return day % self.max_days

    def _get_array_for_day(self, array, day, tier=None):
        """
        Get the array for a specific day and tier.

        Args:
            array (np.ndarray): The array to get the data from.
            day (int): The day to get the data for.
            tier (int, optional): The tier to get the data for. Defaults to None.

        Returns:
            np.ndarray: The array for the specified day and tier.
        """
        if tier is None:
            return array[:, self._get_day_index(day)]
        else:
            return array[:, self._get_day_index(day), tier]

    def _set_array_for_day(self, array, day, value, tier=None):
        """
        Set the array for a specific day and tier.

        Args:
            array (np.ndarray): The array to set the data in.
            day (int): The day to set the data for.
            value (np.ndarray): The value to set in the array.
            tier (int, optional): The tier to set the data for. Defaults to None.
        """
        if tier is None:
            array[:, self._get_day_index(day)] = value
        else:
            array[:, self._get_day_index(day), tier] = value

    async def save_state(self):
        """
        Save the current state of the ScoringSystem to the database, including the amount_wagered and tiers arrays.
        """
        try:
            

            # Serialize invalid_uids and valid_uids
            invalid_uids_json = json.dumps(list(int(uid) for uid in self.invalid_uids))
            valid_uids_json = json.dumps(list(int(uid) for uid in self.valid_uids))

            # Serialize amount_wagered as a list of lists
            amount_wagered_serialized = json.dumps(self.amount_wagered.tolist())

            # Serialize tiers array as a list of lists
            tiers_serialized = json.dumps(self.tiers.tolist())

            # Insert or update the latest state in score_state table
            insert_state_query = """
                INSERT INTO score_state 
                (current_day, current_date, reference_date, invalid_uids, valid_uids, last_update_date, amount_wagered, tiers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(state_id) DO UPDATE SET
                    current_day=excluded.current_day,
                    current_date=excluded.current_date,
                    reference_date=excluded.reference_date,
                    invalid_uids=excluded.invalid_uids,
                    valid_uids=excluded.valid_uids,
                    last_update_date=excluded.last_update_date,
                    amount_wagered=excluded.amount_wagered,
                    tiers=excluded.tiers
            """
            params = (
                self.current_day,
                self.current_date.isoformat() if self.current_date else None,
                self.reference_date.isoformat(),
                invalid_uids_json,
                valid_uids_json,
                self.last_update_date.isoformat() if self.last_update_date else None,  # Store as date string
                amount_wagered_serialized,  # Serialized amount_wagered
                tiers_serialized  # Serialized tiers
            )

            await self.db_manager.execute_query(insert_state_query, params)

            # Now save scores
            await self.save_scores()

            
            bt.logging.info("ScoringSystem state saved to database, including amount_wagered and tiers.")

        except Exception as e:
            bt.logging.error(f"Error saving state to database: {e}")
            raise

    async def load_state(self):
        """Load state from database asynchronously, including historical score reconstruction if needed."""
        try:
            # First, check for the most recent state with current_day > 0
            fetch_state_query = """
                SELECT current_day, current_date, reference_date, invalid_uids, valid_uids, 
                    last_update_date, amount_wagered, tiers, state_id
                FROM score_state
                WHERE current_day > 0
                ORDER BY state_id DESC
                LIMIT 1
            """
            state = await self.db_manager.fetch_one(fetch_state_query, None)

            # If no state with current_day > 0, fall back to the most recent state
            if not state:
                fetch_state_query = """
                    SELECT current_day, current_date, reference_date, invalid_uids, valid_uids, 
                        last_update_date, amount_wagered, tiers, state_id
                    FROM score_state
                    ORDER BY state_id DESC
                    LIMIT 1
                """
                state = await self.db_manager.fetch_one(fetch_state_query, None)
            
            if state:
                bt.logging.info(f"Found existing state in database with state_id={state['state_id']}")
                self.current_day = state["current_day"]
                self.current_date = datetime.fromisoformat(state["current_date"])
                self.reference_date = datetime.fromisoformat(state["reference_date"])
                self.last_update_date = (datetime.fromisoformat(state["last_update_date"]).date() 
                                       if state["last_update_date"] else None)
                
                # Load UIDs and tiers first
                self.invalid_uids = set(json.loads(state["invalid_uids"]))
                self.valid_uids = set(json.loads(state["valid_uids"]))
                self.tiers = np.array(json.loads(state["tiers"]))
                
                # Load scores with skip_rebuild=True since we're in initialization
                await self.load_scores(skip_rebuild=True)
                
                # Verify and populate amount_wagered
                try:
                    amount_wagered_data = json.loads(state["amount_wagered"])
                    if not amount_wagered_data or len(amount_wagered_data) != self.num_miners:
                        bt.logging.warning("Invalid amount_wagered data in state, repopulating...")
                        await self.populate_amount_wagered()
                    else:
                        self.amount_wagered = np.array(amount_wagered_data)
                except (json.JSONDecodeError, TypeError):
                    bt.logging.warning("Corrupted amount_wagered data in state, repopulating...")
                    await self.populate_amount_wagered()
                
                # Add validation check for current_day
                if self.current_day == 0:
                    bt.logging.warning("Loading state with current_day=0, checking for more recent states...")
                    check_query = """
                    SELECT MAX(current_day) as max_day
                    FROM score_state
                    WHERE current_day > 0
                    """
                    max_day = await self.db_manager.fetch_one(check_query, None)
                    if max_day and max_day['max_day'] is not None:
                        bt.logging.warning(f"Found more recent state with day {max_day['max_day']}, but loading day 0 state")
                        # Load the state with the highest current_day
                        fetch_state_query = """
                            SELECT current_day, current_date, reference_date, invalid_uids, valid_uids, 
                                last_update_date, amount_wagered, tiers, state_id
                            FROM score_state
                            WHERE current_day = ?
                            ORDER BY state_id DESC
                            LIMIT 1
                        """
                        state = await self.db_manager.fetch_one(fetch_state_query, (max_day['max_day'],))
                        if state:
                            self.current_day = state["current_day"]
                            bt.logging.info(f"Updated to load state with current_day={self.current_day}")
            else:
                bt.logging.warning("No existing state found, initializing fresh state...")
                await self.populate_amount_wagered()
                await self.load_scores()  # Allow rebuild in this case

        except Exception as e:
            bt.logging.error(f"Error loading state from database: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            bt.logging.warning("Falling back to fresh state initialization...")
            await self.populate_amount_wagered()
            await self.load_scores()
            raise

    async def save_scores(self):
        num_miners, num_days, num_scores = self.composite_scores.shape
        bt.logging.info(f"Saving scores for {num_miners} miners, {num_days} days, {num_scores} scores (1 daily + 5 tiers)")
        
        score_records = []
        for miner in range(num_miners):
            for day in range(num_days):
                # Daily scores
                clv = self.clv_scores[miner, day]
                roi = self.roi_scores[miner, day]
                entropy = self.entropy_scores[miner, day]
                sortino = self.sortino_scores[miner, day]
                composite_daily = self.composite_scores[miner, day, 0]  # Index 0 for daily composite
                
                score_records.append((
                    miner,
                    day,
                    'daily',
                    clv,
                    roi,
                    entropy,
                    composite_daily,
                    sortino
                ))
                
                # Tier-specific composite scores
                for score_index in range(1, num_scores-1):
                    tier = self.tier_mapping[score_index]
                    composite = self.composite_scores[miner, day, score_index]
                    score_records.append((
                        miner,
                        day,
                        tier,
                        None,   # clv_score not applicable
                        None,   # roi_score not applicable
                        None,   # entropy_score not applicable
                        composite,
                        None    # sortino_score not applicable
                    ))
        
        # Batch insert using executemany with conflict resolution based on miner_uid, day_id, and score_type
        insert_score_query = """
            INSERT INTO scores 
            (miner_uid, day_id, score_type, clv_score, roi_score, entropy_score, composite_score, sortino_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(miner_uid, day_id, score_type) DO UPDATE SET
                clv_score = excluded.clv_score,
                roi_score = excluded.roi_score,
                entropy_score = excluded.entropy_score,
                composite_score = excluded.composite_score,
                sortino_score = excluded.sortino_score
        """
        await self.db_manager.executemany(insert_score_query, score_records)
        bt.logging.info(f"Saved {len(score_records)} score records")

    async def load_scores(self, skip_rebuild=False):
        """
        Load the scores from the database. If missing historical scores are detected,
        attempt to rebuild them from historical prediction data.
        """
        try:
            # First do a simple check for non-zero scores
            quick_check_query = """
                SELECT COUNT(*) as valid_scores
                FROM scores 
                WHERE composite_score != 0
            """
            result = await self.db_manager.fetch_one(quick_check_query)
            valid_scores = result["valid_scores"] if result else 0
            
            bt.logging.info(f"Found {valid_scores} non-zero composite scores in database")
            
            if valid_scores == 0 and not skip_rebuild:
                bt.logging.warning("No valid scores found in database, checking for historical data...")
                
                # Check if we have historical prediction data to rebuild from
                check_predictions_query = """
                    SELECT 
                        COUNT(DISTINCT p.prediction_id) as prediction_count,
                        COUNT(DISTINCT DATE(p.prediction_date)) as days_with_predictions,
                        MIN(p.prediction_date) as earliest_prediction,
                        MAX(p.prediction_date) as latest_prediction
                    FROM predictions p
                    JOIN game_data g ON p.game_id = g.external_id
                    WHERE 
                        p.prediction_date >= ? 
                        AND p.prediction_date <= ?
                        AND g.outcome IS NOT NULL
                        AND g.outcome != 'Unfinished'
                """
                params = (
                    (self.current_date - timedelta(days=self.max_days)).isoformat(),
                    self.current_date.isoformat()
                )
                
                pred_result = await self.db_manager.fetch_one(check_predictions_query, params)
                
                if pred_result and pred_result["prediction_count"] > 0:
                    bt.logging.info(
                        f"Found historical data:\n"
                        f"- {pred_result['prediction_count']} predictions\n"
                        f"- {pred_result['days_with_predictions']} days with predictions\n"
                        f"- Date range: {pred_result['earliest_prediction']} to {pred_result['latest_prediction']}"
                    )
                    
                    # Clear existing scores before rebuild
                    clear_scores_query = "DELETE FROM scores"
                    await self.db_manager.execute_query(clear_scores_query)
                    bt.logging.info("Cleared existing scores, starting rebuild...")
                    
                    await self.rebuild_historical_scores()
                    bt.logging.info("Historical scores rebuilt successfully")
                    return
                else:
                    bt.logging.warning("No historical prediction data found for score rebuild")
            else:
                # Do a more detailed check for completeness
                detailed_check_query = """
                    SELECT 
                        COUNT(DISTINCT day_id) as days_with_scores,
                        COUNT(*) as total_entries,
                        SUM(CASE WHEN composite_score != 0 THEN 1 ELSE 0 END) as valid_entries
                    FROM scores 
                    WHERE CASE 
                        WHEN ? <= ? THEN  -- Normal case: oldest <= current
                            day_id BETWEEN ? AND ?
                        ELSE  -- Wrapped case: oldest > current
                            day_id >= ? OR day_id <= ?
                        END
                """
                
                current_day_index = self.current_day
                
                # Get the oldest prediction date within our max_days window
                oldest_prediction_query = """
                    SELECT MIN(prediction_date) as oldest_date
                    FROM predictions 
                    WHERE prediction_date >= date(?, '-45 days')
                    AND prediction_date <= ?
                """
                oldest_result = await self.db_manager.fetch_one(
                    oldest_prediction_query, 
                    (self.current_date.isoformat(), self.current_date.isoformat())
                )
                
                if oldest_result and oldest_result['oldest_date']:
                    oldest_date = datetime.fromisoformat(oldest_result['oldest_date']).date()
                    days_since_oldest = (self.current_date.date() - oldest_date).days
                    oldest_day_index = (current_day_index - min(days_since_oldest, self.max_days - 1)) % self.max_days
                else:
                    oldest_day_index = (current_day_index - self.max_days + 1) % self.max_days
                
                # Calculate total days accounting for circular buffer
                total_days = (
                    current_day_index - oldest_day_index + 1
                    if current_day_index >= oldest_day_index
                    else (self.max_days - oldest_day_index) + current_day_index + 1
                )
                
                bt.logging.info(f"Checking scores from day {oldest_day_index} to {current_day_index} (total: {total_days} days)")
                
                # Use total_days for expected entries calculation
                expected_entries = self.num_miners * 6 * total_days  # daily + 5 tier scores per miner per day
                
                result = await self.db_manager.fetch_one(
                    detailed_check_query, 
                    (
                        oldest_day_index, current_day_index,  # For the WHEN comparison
                        oldest_day_index, current_day_index,  # For the BETWEEN case
                        oldest_day_index, current_day_index   # For the OR case
                    )
                )
                
                if result:
                    bt.logging.info(
                        f"Score completeness check:\n"
                        f"- Days with scores: {result['days_with_scores']}/{total_days}\n"
                        f"- Total entries: {result['total_entries']}/{expected_entries}\n"
                        f"- Valid entries: {result['valid_entries']}/{expected_entries}"
                    )
                    
                    if (result['days_with_scores'] < total_days or 
                        result['total_entries'] < expected_entries or
                        result['valid_entries'] < expected_entries * 0.5):  # At least 60% should be valid
                        bt.logging.warning("Incomplete or invalid scores detected, initiating rebuild...")
                        await self.rebuild_historical_scores()
                        bt.logging.info("Historical scores rebuilt successfully")
                        return
            
            # Load scores from database into memory arrays
            fetch_scores_query = """
                SELECT miner_uid, day_id, score_type, clv_score, roi_score, 
                       entropy_score, composite_score, sortino_score 
                FROM scores
                WHERE composite_score != 0
                ORDER BY day_id, miner_uid
            """
            
            scores = await self.db_manager.fetch_all(fetch_scores_query)
            
            # Reset score arrays before populating
            self.clv_scores.fill(0)
            self.roi_scores.fill(0)
            self.entropy_scores.fill(0)
            self.sortino_scores.fill(0)
            self.composite_scores.fill(0)
            
            populated_count = 0
            for score in scores:
                miner_uid = score["miner_uid"]
                day_id = score["day_id"]
                score_type = score["score_type"]
                
                if 0 <= miner_uid < self.num_miners and 0 <= day_id < self.max_days:
                    populated_count += 1
                    if score_type == 'daily':
                        self.clv_scores[miner_uid, day_id] = score["clv_score"] or 0.0
                        self.roi_scores[miner_uid, day_id] = score["roi_score"] or 0.0
                        self.entropy_scores[miner_uid, day_id] = score["entropy_score"] or 0.0
                        self.sortino_scores[miner_uid, day_id] = score["sortino_score"] or 0.0
                        self.composite_scores[miner_uid, day_id, 0] = score["composite_score"] or 0.0
                    else:
                        # Map tier scores to the correct index
                        tier_index = list(self.tier_mapping.values()).index(score_type)
                        if 1 <= tier_index < self.composite_scores.shape[2]:
                            self.composite_scores[miner_uid, day_id, tier_index] = score["composite_score"] or 0.0
            
            bt.logging.info(f"Populated {populated_count} score entries from database")
            
        except Exception as e:
            bt.logging.error(f"Error loading scores from database: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _ensure_datetime(self, date):
        if isinstance(date, str):
            return datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        elif isinstance(date, datetime) and date.tzinfo is None:
            return date.replace(tzinfo=timezone.utc)
        return date

    def _ensure_date(self, date_input):
        if isinstance(date_input, str):
            return datetime.fromisoformat(date_input).date()
        elif isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, date):
            return date_input
        else:
            raise TypeError("Date input must be a string, datetime, or date object.")

    async def full_reset(self):
        """
        Perform a full reset of the scoring system, clearing all state and history.
        """
        bt.logging.info("Performing full reset of scoring system...")

        # Reset all score arrays
        self.clv_scores.fill(0)
        self.roi_scores.fill(0)
        self.sortino_scores.fill(0)
        self.amount_wagered.fill(0)
        self.entropy_scores.fill(0)
        self.tiers.fill(1)  # Reset all miners to tier 1
        self.composite_scores.fill(0)

        # Reset current day and date
        self.current_day = 0
        self.current_date = datetime.now(timezone.utc)
        self.last_update_date = None

        # Reset UID sets
        self.invalid_uids = set()
        self.valid_uids = set()
        self.empty_uids = set(range(self.num_miners))

        # Reset entropy system
        self.entropy_system = EntropySystem(self.num_miners, self.max_days)

        # Clear database state
        await self._clear_database_state()
        
        # Repopulate amount_wagered from historical data
        await self.populate_amount_wagered()

        bt.logging.info("Scoring system full reset completed.")

    async def _clear_database_state(self):
        """
        Clear all scoring-related state from the database. 
        """
        try:
            # Clear score_state table
            await self.db_manager.execute_query("DELETE FROM score_state", None)

            # Clear scores table
            await  self.db_manager.execute_query("DELETE FROM scores", None)

            #Clear score state table
            await self.db_manager.execute_query("DELETE FROM score_state", None)

            # Clear miner_stats table
            await self.db_manager.execute_query("DELETE FROM miner_stats", None)

            bt.logging.info("Database state cleared successfully.")
        except Exception as e:
            bt.logging.error(f"Error clearing database state: {e}")
            raise

    def _calculate_clv_scores(self, predictions, closing_line_odds):
        """
        Calculate Closing Line Value (CLV) scores for miners.

        Args:
            predictions (np.ndarray): Structured data with shape (num_predictions, 6).
            closing_line_odds (np.ndarray): Closing line odds with shape (num_games, 3).

        Returns:
            np.ndarray: CLV scores with shape (num_miners,).
        """
        if predictions.size == 0 or closing_line_odds.size == 0:
            bt.logging.error("Predictions or closing line odds are empty.")
            return np.zeros(self.num_miners)

        clv_scores = np.zeros(self.num_miners)
        prediction_counts = np.zeros(self.num_miners)

        # Create a mapping from external_id to index in closing_line_odds
        unique_external_ids = np.unique(predictions[:, 1]).astype(int).tolist()
        external_id_to_index = {external_id: idx for idx, external_id in enumerate(unique_external_ids)}

        for pred in predictions:
            miner_id, external_id, predicted_outcome, predicted_odds, payout, wager = pred
            miner_id = int(miner_id)
            external_id = int(external_id)
            predicted_outcome = int(predicted_outcome)

            if 0 <= miner_id < self.num_miners:
                if (
                    external_id in external_id_to_index
                    and predicted_outcome < closing_line_odds.shape[1]
                ):
                    closing_odds_index = external_id_to_index[external_id]
                    closing_odds = closing_line_odds[
                        closing_odds_index, predicted_outcome
                    ]

                    if closing_odds > 0:
                        clv = predicted_odds / closing_odds
                        if np.isfinite(clv):
                            clv_scores[miner_id] += clv
                            prediction_counts[miner_id] += 1
                        else:
                            bt.logging.warning(
                                f"Invalid CLV value for miner {miner_id} on external_id {external_id}."
                            )
                    elif predicted_outcome == 2:
                        continue  # Tie outcome
                        # bt.logging.trace(f"No tie odds for game {game_id}. Skipping CLV calculation for this prediction.")
                    else:
                        bt.logging.warning(
                            f"Closing odds are zero for external_id {external_id}, outcome {predicted_outcome}."
                        )
                else:
                    bt.logging.warning(
                        f"Invalid external_id or predicted_outcome: {external_id}, {predicted_outcome}"
                    )
            else:
                bt.logging.warning(
                    f"Invalid miner_id {miner_id} encountered. Skipping this prediction."
                )

        # Avoid division by zero and compute average CLV per miner
        mask = prediction_counts > 0
        clv_scores[mask] /= prediction_counts[mask]

        return clv_scores

    def _calculate_roi_scores(self, predictions, results):
        """
        Calculate Return on Investment (ROI) scores for miners.

        Args:
            predictions (np.ndarray): Structured prediction data with shape (num_predictions, 6).
            results (np.ndarray): Array of game results with shape (num_games, 2).

        Returns:
            np.ndarray: ROI scores with shape (num_miners,), representing percentage returns.
        """
        if predictions.size == 0 or results.size == 0:
            bt.logging.error("Predictions or game results are empty.")
            return np.zeros(self.num_miners)

        roi_scores = np.zeros(self.num_miners)
        prediction_counts = np.zeros(self.num_miners)

        # Create a dictionary mapping external_id to outcome
        game_outcomes = dict(results)

        for pred in predictions:
            miner_id, game_id, predicted_outcome, predicted_odds, payout, wager = pred
            miner_id = int(miner_id)
            game_id = int(game_id)

            # Fetch the actual outcome
            actual_outcome = game_outcomes.get(game_id)
            if actual_outcome is None:
                bt.logging.error(
                    f"No actual outcome found for game_id {game_id}. Skipping ROI calculation for miner {miner_id}."
                )
                continue

            if wager == 0:
                bt.logging.error(
                    f"Wager is zero for miner {miner_id} on game_id {game_id}. Skipping ROI calculation."
                )
                continue

            roi = (payout - wager) / wager  # ROI as a percentage

            # bt.logging.debug(
            #     f"Miner {miner_id} | Game ID (External ID) {game_id} | Predicted Outcome: {predicted_outcome} | "
            #     f"Actual Outcome: {actual_outcome} | Wager: {wager} | Payout: {payout} | ROI: {roi}"
            # )

            if np.isfinite(roi):
                roi_scores[miner_id] += roi
                prediction_counts[miner_id] += 1
            else:
                bt.logging.error(
                    f"Invalid ROI value ({roi}) for miner {miner_id} on game_id {game_id}."
                )

        # Compute average ROI per miner without normalization
        mask = prediction_counts > 0
        roi_scores[mask] /= prediction_counts[mask]

        bt.logging.info(
            f"ROI Scores - min: {roi_scores.min():.4f}, max: {roi_scores.max():.4f}, mean: {roi_scores.mean():.4f}"
        )

        return roi_scores

    def _calculate_risk_scores(self, predictions, results):
        """
        Calculate Risk/Reward (R/R) scores for miners based on daily predictions and results.
        
        Args:
            predictions (np.ndarray): Structured prediction data with shape (num_predictions, 6).
            results (np.ndarray): Array of game results with shape (num_games, 2).
        
        Returns:
            np.ndarray: R/R scores with shape (num_miners,).
        """
        risk_scores = np.zeros(self.num_miners)
        game_outcomes = dict(results)
        
        for pred in predictions:
            miner_id, game_id, predicted_outcome, predicted_odds, payout, wager = pred
            miner_id = int(miner_id)
            game_id = int(game_id)
            
            actual_outcome = game_outcomes.get(game_id)
            if actual_outcome is None or wager <= 0 or predicted_odds <= 0:
                continue
            
            roi = (payout - wager) / wager # ROI as a percentage
            
            implied_prob = 1 / predicted_odds
            adjusted_prob = implied_prob * (1 - 0.05) #adjust for an average 5% house edge
            risk_score = min(1 - adjusted_prob, 1) #risk is the inverse of the adjusted probability, max of 100%
            inverse_risk_score = (1 - risk_score) + 0.00001 #add a small buffer to avoid division by zero, jic

            if roi > 0:
                rr_score = roi / risk_score # "risk adjusted ROI" - roi % must exceed risk to be >1, will generally be < 1 
            else:
                rr_score = max(roi / inverse_risk_score, -10)  # lost prediction, risk adjusted ROI is negative and inversely proportional to risk (more risk = closer to -1), capped at -10 penalty

            risk_scores[miner_id] += rr_score
        
        # Calculate the average R/R score for each miner
        prediction_counts = np.bincount(
            predictions[:, 0].astype(int), minlength=self.num_miners
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            risk_scores = np.divide(
                risk_scores, prediction_counts, 
                out=np.zeros_like(risk_scores), 
                where=prediction_counts != 0
            )
        
        return risk_scores

    async def _update_composite_scores(self, historical_day=None):
        day_index = self.current_day if historical_day is None else historical_day
        
        # Get miners with predictions from miner_stats with debug info
        query = """
        SELECT 
            COUNT(*) as total_miners,
            SUM(CASE WHEN miner_last_prediction_date IS NOT NULL THEN 1 ELSE 0 END) as miners_with_dates,
            SUM(CASE WHEN miner_lifetime_predictions > 0 THEN 1 END) as miners_with_predictions
        FROM miner_stats
        """
        stats = await self.db_manager.fetch_one(query)
        bt.logging.info(f"Database stats: {stats}")

        # Get active miners
        query = """
        SELECT miner_uid FROM miner_stats 
        WHERE miner_last_prediction_date IS NOT NULL
        OR miner_lifetime_predictions > 0
        """
        active_miners = await self.db_manager.fetch_all(query)
        has_prediction_history = np.zeros(self.num_miners, dtype=bool)
        for miner in active_miners:
            has_prediction_history[miner['miner_uid']] = True

        # Create a 2D valid_history array
        valid_history = np.zeros((self.num_miners, self.max_days), dtype=bool)
        valid_history[:, day_index] = has_prediction_history  # Set current day's history

        # Get component scores for the day
        clv = self.clv_scores[:, day_index]
        roi = self.roi_scores[:, day_index]
        entropy = self.entropy_scores[:, day_index]
        sortino = self.sortino_scores[:, day_index]

        # Log non-zero scores before normalization
        bt.logging.info(f"Miners with non-zero CLV scores: {np.sum(clv != 0)}")
        bt.logging.info(f"Miners with non-zero ROI scores: {np.sum(roi != 0)}")
        bt.logging.info(f"Miners with non-zero entropy scores: {np.sum(entropy != 0)}")
        bt.logging.info(f"Miners with non-zero sortino scores: {np.sum(sortino != 0)}")

        def normalize_with_negatives(scores, has_history):
            """
            Normalize scores while preserving non-zero values and handling negatives.
            """
            normalized = scores.copy()
            non_zero_mask = scores != 0
            
            if not np.any(non_zero_mask):
                return normalized
                
            # Get scores for miners with both history and non-zero values
            valid_scores = scores[non_zero_mask]  # Changed to use all non-zero scores
            if len(valid_scores) == 0:
                return normalized
                
            min_score = valid_scores.min()
            max_score = valid_scores.max()
            score_range = max_score - min_score
            
            if score_range > 0:
                # First shift all non-zero scores to be positive
                if min_score < 0:
                    normalized[non_zero_mask] = scores[non_zero_mask] - min_score
                    max_score = max_score - min_score
                    
                # Then normalize to [0,1] range
                normalized[non_zero_mask] = normalized[non_zero_mask] / max_score
                
                # Ensure we don't lose scores due to floating point errors
                normalized[non_zero_mask] = np.maximum(normalized[non_zero_mask], 1e-10)
            
            bt.logging.debug(f"Normalization stats:")
            bt.logging.debug(f"  Original non-zero scores: {np.sum(scores != 0)}")
            bt.logging.debug(f"  Normalized non-zero scores: {np.sum(normalized != 0)}")
            bt.logging.debug(f"  Min: {min_score:.4f}, Max: {max_score:.4f}, Range: {score_range:.4f}")
            
            # Apply history mask after normalization
            normalized[~has_history] = 0
            
            return normalized

        # Normalize components
        norm_clv = normalize_with_negatives(clv, has_prediction_history)
        norm_roi = normalize_with_negatives(roi, has_prediction_history)
        norm_entropy = normalize_with_negatives(entropy, has_prediction_history)
        norm_sortino = normalize_with_negatives(sortino, has_prediction_history)

        # Calculate composite score
        composite = (
            self.clv_weight * norm_clv
            + self.roi_weight * norm_roi
            + self.sortino_weight * norm_sortino
            + self.entropy_weight * norm_entropy
        )

        # Only zero out scores once for miners with no history
        composite[~has_prediction_history] = 0
        
        bt.logging.info(f"Miners with non-zero composite scores: {np.sum(composite != 0)}")
        
        # Store the base composite score
        self.composite_scores[:, day_index, 0] = composite

        # Calculate rolling averages for each tier - simplified to just average over window
        for tier in range(1, 6):
            window = self.tier_configs[tier + 1]["window"]
            start_day = (day_index - window + 1) % self.max_days
            
            # Handle circular buffer for window scores
            if start_day <= day_index:
                window_scores = self.composite_scores[:, start_day:day_index + 1, 0]
            else:
                window_scores = np.concatenate([
                    self.composite_scores[:, start_day:, 0],
                    self.composite_scores[:, :day_index + 1, 0]
                ], axis=1)
            
            # Simple average over window, only considering non-zero scores
            valid_scores = window_scores != 0
            valid_days = np.sum(valid_scores, axis=1)
            rolling_avg = np.zeros(self.num_miners)
            mask = valid_days > 0
            
            if np.any(mask):
                rolling_avg[mask] = np.sum(window_scores[mask] * valid_scores[mask], axis=1) / valid_days[mask]
            
            self.composite_scores[:, day_index, tier] = rolling_avg

        # Add logging about prediction history
        bt.logging.debug(f"Miners with prediction history for day {day_index}: {np.sum(has_prediction_history)}/{len(has_prediction_history)}")
        bt.logging.debug(f"Composite scores for day {day_index}:")
        bt.logging.debug(
            f"CLV: min={clv[has_prediction_history].min() if np.any(has_prediction_history) else 0:.4f}, "
            f"max={clv[has_prediction_history].max() if np.any(has_prediction_history) else 0:.4f}, "
            f"mean={clv[has_prediction_history].mean() if np.any(has_prediction_history) else 0:.4f}"
        )
        bt.logging.debug(
            f"ROI: min={roi[has_prediction_history].min() if np.any(has_prediction_history) else 0:.4f}, "
            f"max={roi[has_prediction_history].max() if np.any(has_prediction_history) else 0:.4f}, "
            f"mean={roi[has_prediction_history].mean() if np.any(has_prediction_history) else 0:.4f}"
        )
        bt.logging.debug(
            f"Sortino: min={sortino[has_prediction_history].min() if np.any(has_prediction_history) else 0:.4f}, "
            f"max={sortino[has_prediction_history].max() if np.any(has_prediction_history) else 0:.4f}, "
            f"mean={sortino[has_prediction_history].mean() if np.any(has_prediction_history) else 0:.4f}"
        )
        bt.logging.debug(
            f"Entropy: min={entropy[has_prediction_history].min() if np.any(has_prediction_history) else 0:.4f}, "
            f"max={entropy[has_prediction_history].max() if np.any(has_prediction_history) else 0:.4f}, "
            f"mean={entropy[has_prediction_history].mean() if np.any(has_prediction_history) else 0:.4f}"
        )

    async def initialize(self):
        """Async initialization method to be called after constructor"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                self.init = await self.load_state()
                await self.scoring_data.initialize()
                self.advance_day(self.current_date.date())
                await self.populate_amount_wagered()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    bt.logging.warning(f"Initialization attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    bt.logging.error(f"Failed to initialize after {max_retries} attempts")
                    raise

    async def rebuild_historical_scores(self):
        """Rebuilds historical scores for the past 45 days using existing prediction and game data."""
        try:
            bt.logging.info("Starting complete historical score rebuild...")
            
            end_date = self.current_date
            start_date = end_date - timedelta(days=self.max_days)
            
            bt.logging.info(f"Rebuilding scores from {start_date} to {end_date}")
            
            # Track valid history periods
            valid_history = np.zeros((self.num_miners, self.max_days), dtype=bool)
            
            # Process each day
            for days_ago in range(self.max_days - 1, -1, -1):
                process_date = end_date - timedelta(days=days_ago)
                historical_day = (self.current_day - days_ago) % self.max_days
                
                # Get games and predictions like the normal scoring run
                games_query = """
                    SELECT g.*
                    FROM game_data g
                    WHERE DATE(g.event_start_date) = DATE(?)
                    AND g.outcome IS NOT NULL
                    AND g.outcome != 'Unfinished'
                    AND g.outcome != 3
                """
                
                day_games = await self.db_manager.fetch_all(games_query, (process_date.isoformat(),))
                game_ids = [g['external_id'] for g in day_games]
                
                # Modified predictions query to get predictions for specific games
                predictions_query = """
                    SELECT p.*
                    FROM predictions p
                    WHERE p.game_id IN ({})
                """.format(','.join('?' * len(game_ids)))
                
                day_predictions = await self.db_manager.fetch_all(predictions_query, game_ids)

                bt.logging.info(
                    f"Processing {process_date}: {len(day_games)} games, "
                    f"{len(day_predictions)} predictions with valid IDs"
                )

                # Add a diagnostic query to understand the missing IDs
                diagnostic_query = """
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN prediction_id IS NULL THEN 1 ELSE 0 END) as null_ids,
                        SUM(CASE WHEN prediction_id IS NOT NULL THEN 1 ELSE 0 END) as valid_ids
                    FROM predictions 
                    WHERE game_id IN ({})
                """.format(','.join('?' * len(game_ids)))
                
                diagnostic_result = await self.db_manager.fetch_one(diagnostic_query, game_ids)
                if diagnostic_result:
                    bt.logging.info(
                        f"Prediction ID statistics for {process_date}:\n"
                        f"  Total predictions: {diagnostic_result['total_predictions']}\n"
                        f"  Missing IDs: {diagnostic_result['null_ids']}\n"
                        f"  Valid IDs: {diagnostic_result['valid_ids']}"
                    )

                if not day_games or not day_predictions:
                    continue
                    
                try:
                    # Format arrays exactly as in normal scoring
                    game_ids = [g['external_id'] for g in day_games]
                    game_id_to_idx = {gid: idx for idx, gid in enumerate(game_ids)}
                    
                    # Track game IDs for debugging
                    game_ids = set(g['external_id'] for g in day_games)
                    bt.logging.info(f"Processing games: {game_ids}")

                    # Add games to entropy system first and wait for all to complete
                    game_tasks = []
                    for game in day_games:
                        try:
                            odds = [
                                float(game['team_a_odds']), 
                                float(game['team_b_odds']), 
                                float(game['tie_odds']) if game['tie_odds'] is not None else 0.0
                            ]
                            task = self.entropy_system.add_new_game(
                                game_id=game['external_id'],
                                num_outcomes=3 if game['tie_odds'] else 2,
                                odds=odds
                            )
                            game_tasks.append(task)
                            bt.logging.debug(f"Added task for game {game['external_id']}")
                        except Exception as e:
                            bt.logging.error(f"Error adding game {game['external_id']} to entropy system: {e}")
                            continue
                    
                    # Wait for all games to be added
                    await asyncio.gather(*game_tasks)
                    bt.logging.info("All game tasks completed")

                    # Now process predictions, but only for games we just added
                    pred_array = np.array([
                        [
                            int(p['miner_uid']), 
                            int(p['game_id']), 
                            int(p['predicted_outcome']),
                            float(p['predicted_odds']) if p['predicted_odds'] is not None else 0.0,
                            float(p.get('payout', 0)) if p.get('payout') is not None else 0.0,
                            float(p['wager']) if p['wager'] is not None else 0.0
                        ]
                        for p in day_predictions
                        if int(p['game_id']) in game_ids  # Only include predictions for games we added
                        and p['predicted_odds'] is not None 
                        and p['wager'] is not None
                    ])
                    
                    # Create results and closing line odds arrays
                    results = np.array([
                        [int(g['external_id']), int(g['outcome'])]
                        for g in day_games
                        if g['outcome'] not in ['Unfinished', None, 3]
                    ])
                    
                    closing_line_odds = np.array([
                        [float(g['team_a_odds']), float(g['team_b_odds']), 
                         float(g['tie_odds']) if g['tie_odds'] else 0.0]
                        for g in day_games
                    ])

                    if pred_array.size > 0:
                        # Mark this day as having valid history for these miners
                        active_miners = np.unique(pred_array[:, 0].astype(int))
                        valid_history[active_miners, historical_day] = True
                        
                        # Calculate scores for this day
                        roi_scores = self._calculate_roi_scores(pred_array, results)
                        clv_scores = self._calculate_clv_scores(pred_array, closing_line_odds)
                        sortino_scores = self._calculate_risk_scores(pred_array, results)
                        
                        # Update score arrays
                        self.roi_scores[:, historical_day] = roi_scores
                        self.clv_scores[:, historical_day] = clv_scores
                        self.sortino_scores[:, historical_day] = sortino_scores
                        
                        # Add predictions to entropy system - only process predictions that have IDs
                        valid_predictions = [
                            pred for pred in day_predictions 
                            if 'prediction_id' in pred and pred['prediction_id'] is not None
                        ]
                        
                        if len(valid_predictions) != len(day_predictions):
                            bt.logging.warning(
                                f"Skipping {len(day_predictions) - len(valid_predictions)} predictions "
                                f"without prediction IDs for {process_date}"
                            )
                        
                        for pred in valid_predictions:
                            try:
                                self.entropy_system.add_prediction(
                                    prediction_id=pred['prediction_id'],
                                    miner_uid=pred['miner_uid'],
                                    game_id=pred['game_id'],
                                    predicted_outcome=pred['predicted_outcome'],
                                    wager=float(pred['wager']),
                                    predicted_odds=float(pred['predicted_odds']),
                                    prediction_date=pred['prediction_date'],
                                    historical_rebuild=True
                                )
                            except Exception as e:
                                bt.logging.error(
                                    f"Error adding prediction to entropy system:\n"
                                    f"  Prediction ID: {pred.get('prediction_id')}\n"
                                    f"  Game ID: {pred.get('game_id')}\n"
                                    f"  Miner: {pred.get('miner_uid')}\n"
                                    f"  Error: {str(e)}"
                                )
                                continue
                        
                        # Get entropy scores
                        entropy_scores = self.entropy_system.get_current_ebdr_scores(
                            datetime.combine(process_date, datetime.min.time()).replace(tzinfo=timezone.utc),
                            historical_day,
                            game_ids
                        )
                        
                        self.entropy_scores[:, historical_day] = entropy_scores[:, historical_day]
                        
                        # Update composite scores for this day
                        await self._update_composite_scores(historical_day)
                        
                        # After processing each day, ensure continuity in tier scores
                        for tier in range(1, 6):
                            window = self.tier_configs[tier + 1]["window"]
                            start_window = (historical_day - window + 1) % self.max_days
                            
                            # Check which miners have any history in this window
                            miners_with_history = np.any(valid_history[:, max(0, historical_day - window + 1):historical_day + 1], axis=1)
                            
                            if np.any(miners_with_history):
                                # Ensure all days in window have scores for these miners
                                for d in range(max(0, historical_day - window + 1), historical_day + 1):
                                    day_idx = d % self.max_days
                                    if not np.any(self.composite_scores[miners_with_history, day_idx, tier]):
                                        # Recalculate tier score for this day
                                        await self._update_composite_scores(day_idx)
                                        bt.logging.debug(f"Filled gap in tier {tier} scores for day {day_idx}")
                    
                except Exception as e:
                    bt.logging.error(f"Error processing day {process_date}: {str(e)}")
                    bt.logging.error(traceback.format_exc())
                    continue
            
            return True
            
        except Exception as e:
            bt.logging.error(f"Error in historical rebuild: {e}")
            bt.logging.error(traceback.format_exc())
            return False


    async def retroactively_calculate_entropy_scores(self):
        """Retroactively calculate entropy scores for all historical predictions."""
        try:
            # First get all games from the last 45 days
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
            games_query = """
                SELECT 
                    id, external_id, team_a_odds, team_b_odds, tie_odds,
                    event_start_date, outcome
                FROM game_data 
                WHERE event_start_date >= ?
                ORDER BY event_start_date ASC
            """
            games = await self.db_manager.fetch_all(games_query, (cutoff_date,))
            bt.logging.info(f"Found {len(games)} games to process")

            # Add games to entropy system
            for game in games:
                try:
                    odds = [
                        float(game['team_a_odds']), 
                        float(game['team_b_odds']), 
                        float(game['tie_odds']) if game['tie_odds'] else 0.0
                    ]
                    await self.entropy_system.add_new_game(
                        game_id=game['external_id'],
                        num_outcomes=3 if game['tie_odds'] else 2,
                        odds=odds
                    )
                    
                    # If game is finished, mark it as closed
                    if game['outcome'] not in ['Unfinished', None]:
                        self.entropy_system.close_game(game['external_id'])
                        
                except Exception as e:
                    bt.logging.error(f"Error adding game {game['external_id']} to entropy system: {e}")
                    continue

            # Now get all predictions for these games
            predictions_query = """
                SELECT 
                    p.prediction_id, p.miner_uid, p.game_id, 
                    p.predicted_outcome, p.predicted_odds, p.wager,
                    p.prediction_date, p.processed
                FROM predictions p
                JOIN game_data g ON p.game_id = g.external_id
                WHERE g.event_start_date >= ?
                ORDER BY p.prediction_date ASC
            """
            predictions = await self.db_manager.fetch_all(predictions_query, (cutoff_date,))
            bt.logging.info(f"Found {len(predictions)} predictions to process")

            # Process predictions in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(predictions), chunk_size):
                chunk = predictions[i:i+chunk_size]
                bt.logging.info(f"Processing predictions chunk {i//chunk_size + 1}")
                
                for pred in chunk:
                    try:
                        self.entropy_system.add_prediction(
                            prediction_id=pred['prediction_id'],
                            miner_uid=pred['miner_uid'],
                            game_id=pred['game_id'],
                            predicted_outcome=pred['predicted_outcome'],
                            wager=float(pred['wager']),
                            predicted_odds=float(pred['predicted_odds']),
                            prediction_date=pred['prediction_date']
                        )
                    except Exception as e:
                        bt.logging.error(f"Error processing prediction {pred['prediction_id']}: {e}")
                        continue

            # Save the entropy system state
            self.entropy_system.save_state()
            bt.logging.info("Completed retroactive entropy score calculation")

        except Exception as e:
            bt.logging.error(f"Error in retroactive entropy calculation: {e}")
            bt.logging.error(traceback.format_exc())

    async def save_scores_for_day(self, day_id):
        """Save scores for a specific day to the database."""
        try:
            # First ensure miner_stats entries exist for all miners
            miner_stats_query = """
            INSERT OR IGNORE INTO miner_stats (miner_uid)
            VALUES (?)
            """
            miner_stats_params = [(i,) for i in range(self.num_miners)]
            await self.db_manager.executemany(miner_stats_query, miner_stats_params)

            # Delete existing scores for this day
            await self.db_manager.execute_query(
                "DELETE FROM scores WHERE day_id = ?",
                (day_id,)
            )
            
            # Batch insert scores with UPSERT
            score_records = []
            for miner_id in range(self.num_miners):
                # Daily scores
                score_records.append((
                    miner_id, day_id, 'daily',
                    float(self.clv_scores[miner_id, day_id]),
                    float(self.roi_scores[miner_id, day_id]),
                    float(self.entropy_scores[miner_id, day_id]),
                    float(self.composite_scores[miner_id, day_id, 0]),
                    float(self.sortino_scores[miner_id, day_id])
                ))
                
                # Tier-specific scores
                for tier_idx in range(1, self.composite_scores.shape[2]):
                    if tier_idx in self.tier_mapping:
                        score_records.append((
                            miner_id, day_id, self.tier_mapping[tier_idx],
                            None,  # clv_score
                            None,  # roi_score
                            None,  # entropy_score
                            float(self.composite_scores[miner_id, day_id, tier_idx]),
                            None   # sortino_score
                        ))

            # Insert with conflict resolution
            insert_query = """
            INSERT INTO scores 
                (miner_uid, day_id, score_type, clv_score, roi_score, entropy_score, composite_score, sortino_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(miner_uid, day_id, score_type) DO UPDATE SET
                clv_score=excluded.clv_score,
                roi_score=excluded.roi_score,
                entropy_score=excluded.entropy_score,
                composite_score=excluded.composite_score,
                sortino_score=excluded.sortino_score
            """
            
            await self.db_manager.executemany(insert_query, score_records)
            
        except Exception as e:
            bt.logging.error(f"Error saving scores for day {day_id}: {str(e)}")
            raise

    # Diagnostic query to understand miner_stats data
    async def debug_miner_stats(self):
        # Diagnostic query to understand miner_stats data
        debug_query = """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(CASE WHEN miner_uid < 256 THEN 1 END) as valid_miners,
            COUNT(CASE WHEN miner_uid < 256 AND miner_last_prediction_date IS NOT NULL THEN 1 END) as miners_with_dates,
            COUNT(CASE WHEN miner_uid < 256 AND miner_lifetime_predictions > 0 THEN 1 END) as miners_with_predictions,
            COUNT(CASE WHEN miner_uid >= 256 THEN 1 END) as invalid_miners
        FROM miner_stats;
        """
        debug_stats = await self.db_manager.fetch_one(debug_query)
        bt.logging.info(f"Detailed stats: {debug_stats}")

        # Also let's see some sample data
        sample_query = """
        SELECT miner_uid, miner_last_prediction_date, miner_lifetime_predictions
        FROM miner_stats
        WHERE miner_uid < 256
        ORDER BY miner_last_prediction_date DESC
        LIMIT 5;
        """
        sample_data = await self.db_manager.fetch_all(sample_query)
        bt.logging.info(f"Sample miner data: {sample_data}")