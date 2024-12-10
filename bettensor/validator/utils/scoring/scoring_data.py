import os
import json
import time
import asyncio
import traceback
import threading
from datetime import datetime, timezone
import bittensor as bt
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import async_timeout
import pytz
import numpy as np
from datetime import datetime, timedelta, timezone
from bettensor.validator.utils.database.database_manager import DatabaseManager
from typing import List, Tuple, Dict
from collections import defaultdict
import random


class ScoringData:
    def __init__(self, scoring_system):
        self.scoring_system = scoring_system
        self.db_manager = scoring_system.db_manager
        self.validator = scoring_system.validator
        self.miner_stats = defaultdict(lambda: {
            'clv': 0.0,
            'roi': 0.0,
            'entropy': 0.0,
            'sortino': 0.0,
            'composite_daily': 0.0,
            'tier_scores': {}
        })

    async def initialize(self):
        """Async initialization method to be called after constructor"""
        await self.init_miner_stats()

    @property
    def current_day(self):
        return int(self.scoring_system.current_day)

    @property
    def num_miners(self):
        return self.scoring_system.num_miners

    @property
    def tiers(self):
        return self.scoring_system.tiers

    async def preprocess_for_scoring(self, date_str):
        bt.logging.debug(f"Preprocessing for scoring on date: {date_str}")

        # Step 1: Get closed games for that day (Outcome != 3)
        closed_games = await self._fetch_closed_game_data(date_str)

        if len(closed_games) == 0:
            bt.logging.warning("No closed games found for the given date.")
            return np.array([]), np.array([]), np.array([])

        game_ids = [game["external_id"] for game in closed_games]

        # Step 2: Get all predictions for each of those closed games
        predictions = await self._fetch_predictions(game_ids)
 
        # Step 3: Ensure predictions have their payout calculated and outcome updated
        bt.logging.debug("Updating predictions with payout")
        predictions = await self._update_predictions_with_payout(predictions, closed_games)

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
                    float(game["closing_line_odds"][0]),  # team_a_odds
                    float(game["closing_line_odds"][1]),  # team_b_odds
                    float(game["closing_line_odds"][2]),  # tie_odds
                ]
                for game in closed_games
            ]
        )

        bt.logging.debug(f"Structured predictions shape: {structured_predictions.shape}")
        bt.logging.debug(f"Closing line odds shape: {closing_line_odds.shape}")
        bt.logging.debug(f"Results shape: {results.shape}")

        return structured_predictions, closing_line_odds, results

    async def _fetch_closed_game_data(self, date_str):
        """Fetch closed game data for a specific date."""
        try:
            # Parse the input date
            target_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            
            # Format start and end dates
            formatted_start = target_date.replace(hour=0, minute=0, second=0).isoformat()
            formatted_end = target_date.replace(hour=23, minute=59, second=59).isoformat()
            
            query = """
                SELECT 
                    external_id,
                    event_start_date,
                    team_a_odds,
                    team_b_odds,
                    tie_odds,
                    outcome
                FROM game_data
                WHERE event_start_date BETWEEN :start_date AND :end_date
                AND outcome IS NOT NULL
                AND outcome != 'Unfinished'
                AND outcome != 3
                ORDER BY event_start_date ASC
            """
            
            params = {
                "start_date": formatted_start,
                "end_date": formatted_end
            }
            
            games = await self.db_manager.fetch_all(query, params)
            
            if not games:
                bt.logging.warning(f"No closed games found for date {date_str}")
                return np.array([])
            
            # Process the games to combine odds into the expected format
            processed_games = []
            for game in games:
                processed_game = {
                    'external_id': game['external_id'],
                    'event_start_date': game['event_start_date'],
                    'closing_line_odds': [
                        game['team_a_odds'],
                        game['team_b_odds'],
                        game['tie_odds'] if game['tie_odds'] is not None else float('inf')
                    ],
                    'outcome': game['outcome']
                }
                processed_games.append(processed_game)
            
            bt.logging.info(f"Found {len(processed_games)} closed games for date {date_str}")
            return np.array(processed_games)
            
        except Exception as e:
            bt.logging.error(f"Error fetching closed game data: {e}")
            bt.logging.error(traceback.format_exc())
            return np.array([])

    async def _fetch_predictions(self, game_ids):
        """Fetch predictions for given game IDs"""
        if not game_ids:
            return []
        
        query = """
            SELECT * FROM predictions
            WHERE game_id IN ({})
        """.format(','.join(':id_' + str(i) for i in range(len(game_ids))))

        # Convert list of game_ids to dictionary of named parameters
        params = {f'id_{i}': game_id for i, game_id in enumerate(game_ids)}
        
        return await self.db_manager.fetch_all(query, params)

    async def _update_predictions_with_payout(self, predictions, closed_games):
        bt.logging.debug(f"Predictions: {len(predictions)}")
        bt.logging.debug(f"Closed games: {len(closed_games)}")
        game_outcomes = {game["external_id"]: game["outcome"] for game in closed_games}
        bt.logging.debug(f"Game outcomes: {game_outcomes}")
        skipped_with_payout = 0
        skipped_unfinished = 0
        processed_predictions = 0
        
        for pred in predictions:
            game_id = pred["game_id"]
            miner_uid = pred["miner_uid"]
            outcome = game_outcomes.get(game_id)
            
            # Track different skip reasons
            if pred["payout"] is not None:
                skipped_with_payout += 1
                continue
                
            if outcome in [None, "Unfinished", 3]:
                skipped_unfinished += 1
                continue
            
            wager = float(pred["wager"])
            predicted_outcome = pred["predicted_outcome"]
            
            # Calculate payout for valid outcomes
            if int(predicted_outcome) == int(outcome):
                payout = wager * float(pred["predicted_odds"])
                bt.logging.debug(f"Correct prediction for miner {miner_uid}: Payout set to {payout}")
            else:
                payout = 0.0
                bt.logging.debug(f"Incorrect prediction for miner {miner_uid}: Payout set to 0.0")
            
            # Update prediction record
            await self.db_manager.execute_query(
                """
                UPDATE predictions
                SET payout = ?, outcome = ?
                WHERE prediction_id = ?
                """,
                (payout, outcome, pred["prediction_id"]),
            )
            pred["payout"] = payout
            pred["outcome"] = outcome
            processed_predictions += 1

        bt.logging.info(f"Prediction processing summary:")
        bt.logging.info(f"- Total predictions: {len(predictions)}")
        bt.logging.info(f"- Processed: {processed_predictions}")
        bt.logging.info(f"- Skipped (already had payout): {skipped_with_payout}")
        bt.logging.info(f"- Skipped (unfinished games): {skipped_unfinished}")
        
        return predictions

    async def validate_data_integrity(self):
        """Validate that all predictions reference closed games with valid outcomes."""
        invalid_predictions = await self.db_manager.fetch_all(
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
            bt.logging.error(f"Sample of invalid predictions: {invalid_predictions[:5]}")
            raise ValueError("Data integrity check failed: Some predictions reference invalid or open games.")
        else:
            bt.logging.debug("All predictions reference valid closed games.")

    async def init_miner_stats(self):
        """Initialize or update miner stats with proper retry logic and transaction management."""
        bt.logging.trace("Initializing Miner Stats")
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Use a longer timeout for the entire operation
                async with async_timeout.timeout(60):  # 60 second timeout
                    async with self.db_manager.get_long_running_session() as session:
                        # First ensure all miners have a basic entry
                        insert_base_query = text("""
                        INSERT OR IGNORE INTO miner_stats (
                            miner_uid, miner_hotkey, miner_coldkey, miner_status,
                            miner_rank, miner_cash, miner_current_incentive, miner_current_tier,
                            miner_current_scoring_window, miner_current_composite_score,
                            miner_current_sharpe_ratio, miner_current_sortino_ratio,
                            miner_current_roi, miner_current_clv_avg, miner_lifetime_earnings,
                            miner_lifetime_wager_amount, miner_lifetime_roi, miner_lifetime_predictions,
                            miner_lifetime_wins, miner_lifetime_losses, miner_win_loss_ratio,
                            miner_last_prediction_date
                        ) VALUES (
                            :miner_uid, :hotkey, :coldkey, :status,
                            0, 0.0, 0.0, 1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, NULL
                        )
                        """)
                        
                        # Prepare base values as dictionaries with named parameters
                        base_values = [
                            {
                                "miner_uid": uid,
                                "hotkey": self.validator.metagraph.hotkeys[uid],
                                "coldkey": self.validator.metagraph.coldkeys[uid],
                                "status": 'active' if self.validator.metagraph.active[uid] else 'inactive'
                            }
                            for uid in range(min(len(self.validator.metagraph.hotkeys), self.num_miners))
                        ]
                        
                        if base_values:
                            # Process in smaller batches
                            batch_size = 25
                            for i in range(0, len(base_values), batch_size):
                                batch = base_values[i:i + batch_size]
                                bt.logging.debug(f"Processing batch {i//batch_size + 1} of {(len(base_values) + batch_size - 1)//batch_size}")
                                
                                try:
                                    await session.execute(insert_base_query, batch)
                                    await session.commit()
                                    
                                    # Also insert into backup table
                                    backup_query = text(insert_base_query.text.replace('miner_stats', 'miner_stats_backup'))
                                    await session.execute(backup_query, batch)
                                    await session.commit()
                                except Exception as e:
                                    bt.logging.error(f"Error processing batch: {e}")
                                    raise

                        # Update lifetime statistics
                        update_lifetime_query = text("""
                        WITH prediction_stats AS (
                            SELECT 
                                miner_uid,
                                COUNT(*) as total_predictions,
                                SUM(CASE WHEN payout > 0 THEN 1 ELSE 0 END) as wins,
                                SUM(CASE WHEN payout = 0 THEN 1 ELSE 0 END) as losses,
                                SUM(payout) as total_earnings,
                                SUM(wager) as total_wager,
                                MAX(prediction_date) as last_prediction
                            FROM predictions
                            GROUP BY miner_uid
                        )
                        UPDATE miner_stats
                        SET
                            miner_lifetime_predictions = COALESCE(ps.total_predictions, 0),
                            miner_lifetime_wins = COALESCE(ps.wins, 0),
                            miner_lifetime_losses = COALESCE(ps.losses, 0),
                            miner_lifetime_earnings = COALESCE(ps.total_earnings, 0),
                            miner_lifetime_wager_amount = COALESCE(ps.total_wager, 0),
                            miner_win_loss_ratio = CASE 
                                WHEN COALESCE(ps.losses, 0) > 0 
                                THEN CAST(COALESCE(ps.wins, 0) AS REAL) / COALESCE(ps.losses, 0)
                                ELSE COALESCE(ps.wins, 0)
                            END,
                            miner_last_prediction_date = ps.last_prediction
                        FROM prediction_stats ps
                        WHERE miner_stats.miner_uid = ps.miner_uid;
                        """)
                        
                        await session.execute(update_lifetime_query)
                        await session.commit()
                        bt.logging.debug("Updated lifetime statistics for miners.")

                        # Clean up and sync backup table
                        cleanup_query = text("""
                        DELETE FROM miner_stats 
                        WHERE miner_uid >= 256 
                        OR miner_uid IN (
                            SELECT miner_uid 
                            FROM miner_stats 
                            GROUP BY miner_uid 
                            HAVING COUNT(*) > 1
                        );
                        """)
                        await session.execute(cleanup_query)
                        await session.commit()
                        
                        # Same cleanup for backup table
                        cleanup_backup_query = text(cleanup_query.text.replace('miner_stats', 'miner_stats_backup'))
                        await session.execute(cleanup_backup_query)
                        await session.commit()
                        
                        # Sync backup table with main table
                        sync_query = text("""
                        INSERT OR REPLACE INTO miner_stats_backup
                        SELECT * FROM miner_stats;
                        """)
                        await session.execute(sync_query)
                        await session.commit()
                        
                        bt.logging.info("Successfully initialized miner stats")
                        return
                        
            except (asyncio.TimeoutError, SQLAlchemyError) as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    bt.logging.warning(f"Initialization attempt {attempt + 1} failed, retrying in {delay:.1f}s: {str(e)}")
                    await asyncio.sleep(delay)
                    continue
                bt.logging.error(f"Error initializing miner stats after {max_retries} attempts: {e}")
                raise
            except Exception as e:
                bt.logging.error(f"Error initializing miner stats: {e}")
                bt.logging.error(traceback.format_exc())
                raise

    async def _update_lifetime_statistics(self):
        """Update lifetime statistics for miners with proper prediction history tracking."""
        update_lifetime_query = """
        WITH prediction_stats AS (
            SELECT 
                miner_uid,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN payout > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN payout = 0 THEN 1 ELSE 0 END) as losses,
                SUM(payout) as total_earnings,
                SUM(wager) as total_wager,
                MAX(prediction_date) as last_prediction
            FROM predictions
            GROUP BY miner_uid
        )
        UPDATE miner_stats
        SET
            miner_lifetime_predictions = COALESCE(ps.total_predictions, 0),
            miner_lifetime_wins = COALESCE(ps.wins, 0),
            miner_lifetime_losses = COALESCE(ps.losses, 0),
            miner_lifetime_earnings = COALESCE(ps.total_earnings, 0),
            miner_lifetime_wager_amount = COALESCE(ps.total_wager, 0),
            miner_win_loss_ratio = CASE 
                WHEN COALESCE(ps.losses, 0) > 0 
                THEN CAST(COALESCE(ps.wins, 0) AS REAL) / COALESCE(ps.losses, 0)
                ELSE COALESCE(ps.wins, 0)
            END,
            miner_last_prediction_date = ps.last_prediction
        FROM prediction_stats ps
        WHERE miner_stats.miner_uid = ps.miner_uid;
        """
        
        await self.db_manager.execute_query(update_lifetime_query)
        bt.logging.debug("Updated lifetime statistics for miners.")

    async def update_miner_stats(self, current_day):
        try:
            bt.logging.info(f"Updating miner stats for day {current_day}...")
            
            # Update miner_hotkey and miner_coldkey from metagraph
            updates = []
            column_names = ['miner_uid', 'miner_hotkey', 'miner_coldkey']
            
            for miner_uid in range(min(256, len(self.validator.metagraph.hotkeys))):
                hotkey = self.validator.metagraph.hotkeys[miner_uid]
                coldkey = self.validator.metagraph.coldkeys[miner_uid]
                updates.append((miner_uid, hotkey, coldkey))

            # Batch update with proper column names
            await self.db_manager.executemany(
                """INSERT INTO miner_stats (miner_uid, miner_hotkey, miner_coldkey)
                   VALUES (?, ?, ?)
                   ON CONFLICT(miner_uid) DO UPDATE SET
                   miner_hotkey = EXCLUDED.miner_hotkey,
                   miner_coldkey = EXCLUDED.miner_coldkey""",
                updates,
                column_names=column_names
            )

            # Continue with rest of the updates...
            await self._update_lifetime_statistics()
            tiers_dict = self.get_current_tiers()
            
            # Update current tiers
            tier_updates = []
            tier_column_names = ['tier', 'miner_uid']
            for miner_uid, current_tier in tiers_dict.items():
                if miner_uid < 256:  # Only update valid UIDs
                    tier_updates.append((int(current_tier - 1), miner_uid))
            
            await self.db_manager.executemany(
                """UPDATE miner_stats
                   SET miner_current_tier = ?
                   WHERE miner_uid = ?""",
                tier_updates,
                column_names=tier_column_names
            )
            
            await self._update_current_daily_scores(current_day, tiers_dict)
            await self._update_additional_fields()
            
            bt.logging.info("Miner stats update completed successfully.")

        except Exception as e:
            bt.logging.error(f"Error updating miner stats: {e}")
            raise
    
    def safe_format(self, value, decimal_places=4):
        return f"{value:.{decimal_places}f}" if value is not None else 'None'

    async def _update_current_daily_scores(self, current_day, tiers_dict):
        """
        Update the current daily scores for each miner.
        Uses tier-specific composite scores based on miner's current tier.
        
        Args:
            current_day (int): The current day index.
            tiers_dict (Dict[int, int]): Mapping of miner_uid to current_tier.
        """
        bt.logging.info(f"Updating current daily scores for miners for day {current_day}...")

        # Fetch current day's scores from the 'scores' table
        fetch_scores_query = """
            SELECT miner_uid, score_type, clv_score, roi_score, sortino_score, entropy_score, composite_score
            FROM scores
            WHERE day_id = :day_id
        """
        scores = await self.db_manager.fetch_all(fetch_scores_query, {"day_id": current_day})

        # Organize scores by miner_uid and score_type
        miner_scores = defaultdict(lambda: defaultdict(dict))
        for score in scores:
            miner_uid = score["miner_uid"]
            score_type = score["score_type"]
            miner_scores[miner_uid][score_type] = {
                "clv_score": score["clv_score"],
                "roi_score": score["roi_score"],
                "sortino_score": score["sortino_score"],
                "entropy_score": score["entropy_score"],
                "composite_score": score["composite_score"]
            }

        # Prepare update records
        update_current_scores_query = """
            UPDATE miner_stats
            SET
                miner_current_clv_avg = ?,
                miner_current_roi = ?,
                miner_current_sortino_ratio = ?,
                miner_current_entropy_score = ?,
                miner_current_composite_score = ?
            WHERE miner_uid = ?
        """
        update_records = []

        for miner_uid, current_tier in tiers_dict.items():
            # Get daily scores for component metrics
            daily_scores = miner_scores[miner_uid].get('daily', {})
            
            # Get tier-specific scores
            tier_scores = miner_scores[miner_uid].get(f'tier_{current_tier}', {})
            
            # Use component scores from daily calculation
            clv_avg = daily_scores.get('clv_score')
            roi = daily_scores.get('roi_score')
            sortino = daily_scores.get('sortino_score')
            entropy = daily_scores.get('entropy_score')
            
            # Use composite score from tier-specific calculation
            composite_score = tier_scores.get('composite_score')

            update_records.append((clv_avg, roi, sortino, entropy, composite_score, miner_uid))

        # Execute batch update
        await self.db_manager.executemany(update_current_scores_query, update_records)
        bt.logging.debug("Current daily scores updated for all miners.")

        # Log some statistics
        composite_scores = [record[4] for record in update_records if record[4] is not None]
        if composite_scores:
            bt.logging.info(f"Tier-specific composite score stats - min: {min(composite_scores):.4f}, "
                          f"max: {max(composite_scores):.4f}, "
                          f"mean: {sum(composite_scores) / len(composite_scores):.4f}")
        else:
            bt.logging.warning("No valid composite scores found.")

        # Log the number of miners with non-zero scores
        non_zero_scores = sum(1 for record in update_records if any(record[:5]))
        bt.logging.info(f"Number of miners with non-zero scores: {non_zero_scores}")

    async def _update_additional_fields(self):
        """
        Update additional miner fields such as rank, status, and cash.
        """
        bt.logging.info("Updating additional miner fields...")

        # Placeholder for additional field updates
        update_additional_fields_query = """
            UPDATE miner_stats
            SET
                miner_rank = ?,
                miner_status = ?,
                miner_cash = ?,
                miner_current_incentive = ?
            WHERE miner_uid = ?
        """
        additional_records = []
        for miner_uid in range(len(self.validator.metagraph.hotkeys)-1):
            miner_rank = self.get_miner_rank(miner_uid)
            miner_status = self.get_miner_status(miner_uid)
            miner_cash = await self.calculate_miner_cash(miner_uid)
            miner_current_incentive = self.get_miner_current_incentive(miner_uid)
            additional_records.append((miner_rank, miner_status, miner_cash, miner_current_incentive, miner_uid))
        
        await self.db_manager.executemany(update_additional_fields_query, additional_records)
        bt.logging.debug("Additional miner fields updated.")

    def get_current_tiers(self):
        try:
            current_day = self.current_day
            bt.logging.debug(f"Current day: {current_day}")
            bt.logging.debug(f"Tiers shape: {self.tiers.shape}")
            if current_day < 0 or current_day >= self.tiers.shape[1]:
                bt.logging.error(f"Invalid current_day: {current_day}")
                return {}
            tiers = self.tiers[:, current_day]
            bt.logging.debug(f"Tiers: {tiers}")
            return {miner_uid: tier for miner_uid, tier in enumerate(tiers)}
        except Exception as e:
            bt.logging.error(f"Error in get_current_tiers: {str(e)}")
            return {}

    def get_miner_rank(self, miner_uid: int) -> int:
        """
        Get the rank for a miner.
        
        Args:
            miner_uid (int): The miner's UID.
        
        Returns:
            int: Calculated rank.
        """
        rank = self.validator.metagraph.R[miner_uid]
        return int(rank)

    def get_miner_status(self, miner_uid: int) -> str:
        """
        Get the current status of a miner.
        
        Args:
            miner_uid (int): The miner's UID.
        
        Returns:
            str: Status of the miner.
        """
        active = self.validator.metagraph.active[miner_uid]
        return "active" if active else "inactive"

    async def calculate_miner_cash(self, miner_uid: int) -> float:
        """
        Calculate the current cash for a miner by subtracting
        the sum of their wagers made since 00:00 UTC from today.

        Args:
            miner_uid (int): The miner's UID.

        Returns:
            float: Current cash of the miner.
        """
        # Calculate the start of today in UTC
        now_utc = datetime.now(timezone.utc)
        start_of_today = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

        # Query to sum wagers since the start of today
        query = """
            SELECT SUM(wager) as total_wager
            FROM predictions
            WHERE miner_uid = ?
              AND prediction_date >= ?
        """
        result = await self.db_manager.fetch_one(query, (miner_uid, start_of_today))
        total_wager = result['total_wager'] if result['total_wager'] is not None else 0.0

        return 1000 - float(total_wager)

    def get_miner_current_incentive(self, miner_uid: int) -> float:
        incentive = self.validator.metagraph.incentive[miner_uid]
        return float(incentive)

    async def cleanup_miner_stats(self):
        """Clean up and synchronize miner_stats and miner_stats_backup tables"""
        try:
            bt.logging.info("Starting miner stats cleanup...")
            
            # First, remove any invalid entries (UID >= 256 or duplicates)
            cleanup_query = """
            DELETE FROM miner_stats 
            WHERE miner_uid >= 256 
            OR miner_uid IN (
                SELECT miner_uid 
                FROM miner_stats 
                GROUP BY miner_uid 
                HAVING COUNT(*) > 1
            );
            """
            await self.db_manager.execute_query(cleanup_query)
            
            # Same cleanup for backup table
            cleanup_backup_query = """
            DELETE FROM miner_stats_backup 
            WHERE miner_uid >= 256 
            OR miner_uid IN (
                SELECT miner_uid 
                FROM miner_stats_backup 
                GROUP BY miner_uid 
                HAVING COUNT(*) > 1
            );
            """
            await self.db_manager.execute_query(cleanup_backup_query)
            
            # Sync backup table with main table
            sync_query = """
            INSERT OR REPLACE INTO miner_stats_backup
            SELECT * FROM miner_stats;
            """
            await self.db_manager.execute_query(sync_query)
            
            # Verify sync
            verify_query = """
            SELECT 
                COUNT(*) as total_rows,
                COUNT(CASE WHEN miner_uid < 256 THEN 1 END) as valid_miners,
                COUNT(CASE WHEN miner_uid < 256 AND miner_last_prediction_date IS NOT NULL THEN 1 END) as miners_with_dates,
                COUNT(CASE WHEN miner_uid < 256 AND miner_lifetime_predictions > 0 THEN 1 END) as miners_with_predictions
            FROM miner_stats_backup;
            """
            backup_stats = await self.db_manager.fetch_one(verify_query)
            bt.logging.info(f"Backup table stats after sync: {backup_stats}")
            
        except Exception as e:
            bt.logging.error(f"Error during miner stats cleanup: {e}")
            bt.logging.error(traceback.format_exc())
            raise










