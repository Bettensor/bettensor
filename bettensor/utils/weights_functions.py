import math
import numpy as np
import torch
import sqlite3
from datetime import datetime, timezone, timedelta
import bittensor as bt
from bettensor import __spec_version__
import time

class WeightSetter:
    def __init__(self, metagraph, wallet, subtensor, neuron_config, db_path):
        self.metagraph = metagraph
        self.wallet = wallet
        self.subtensor = subtensor
        self.neuron_config = neuron_config
        self.decay_factors = self.compute_decay_factors()
        self.db_path = db_path
        self.initialize_daily_stats_table()
        self.initialize_daily_database()
    
    def connect_db(self):
        return sqlite3.connect(self.db_path)
    
    def initialize_daily_stats_table(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_miner_stats (
                    date DATE,
                    minerId TEXT,
                    total_predictions INT,
                    correct_predictions INT,
                    total_wager REAL,
                    total_earnings REAL,
                    PRIMARY KEY (date, minerId)
                )
            """)
            
            cursor.execute("SELECT COUNT(*) FROM daily_miner_stats")
            if cursor.fetchone()[0] == 0:
                start_date = datetime(2024, 7, 28).date()
                end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
                current_date = start_date
                
                while current_date <= end_date:
                    self.update_daily_stats(current_date)
                    current_date += timedelta(days=1)
                
                bt.logging.info(f"Initialized daily stats from {start_date} to {end_date}")
            else:
                bt.logging.info("daily_miner_stats table already exists and contains data")
        
        except Exception as e:
            bt.logging.error(f"Error initializing daily stats table: {e}")
            conn.rollback()
        finally:
            conn.close()

    def initialize_daily_database(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Create the daily_miner_stats table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_miner_stats (
                    date DATE,
                    minerId TEXT,
                    total_predictions INT,
                    correct_predictions INT,
                    total_wager REAL,
                    total_earnings REAL,
                    PRIMARY KEY (date, minerId)
                )
            """)
            
            conn.commit()
            bt.logging.info("Daily miner stats table initialized successfully")
        except Exception as e:
            bt.logging.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()

    def update_daily_stats(self, date):
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            today = datetime.now(timezone.utc).date()
            if date >= datetime(2024, 7, 28).date() and date < today:
                cursor.execute("""
                    WITH daily_wagers AS (
                        SELECT 
                            minerId,
                            SUM(wager) as total_daily_wager
                        FROM predictions
                        WHERE DATE(predictionDate) = ?
                        GROUP BY minerId
                    )
                    INSERT INTO daily_miner_stats (date, minerId, total_predictions, correct_predictions, total_wager, total_earnings)
                    SELECT 
                        DATE(p.predictionDate) as date,
                        p.minerId,
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN p.predictedOutcome = p.outcome THEN 1 ELSE 0 END) as correct_predictions,
                        SUM(p.wager) as total_wager,
                        SUM(CASE 
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '0' AND p.wager > 0 THEN p.wager * p.teamAodds
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '1' AND p.wager > 0 THEN p.wager * p.teamBodds
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '2' AND p.wager > 0 THEN p.wager * p.tieOdds
                            ELSE 0
                        END) as total_earnings
                    FROM predictions p
                    JOIN daily_wagers dw ON p.minerId = dw.minerId
                    WHERE DATE(p.predictionDate) = ? AND dw.total_daily_wager <= 1000
                    GROUP BY DATE(p.predictionDate), p.minerId
                    ON CONFLICT(date, minerId) DO UPDATE SET
                        total_predictions = excluded.total_predictions,
                        correct_predictions = excluded.correct_predictions,
                        total_wager = excluded.total_wager,
                        total_earnings = excluded.total_earnings
                """, (date.isoformat(), date.isoformat()))
                
                conn.commit()
                bt.logging.debug(f"Updated daily stats for {date.isoformat()}")
            elif date >= today:
                bt.logging.debug(f"Skipped updating daily stats for {date.isoformat()} (current or future date)")
            else:
                bt.logging.debug(f"Skipped updating daily stats for {date.isoformat()} (before July 28, 2024)")
        except Exception as e:
            bt.logging.error(f"Error updating daily stats for {date.isoformat()}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def recalculate_daily_profits(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            start_date = datetime(2024, 7, 28).date().isoformat()

            # Check if there's any data in the predictions table
            cursor.execute("SELECT COUNT(*) FROM predictions")
            if cursor.fetchone()[0] == 0:
                bt.logging.warning("No predictions data available. Skipping daily profits recalculation.")
                return
            
            cursor.execute("""
                UPDATE daily_miner_stats
                SET total_earnings = 0
                WHERE date >= ?
            """, (start_date,))
            
            cursor.execute("""
                WITH daily_wagers AS (
                    SELECT 
                        DATE(predictionDate) as date,
                        minerId,
                        SUM(wager) as total_daily_wager
                    FROM predictions
                    WHERE DATE(predictionDate) >= ?
                    GROUP BY DATE(predictionDate), minerId
                ),
                daily_earnings AS (
                    SELECT 
                        DATE(p.predictionDate) as date,
                        p.minerId,
                        SUM(CASE 
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '0' AND p.wager > 0 THEN p.wager * p.teamAodds
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '1' AND p.wager > 0 THEN p.wager * p.teamBodds
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '2' AND p.wager > 0 THEN p.wager * p.tieOdds
                            ELSE 0
                        END) as recalculated_earnings
                    FROM predictions p
                    JOIN daily_wagers dw ON p.minerId = dw.minerId AND DATE(p.predictionDate) = dw.date
                    WHERE p.outcome != 'Unfinished' AND p.wager > 0 AND dw.total_daily_wager <= 1000
                        AND DATE(p.predictionDate) >= ?
                    GROUP BY DATE(p.predictionDate), p.minerId
                )
                UPDATE daily_miner_stats
                SET total_earnings = daily_earnings.recalculated_earnings
                FROM daily_earnings
                WHERE daily_miner_stats.date = daily_earnings.date
                AND daily_miner_stats.minerId = daily_earnings.minerId
                AND daily_miner_stats.date >= ?
            """, (start_date, start_date, start_date))
            
            conn.commit()
            bt.logging.info(f"Successfully recalculated daily profits from {start_date} to now")
        except Exception as e:
            bt.logging.error(f"Error recalculating daily profits: {e}")
            bt.logging.error(f"Error recalculating daily profits: {e.with_traceback()}")
            conn.rollback()
        finally:
            conn.close()

    def logarithmic_penalty(self, count, min_count):
        if count >= min_count:
            return 1.0
        elif count == 0:
            return 0.0
        else:
            return math.log10(count + 1) / math.log10(min_count)
    
    def compute_decay_factors(self):
        decay_rate = 0.46
        max_age_days = 365
        return {
            age: math.exp(-decay_rate * min(age, max_age_days))
            for age in range(max_age_days + 1)
        }
    
    @staticmethod
    def exponential_decay_returns(scale: int) -> np.ndarray:
        top_miner_benefit = 0.90 # The percentage of top miners that share the top_miner_percent rewards
        top_miner_percent = 0.50 # The percentage of rewards that go to the top miners

        top_miner_benefit = np.clip(top_miner_benefit, a_min=0, a_max=0.99999999)
        top_miner_percent = np.clip(top_miner_percent, a_min=0.00000001, a_max=1)
        scale = max(1, scale)
        if scale == 1:
            return np.array([1])

        k = -np.log(1 - top_miner_benefit) / (top_miner_percent * scale)
        xdecay = np.linspace(0, scale-1, scale)
        decayed_returns = np.exp((-k) * xdecay)

        return decayed_returns / np.sum(decayed_returns)

    def get_daily_profits(self, start_date, end_date):
        conn = self.connect_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, SUM(total_earnings) as daily_profit
            FROM daily_miner_stats
            WHERE date BETWEEN ? AND ?
            GROUP BY date
            ORDER BY date
        """, (start_date, end_date))
        
        daily_profits = cursor.fetchall()
        conn.close()
        
        return daily_profits

    def update_daily_stats_if_new_day(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Get the date of the last update
            cursor.execute("SELECT MAX(date) FROM daily_miner_stats")
            last_update = cursor.fetchone()[0]
            
            # Get today's date
            today = datetime.now(timezone.utc).date()
            
            if last_update is None or datetime.strptime(last_update, '%Y-%m-%d').date() < today:
                # It's a new day, update the stats
                yesterday = (today - timedelta(days=1)).isoformat()
                
                cursor.execute("""
                    INSERT INTO daily_miner_stats (date, minerId, total_predictions, correct_predictions, total_wager, total_earnings)
                    SELECT 
                        DATE(p.predictionDate) as date,
                        p.minerId,
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN p.predictedOutcome = p.outcome THEN 1 ELSE 0 END) as correct_predictions,
                        SUM(p.wager) as total_wager,
                        SUM(CASE 
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '0' THEN p.wager * p.teamAodds
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '1' THEN p.wager * p.teamBodds
                            WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '2' THEN p.wager * p.tieOdds
                            ELSE 0
                        END) as total_earnings
                    FROM predictions p
                    WHERE DATE(p.predictionDate) = ?
                    GROUP BY DATE(p.predictionDate), p.minerId
                    ON CONFLICT(date, minerId) DO UPDATE SET
                        total_predictions = excluded.total_predictions,
                        correct_predictions = excluded.correct_predictions,
                        total_wager = excluded.total_wager,
                        total_earnings = excluded.total_earnings
                """, (yesterday,))
                
                conn.commit()
                bt.logging.info(f"Updated daily stats for {yesterday}")
                return True
            else:
                bt.logging.info("Daily stats are up to date")
                return False
        except Exception as e:
            bt.logging.error(f"Error updating daily stats: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_current_day_performance(self, cursor):
        today = datetime.now(timezone.utc).date().isoformat()
        
        cursor.execute("""
            WITH daily_wagers AS (
                SELECT minerId, SUM(wager) as total_daily_wager
                FROM predictions
                WHERE DATE(predictionDate) = ?
                GROUP BY minerId
            )
            SELECT p.minerId, COUNT(*) as total_predictions,
                SUM(CASE WHEN p.predictedOutcome = p.outcome THEN 1 ELSE 0 END) as correct_predictions,
                SUM(p.wager) as total_wager,
                SUM(CASE 
                    WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '0' AND p.wager > 0 THEN p.wager * p.teamAodds
                    WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '1' AND p.wager > 0 THEN p.wager * p.teamBodds
                    WHEN p.predictedOutcome = p.outcome AND p.predictedOutcome = '2' AND p.wager > 0 THEN p.wager * p.tieOdds
                    ELSE 0
                END) as total_earnings
            FROM predictions p
            JOIN daily_wagers dw ON p.minerId = dw.minerId
            WHERE DATE(p.predictionDate) = ? AND p.outcome != 'Unfinished' AND dw.total_daily_wager <= 1000
            GROUP BY p.minerId
        """, (today, today))
        
        return cursor.fetchall()

    def get_historical_performance(self, cursor, start_date):
        today = datetime.now(timezone.utc).date()
        cursor.execute("""
            SELECT minerId, date, total_predictions, correct_predictions, total_wager, total_earnings
            FROM daily_miner_stats
            WHERE date >= ? AND date < ?
        """, (start_date, today.isoformat()))
        
        return cursor.fetchall()

    def calculate_miner_scores(self, db_path):
        earnings = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=365)).date().isoformat()
        
        # Fetch historical data (excluding today)
        daily_stats = self.get_historical_performance(cursor, start_date)
        
        # Fetch current day data
        current_day_stats = self.get_current_day_performance(cursor)
        
        miner_performance = {}
        miner_prediction_counts = {}
        miner_id_to_index = {miner_id: idx for idx, miner_id in enumerate(self.metagraph.hotkeys)}

        min_prediction_count = 1

        # Process historical data
        for row in daily_stats:
            miner_id, date, total_predictions, correct_predictions, total_wager, total_earnings = row
            
            age_days = (now.date() - datetime.strptime(date, '%Y-%m-%d').date()).days
            decay_factor = self.decay_factors[min(age_days, 365)]
            
            if miner_id not in miner_performance:
                miner_performance[miner_id] = 0.0
                miner_prediction_counts[miner_id] = 0
            
            miner_prediction_counts[miner_id] += total_predictions
            miner_performance[miner_id] += total_earnings * decay_factor

        # Process current day data
        for row in current_day_stats:
            miner_id, total_predictions, correct_predictions, total_wager, total_earnings = row
            
            if miner_id not in miner_performance:
                miner_performance[miner_id] = 0.0
                miner_prediction_counts[miner_id] = 0
            
            miner_prediction_counts[miner_id] += total_predictions
            miner_performance[miner_id] += total_earnings

        # Apply penalty and calculate final scores
        for miner_id, total_earned in miner_performance.items():
            prediction_count = miner_prediction_counts[miner_id]
            penalty_factor = self.logarithmic_penalty(prediction_count, min_prediction_count)
            miner_performance[miner_id] *= penalty_factor

        sorted_miners = sorted(miner_performance.items(), key=lambda x: x[1], reverse=True)
        decay_factors = self.exponential_decay_returns(len(sorted_miners))
        
        final_scores = {}
        for (miner_id, score), decay_factor in zip(sorted_miners, decay_factors):
            final_scores[miner_id] = score * decay_factor

        for miner_id, final_score in final_scores.items():
            if miner_id in miner_id_to_index:
                idx = miner_id_to_index[miner_id]
                earnings[idx] = final_score

        bt.logging.info("Miner performance scores before normalization:")
        bt.logging.info(final_scores)

        conn.close()
        return earnings

    def update_all_daily_stats(self):
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Get the earliest prediction date
            cursor.execute("SELECT MIN(DATE(predictionDate)) FROM predictions")
            start_date = cursor.fetchone()[0]
            
            if start_date is None:
                bt.logging.info("No predictions found in the database.")
                return

            # Ensure start_date is not earlier than July 28, 2024
            start_date = max(datetime.strptime(start_date, '%Y-%m-%d').date(), datetime(2024, 7, 28).date())

            # Get yesterday's date
            end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
            
            current_date = start_date
            
            while current_date <= end_date:
                self.update_daily_stats(current_date)
                current_date += timedelta(days=1)

            bt.logging.info(f"Updated daily stats from {start_date} to {end_date}")
        except Exception as e:
            bt.logging.error(f"Error updating all daily stats: {e}")
            conn.rollback()
        finally:
            conn.close()

    def set_weights(self, db_path):
        # Update daily stats if it's a new day
        self.update_all_daily_stats()

        earnings = self.calculate_miner_scores(db_path)
        weights = torch.nn.functional.normalize(earnings, p=1.0, dim=0)
        np.set_printoptions(precision=8, suppress=True)
        weights_np = weights.numpy()
        bt.logging.info(f"Normalized weights: {weights_np}")

        bt.logging.info(f"Normalized weights: {weights}")
        uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        stake = float(self.metagraph.S[uid])
        if stake < 1000.0:
            bt.logging.error("Insufficient stake. Failed in setting weights.")
            return False

        NUM_RETRIES = 3 
        for i in range(NUM_RETRIES):
            bt.logging.info(f"Attempting to set weights, attempt {i+1} of {NUM_RETRIES}")
            try:
                result = self.subtensor.set_weights(
                    netuid=self.neuron_config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=weights,
                    version_key=__spec_version__,
                    wait_for_inclusion=False,
                    wait_for_finalization=True
                )
                bt.logging.trace(f"Set weights result: {result}")
                
                if isinstance(result, tuple) and len(result) >= 1:
                    success = result[0]
                    if success:
                        bt.logging.info("Successfully set weights.")
                        return True
                else:
                    bt.logging.warning(f"Unexpected result format in setting weights: {result}")
            except Exception as e:
                bt.logging.error(f"Error setting weights: {str(e)}")
            
            if i < NUM_RETRIES - 1:
                time.sleep(10)
        
        bt.logging.error("Failed to set weights after all attempts.")
        return False