import math
import numpy as np
import torch
import sqlite3
from datetime import datetime, timezone, timedelta
import asyncio
import bittensor as bt

class WeightSetter:
    def __init__(self, metagraph, wallet, subtensor, neuron_config, loop, thread_executor, db_path):
        self.metagraph = metagraph
        self.wallet = wallet
        self.subtensor = subtensor
        self.neuron_config = neuron_config
        self.loop = loop
        self.thread_executor = thread_executor
        self.decay_factors = self.compute_decay_factors()
        self.db_path = db_path
    
    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def logarithmic_penalty(self, count, min_count):
        if count >= min_count:
            return 1.0
        else:
            return math.log10(count + 1) / math.log10(min_count)
    
    def compute_decay_factors(self):
        decay_rate = 0.05
        max_age_days = 365
        return {
            age: math.exp(-decay_rate * min(age, max_age_days))
            for age in range(max_age_days + 1)
        }
    
    @staticmethod
    def exponential_decay_returns(scale: int) -> np.ndarray:
        top_miner_benefit = 0.90
        top_miner_percent = 0.40

        top_miner_benefit = np.clip(top_miner_benefit, a_min=0, a_max=0.99999999)
        top_miner_percent = np.clip(top_miner_percent, a_min=0.00000001, a_max=1)
        scale = max(1, scale)
        if scale == 1:
            return np.array([1])

        k = -np.log(1 - top_miner_benefit) / (top_miner_percent * scale)
        xdecay = np.linspace(0, scale-1, scale)
        decayed_returns = np.exp((-k) * xdecay)

        return decayed_returns / np.sum(decayed_returns)

    def get_current_day_performance(self, cursor):
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        
        cursor.execute("""
            SELECT minerId, COUNT(*) as total_predictions,
                SUM(CASE WHEN predictedOutcome = outcome THEN 1 ELSE 0 END) as correct_predictions,
                SUM(wager) as total_wager,
                SUM(CASE 
                    WHEN predictedOutcome = outcome AND predictedOutcome = '0' THEN wager * teamAodds
                    WHEN predictedOutcome = outcome AND predictedOutcome = '1' THEN wager * teamBodds
                    WHEN predictedOutcome = outcome AND predictedOutcome = '2' THEN wager * tieOdds
                    ELSE 0
                END) as total_earnings
            FROM predictions
            WHERE DATE(predictionDate) = ?
            GROUP BY minerId
        """, (today,))
        
        return cursor.fetchall()

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

    def get_historical_performance(self, cursor, start_date):
        cursor.execute("""
            SELECT minerId, date, total_predictions, correct_predictions, total_wager, total_earnings
            FROM daily_miner_stats
            WHERE date >= ?
        """, (start_date,))
        
        return cursor.fetchall()

    def calculate_miner_scores(self, db_path):
        earnings = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=365)).date().isoformat()
        
        # Fetch historical data
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
            miner_performance[miner_id] += total_earnings  # No decay for current day

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

            # Get today's date
            end_date = datetime.now(timezone.utc).date()
            
            current_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            
            while current_date <= end_date:
                self.update_daily_stats(current_date)
                current_date += timedelta(days=1)

            bt.logging.info(f"Updated daily stats from {start_date} to {end_date}")
        except Exception as e:
            bt.logging.error(f"Error updating all daily stats: {e}")
            conn.rollback()
        finally:
            conn.close()

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    async def set_weights(self, db_path):
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
            return

        NUM_RETRIES = 3 
        for i in range(NUM_RETRIES):
            bt.logging.info(f"Attempting to set weights, attempt {i+1} of {NUM_RETRIES}")
            try:
                result = await asyncio.wait_for(
                    self.run_sync_in_async(lambda: self.subtensor.set_weights(
                        netuid=self.neuron_config.netuid,
                        wallet=self.wallet,
                        uids=self.metagraph.uids,
                        weights=weights,
                        wait_for_inclusion=False,
                        wait_for_finalization=True,
                    )),
                    timeout=90
                )
                bt.logging.trace(f"Set weights result: {result}")
                
                if isinstance(result, tuple) and len(result) >= 1:
                    success = result[0]
                    if success:
                        bt.logging.info("Successfully set weights.")
                        return
                else:
                    bt.logging.warning(f"Unexpected result format in setting weights: {result}")
            except TimeoutError:
                bt.logging.error("Timeout occurred while setting weights.")
            except Exception as e:
                bt.logging.error(f"Error setting weights: {str(e)}")
            
            if i < NUM_RETRIES - 1:
                await asyncio.sleep(1)
        
        bt.logging.error("Failed to set weights after all attempts.")