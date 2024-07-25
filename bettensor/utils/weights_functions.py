import math
import numpy as np
import torch
import sqlite3
from datetime import datetime, timezone
import asyncio
import bittensor as bt

class WeightSetter:
    def __init__(self, metagraph, wallet, subtensor, neuron_config, loop, thread_executor):
        self.metagraph = metagraph
        self.wallet = wallet
        self.subtensor = subtensor
        self.neuron_config = neuron_config
        self.loop = loop
        self.thread_executor = thread_executor
        self.decay_factors = self.compute_decay_factors()

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

    def calculate_miner_scores(self, db_path):
        earnings = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        now = datetime.now(timezone.utc)

        cursor.execute("SELECT externalId, eventStartDate FROM game_data")
        game_data_rows = cursor.fetchall()
        game_date_map = {row[0]: datetime.fromisoformat(row[1]) for row in game_data_rows}

        cursor.execute(
            "SELECT predictionID, teamGameID, minerId, predictedOutcome, outcome, teamA, teamB, wager, teamAodds, teamBodds, tieOdds FROM predictions"
        )
        prediction_rows = cursor.fetchall()

        conn.close()

        miner_performance = {}
        miner_prediction_counts = {}
        miner_id_to_index = {miner_id: idx for idx, miner_id in enumerate(self.metagraph.hotkeys)}

        min_prediction_count = 1

        for row in prediction_rows:
            (prediction_id, team_game_id, miner_id, predicted_outcome, outcome, team_a, team_b, wager, team_a_odds, team_b_odds, tie_odds) = row

            if team_game_id in game_date_map:
                event_date = game_date_map[team_game_id]
                age_days = max((now - event_date).days, 0)
                decay_factor = self.decay_factors[min(age_days, 365)]

                if miner_id not in miner_performance:
                    miner_performance[miner_id] = 0.0
                    miner_prediction_counts[miner_id] = 0

                miner_prediction_counts[miner_id] += 1

                if predicted_outcome == outcome:
                    if predicted_outcome == "0":
                        earned = wager * team_a_odds
                    elif predicted_outcome == "1":
                        earned = wager * team_b_odds
                    elif predicted_outcome == "Tie":
                        earned = wager * tie_odds
                    else:
                        bt.logging.warning(f"Outcome for {team_game_id} not found. Please notify Bettensor Developers, as this is likely a larger API issue.")
                        continue

                    miner_performance[miner_id] += earned * decay_factor

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

        bt.logging.trace("Miner performance scores before normalization:")
        bt.logging.trace(final_scores)

        return earnings

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    async def set_weights(self, db_path):
        earnings = self.calculate_miner_scores(db_path)
        weights = torch.nn.functional.normalize(earnings, p=1.0, dim=0)
        np.set_printoptions(precision=8, suppress=True)
        weights_np = weights.numpy()
        bt.logging.info(f"Normalized weights: {weights_np}")

        bt.logging.info(f"Normalized weights: {weights}")
        uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        stake = float(self.metagraph.S[uid])
        if stake 1000.0:
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