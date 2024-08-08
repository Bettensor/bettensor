import threading
from typing import Dict, List, Any, Optional
import bittensor as bt
from datetime import datetime, timezone, timedelta
import traceback

import torch
from bettensor.protocol import TeamGamePrediction  # Add this import

class MinerStatsHandler:
    def __init__(self, db_manager, state_manager):
        bt.logging.trace("Initializing MinerStatsHandler")
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.recalculate_stats_from_predictions()
        bt.logging.trace("MinerStatsHandler initialization complete")

    def recalculate_stats_from_predictions(self):
        bt.logging.info("Recalculating miner stats from existing predictions")
        
        # First, let's check all predictions
        all_predictions_query = """
        SELECT minerHotkey, COUNT(*) as count
        FROM predictions
        GROUP BY minerHotkey
        """
        all_predictions_result = self.db_manager.execute_query(all_predictions_query)
        bt.logging.debug(f"All predictions by miner hotkey: {all_predictions_result}")
        
        # Now, let's check predictions for this miner
        query = """
        SELECT predictionid AS "predictionID", teamgameid AS "teamGameID", minerid AS "minerID", minerhotkey AS "minerHotkey", 
               predictiondate AS "predictionDate", predictedoutcome AS "predictedOutcome", teama AS "teamA", teamb AS "teamB", 
               wager, teamaodds AS "teamAodds", teambodds AS "teamBodds", tieodds AS "tieOdds", outcome
        FROM predictions
        WHERE minerHotkey = %s
        """
        try:
            results = self.db_manager.execute_query(query, (self.state_manager.miner_hotkey,))
            bt.logging.debug(f"Query results for miner {self.state_manager.miner_hotkey}: {results}")
            
            if not results:
                bt.logging.warning(f"No predictions found for miner with hotkey {self.state_manager.miner_hotkey}")
                return

            # Convert query results to TeamGamePrediction objects
            predictions = [TeamGamePrediction(
                predictionID=row['predictionID'],
                teamGameID=row['teamGameID'],
                minerID=row['minerID'],
                predictionDate=row['predictionDate'].isoformat() if isinstance(row['predictionDate'], datetime) else row['predictionDate'],
                predictedOutcome=row['predictedOutcome'],
                teamA=row.get('teamA'),
                teamB=row.get('teamB'),
                wager=row['wager'],
                teamAodds=row['teamAodds'],
                teamBodds=row['teamBodds'],
                tieOdds=row.get('tieOdds'),
                outcome=row['outcome']
            ) for row in results]
            
            stats = self.calculate_stats(predictions)
            bt.logging.info(f"Calculated stats: {stats}")
            self.state_manager.update_state(stats)
            bt.logging.info(f"Updated state: {self.state_manager.state}")
        except Exception as e:
            bt.logging.error(f"Error in recalculate_stats_from_predictions: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    def calculate_stats(self, predictions: List[TeamGamePrediction]) -> Dict[str, Any]:
        total_predictions = len(predictions)
        total_wins = 0
        total_losses = 0
        total_earnings = 0.0
        total_wager = 0.0
        last_prediction_date = None
        outcomes = set()

        bt.logging.debug(f"Calculating stats for {total_predictions} predictions")

        for pred in predictions:
            bt.logging.debug(f"Processing prediction: {pred}")
            total_wager += float(pred.wager)
            outcomes.add(pred.outcome)

            if pred.outcome.startswith('Wager Won'):
                total_wins += 1
                if pred.predictedOutcome == pred.teamA:
                    odds = float(pred.teamAodds)
                elif pred.predictedOutcome == pred.teamB:
                    odds = float(pred.teamBodds)
                else:
                    odds = float(pred.tieOdds) if pred.tieOdds is not None else 0
                total_earnings += float(pred.wager) * odds
            elif pred.outcome.startswith('Wager Lost'):
                total_losses += 1

            pred_date = datetime.fromisoformat(pred.predictionDate)
            if last_prediction_date is None or pred_date > last_prediction_date:
                last_prediction_date = pred_date

        win_loss_ratio = total_wins / total_losses if total_losses > 0 else 0.0

        stats = {
            'miner_lifetime_predictions': total_predictions,
            'miner_lifetime_wins': total_wins,
            'miner_lifetime_losses': total_losses,
            'miner_lifetime_earnings': total_earnings,
            'miner_lifetime_wager': total_wager,
            'miner_win_loss_ratio': win_loss_ratio,
            'miner_last_prediction_date': last_prediction_date.isoformat() if last_prediction_date else None
        }

        return stats

    # ... (other methods remain the same)

class MinerStateManager:
    DAILY_CASH = 1000.0

    def __init__(self, db_manager, miner_hotkey: str, miner_uid: str, stats_handler):
        bt.logging.trace("Initializing MinerStateManager")
        self.db_manager = db_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = miner_uid
        self.stats_handler = stats_handler
        self.lock = threading.Lock()
        self.state = self.load_state()
        self.reconcile_state()

    def load_state(self) -> Dict[str, Any]:
        bt.logging.trace("Loading miner state")
        query = "SELECT * FROM miner_stats WHERE miner_hotkey = %s"
        result = self.db_manager.execute_query(query, (self.miner_hotkey,))
        if result:
            state = result[0]
            # Ensure last_daily_reset is a string
            state['last_daily_reset'] = state['last_daily_reset'].isoformat() if isinstance(state['last_daily_reset'], datetime) else state['last_daily_reset']
            # Ensure miner_last_prediction_date is a string
            if state['miner_last_prediction_date'] is not None:
                state['miner_last_prediction_date'] = state['miner_last_prediction_date'].isoformat() if isinstance(state['miner_last_prediction_date'], datetime) else state['miner_last_prediction_date']
            return state
        else:
            return self.initialize_state()

    def initialize_state(self) -> Dict[str, Any]:
        bt.logging.trace("Initializing new miner state")
        state = {
            'miner_hotkey': self.miner_hotkey,
            'miner_uid': self.miner_uid,
            'miner_cash': self.DAILY_CASH,
            'miner_current_incentive': 0.0,
            'miner_last_prediction_date': None,
            'miner_lifetime_earnings': 0.0,
            'miner_lifetime_wager': 0.0,
            'miner_lifetime_predictions': 0,
            'miner_lifetime_wins': 0,
            'miner_lifetime_losses': 0,
            'miner_win_loss_ratio': 0.0,
            'last_daily_reset': datetime.now(timezone.utc).isoformat()  # Ensure this is a string
        }
        self.save_state(state)
        return state

    def save_state(self, state: Optional[Dict[str, Any]] = None):
        if state is None:
            state = self.state
        bt.logging.trace("Saving miner state")
        query = """
        INSERT INTO miner_stats (miner_hotkey, miner_uid, miner_cash, miner_current_incentive, miner_last_prediction_date, 
                                 miner_lifetime_earnings, miner_lifetime_wager, miner_lifetime_predictions, miner_lifetime_wins, 
                                 miner_lifetime_losses, miner_win_loss_ratio, last_daily_reset)
        VALUES (%(miner_hotkey)s, %(miner_uid)s, %(miner_cash)s, %(miner_current_incentive)s, %(miner_last_prediction_date)s, 
                %(miner_lifetime_earnings)s, %(miner_lifetime_wager)s, %(miner_lifetime_predictions)s, %(miner_lifetime_wins)s, 
                %(miner_lifetime_losses)s, %(miner_win_loss_ratio)s, %(last_daily_reset)s)
        ON CONFLICT (miner_hotkey) DO UPDATE SET
            miner_uid = EXCLUDED.miner_uid,
            miner_cash = EXCLUDED.miner_cash,
            miner_current_incentive = EXCLUDED.miner_current_incentive,
            miner_last_prediction_date = EXCLUDED.miner_last_prediction_date,
            miner_lifetime_earnings = EXCLUDED.miner_lifetime_earnings,
            miner_lifetime_wager = EXCLUDED.miner_lifetime_wager,
            miner_lifetime_predictions = EXCLUDED.miner_lifetime_predictions,
            miner_lifetime_wins = EXCLUDED.miner_lifetime_wins,
            miner_lifetime_losses = EXCLUDED.miner_lifetime_losses,
            miner_win_loss_ratio = EXCLUDED.miner_win_loss_ratio,
            last_daily_reset = EXCLUDED.last_daily_reset
        """
        self.db_manager.execute_query(query, state)

    def reconcile_state(self):
        bt.logging.trace("Reconciling miner state")
        now = datetime.now(timezone.utc)
        last_reset = datetime.fromisoformat(self.state['last_daily_reset'])
        # Ensure last_reset is timezone-aware
        if last_reset.tzinfo is None:
            last_reset = last_reset.replace(tzinfo=timezone.utc)
        if now - last_reset >= timedelta(days=1):
            self.reset_daily_cash()

    def reset_daily_cash(self):
        bt.logging.trace("Resetting daily cash")
        with self.lock:
            self.state['miner_cash'] = self.DAILY_CASH
            self.state['last_daily_reset'] = datetime.now(timezone.utc).isoformat()
            self.save_state()

    def update_on_prediction(self, prediction_data):
        with self.lock:
            bt.logging.trace(f"Updating state for prediction: {prediction_data}")
            self.state['miner_lifetime_predictions'] += 1
            if prediction_data['wager'] > 0:
                self.state['miner_cash'] -= prediction_data['wager']
                self.state['miner_lifetime_wager'] += prediction_data['wager']
            self.state['miner_last_prediction_date'] = datetime.now(timezone.utc).isoformat()
            self.save_state()

    def update_on_game_result(self, result_data):
        with self.lock:
            bt.logging.trace(f"Updating state for game result: {result_data}")
            if 'Wager Won' in result_data['outcome']:
                self.state['miner_lifetime_wins'] += 1
                self.state['miner_lifetime_earnings'] += result_data['earnings']
            elif 'Wager Lost' in result_data['outcome']:
                self.state['miner_lifetime_losses'] += 1
            self.update_win_loss_ratio()
            self.save_state()

    def update_win_loss_ratio(self):
        total_games = self.state['miner_lifetime_wins'] + self.state['miner_lifetime_losses']
        self.state['miner_win_loss_ratio'] = self.state['miner_lifetime_wins'] / total_games if total_games > 0 else 0.0

    def update_current_incentive(self, incentive):
        with self.lock:
            bt.logging.trace(f"Updating current incentive: {incentive}")
            if isinstance(incentive, torch.Tensor):
                incentive = incentive.item() if incentive.numel() == 1 else None
            self.state['miner_current_incentive'] = incentive
            self.save_state()

    def get_current_incentive(self):
        return self.state['miner_current_incentive']

    def get_stats(self):
        bt.logging.trace("Getting miner stats")
        with self.lock:
            return self.state.copy()

    def periodic_db_update(self):
        bt.logging.trace("Performing periodic database update")
        try:
            self.save_state()
            self.stats_handler.recalculate_stats_from_predictions()
        except Exception as e:
            bt.logging.error(f"Error in periodic_db_update: {e}")
            bt.logging.error(traceback.format_exc())

    def deduct_wager(self, amount):
        with self.lock:
            bt.logging.trace(f"Attempting to deduct wager: {amount}")
            if self.state['miner_cash'] >= amount:
                self.state['miner_cash'] -= amount
                self.save_state()
                bt.logging.trace(f"Wager deducted. New balance: {self.state['miner_cash']}")
                return True
            bt.logging.trace(f"Insufficient funds. Current balance: {self.state['miner_cash']}")
            return False

    def update_state(self, new_state: Dict):
        with self.lock:
            bt.logging.trace(f"Updating state with: {new_state}")
            self.state.update(new_state)
            self.save_state()

    def get_miner_cash(self, miner_hotkey):
        bt.logging.trace(f"Getting miner cash for hotkey: {miner_hotkey}")
        query = "SELECT miner_cash FROM miner_stats WHERE miner_hotkey = %s"
        result = self.db_manager.execute_query(query, (miner_hotkey,))
        if result and result[0]:
            return result[0][0]
        return 0

    def update_stats_from_predictions(self):
        bt.logging.info("Updating stats from predictions")
        self.stats_handler.recalculate_stats_from_predictions()

    def update_predictions_with_hotkey(self):
        bt.logging.trace("Updating predictions with miner hotkey")
        query = """
        UPDATE predictions
        SET minerHotkey = %s
        WHERE minerID = %s AND minerHotkey IS NULL
        """
        try:
            self.db_manager.execute_query(query, params=(self.miner_hotkey, self.miner_uid))
            bt.logging.info("Successfully updated predictions with miner hotkey")
        except Exception as e:
            bt.logging.error(f"Error updating predictions with hotkey: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    def ensure_correct_column_names(self):
        bt.logging.info("Ensuring correct column names in predictions table")
        query = """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'predictions' AND column_name = 'minerid') THEN
                ALTER TABLE predictions ADD COLUMN minerID VARCHAR(255);
            END IF;
            
            IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'predictions' AND column_name = 'miner_uid') THEN
                ALTER TABLE predictions DROP COLUMN miner_uid;
            END IF;
        END $$;
        """
        try:
            self.db_manager.execute_query(query)
            bt.logging.info("Successfully ensured correct column names")
        except Exception as e:
            bt.logging.error(f"Error ensuring correct column names: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")

    def update_predictions_with_minerid(self):
        bt.logging.trace("Updating predictions with miner ID")
        query = """
        UPDATE predictions
        SET minerID = %s
        WHERE minerID IS NULL
        """
        try:
            self.db_manager.execute_query(query, params=(self.miner_uid,))
            bt.logging.info("Successfully updated predictions with miner ID")
        except Exception as e:
            bt.logging.error(f"Error updating predictions with miner ID: {str(e)}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")