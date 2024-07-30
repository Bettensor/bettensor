import os
import threading
from typing import Dict, List
import bittensor as bt
from datetime import datetime, timezone, timedelta
import pytz
import torch
import time
import traceback

from bettensor.protocol import TeamGame

class MinerStateManager:
    def __init__(self, db_manager, miner_hotkey, miner_uid):
        self.db_manager = db_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = miner_uid
        self.state = self.load_state()
        self.last_cash_reset = None
        self.lock = threading.Lock()
        self.stats_handler = MinerStatsHandler(db_manager, self)
        bt.logging.info(f"Initial state loaded: {self.state}")
        self._reconcile_state()

    def load_state(self):
        bt.logging.info("Loading miner state from database")
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM miner_stats
                    WHERE miner_hotkey = ?
                """, (self.miner_hotkey,))
                result = cursor.fetchone()
            
            if result:
                # Convert the result to a dictionary
                columns = [column[0] for column in cursor.description]
                result_dict = dict(zip(columns, result))
                
                state = {
                    'miner_hotkey': self.miner_hotkey,
                    'miner_uid': self.miner_uid,
                    'miner_cash': result_dict['miner_cash'],
                    'last_daily_reset': result_dict['last_daily_reset'],
                    'miner_lifetime_predictions': result_dict['miner_lifetime_predictions'],
                    'miner_lifetime_wins': result_dict['miner_lifetime_wins'],
                    'miner_lifetime_losses': result_dict['miner_lifetime_losses'],
                    'miner_lifetime_earnings': result_dict['miner_lifetime_earnings'],
                    'miner_lifetime_wager': result_dict['miner_lifetime_wager'],
                    'miner_win_loss_ratio': result_dict['miner_win_loss_ratio'],
                    'miner_current_incentive': result_dict['miner_current_incentive'],
                    'miner_rank': result_dict.get('miner_rank', 0),
                }
                bt.logging.info(f"Loaded state: {state}")
            else:
                bt.logging.info("No existing state found, initializing with default values")
                state = {
                    'miner_hotkey': self.miner_hotkey,
                    'miner_uid': self.miner_uid,
                    'miner_cash': 1000.0,
                    'last_daily_reset': datetime.now(timezone.utc).isoformat(),
                    'miner_lifetime_predictions': 0,
                    'miner_lifetime_wins': 0,
                    'miner_lifetime_losses': 0,
                    'miner_lifetime_earnings': 0.0,
                    'miner_lifetime_wager': 0.0,
                    'miner_win_loss_ratio': 0.0,
                    'miner_current_incentive': 0.0,
                    'miner_rank': 0,
                }
                self.save_state(state)
        except Exception as e:
            bt.logging.error(f"Error loading miner state: {e}")
            bt.logging.error(traceback.format_exc())
            state = {
                'miner_hotkey': self.miner_hotkey,
                'miner_uid': self.miner_uid,
                'miner_cash': 1000.0,
                'last_daily_reset': datetime.now(timezone.utc).isoformat(),
                'miner_lifetime_predictions': 0,
                'miner_lifetime_wins': 0,
                'miner_lifetime_losses': 0,
                'miner_lifetime_earnings': 0.0,
                'miner_lifetime_wager': 0.0,
                'miner_win_loss_ratio': 0.0,
                'miner_current_incentive': 0.0,
                'miner_rank': 0,
            }
        return state

    def save_state(self, state=None):
        if state is None:
            state = self.state
        bt.logging.info(f"Saving state to database: {state}")
        try:
            self.update_miner_stats(state)
        except Exception as e:
            bt.logging.error(f"Error saving miner state: {e}")
            bt.logging.error(traceback.format_exc())

    def update_miner_stats(self, stats):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO miner_stats
                (miner_hotkey, miner_uid, miner_cash, last_daily_reset, miner_lifetime_predictions,
                miner_lifetime_wins, miner_lifetime_losses, miner_lifetime_earnings,
                miner_lifetime_wager, miner_win_loss_ratio, miner_current_incentive, miner_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.miner_hotkey,
                stats['miner_uid'],
                stats['miner_cash'],
                stats['last_daily_reset'],
                stats['miner_lifetime_predictions'],
                stats['miner_lifetime_wins'],
                stats['miner_lifetime_losses'],
                stats['miner_lifetime_earnings'],
                stats['miner_lifetime_wager'],
                stats['miner_win_loss_ratio'],
                stats['miner_current_incentive'],
                stats.get('miner_rank', 0)
            ))

    def _reconcile_state(self):
        bt.logging.info("Entering _reconcile_state")
        try:
            now = datetime.now(timezone.utc)
            last_reset = self.state.get('last_daily_reset')
            if isinstance(last_reset, str):
                last_reset = datetime.fromisoformat(last_reset).replace(tzinfo=timezone.utc)
            elif not isinstance(last_reset, datetime):
                last_reset = now - timedelta(days=1)  # Set to yesterday if not set
            
            bt.logging.info(f"Now: {now}, Last reset: {last_reset}")
            bt.logging.info(f"Current cash before reconcile: {self.state['miner_cash']}")

            # Only reset cash if it hasn't been reset today and it's a new day
            if now.date() > last_reset.date():
                bt.logging.info("Conditions met for resetting daily cash")
                self.reset_daily_cash()
            else:
                bt.logging.info("Conditions not met for resetting daily cash")

            bt.logging.info(f"Current cash after reconcile: {self.state['miner_cash']}")
        except Exception as e:
            bt.logging.error(f"Error in _reconcile_state: {e}")
        bt.logging.info("Exiting _reconcile_state")

    def reset_daily_cash(self):
        bt.logging.info("Resetting daily cash")
        self.state['miner_cash'] = 1000.0
        self.state['last_daily_reset'] = datetime.now(timezone.utc).isoformat()
        self.save_state()
        bt.logging.info(f"Daily cash reset. New cash: {self.state['miner_cash']}, Last reset: {self.state['last_daily_reset']}")

    def _get_last_prediction_date(self):
        if not self.db_manager:
            return None
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT MAX(predictionDate) FROM predictions
                WHERE minerID = ?
            """, (self.miner_uid,))
            result = cursor.fetchone()
            if result[0]:
                return datetime.fromisoformat(result[0]).replace(tzinfo=timezone.utc)
        return None

    def update_on_prediction(self, prediction_data):
        with self.lock:
            bt.logging.info(f"Cash before prediction: {self.state['miner_cash']}")
            self.state['miner_lifetime_predictions'] += 1
            self.state['miner_cash'] -= prediction_data['wager']
            self.state['miner_lifetime_wager'] += prediction_data['wager']
            self.state['miner_last_prediction_date'] = datetime.now(timezone.utc).isoformat()
            bt.logging.info(f"Cash after prediction: {self.state['miner_cash']}")
            self.save_state()

    def update_on_game_result(self, result_data):
        with self.lock:
            if result_data['outcome'] == 'win':
                self.state['miner_lifetime_wins'] += 1
                # Only update lifetime earnings, not cash
                self.state['miner_lifetime_earnings'] += result_data['earnings']
            else:
                self.state['miner_lifetime_losses'] += 1
            
            total_games = self.state['miner_lifetime_wins'] + self.state['miner_lifetime_losses']
            if total_games > 0:
                self.state['miner_win_loss_ratio'] = self.state['miner_lifetime_wins'] / total_games
            
            self.save_state()

    def update_current_incentive(self, incentive):
        with self.lock:
            self.state['miner_current_incentive'] = incentive
            self.save_state()

    def get_current_incentive(self):
        with self.lock:
            return self.state['miner_current_incentive']

    def get_stats(self):
        bt.logging.trace("Entering get_stats method")
        with self.lock:
            stats = self.state.copy()
        bt.logging.trace(f"Returning stats: {stats}")
        bt.logging.trace("Exiting get_stats method")
        return stats

    def periodic_db_update(self):
        bt.logging.trace("Performing periodic database update")
        try:
            self.save_state()
            self.stats_handler.recalculate_stats()
        except Exception as e:
            bt.logging.error(f"Error in periodic_db_update: {e}")
            bt.logging.error(traceback.format_exc())

    def deduct_wager(self, amount):
        with self.lock:
            current_cash = self.state['miner_cash']
            if current_cash >= amount:
                self.state['miner_cash'] = current_cash - amount
                bt.logging.info(f"Deducted wager of {amount}. New balance: {self.state['miner_cash']}")
                self.save_state()
                return True
            else:
                bt.logging.warning(f"Insufficient funds. Current balance: {current_cash}, Wager amount: {amount}")
                return False

    def update_stats_from_predictions(self, updated_predictions: List[Dict], updated_games: Dict[str, TeamGame]):
        bt.logging.debug("Updating miner stats based on new prediction outcomes")
        try:
            for pred in updated_predictions:
                game = updated_games[pred['teamGameID']]
                self.update_stats_on_game_result(pred, game)
            
            bt.logging.info(f"Updated stats for {len(updated_predictions)} predictions")
        except Exception as e:
            bt.logging.error(f"Error updating stats from predictions: {e}")

    def update_stats_on_game_result(self, prediction, game):
        wager = prediction['wager']
        predicted_outcome = prediction['predictedOutcome']
        game_outcome = game.outcome

        if predicted_outcome == game_outcome:
            self.state['miner_lifetime_wins'] += 1
            odds = prediction['teamAodds'] if game_outcome == "Team A Win" else prediction['teamBodds'] if game_outcome == "Team B Win" else prediction['tieOdds']
            earnings = wager * (odds - 1)
            self.state['miner_lifetime_earnings'] += earnings
        else:
            self.state['miner_lifetime_losses'] += 1
            self.state['miner_lifetime_earnings'] -= wager

        self.state['miner_lifetime_predictions'] += 1
        total_games = self.state['miner_lifetime_wins'] + self.state['miner_lifetime_losses']
        self.state['miner_win_loss_ratio'] = self.state['miner_lifetime_wins'] / total_games if total_games > 0 else 0

        self.save_state()


class MinerStatsHandler:
    def __init__(self, db_manager, state_manager):
        self.db_manager = db_manager
        self.state_manager = state_manager

    def recalculate_stats(self):
        bt.logging.trace("Entering recalculate_stats method")
        if not self.db_manager:
            bt.logging.trace("Skipping recalculate_stats (db_manager not available)")
            return

        try:
            with self.db_manager.get_cursor() as cursor:
                bt.logging.trace("Executing SELECT query for recalculating stats")
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as total_wins,
                        SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as total_losses,
                        SUM(CASE WHEN outcome = 'win' THEN wager * teamAodds ELSE 0 END) as total_earnings,
                        SUM(wager) as total_wager
                    FROM predictions
                    WHERE minerID = ?
                """, (self.state_manager.miner_uid,))
                
                result = cursor.fetchone()
                bt.logging.trace(f"Query result: {result}")

                if result:
                    with self.state_manager.lock:
                        current_state = self.state_manager.state
                        current_state['miner_lifetime_predictions'] = result[0] or 0
                        current_state['miner_lifetime_wins'] = result[1] or 0
                        current_state['miner_lifetime_losses'] = result[2] or 0
                        current_state['miner_lifetime_earnings'] = result[3] or 0.0
                        current_state['miner_lifetime_wager'] = result[4] or 0.0
                        
                        # Calculate win_loss_ratio
                        total_games = current_state['miner_lifetime_wins'] + current_state['miner_lifetime_losses']
                        if total_games > 0:
                            current_state['miner_win_loss_ratio'] = current_state['miner_lifetime_wins'] / total_games
                        else:
                            current_state['miner_win_loss_ratio'] = 0.0
                        
                        self.state_manager.state = current_state
                    
                    bt.logging.trace("Saving updated state")
                    self.state_manager.save_state()
                else:
                    bt.logging.trace("No results found for recalculating stats")
        except Exception as e:
            bt.logging.error(f"Error in recalculate_stats: {e}")
        bt.logging.trace("Exiting recalculate_stats method")