import os
import threading
from typing import Dict, List
import bittensor as bt
from datetime import datetime, timezone, timedelta
import pytz
import torch
import time
import traceback
import psycopg2
import sqlite3

from bettensor.protocol import TeamGame


class MinerStatsHandler:
    def __init__(self, db_manager, state_manager):
        self.db_manager = db_manager
        self.state_manager = state_manager
        self.recalculate_stats_from_predictions()

    def recalculate_stats_from_predictions(self):
        bt.logging.info("Recalculating miner stats from existing predictions")
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN outcome LIKE 'Wager Won%%' THEN 1 ELSE 0 END) as total_wins,
                SUM(CASE WHEN outcome LIKE 'Wager Lost%%' THEN 1 ELSE 0 END) as total_losses,
                SUM(CASE WHEN outcome LIKE 'Wager Won%%' THEN 
                    wager * (
                        CASE 
                            WHEN predictedOutcome = teamA THEN teamAodds
                            WHEN predictedOutcome = teamB THEN teamBodds
                            ELSE tieOdds
                        END
                    ) - wager 
                ELSE 0 END) as total_earnings,
                SUM(wager) as total_wager,
                MAX(predictionDate) as last_prediction_date
            FROM predictions
            WHERE minerID = %s
        """, (self.state_manager.miner_uid,))
        
        result = cursor.fetchone()
        
        if result:
            with self.state_manager.lock:
                self.state_manager.state['miner_lifetime_predictions'] = result[0] or 0
                self.state_manager.state['miner_lifetime_wins'] = result[1] or 0
                self.state_manager.state['miner_lifetime_losses'] = result[2] or 0
                self.state_manager.state['miner_lifetime_earnings'] = result[3] or 0.0
                self.state_manager.state['miner_lifetime_wager'] = result[4] or 0.0
                self.state_manager.state['miner_last_prediction_date'] = result[5]
                
                total_games = self.state_manager.state['miner_lifetime_wins'] + self.state_manager.state['miner_lifetime_losses']
                if total_games > 0:
                    self.state_manager.state['miner_win_loss_ratio'] = self.state_manager.state['miner_lifetime_wins'] / total_games
                else:
                    self.state_manager.state['miner_win_loss_ratio'] = 0.0
                
            self.state_manager.save_state()
            bt.logging.info("Miner stats recalculated from existing predictions")
        else:
            bt.logging.info("No predictions found for recalculation")

    def recalculate_stats(self):
        bt.logging.trace("Entering recalculate_stats method")
        if not self.db_manager:
            bt.logging.trace("Skipping recalculate_stats (db_manager not available)")
            return

        try:
            self.recalculate_stats_from_predictions()
        except Exception as e:
            bt.logging.error(f"Error in recalculate_stats: {e}")
        
        bt.logging.trace("Exiting recalculate_stats method")

class MinerStateManager:
    def __init__(self, db_manager, miner_hotkey, miner_uid):
        self.db_manager = db_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = miner_uid
        self._ensure_miner_stats_table_exists()
        self.state = self.load_state()
        self.lock = threading.Lock()
        self.stats_handler = MinerStatsHandler(db_manager, self)
        bt.logging.trace(f"Initial state loaded: {self.state}")
        self._reconcile_state()
        self.stats_handler.recalculate_stats_from_predictions()

    def _ensure_miner_stats_table_exists(self):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS miner_stats (
                    miner_hotkey TEXT PRIMARY KEY,
                    miner_uid INTEGER,
                    miner_rank INTEGER,
                    miner_cash REAL,
                    miner_current_incentive REAL,
                    miner_last_prediction_date TIMESTAMP,
                    miner_lifetime_earnings REAL,
                    miner_lifetime_wager REAL,
                    miner_lifetime_predictions INTEGER,
                    miner_lifetime_wins INTEGER,
                    miner_lifetime_losses INTEGER,
                    miner_win_loss_ratio REAL,
                    last_daily_reset TIMESTAMP
                )
            """)

    def _load_state_from_db(self):
        bt.logging.trace("Loading miner state from database")
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM miner_stats WHERE miner_uid = %s", (self.miner_uid,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
            bt.logging.trace("No existing state found, will initialize with default values")
            return None
        except psycopg2.Error as e:
            bt.logging.error(f"Error loading miner state: {e}")
            bt.logging.error(traceback.format_exc())
            return None

    def load_state(self):
        loaded_state = self._load_state_from_db()
        if loaded_state:
            return loaded_state
        else:
            return self._initialize_default_state()

    def _initialize_default_state(self):
        bt.logging.trace("Initializing default state")
        state = {
            'miner_hotkey': self.miner_hotkey,
            'miner_uid': self.miner_uid,
            'miner_rank': 0,
            'miner_cash': 1000.0,
            'miner_current_incentive': 0.0,
            'miner_last_prediction_date': None,
            'miner_lifetime_earnings': 0.0,
            'miner_lifetime_wager': 0.0,
            'miner_lifetime_predictions': 0,
            'miner_lifetime_wins': 0,
            'miner_lifetime_losses': 0,
            'miner_win_loss_ratio': 0.0,
            'last_daily_reset': datetime.now(timezone.utc).isoformat(),
        }
        self.save_state(state)
        return state

    def save_state(self, state=None):
        if state is None:
            state = self.state
        bt.logging.trace(f"Saving state to database: {state}")
        max_retries = 5
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                self.update_miner_stats(state)
                break
            except psycopg2.OperationalError as e:
                if "deadlock detected" in str(e) and attempt < max_retries - 1:
                    bt.logging.warning(f"Deadlock detected. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    bt.logging.error(f"Error saving miner state: {e}")
                    break
            except Exception as e:
                bt.logging.error(f"Error saving miner state: {e}")
                bt.logging.error(traceback.format_exc())
                break

    def update_miner_stats(self, stats):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO miner_stats
                (miner_hotkey, miner_uid, miner_rank, miner_cash, miner_current_incentive,
                miner_last_prediction_date, miner_lifetime_earnings, miner_lifetime_wager,
                miner_lifetime_predictions, miner_lifetime_wins, miner_lifetime_losses,
                miner_win_loss_ratio, last_daily_reset)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (miner_hotkey) DO UPDATE SET
                miner_uid = EXCLUDED.miner_uid,
                miner_rank = EXCLUDED.miner_rank,
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
            """, (
                self.miner_hotkey,
                self.miner_uid,
                stats.get('miner_rank', 0),
                stats['miner_cash'],
                stats['miner_current_incentive'],
                stats.get('miner_last_prediction_date'),
                stats['miner_lifetime_earnings'],
                stats['miner_lifetime_wager'],
                stats['miner_lifetime_predictions'],
                stats['miner_lifetime_wins'],
                stats['miner_lifetime_losses'],
                stats['miner_win_loss_ratio'],
                stats['last_daily_reset']
            ))

    def _reconcile_state(self):
        bt.logging.trace("Reconciling miner state")
        now = datetime.now(timezone.utc)
        last_reset = self.state.get('last_daily_reset')
        if isinstance(last_reset, str):
            last_reset = datetime.fromisoformat(last_reset).replace(tzinfo=timezone.utc)
        elif not isinstance(last_reset, datetime):
            last_reset = now - timedelta(days=1)  # Set to yesterday if not set
        
        if now.date() > last_reset.date():
            bt.logging.trace("Performing daily cash reset")
            self.reset_daily_cash()
        else:
            bt.logging.trace("No need for daily cash reset")
        
        self.recalculate_miner_cash()

    def reset_daily_cash(self):
        bt.logging.trace("Resetting daily cash")
        self.state['miner_cash'] = 1000.0
        self.state['last_daily_reset'] = datetime.now(timezone.utc).isoformat()
        self.save_state()

    def recalculate_miner_cash(self):
        bt.logging.trace("Recalculating miner cash")
        current_time = datetime.now(timezone.utc)
        reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT COALESCE(SUM(wager), 0) as total_wager
                FROM predictions
                WHERE minerID = %s AND predictionDate >= %s
            """, (self.miner_uid, reset_time.isoformat()))
            
            result = cursor.fetchone()
            total_wager = result[0] if result[0] is not None else 0
            
            self.state['miner_cash'] = 1000 - total_wager
            self.save_state()
        bt.logging.trace(f"Recalculated miner cash: {self.state['miner_cash']}")

    def _get_last_prediction_date(self):
        if not self.db_manager:
            return None
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT MAX(predictionDate) FROM predictions
                WHERE minerID = %s
            """, (self.miner_uid,))
            result = cursor.fetchone()
            if result[0]:
                return datetime.fromisoformat(result[0]).replace(tzinfo=timezone.utc)
        return None

    def update_on_prediction(self, prediction_data):
        with self.lock:
            bt.logging.trace(f"Cash before prediction: {self.state['miner_cash']}")
            self.state['miner_lifetime_predictions'] += 1
            self.state['miner_cash'] -= prediction_data['wager']
            self.state['miner_lifetime_wager'] += prediction_data['wager']
            self.state['miner_last_prediction_date'] = datetime.now(timezone.utc).isoformat()
            bt.logging.trace(f"Cash after prediction: {self.state['miner_cash']}")
            self.save_state()

    def update_on_game_result(self, result_data):
        with self.lock:
            bt.logging.debug(f"Updating state for game result: {result_data}")
            if 'Wager Won' in result_data['outcome']:
                self.state['miner_lifetime_wins'] += 1
                self.state['miner_lifetime_earnings'] += result_data['earnings']
            elif 'Wager Lost' in result_data['outcome']:
                self.state['miner_lifetime_losses'] += 1
                # Do not subtract earnings for losses

            total_games = self.state['miner_lifetime_wins'] + self.state['miner_lifetime_losses']
            self.state['miner_win_loss_ratio'] = self.state['miner_lifetime_wins'] / total_games if total_games > 0 else 0
            bt.logging.debug(f"Updated state: wins={self.state['miner_lifetime_wins']}, losses={self.state['miner_lifetime_losses']}, earnings={self.state['miner_lifetime_earnings']}")
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
                bt.logging.trace(f"Deducted wager of {amount}. New balance: {self.state['miner_cash']}")
                self.save_state()
                return True
            else:
                bt.logging.trace(f"Insufficient funds. Current balance: {current_cash}, Wager amount: {amount}")
                return False

    def update_stats_from_predictions(self, predictions, updated_games):
        for prediction in predictions:
            game = updated_games.get(prediction.teamGameID)
            if game:
                if prediction.outcome == 'Wager Won':
                    odds = (prediction.teamAodds if game.outcome == 0 else 
                            prediction.teamBodds if game.outcome == 1 else 
                            prediction.tieOdds)
                    earnings = prediction.wager * (odds - 1)
                else:
                    earnings = -prediction.wager
                
                self.update_on_game_result({
                    'outcome': prediction.outcome,
                    'earnings': earnings,
                    'wager': prediction.wager,
                    'prediction': prediction
                })