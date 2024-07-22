import os
import threading
import bittensor as bt
from datetime import datetime, timedelta
import pytz
import torch

class MinerStateManager:
    def __init__(self, db_manager, miner_hotkey, miner_uid):
        self.db_manager = db_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = miner_uid
        self.state_file = f"data/miner_state_{miner_uid}.pt"
        self.lock = threading.Lock()
        self.stats_handler = MinerStatsHandler(self)
        self.load_state()

    def load_state(self):
        with self.lock:
            file_state = self._load_state_from_file()
            db_state = self._load_state_from_db()
            
            if file_state and db_state:
                # Use the most recent state
                if file_state['last_daily_reset'] > db_state['last_daily_reset']:
                    self.state = file_state
                else:
                    self.state = db_state
            elif file_state:
                self.state = file_state
            elif db_state:
                self.state = db_state
            else:
                self.state = self._get_initial_state()
            
            self._reconcile_state()
            self.save_state()  # Save the reconciled state to both file and DB

    def _load_state_from_file(self):
        if os.path.exists(self.state_file):
            return torch.load(self.state_file)
        return None

    def _load_state_from_db(self):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM miner_stats WHERE miner_hotkey = ?", (self.miner_hotkey,))
            row = cursor.fetchone()
            if row:
                return {
                    "miner_cash": torch.tensor(row[2], dtype=torch.float32),
                    "miner_lifetime_earnings": torch.tensor(row[3], dtype=torch.float32),
                    "miner_lifetime_predictions": torch.tensor(row[4], dtype=torch.int64),
                    "miner_lifetime_wins": torch.tensor(row[5], dtype=torch.int64),
                    "miner_lifetime_losses": torch.tensor(row[6], dtype=torch.int64),
                    "last_daily_reset": row[7],
                }
            return None

    def save_state(self):
        with self.lock:
            torch.save(self.state, self.state_file)
            self.stats_handler.update_miner_row(self.get_stats())

    def _get_initial_state(self):
        return {
            "miner_cash": torch.tensor(1000, dtype=torch.float32),
            "miner_lifetime_earnings": torch.tensor(0, dtype=torch.float32),
            "miner_lifetime_predictions": torch.tensor(0, dtype=torch.int64),
            "miner_lifetime_wins": torch.tensor(0, dtype=torch.int64),
            "miner_lifetime_losses": torch.tensor(0, dtype=torch.int64),
            "last_daily_reset": datetime.now(pytz.utc).isoformat(),
        }

    def _reconcile_state(self):
        now = datetime.now(pytz.utc)
        last_reset = datetime.fromisoformat(self.state['last_daily_reset'])
        last_prediction = self._get_last_prediction_date()

        if now.date() > last_reset.date() and (last_prediction is None or last_prediction.date() < now.date()):
            self.reset_daily_cash()

        # Ensure state matches database
        self.stats_handler.recalculate_stats()

    def _get_last_prediction_date(self):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT MAX(predictionDate) FROM predictions
                WHERE minerID = ?
            """, (self.miner_uid,))
            result = cursor.fetchone()
            if result[0]:
                return datetime.fromisoformat(result[0])
        return None

    def update_on_prediction(self, prediction):
        with self.lock:
            self.state['miner_lifetime_predictions'] += 1
            self.state['miner_cash'] -= prediction['wager']
            self.save_state()
        self.stats_handler.recalculate_stats()

    def update_on_game_result(self, game_result):
        with self.lock:
            if game_result['outcome'] == 'win':
                self.state['miner_lifetime_wins'] += 1
                self.state['miner_lifetime_earnings'] += game_result['earnings']
                self.state['miner_cash'] += game_result['earnings']
            else:
                self.state['miner_lifetime_losses'] += 1
            self.save_state()
        self.stats_handler.recalculate_stats()

    def reset_daily_cash(self):
        with self.lock:
            self.state['miner_cash'] = torch.tensor(1000, dtype=torch.float32)
            self.state['last_daily_reset'] = datetime.now(pytz.utc).isoformat()
            self.save_state()
        self.stats_handler.recalculate_stats()

    def get_stats(self):
        with self.lock:
            return {
                "miner_hotkey": self.miner_hotkey,
                "miner_uid": self.miner_uid,
                "miner_cash": self.state['miner_cash'].item(),
                "miner_lifetime_earnings": self.state['miner_lifetime_earnings'].item(),
                "miner_lifetime_predictions": self.state['miner_lifetime_predictions'].item(),
                "miner_lifetime_wins": self.state['miner_lifetime_wins'].item(),
                "miner_lifetime_losses": self.state['miner_lifetime_losses'].item(),
                "miner_win_loss_ratio": self._calculate_win_loss_ratio(),
                "miner_last_prediction_date": self.state['last_daily_reset'],
            }

    def _calculate_win_loss_ratio(self):
        total_games = self.state['miner_lifetime_wins'] + self.state['miner_lifetime_losses']
        if total_games == 0:
            return 0
        return round((self.state['miner_lifetime_wins'] / total_games).item(), 3)

class MinerStatsHandler:
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.db_manager = state_manager.db_manager

    def return_miner_stats(self, miner_hotkey):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM miner_stats WHERE miner_hotkey = ?", (miner_hotkey,))
            row = cursor.fetchone()
            if row:
                return row
        return None

    def init_miner_row(self):
        with self.db_manager.get_cursor() as cursor:
            # First, check if the row already exists
            cursor.execute("SELECT * FROM miner_stats WHERE miner_hotkey = ?", (self.state_manager.miner_hotkey,))
            existing_row = cursor.fetchone()
            
            if existing_row:
                bt.logging.info(f"Miner row already exists for hotkey: {self.state_manager.miner_hotkey}")
                return existing_row
            
            # If no existing row, insert a new one
            cursor.execute("""
                INSERT INTO miner_stats 
                (miner_hotkey, miner_uid, miner_cash, miner_lifetime_earnings, 
                miner_lifetime_predictions, miner_lifetime_wins, miner_lifetime_losses, 
                miner_win_loss_ratio, miner_last_prediction_date) 
                VALUES (?, ?, 1000, 0, 0, 0, 0, 0, ?)
            """, (self.state_manager.miner_hotkey, self.state_manager.miner_uid, datetime.now(pytz.utc).isoformat()))
            cursor.connection.commit()
            bt.logging.info(f"Initialized new miner row for hotkey: {self.state_manager.miner_hotkey}")
        
        return self.return_miner_stats(self.state_manager.miner_hotkey)

    def update_miner_row(self, stats):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                UPDATE miner_stats
                SET miner_cash = ?,
                    miner_lifetime_earnings = ?,
                    miner_lifetime_predictions = ?,
                    miner_lifetime_wins = ?,
                    miner_lifetime_losses = ?,
                    miner_win_loss_ratio = ?,
                    miner_last_prediction_date = ?
                WHERE miner_hotkey = ?
            """, (
                stats['miner_cash'],
                stats['miner_lifetime_earnings'],
                stats['miner_lifetime_predictions'],
                stats['miner_lifetime_wins'],
                stats['miner_lifetime_losses'],
                stats['miner_win_loss_ratio'],
                stats['miner_last_prediction_date'],
                stats['miner_hotkey']
            ))
            cursor.connection.commit()

    def recalculate_stats(self):
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as total_wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as total_losses,
                    SUM(CASE WHEN outcome = 'win' THEN wager * teamAodds ELSE 0 END) as total_earnings
                FROM predictions
                WHERE minerID = ?
            """, (self.state_manager.miner_uid,))
            result = cursor.fetchone()

            if result:
                with self.state_manager.lock:
                    self.state_manager.state['miner_lifetime_predictions'] = torch.tensor(result[0], dtype=torch.int64)
                    self.state_manager.state['miner_lifetime_wins'] = torch.tensor(result[1], dtype=torch.int64)
                    self.state_manager.state['miner_lifetime_losses'] = torch.tensor(result[2], dtype=torch.int64)
                    self.state_manager.state['miner_lifetime_earnings'] = torch.tensor(result[3], dtype=torch.float32)
                    self.state_manager.save_state()

    def reset_daily_cash_on_startup(self):
        self.state_manager._reconcile_state()