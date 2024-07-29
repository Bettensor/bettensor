import os
import threading
from typing import Dict, List
import bittensor as bt
from datetime import datetime, timedelta, timezone
import pytz
import torch
import time

from bettensor.protocol import TeamGame

class MinerStateManager:
    def __init__(self, db_manager, miner_hotkey, miner_uid):
        bt.logging.info("Initializing MinerStateManager")
        self.db_manager = db_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = miner_uid
        self.state_file = f"data/miner_state_{miner_uid}.pt"
        self.lock = threading.RLock()
        self.state = self._get_initial_state()
        
        bt.logging.info("Creating MinerStatsHandler")
        self.stats_handler = MinerStatsHandler(self.db_manager, self)
        
        bt.logging.info("Initializing miner row")
        self.stats_handler.init_miner_row()
        
        if self.db_manager:
            bt.logging.info("Loading state")
            self.load_state()
        
        bt.logging.info("MinerStateManager initialization complete")

    def load_state(self):
        bt.logging.debug("Starting load_state")
        file_state = self._load_state_from_file()
        db_state = self._load_state_from_db()

        bt.logging.debug(f"File state: {file_state}")
        bt.logging.debug(f"DB state: {db_state}")

        if file_state and not db_state:
            bt.logging.debug("Only file state exists")
            self.state = file_state
            if isinstance(self.state['last_daily_reset'], str):
                self.state['last_daily_reset'] = torch.tensor(self._date_to_timestamp(self.state['last_daily_reset']), dtype=torch.float64)
        elif db_state and not file_state:
            bt.logging.debug("Only DB state exists")
            self.state = db_state
        elif db_state and file_state:
            bt.logging.debug("Both states exist, using the most recent")
            db_timestamp = db_state['last_daily_reset'].item()
            file_timestamp = self._date_to_timestamp(file_state['last_daily_reset']) if isinstance(file_state['last_daily_reset'], str) else file_state['last_daily_reset'].item()
            self.state = db_state if db_timestamp > file_timestamp else file_state
        else:
            bt.logging.debug("No existing state, initializing new state")
            self.state = self._initialize_new_state()

        bt.logging.debug(f"Final state: {self.state}")
        self._reconcile_state()
        bt.logging.debug("load_state complete")

    def _load_state_from_file(self):
        if os.path.exists(self.state_file):
            return torch.load(self.state_file)
        return None

    def _date_to_timestamp(self, date_str):
        try:
            bt.logging.debug(f"Attempting to parse date string: {date_str}")
            dt = datetime.fromisoformat(date_str)
            timestamp = dt.timestamp()
            bt.logging.debug(f"Successfully converted date to timestamp: {timestamp}")
            return timestamp
        except Exception as e:
            bt.logging.error(f"Error converting date to timestamp: {e}")
            bt.logging.error(f"Problematic date string: {date_str}")
            return datetime.now(timezone.utc).timestamp()

    def _timestamp_to_date(self, timestamp):
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        except Exception as e:
            bt.logging.error(f"Error converting timestamp to date: {e}")
            return datetime.now(timezone.utc).isoformat()

    def _load_state_from_db(self):
        if not self.db_manager:
            return None
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM miner_stats WHERE miner_hotkey = ?", (self.miner_hotkey,))
                row = cursor.fetchone()
                if row:
                    bt.logging.debug(f"Raw database row: {row}")
                    state = {}
                    for i, value in enumerate(row):
                        bt.logging.debug(f"Field {i}: {value} (type: {type(value).__name__})")
                    try:
                        state["miner_cash"] = torch.tensor(float(row[4]) if row[4] is not None else 1000.0, dtype=torch.float32)
                        state["miner_lifetime_earnings"] = torch.tensor(float(row[5]) if row[5] is not None else 0.0, dtype=torch.float32)
                        state["miner_lifetime_wager"] = torch.tensor(float(row[7]) if row[7] is not None else 0.0, dtype=torch.float32)
                        state["miner_current_incentive"] = torch.tensor(float(row[8]) if row[8] is not None else 0.0, dtype=torch.float32)
                        state["miner_rank"] = torch.tensor(int(row[3]) if row[3] is not None else 0, dtype=torch.int64)
                        state["miner_lifetime_predictions"] = torch.tensor(int(row[9]) if row[9] is not None else 0, dtype=torch.int64)
                        state["miner_lifetime_wins"] = torch.tensor(int(row[10]) if row[10] is not None else 0, dtype=torch.int64)
                        state["miner_lifetime_losses"] = torch.tensor(int(row[11]) if row[11] is not None else 0, dtype=torch.int64)
                        state["miner_win_loss_ratio"] = torch.tensor(float(row[12]) if row[12] is not None else 0.0, dtype=torch.float32)
                        
                        date_str = row[6]
                        if date_str:
                            try:
                                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                state["miner_last_prediction_date"] = torch.tensor(date_obj.timestamp(), dtype=torch.float64)
                            except Exception as date_error:
                                bt.logging.error(f"Error parsing date string: {date_error}")
                                state["miner_last_prediction_date"] = torch.tensor(datetime.now(timezone.utc).timestamp(), dtype=torch.float64)
                        else:
                            state["miner_last_prediction_date"] = torch.tensor(datetime.now(timezone.utc).timestamp(), dtype=torch.float64)
                        
                        state["last_daily_reset"] = torch.tensor(datetime.now(timezone.utc).timestamp(), dtype=torch.float64)
                        
                        return state
                    except Exception as e:
                        bt.logging.error(f"Error parsing database row: {e}")
                        return None
                else:
                    bt.logging.warning(f"No state found in database for hotkey: {self.miner_hotkey}")
                    return None
        except Exception as e:
            bt.logging.error(f"Error loading state from database: {e}")
            return None

    def save_state(self):
        bt.logging.debug("Entering save_state method")
        try:
            bt.logging.debug(f"Saving state to file: {self.state_file}")
            torch.save(self.state, self.state_file)
            bt.logging.debug("State saved to file successfully")

            if self.db_manager and self.stats_handler:
                bt.logging.debug("Updating miner row in database")
                try:
                    stats = self.get_stats()
                    bt.logging.debug(f"Got stats: {stats}")
                    self.stats_handler.update_miner_row(stats)
                    bt.logging.debug("Miner row updated in database")
                except Exception as e:
                    bt.logging.error(f"Error updating miner row: {e}")
            else:
                bt.logging.debug("Skipping database update (db_manager or stats_handler not available)")
            
            bt.logging.trace("Finished processing in save_state")
        except Exception as e:
            bt.logging.error(f"Unexpected error in save_state: {e}")
        bt.logging.debug("Exiting save_state method")

    def _get_initial_state(self):
        return {
            'miner_cash': torch.tensor(1000.0, dtype=torch.float32),
            'miner_lifetime_earnings': torch.tensor(0.0, dtype=torch.float32),
            'miner_lifetime_wager': torch.tensor(0.0, dtype=torch.float32),
            'miner_current_incentive': torch.tensor(0.0, dtype=torch.float32),
            'miner_rank': torch.tensor(0, dtype=torch.int64),
            'miner_lifetime_predictions': torch.tensor(0, dtype=torch.int64),
            'miner_lifetime_wins': torch.tensor(0, dtype=torch.int64),
            'miner_lifetime_losses': torch.tensor(0, dtype=torch.int64),
            'miner_win_loss_ratio': torch.tensor(0.0, dtype=torch.float32),
            'miner_last_prediction_date': torch.tensor(datetime.now(timezone.utc).timestamp(), dtype=torch.float64),
            'last_daily_reset': torch.tensor(datetime.now(timezone.utc).timestamp(), dtype=torch.float64),
        }

    def _reconcile_state(self):
        bt.logging.trace("Entering _reconcile_state")
        try:
            now = datetime.now(pytz.utc)
            last_reset = self.state['last_daily_reset']
            if isinstance(last_reset, str):
                last_reset = datetime.fromisoformat(last_reset)
            elif not isinstance(last_reset, datetime):
                last_reset = now
            
            last_prediction = self._get_last_prediction_date()

            if now.date() > last_reset.date() and (last_prediction is None or last_prediction.date() < now.date()):
                bt.logging.trace("Resetting daily cash")
                self.reset_daily_cash()

            bt.logging.trace("Recalculating stats")
            self.stats_handler.recalculate_stats()
        except Exception as e:
            bt.logging.error(f"Error in _reconcile_state: {e}")
        bt.logging.trace("Exiting _reconcile_state")

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
                return datetime.fromisoformat(result[0])
        return None

    def update_on_prediction(self, prediction_data):
        with self.lock:
            self.state['miner_lifetime_predictions'] += 1
            self.state['miner_cash'] -= prediction_data['wager']
            self.state['miner_lifetime_wager'] += prediction_data['wager']
            self.state['miner_last_prediction_date'] = torch.tensor(datetime.now(timezone.utc).timestamp(), dtype=torch.float64)
            self.save_state()

    def update_on_game_result(self, result_data):
        with self.lock:
            if result_data['outcome'] == 'win':
                self.state['miner_lifetime_wins'] += 1
                self.state['miner_cash'] += result_data['earnings']
                self.state['miner_lifetime_earnings'] += result_data['earnings']
            else:
                self.state['miner_lifetime_losses'] += 1
            
            total_games = self.state['miner_lifetime_wins'] + self.state['miner_lifetime_losses']
            if total_games > 0:
                self.state['miner_win_loss_ratio'] = self.state['miner_lifetime_wins'] / total_games
            
            self.save_state()

    def reset_daily_cash(self):
        with self.lock:
            self.state['miner_cash'] = torch.tensor(1000, dtype=torch.float32)
            self.state['last_daily_reset'] = torch.tensor(datetime.now(pytz.utc).timestamp(), dtype=torch.float64)
            self.save_state()
        self.stats_handler.recalculate_stats()

    def get_stats(self):
        bt.logging.trace("Entering get_stats method")
        with self.lock:
            stats = {
                "miner_hotkey": self.miner_hotkey,
                "miner_uid": self.miner_uid,
                "miner_cash": float(self.state['miner_cash'].item()),
                "miner_lifetime_earnings": float(self.state['miner_lifetime_earnings'].item()),
                "miner_lifetime_wager": float(self.state['miner_lifetime_wager'].item()),
                "miner_current_incentive": float(self.state['miner_current_incentive'].item()),
                "miner_rank": int(self.state['miner_rank'].item()),
                "miner_lifetime_predictions": int(self.state['miner_lifetime_predictions'].item()),
                "miner_lifetime_wins": int(self.state['miner_lifetime_wins'].item()),
                "miner_lifetime_losses": int(self.state['miner_lifetime_losses'].item()),
                "miner_win_loss_ratio": float(self.state['miner_win_loss_ratio'].item()),
                "miner_last_prediction_date": datetime.fromtimestamp(self.state['miner_last_prediction_date'].item(), tz=timezone.utc).isoformat(),
                "last_daily_reset": datetime.fromtimestamp(self.state['last_daily_reset'].item(), tz=timezone.utc).isoformat()
            }
        bt.logging.trace(f"Returning stats: {stats}")
        bt.logging.trace("Exiting get_stats method")
        return stats

    def _calculate_win_loss_ratio(self):
        total_games = self.state['miner_lifetime_wins'] + self.state['miner_lifetime_losses']
        if total_games == 0:
            return 0
        return round((self.state['miner_lifetime_wins'] / total_games).item(), 3)

    def periodic_db_update(self):
        bt.logging.trace("Performing periodic database update")
        try:
            self.save_state()
            if self.stats_handler:
                self.stats_handler.recalculate_stats()
        except Exception as e:
            bt.logging.error(f"Error in periodic_db_update: {e}")

    def deduct_wager(self, amount):
        with self.lock:
            current_cash = self.state['miner_cash'].item()
            if current_cash >= amount:
                self.state['miner_cash'] = torch.tensor(current_cash - amount, dtype=torch.float32)
                bt.logging.info(f"Deducted wager of {amount}. New balance: {self.state['miner_cash'].item()}")
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

    def return_miner_stats(self, miner_hotkey):
        if not self.db_manager:
            return None
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM miner_stats WHERE miner_hotkey = ?", (miner_hotkey,))
            row = cursor.fetchone()
            if row:
                return row
        return None

    def init_miner_row(self):
        if not self.db_manager:
            return None
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM miner_stats WHERE miner_hotkey = ?", (self.state_manager.miner_hotkey,))
            existing_row = cursor.fetchone()
            
            if existing_row:
                bt.logging.info(f"Miner row already exists for hotkey: {self.state_manager.miner_hotkey}")
                return existing_row
            
            bt.logging.info(f"Initializing new miner row for hotkey: {self.state_manager.miner_hotkey}")
            cursor.execute("""
                INSERT INTO miner_stats 
                (miner_hotkey, miner_uid, miner_cash, miner_lifetime_earnings, 
                miner_lifetime_wager, miner_current_incentive, miner_rank,
                miner_lifetime_predictions, miner_lifetime_wins, miner_lifetime_losses, 
                miner_win_loss_ratio, miner_last_prediction_date) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.state_manager.miner_hotkey,
                self.state_manager.miner_uid,
                1000.0,  # miner_cash
                0.0,     # miner_lifetime_earnings
                0.0,     # miner_lifetime_wager
                0.0,     # miner_current_incentive
                0,       # miner_rank
                0,       # miner_lifetime_predictions
                0,       # miner_lifetime_wins
                0,       # miner_lifetime_losses
                0.0,     # miner_win_loss_ratio
                datetime.now(timezone.utc).isoformat()  # miner_last_prediction_date
            ))
        
        return self.return_miner_stats(self.state_manager.miner_hotkey)

    def update_miner_row(self, stats):
        if not self.db_manager:
            return
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("""
                UPDATE miner_stats
                SET miner_cash = ?,
                    miner_lifetime_earnings = ?,
                    miner_lifetime_wager = ?,
                    miner_current_incentive = ?,
                    miner_rank = ?,
                    miner_lifetime_predictions = ?,
                    miner_lifetime_wins = ?,
                    miner_lifetime_losses = ?,
                    miner_win_loss_ratio = ?,
                    miner_last_prediction_date = ?,
                    last_daily_reset = ?
                WHERE miner_hotkey = ?
            """, (
                float(stats['miner_cash']),
                float(stats['miner_lifetime_earnings']),
                float(stats['miner_lifetime_wager']),
                float(stats['miner_current_incentive']),
                int(stats['miner_rank']),
                int(stats['miner_lifetime_predictions']),
                int(stats['miner_lifetime_wins']),
                int(stats['miner_lifetime_losses']),
                float(stats['miner_win_loss_ratio']),
                stats['miner_last_prediction_date'],
                stats['last_daily_reset'],
                stats['miner_hotkey']
            ))

    def recalculate_stats(self):
        bt.logging.trace("Entering recalculate_stats method")
        if not self.db_manager:
            bt.logging.trace("Skipping recalculate_stats (db_manager not available)")
            return

        start_time = time.time()
        timeout = 30

        try:
            with self.db_manager.get_cursor() as cursor:
                bt.logging.trace("Executing SELECT query for recalculating stats")
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as total_wins,
                        SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as total_losses,
                        SUM(CASE WHEN outcome = 'win' THEN wager * teamAodds ELSE 0 END) as total_earnings
                    FROM predictions
                    WHERE minerID = ?
                """, (self.state_manager.miner_uid,))
                bt.logging.trace("SELECT query executed")
                
                bt.logging.trace("Fetching result")
                result = cursor.fetchone()
                bt.logging.trace(f"Query result: {result}")

                if result:
                    bt.logging.trace("Updating state with new stats")
                    with self.state_manager.lock:
                        bt.logging.trace("Acquired lock for updating state")
                        self.state_manager.state['miner_lifetime_predictions'] = torch.tensor(result[0] or 0, dtype=torch.int64)
                        self.state_manager.state['miner_lifetime_wins'] = torch.tensor(result[1] or 0, dtype=torch.int64)
                        self.state_manager.state['miner_lifetime_losses'] = torch.tensor(result[2] or 0, dtype=torch.int64)
                        self.state_manager.state['miner_lifetime_earnings'] = torch.tensor(result[3] or 0.0, dtype=torch.float32)
                        bt.logging.trace("State updated with new stats")
                    bt.logging.trace("Released lock after updating state")
                    
                    bt.logging.trace("Saving updated state")
                    self.state_manager.save_state()
                    bt.logging.trace("Updated state saved")
                else:
                    bt.logging.trace("No results found for recalculating stats")

                if time.time() - start_time > timeout:
                    bt.logging.error(f"recalculate_stats timed out after {timeout} seconds")
                    return

        except Exception as e:
            bt.logging.error(f"Error in recalculate_stats: {e}")

        bt.logging.trace("Exiting recalculate_stats method")

    def reset_daily_cash_on_startup(self):
        self.state_manager._reconcile_state()