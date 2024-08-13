import threading
import traceback
from typing import Dict, Any
from psycopg2.extras import RealDictCursor
import bittensor as bt
from datetime import datetime, timezone, timedelta

class MinerStatsHandler:
    def __init__(self, state_manager):
        # bt.logging.trace("Initializing MinerStatsHandler")
        self.state_manager = state_manager
        self.db_manager = state_manager.db_manager
        self.stats = self.state_manager.state
        if 'miner_current_incentive' not in self.stats:
            self.stats['miner_current_incentive'] = 0.0
        self.lock = threading.Lock()
        self.update_stats_from_predictions()
        # bt.logging.trace("MinerStatsHandler initialization complete")

    def load_stats_from_state(self):
        with self.lock:
            self.stats = self.state_manager.get_stats()

    def update_stats(self, new_stats: Dict[str, Any]):
        with self.lock:
            self.stats.update(new_stats)
            self.state_manager.update_state(new_stats)

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return self.stats.copy()

    def update_on_prediction(self, prediction_data: Dict[str, Any]):
        with self.lock:
            self.stats['miner_lifetime_predictions'] = self.stats.get('miner_lifetime_predictions', 0) + 1
            self.stats['miner_lifetime_wager'] = self.stats.get('miner_lifetime_wager', 0) + prediction_data['wager']
            self.stats['miner_last_prediction_date'] = prediction_data['predictionDate']
            self.state_manager.update_state(self.stats)

    def update_on_game_result(self, result_data: Dict[str, Any]):
        with self.lock:
            if result_data['outcome'] == 'Wager Won':
                self.stats['miner_lifetime_wins'] = self.stats.get('miner_lifetime_wins', 0) + 1
                self.stats['miner_lifetime_earnings'] = self.stats.get('miner_lifetime_earnings', 0) + result_data['earnings']
            elif result_data['outcome'] == 'Wager Lost':
                self.stats['miner_lifetime_losses'] = self.stats.get('miner_lifetime_losses', 0) + 1
            
            self.stats['miner_lifetime_wager'] = self.stats.get('miner_lifetime_wager', 0) + result_data['wager']
            self.update_win_loss_ratio()
            self.state_manager.update_state(self.stats)

    def update_win_loss_ratio(self):
        total_games = self.stats['miner_lifetime_wins'] + self.stats['miner_lifetime_losses']
        self.stats['miner_win_loss_ratio'] = self.stats['miner_lifetime_wins'] / total_games if total_games > 0 else 0.0

    def update_current_incentive(self, incentive: float):
        with self.lock:
            self.stats['miner_current_incentive'] = incentive
            self.state_manager.update_state({'miner_current_incentive': incentive})

    def get_current_incentive(self) -> float:
        return self.stats['miner_current_incentive']

    def deduct_wager(self, amount: float) -> bool:
        with self.lock:
            if self.stats['miner_cash'] >= amount:
                self.stats['miner_cash'] -= amount
                self.state_manager.update_state({'miner_cash': self.stats['miner_cash']})
                return True
            return False

    def reset_daily_cash(self):
        with self.lock:
            self.stats['miner_cash'] = self.state_manager.DAILY_CASH
            self.stats['last_daily_reset'] = datetime.now(timezone.utc).isoformat()
            self.state_manager.update_state(self.stats)
            print(f"DEBUG: Reset daily cash to {self.state_manager.DAILY_CASH}")

    def update_stats_from_predictions(self):
        print("DEBUG: Updating miner stats from all predictions")
        with self.lock:
            miner_uid = self.state_manager.miner_uid
            print(f"DEBUG: Miner UID: {miner_uid}")
            if miner_uid is None:
                return

            query = """
            SELECT 
                p.predictedOutcome, p.outcome, p.wager, p.teamAodds, p.teamBodds, p.tieOdds, p.predictionDate,
                g.teamA, g.teamB
            FROM predictions p
            JOIN games g ON p.teamGameID = g.externalID
            WHERE p.minerID = %s
            """
            
            conn, cur = self.db_manager.connection_pool.getconn(), None
            try:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(query, (miner_uid,))
                predictions = cur.fetchall()
                print(f"DEBUG: Fetched {len(predictions)} predictions")

                total_predictions = 0
                total_wins = 0
                total_losses = 0
                total_wager = 0
                total_earnings = 0
                last_prediction_date = None
                daily_wager = 0
                today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                
                for pred in predictions:
                    print(f"DEBUG: Processing prediction: {pred}")
                    total_predictions += 1
                    total_wager += pred['wager']
                    print(f"DEBUG: Total wager so far: {total_wager}")

                    pred_date = pred.get('predictiondate')
                    if pred_date:
                        pred_date = datetime.fromisoformat(pred_date)
                        if pred_date >= today:
                            daily_wager += pred['wager']
                        if last_prediction_date is None or pred_date > last_prediction_date:
                            last_prediction_date = pred_date

                    print(f"DEBUG: Prediction details - Outcome: {pred['outcome']}, Predicted: {pred['predictedoutcome']}, Wager: {pred['wager']}, Odds: {pred['teamaodds']}/{pred['teambodds']}/{pred['tieodds']}, Team A: {pred['teama']}, Team B: {pred['teamb']}")
                    if 'Wager Won' in pred['outcome']:
                        total_wins += 1
                        if pred['predictedoutcome'] == pred['teama']:
                            payout = pred['wager'] * pred['teamaodds']
                        elif pred['predictedoutcome'] == pred['teamb']:
                            payout = pred['wager'] * pred['teambodds']
                        elif pred['predictedoutcome'] == "Tie":
                            payout = pred['wager'] * pred['tieodds']
                        else:
                            payout = 0
                        print(f"DEBUG: Wager won. Payout: {payout}")
                        total_earnings += payout
                    elif 'Wager Lost' in pred['outcome']:
                        total_losses += 1
                        print(f"DEBUG: Wager lost. Payout: 0")
                    else:
                        print(f"DEBUG: Unfinished wager. Outcome: {pred['outcome']}")

                    print(f"DEBUG: Total earnings so far: {total_earnings}")

                # Calculate current cash
                current_cash = self.state_manager.DAILY_CASH - daily_wager

                print(f"DEBUG: Final stats - Total predictions: {total_predictions}, Total wins: {total_wins}, Total losses: {total_losses}")
                print(f"DEBUG: Final stats - Total wager: {total_wager}, Total earnings: {total_earnings}")

                self.stats.update({
                    'miner_lifetime_predictions': total_predictions,
                    'miner_lifetime_wins': total_wins,
                    'miner_lifetime_losses': total_losses,
                    'miner_lifetime_wager': total_wager,
                    'miner_lifetime_earnings': total_earnings,
                    'miner_last_prediction_date': last_prediction_date.isoformat() if last_prediction_date else None,
                    'miner_cash': current_cash
                })
                self.update_win_loss_ratio()
                self.state_manager.update_state(self.stats)
                print(f"DEBUG: Updated stats: {self.stats}")

            except Exception as e:
                print(f"DEBUG: Error updating stats from predictions: {str(e)}")
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
            finally:
                if cur:
                    cur.close()
                if conn:
                    self.db_manager.connection_pool.putconn(conn)

    def check_and_reset_daily_cash(self):
        with self.lock:
            last_reset = datetime.fromisoformat(self.stats.get('last_daily_reset', '2000-01-01T00:00:00+00:00'))
            now = datetime.now(timezone.utc)
            if now.date() > last_reset.date():
                self.reset_daily_cash()
                self.update_stats_from_predictions()  # Recalculate stats after reset

    def reset_daily_cash(self):
        with self.lock:
            self.stats['miner_cash'] = self.state_manager.DAILY_CASH
            self.stats['last_daily_reset'] = datetime.now(timezone.utc).isoformat()
            self.state_manager.update_state(self.stats)
            print(f"DEBUG: Reset daily cash to {self.state_manager.DAILY_CASH}")

    def initialize_default_stats(self):
        default_stats = {
            'miner_lifetime_predictions': 0,
            'miner_lifetime_wins': 0,
            'miner_lifetime_losses': 0,
            'miner_lifetime_wager': 0.0,
            'miner_lifetime_earnings': 0.0,
            'miner_win_loss_ratio': 0.0,
            'miner_last_prediction_date': None
        }
        self.stats.update(default_stats)
        self.state_manager.update_state(default_stats)
        # bt.logging.info("Initialized stats with default values")

class MinerStateManager:
    DAILY_CASH = 1000.0

    def __init__(self, db_manager, miner_hotkey: str, miner_uid: str):
        self.db_manager = db_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = miner_uid if miner_uid != "default" else None
        self.state = self.load_state()

    def load_state(self) -> Dict[str, Any]:
        # bt.logging.trace("Loading miner state from database")
        query = "SELECT * FROM miner_stats WHERE miner_hotkey = %s"
        conn, cur = self.db_manager.connection_pool.getconn(), None
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, (self.miner_hotkey,))
            result = cur.fetchone()
            if result:
                return result
            else:
                return self.initialize_state()
        finally:
            if cur:
                cur.close()
            if conn:
                self.db_manager.connection_pool.putconn(conn)

    def initialize_state(self) -> Dict[str, Any]:
        # bt.logging.trace("Initializing new miner state in database")
        initial_state = {
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
            'last_daily_reset': datetime.now(timezone.utc).isoformat()
        }
        self.save_state(initial_state)
        return initial_state

    def save_state(self, state: Dict[str, Any] = None):
        # bt.logging.info("Saving miner state")
        if state is None:
            state = self.state
        try:
            query = """
            INSERT INTO miner_stats (
                miner_hotkey, miner_uid, miner_cash, miner_current_incentive, 
                miner_last_prediction_date, miner_lifetime_earnings, miner_lifetime_wager, 
                miner_lifetime_predictions, miner_lifetime_wins, miner_lifetime_losses, 
                miner_win_loss_ratio, last_daily_reset
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (miner_hotkey) DO UPDATE SET
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
            params = (
                self.miner_hotkey,
                self.miner_uid if self.miner_uid is not None else None,
                state.get('miner_cash', 0),
                state.get('miner_current_incentive', 0),
                state.get('miner_last_prediction_date'),
                state.get('miner_lifetime_earnings', 0),
                state.get('miner_lifetime_wager', 0),
                state.get('miner_lifetime_predictions', 0),
                state.get('miner_lifetime_wins', 0),
                state.get('miner_lifetime_losses', 0),
                state.get('miner_win_loss_ratio', 0),
                state.get('last_daily_reset')
            )
            conn, cur = self.db_manager.connection_pool.getconn(), None
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                conn.commit()
                # bt.logging.info("Miner state saved successfully")
            finally:
                if cur:
                    cur.close()
                if conn:
                    self.db_manager.connection_pool.putconn(conn)
        except Exception as e:
            # bt.logging.error(f"Error saving miner state: {e}")
            # bt.logging.error(traceback.format_exc())
            raise

    def update_state(self, new_state: Dict[str, Any]):
        # bt.logging.trace("Updating miner state in database")
        set_clause = ', '.join([f"{k} = %s" for k in new_state.keys()])
        query = f"""
        UPDATE miner_stats
        SET {set_clause}
        WHERE miner_hotkey = %s
        """
        params = list(new_state.values()) + [self.miner_hotkey]
        conn, cur = self.db_manager.connection_pool.getconn(), None
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
        finally:
            if cur:
                cur.close()
            if conn:
                self.db_manager.connection_pool.putconn(conn)

    def get_stats(self) -> Dict[str, Any]:
        # bt.logging.trace("Getting miner stats from database")
        query = "SELECT * FROM miner_stats WHERE miner_hotkey = %s"
        conn, cur = self.db_manager.connection_pool.getconn(), None
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, (self.miner_hotkey,))
            result = cur.fetchall()
            return result[0] if result else {}
        finally:
            if cur:
                cur.close()
            if conn:
                self.db_manager.connection_pool.putconn(conn)