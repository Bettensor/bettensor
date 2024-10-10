import threading
import traceback
from typing import Dict, Any
from psycopg2.extras import RealDictCursor
import warnings
from eth_utils.exceptions import ValidationError
import json
import os
import bittensor as bt
from datetime import datetime, timezone, timedelta

class MinerStatsHandler:
    def __init__(self, state_manager=None):
        self.state_manager = state_manager
        self.db_manager = state_manager.db_manager if state_manager else None
        self.stats = self.state_manager.state if state_manager else {}
        if "miner_current_incentive" not in self.stats:
            self.stats["miner_current_incentive"] = 0.0
        self.lock = threading.Lock()
        self.validator_confirmation_file = "validator_confirmation_dict.json"
        self.validator_confirmation_dict = self.load_validator_confirmation_dict()
        if state_manager and state_manager.miner_uid:
            self.update_stats_from_predictions()

    def load_validator_confirmation_dict(self) -> Dict[str, Any]:
        if os.path.exists(self.validator_confirmation_file):
            try:
                with open(self.validator_confirmation_file, "r") as f:
                    data = json.load(f)
                    bt.logging.info("Loaded validator_confirmation_dict from JSON file.")
                    return data
            except Exception as e:
                bt.logging.error(f"Error loading validator_confirmation_dict: {e}")
                return {}
        else:
            bt.logging.info("validator_confirmation_dict JSON file not found. Initializing empty dictionary.")
            return {}

    def save_validator_confirmation_dict(self):
        try:
            with open(self.validator_confirmation_file, "w") as f:
                json.dump(self.validator_confirmation_dict, f, indent=4)
                bt.logging.info("Saved validator_confirmation_dict to JSON file.")
        except Exception as e:
            bt.logging.error(f"Error saving validator_confirmation_dict: {e}")

    def load_stats_from_state(self):
        with self.lock:
            self.stats = self.state_manager.get_stats()
            self.validator_confirmation_dict = self.load_validator_confirmation_dict()

    def update_stats(self, new_stats: Dict[str, Any]):
        with self.lock:
            self.stats.update(new_stats)
            self.state_manager.update_state(new_stats)
            self.db_manager.update_miner_activity(self.state_manager.miner_uid)

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return self.stats.copy()

    def update_on_prediction(self, prediction_data: Dict[str, Any]):
        with self.lock:
            self.stats["miner_lifetime_predictions"] = (
                self.stats.get("miner_lifetime_predictions", 0) + 1
            )
            self.stats["miner_lifetime_wager"] = (
                self.stats.get("miner_lifetime_wager", 0) + prediction_data["wager"]
            )
            self.stats["miner_last_prediction_date"] = prediction_data["predictionDate"]
            self.state_manager.update_state(self.stats)
            self.save_state()

    def update_on_game_result(self, result_data: Dict[str, Any]):
        with self.lock:
            prediction_id = result_data.get("prediction_id")
            if prediction_id is None:
                bt.logging.warning("No prediction_id provided in result_data")
                return

            # Check if this prediction has already been processed
            query = "SELECT processed FROM predictions WHERE predictionID = %s"
            result = self.db_manager.execute_query(query, (prediction_id,))

            if not result or result[0]["processed"]:
                bt.logging.debug(
                    f"Prediction {prediction_id} already processed or not found"
                )
                return

            if result_data["outcome"] == "Wager Won":
                self.stats["miner_lifetime_wins"] = (
                    self.stats.get("miner_lifetime_wins", 0) + 1
                )
                self.stats["miner_lifetime_earnings"] = (
                    self.stats.get("miner_lifetime_earnings", 0)
                    + result_data["earnings"]
                )
            elif result_data["outcome"] == "Wager Lost":
                self.stats["miner_lifetime_losses"] = (
                    self.stats.get("miner_lifetime_losses", 0) + 1
                )

            self.stats["miner_lifetime_wager"] = (
                self.stats.get("miner_lifetime_wager", 0) + result_data["wager"]
            )
            self.update_win_loss_ratio()
            self.state_manager.update_state(self.stats)
            self.save_state()

            # Mark the prediction as processed
            update_query = (
                "UPDATE predictions SET processed = TRUE WHERE predictionID = %s"
            )
            self.db_manager.execute_query(update_query, (prediction_id,))

    def update_win_loss_ratio(self):
        total_games = (
            self.stats["miner_lifetime_wins"] + self.stats["miner_lifetime_losses"]
        )
        self.stats["miner_win_loss_ratio"] = (
            self.stats["miner_lifetime_wins"] / total_games if total_games > 0 else 0.0
        )

    def update_current_incentive(self, incentive: float):
        with self.lock:
            self.stats["miner_current_incentive"] = incentive
            self.state_manager.update_state({"miner_current_incentive": incentive})
            self.save_state()

    def get_current_incentive(self) -> float:
        return self.stats["miner_current_incentive"]

    def deduct_wager(self, amount: float) -> bool:
        with self.lock:
            if self.stats["miner_cash"] >= amount:
                self.stats["miner_cash"] -= amount
                self.state_manager.update_state(
                    {"miner_cash": self.stats["miner_cash"]}
                )
                self.save_state()
                return True
            return False

    def reset_daily_cash(self):
        with self.lock:
            self.stats["miner_cash"] = self.state_manager.DAILY_CASH
            self.stats["last_daily_reset"] = datetime.now(timezone.utc).isoformat()
            self.state_manager.update_state(self.stats)
            self.save_state()
            bt.logging.info(f"Reset daily cash to {self.state_manager.DAILY_CASH}")

    def update_stats_from_predictions(self):
        with self.lock:
            miner_uid = self.state_manager.miner_uid
            if miner_uid is None:
                return

            query = """
            SELECT 
                p.predicted_outcome, p.outcome, p.wager, p.team_a_odds, p.team_b_odds, p.tie_odds, p.prediction_date,
                g.team_a, g.team_b
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.miner_uid = %s
            """

            conn, cur = self.db_manager.connection_pool.getconn(), None
            try:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(query, (miner_uid,))
                predictions = cur.fetchall()

                total_predictions = 0
                total_wins = 0
                total_losses = 0
                total_wager = 0
                total_earnings = 0
                last_prediction_date = None
                daily_wager = 0
                today = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

                for pred in predictions:
                    total_predictions += 1
                    total_wager += pred["wager"]

                    pred_date = pred.get("prediction_date")
                    if pred_date:
                        pred_date = datetime.fromisoformat(pred_date)
                        if pred_date >= today:
                            daily_wager += pred["wager"]
                        if (
                            last_prediction_date is None
                            or pred_date > last_prediction_date
                        ):
                            last_prediction_date = pred_date

                    if "Wager Won" in pred["outcome"]:
                        total_wins += 1
                        if pred["predicted_outcome"] == pred["team_a"]:
                            payout = pred["wager"] * pred["team_a_odds"]
                        elif pred["predicted_outcome"] == pred["team_b"]:
                            payout = pred["wager"] * pred["team_b_odds"]
                        elif pred["predicted_outcome"] == "Tie":
                            payout = pred["wager"] * pred["tie_odds"]
                        else:
                            payout = 0
                        total_earnings += payout
                    elif "Wager Lost" in pred["outcome"]:
                        total_losses += 1

                # Calculate current cash
                current_cash = self.state_manager.DAILY_CASH - daily_wager

                self.stats.update(
                    {
                        "miner_lifetime_predictions": total_predictions,
                        "miner_lifetime_wins": total_wins,
                        "miner_lifetime_losses": total_losses,
                        "miner_lifetime_wager": total_wager,
                        "miner_lifetime_earnings": total_earnings,
                        "miner_last_prediction_date": last_prediction_date.isoformat()
                        if last_prediction_date
                        else None,
                        "miner_cash": current_cash,
                    }
                )
                self.update_win_loss_ratio()
                self.state_manager.update_state(self.stats)
                self.save_state()

            except Exception as e:
                bt.logging.error(f"Error updating stats from predictions: {str(e)}")
                bt.logging.error(traceback.format_exc())
            finally:
                if cur:
                    cur.close()
                if conn:
                    self.db_manager.connection_pool.putconn(conn)

    def check_and_reset_daily_cash(self):
        with self.lock:
            last_reset_str = self.stats.get("last_daily_reset")
            if last_reset_str is None:
                bt.logging.warning(
                    "last_daily_reset is None, initializing to current time"
                )
                last_reset = datetime.now(timezone.utc)
                self.stats["last_daily_reset"] = last_reset.isoformat()
                self.state_manager.update_state(self.stats)
            else:
                last_reset = datetime.fromisoformat(last_reset_str)
            now = datetime.now(timezone.utc)
            if now.date() > last_reset.date():
                self.reset_daily_cash()
                self.update_stats_from_predictions()  # Recalculate stats after reset

    def reset_daily_cash(self):
        with self.lock:
            self.stats["miner_cash"] = self.state_manager.DAILY_CASH
            self.stats["last_daily_reset"] = datetime.now(timezone.utc).isoformat()
            self.state_manager.update_state(self.stats)
            print(f"DEBUG: Reset daily cash to {self.state_manager.DAILY_CASH}")

    def initialize_default_stats(self):
        default_stats = {
            "miner_lifetime_predictions": 0,
            "miner_lifetime_wins": 0,
            "miner_lifetime_losses": 0,
            "miner_lifetime_wager": 0.0,
            "miner_lifetime_earnings": 0.0,
            "miner_win_loss_ratio": 0.0,
            "miner_last_prediction_date": None,
            "last_daily_reset": datetime.now(timezone.utc).isoformat(),
        }
        self.stats.update(default_stats)
        self.state_manager.update_state(default_stats)
        self.save_state()

    def get_miner_cash(self) -> float:
        with self.lock:
            cash = self.stats.get("miner_cash", 0.0)
            bt.logging.info(f"Retrieved miner cash: {cash}")
            return cash

    def update_miner_earnings(self, earnings: float):
        with self.lock:
            self.stats["miner_lifetime_earnings"] = (
                self.stats.get("miner_lifetime_earnings", 0) + earnings
            )
            self.state_manager.update_state(
                {"miner_lifetime_earnings": self.stats["miner_lifetime_earnings"]}
            )
            self.save_state()

    def update_miner_cash(self, amount: float):
        with self.lock:
            self.stats["miner_cash"] = self.stats.get("miner_cash", 0) + amount
            self.state_manager.update_state({"miner_cash": self.stats["miner_cash"]})
            self.save_state()

    def increment_miner_wins(self):
        with self.lock:
            self.stats["miner_lifetime_wins"] = (
                self.stats.get("miner_lifetime_wins", 0) + 1
            )
            self.state_manager.update_state(
                {"miner_lifetime_wins": self.stats["miner_lifetime_wins"]}
            )
            self.save_state()

    def increment_miner_losses(self):
        with self.lock:
            self.stats["miner_lifetime_losses"] = (
                self.stats.get("miner_lifetime_losses", 0) + 1
            )
            self.state_manager.update_state(
                {"miner_lifetime_losses": self.stats["miner_lifetime_losses"]}
            )
            self.save_state()

    def save_state(self):
        self.state_manager.save_state(self.stats)
        self.save_validator_confirmation_dict()

    # Add methods to handle validator_confirmation_dict updates
    def add_validator_confirmation(self, prediction_id: str, validator_hotkey: str):
        with self.lock:
            if prediction_id not in self.validator_confirmation_dict:
                self.validator_confirmation_dict[prediction_id] = {"validators": {validator_hotkey: {'confirmed': False}}}
                bt.logging.info(f"Added new validator confirmation for prediction_id: {prediction_id}")
            elif validator_hotkey not in self.validator_confirmation_dict[prediction_id]["validators"]:
                self.validator_confirmation_dict[prediction_id]["validators"][validator_hotkey] = {'confirmed': False}
                bt.logging.info(f"Added new validator hotkey: {validator_hotkey} for prediction_id: {prediction_id}")
            self.save_validator_confirmation_dict()

class MinerStateManager:
    DAILY_CASH = 1000.0

    def __init__(self, db_manager, miner_hotkey: str, miner_uid: str):
        self.db_manager = db_manager
        self.miner_hotkey = miner_hotkey
        self.miner_uid = miner_uid if miner_uid != "default" else None
        self.state = self.load_state()
    
    def get_stats(self):
        return self.state.copy()

    def load_state(self) -> Dict[str, Any]:
        if not self.miner_uid:
            return {}

        query = "SELECT * FROM miner_stats WHERE miner_hotkey = %s"
        conn, cur = self.db_manager.connection_pool.getconn(), None
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, (self.miner_hotkey,))
            result = cur.fetchone()
            return result if result else {}
        finally:
            if cur:
                cur.close()
            if conn:
                self.db_manager.connection_pool.putconn(conn)

    def initialize_state(self) -> Dict[str, Any]:
        if not self.miner_uid:
            return {}

        bt.logging.info("Initializing new miner state in database")
        initial_state = {
            "miner_hotkey": self.miner_hotkey,
            "miner_uid": self.miner_uid,
            "miner_cash": self.DAILY_CASH,
            "miner_current_incentive": 0.0,
            "miner_last_prediction_date": None,
            "miner_lifetime_earnings": 0.0,
            "miner_lifetime_wager": 0.0,
            "miner_lifetime_predictions": 0,
            "miner_lifetime_wins": 0,
            "miner_lifetime_losses": 0,
            "miner_win_loss_ratio": 0.0,
            "last_daily_reset": datetime.now(timezone.utc).isoformat(),
        }
        self.save_state(initial_state)
        return initial_state

    def save_state(self, state: Dict[str, Any] = None):
        if not self.miner_uid or not state:
            return

        bt.logging.info("Saving miner state")
        try:
            query = """
            INSERT INTO miner_stats (
                miner_hotkey, miner_uid, miner_cash, miner_current_incentive, 
                miner_last_prediction_date, miner_lifetime_earnings, miner_lifetime_wager_amount, 
                miner_lifetime_predictions, miner_lifetime_wins, miner_lifetime_losses, 
                miner_win_loss_ratio, last_daily_reset
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (miner_hotkey) DO UPDATE SET
                miner_uid = EXCLUDED.miner_uid,
                miner_cash = EXCLUDED.miner_cash,
                miner_current_incentive = EXCLUDED.miner_current_incentive,
                miner_last_prediction_date = EXCLUDED.miner_last_prediction_date,
                miner_lifetime_earnings = EXCLUDED.miner_lifetime_earnings,
                miner_lifetime_wager_amount = EXCLUDED.miner_lifetime_wager_amount,
                miner_lifetime_predictions = EXCLUDED.miner_lifetime_predictions,
                miner_lifetime_wins = EXCLUDED.miner_lifetime_wins,
                miner_lifetime_losses = EXCLUDED.miner_lifetime_losses,
                miner_win_loss_ratio = EXCLUDED.miner_win_loss_ratio,
                last_daily_reset = EXCLUDED.last_daily_reset
            """
            params = (
                self.miner_hotkey,
                self.miner_uid,
                state.get("miner_cash", 0),
                state.get("miner_current_incentive", 0),
                state.get("miner_last_prediction_date"),
                state.get("miner_lifetime_earnings", 0),
                state.get("miner_lifetime_wager", 0),
                state.get("miner_lifetime_predictions", 0),
                state.get("miner_lifetime_wins", 0),
                state.get("miner_lifetime_losses", 0),
                state.get("miner_win_loss_ratio", 0),
                state.get("last_daily_reset"),
            )
            self.db_manager.execute_query(query, params)
            bt.logging.info("Miner state saved successfully")
        except Exception as e:
            bt.logging.error(f"Error saving miner state: {e}")
            bt.logging.error(traceback.format_exc())

    def update_state(self, new_state):
        self.state.update(new_state)
        self.save_state(self.state)